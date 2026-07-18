//! Flat Data-Oriented Vulkan render backend.
//!
//! Encapsulates ALL Vulkan resources behind a single struct whose field
//! declaration order dictates drop order.  No resource lifecycle
//! management code lives in `Application` — it delegates to this module
//! exclusively.
//!
//! # Drop Order (reverse declaration = last-to-first)
//!
//! `vulkan` → `swapchain` → `blur_target` → `geometry_pool` →
//! `draw_count_buffers` → `indirect_buffers` → `instance_buffers` →
//! `ubo_buffer`/`ubo_memory` → `skinning_pipeline` → `compute_pipeline` →
//! `skinning_instances` → descriptor pools → `ui_backend` →
//! `pipeline` → `post_process` → `offscreen_target` →
//! `sdr_target` → `bloom_target` → remaining fields

use ash::vk;
use std::ffi::c_void;

use crate::platform::Window;
use crate::renderer::vulkan::{
    bloom::BloomTarget,
    buffer::Buffer,
    compute_cloth::{ComputeClothPipeline, ClothGpuInstance, MAX_CLOTH_INSTANCES},
    compute_cull::ComputeCullPipeline,
    compute_skinning::{self, ComputeSkinningPipeline},
    GeometryPool, OffscreenTarget, Pipeline, PostProcessPipeline, Swapchain, UiBackend,
    VulkanDevice, shadow_pass::ShadowPass,
};
use crate::renderer::vulkan::render_graph::{RenderGraph, ResourceTracker, ResourceHandle, ResourceState};
use crate::ecs::{
    CameraComponent, LightComponent, RenderComponent, TransformComponent,
};
use crate::math::vec::Vec3;

pub struct RenderBackend {
    // === Lifetime 1: base device (last to drop) ===
    pub vulkan: VulkanDevice,
    pub swapchain: Swapchain,

    // === Lifetime 2: render targets ===
    pub blur_target: OffscreenTarget,
    pub bloom_target: BloomTarget,
    pub sdr_target: OffscreenTarget,
    pub offscreen_target: OffscreenTarget,

    // === Lifetime 3: geometry ===
    pub geometry_pool: GeometryPool,

    // === Lifetime 4: multi-buffered GPU buffers ===
    pub draw_count_buffers: [Buffer; 2],
    pub indirect_buffers: [Buffer; 2],
    pub instance_buffers: [Buffer; 2],
    pub instance_mapped: [*mut c_void; 2],

    // === Lifetime 5: uniform buffer ===
    pub ubo_buffer: vk::Buffer,
    pub ubo_memory: vk::DeviceMemory,
    pub ubo_mapped: *mut c_void,

    // === Lifetime 6: compute/skinning pipelines ===
    pub skinning_pipeline: Option<ComputeSkinningPipeline>,
    pub skinning_instances: Vec<compute_skinning::SkinningInstance>,
    pub skinning_descriptor_pool: vk::DescriptorPool,
    pub compute_pipeline: Option<ComputeCullPipeline>,
    pub compute_descriptor_pool: vk::DescriptorPool,
    /// GPU cloth simulation pipeline.  `None` if shaders could not be loaded.
    pub cloth_pipeline: Option<ComputeClothPipeline>,
    /// Per-entity GPU cloth instances.  Pre-allocated to `MAX_CLOTH_INSTANCES`.
    pub cloth_instances: Vec<ClothGpuInstance>,

    // === Lifetime 7: descriptor pools and sets ===
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: [vk::DescriptorSet; 2],
    pub global_texture_descriptor_sets: [vk::DescriptorSet; 2],
    pub post_process_descriptor_pool: vk::DescriptorPool,
    pub tonemap_descriptor_set: vk::DescriptorSet,
    pub bloom_descriptor_sets: Vec<vk::DescriptorSet>,
    pub blur_descriptor_sets: Vec<vk::DescriptorSet>,

    // === Lifetime 8: UI ===
    pub ui_backend: UiBackend,

    // === Lifetime 9: pipelines ===
    pub pipeline: Option<Pipeline>,
    pub post_process: PostProcessPipeline,
    pub shadow_pass: Option<ShadowPass>,

    // === Lifetime 10: per-mesh descriptor sets ===
    pub compute_descriptor_sets: Vec<Option<vk::DescriptorSet>>,
    pub material_descriptor_sets: Vec<Option<vk::DescriptorSet>>,

    // === Lifetime 11: misc state ===
    pub offscreen_texture_id: u32,
    pub current_frame: usize,
    pub bloom_threshold: f32,
    pub resource_tracker: ResourceTracker,

    // === Lifetime 12: pre-allocated CPU scratch buffers ===
    /// Pre-allocated buffer for instance data built each frame.
    /// Sized to `max_instances` (100_000) at construction.  `clear()`
    /// + `push()` reuses the capacity every frame — zero heap allocations
    /// on the hot render path.
    pub instance_data_buffer: Vec<crate::renderer::vulkan::pipeline::InstanceData>,
    
    // === Debug Toggles ===
    pub debug_cull: bool,
}

impl RenderBackend {
    /// Create all Vulkan resources.
    ///
    /// # Safety
    /// Must be called exactly once during engine initialization.
    /// `asset_manager` is mutably borrowed to load default textures and
    /// models into the fresh geometry pool.
    pub fn new(
        window: &Window,
        width: i32,
        height: i32,
        asset_manager: &mut crate::asset_manager::AssetManager,
        ui_font: &crate::ui::font::Font,
    ) -> Option<Self> {
        let vulkan = VulkanDevice::new()?;
        let swapchain = Swapchain::new(&vulkan, window, width as u32, height as u32)?;

        let target_width = swapchain.extent.width;
        let target_height = swapchain.extent.height;

        let offscreen_target = OffscreenTarget::new(
            &vulkan,
            target_width,
            target_height,
            vk::Format::R16G16B16A16_SFLOAT,
        )?;
        let sdr_target = OffscreenTarget::new(
            &vulkan,
            target_width,
            target_height,
            vk::Format::R8G8B8A8_UNORM,
        )?;
        let mip_levels = 6u32;
        let bloom_target = BloomTarget::new(&vulkan, target_width / 2, target_height / 2, mip_levels)?;

        let post_process = PostProcessPipeline::new(
            &vulkan,
            vk::Format::R8G8B8A8_UNORM,
            &asset_manager.vfs,
        )?;

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(4),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(32);
        let post_process_descriptor_pool = unsafe {
            vulkan
                .device
                .create_descriptor_pool(&pool_info, None)
                .ok()?
        };

        let tonemap_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(post_process_descriptor_pool)
            .set_layouts(std::slice::from_ref(&post_process.tonemap_descriptor_set_layout));
        let tonemap_descriptor_set =
            unsafe { vulkan.device.allocate_descriptor_sets(&tonemap_alloc_info).ok()? }[0];

        let mut bloom_layouts = Vec::with_capacity((mip_levels + 1) as usize);
        for _ in 0..=mip_levels {
            bloom_layouts.push(post_process.bloom_descriptor_set_layout);
        }
        let bloom_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(post_process_descriptor_pool)
            .set_layouts(&bloom_layouts);
        let bloom_descriptor_sets =
            unsafe { vulkan.device.allocate_descriptor_sets(&bloom_alloc_info).ok()? };

        let pipeline = Pipeline::new(
            &vulkan,
            vk::Format::R16G16B16A16_SFLOAT,
            &asset_manager.vfs,
            "shaders/vert.spv",
            "shaders/frag.spv",
        );

        let compute_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1000),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2000),
        ];
        let compute_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&compute_pool_sizes)
            .max_sets(1000);
        let compute_descriptor_pool = unsafe {
            vulkan
                .device
                .create_descriptor_pool(&compute_pool_info, None)
                .unwrap()
        };

        let mut compute_pipeline = ComputeCullPipeline::new(&vulkan, &asset_manager.vfs);

        let skinning_pipeline = ComputeSkinningPipeline::new(&vulkan, &asset_manager.vfs);

        let skinning_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1500),
        ];
        let skinning_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&skinning_pool_sizes)
            .max_sets(500);
        let skinning_descriptor_pool = unsafe {
            vulkan
                .device
                .create_descriptor_pool(&skinning_pool_info, None)
                .unwrap()
        };

        let mut descriptor_pool = vk::DescriptorPool::null();
        let mut descriptor_set = [vk::DescriptorSet::null(), vk::DescriptorSet::null()];

        // Load default textures needed for UBO descriptors.
        // These must exist before we bind the descriptor sets below.
        asset_manager.load_checkerboard(&vulkan, "default");
        asset_manager.load_checkerboard(&vulkan, "fallback");
        asset_manager.load_procedural_env(&vulkan, "env_default");
        asset_manager.load_solid_color(&vulkan, "shadow_default", 255, 255, 255, 255);

        let ubo_data = if let Some(pipe) = &pipeline {
            asset_manager.load_checkerboard(&vulkan, "fallback");
            let tex = asset_manager
                .get_texture("default")
                .or_else(|| asset_manager.get_texture("fallback"));
            if let Some(_tex) = tex {
                let pool_sizes = [
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(100),
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1000),
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(10),
                ];
                let pool_info = vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                    .pool_sizes(&pool_sizes)
                    .max_sets(1000);
                descriptor_pool =
                    unsafe { vulkan.device.create_descriptor_pool(&pool_info, None).unwrap() };

                let layouts = [pipe.descriptor_set_layout, pipe.descriptor_set_layout];
                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts);
                let sets = unsafe { vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap() };
                descriptor_set = [sets[0], sets[1]];

                let ubo_size =
                    std::mem::size_of::<crate::renderer::vulkan::pipeline::GlobalUbo>() as u64;
                let ubo_buffer = unsafe {
                    vulkan
                        .device
                        .create_buffer(
                            &vk::BufferCreateInfo::default()
                                .size(ubo_size)
                                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                                .sharing_mode(vk::SharingMode::EXCLUSIVE),
                            None,
                        )
                        .unwrap()
                };
                let mem_req = unsafe { vulkan.device.get_buffer_memory_requirements(ubo_buffer) };
                let mem_type_index = vulkan
                    .find_memory_type(
                        mem_req.memory_type_bits,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
                    .unwrap();
                let alloc_info = vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_req.size)
                    .memory_type_index(mem_type_index);
                let ubo_memory =
                    unsafe { vulkan.device.allocate_memory(&alloc_info, None).unwrap() };
                unsafe {
                    vulkan
                        .device
                        .bind_buffer_memory(ubo_buffer, ubo_memory, 0)
                        .unwrap()
                };

                let ubo_mapped = unsafe {
                    vulkan
                        .device
                        .map_memory(ubo_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                        .unwrap_or(std::ptr::null_mut())
                };

                for set in descriptor_set.iter() {
                    let ubo_info = vk::DescriptorBufferInfo::default()
                        .buffer(ubo_buffer)
                        .offset(0)
                        .range(ubo_size);
                    let env_tex = asset_manager.get_texture("env_default").unwrap();
                    let shadow_tex = asset_manager.get_texture("shadow_default").unwrap();
                    let env_image_info = vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(env_tex.view)
                        .sampler(env_tex.sampler);
                    let shadow_image_info = vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(shadow_tex.view)
                        .sampler(shadow_tex.sampler);
                    let writes = [
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(std::slice::from_ref(&ubo_info)),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(std::slice::from_ref(&env_image_info)),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(2)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(std::slice::from_ref(&shadow_image_info)),
                    ];
                    unsafe { vulkan.device.update_descriptor_sets(&writes, &[]) };
                }

                let counts = [1000u32];
                let mut variable_info =
                    vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
                        .descriptor_counts(&counts);
                let alloc_info_bindless = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(std::slice::from_ref(&pipe.material_descriptor_set_layout))
                    .push_next(&mut variable_info);
                let bindless_set = unsafe {
                    vulkan
                        .device
                        .allocate_descriptor_sets(&alloc_info_bindless)
                        .unwrap()[0]
                };

                let fallback_tex = asset_manager.get_texture("fallback").unwrap();
                let image_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(fallback_tex.view)
                    .sampler(fallback_tex.sampler);
                let write_desc = vk::WriteDescriptorSet::default()
                    .dst_set(bindless_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&image_info));
                unsafe { vulkan.device.update_descriptor_sets(&[write_desc], &[]) };

                (Some((ubo_buffer, ubo_memory, ubo_mapped)), bindless_set)
            } else {
                (None, vk::DescriptorSet::null())
            }
        } else {
            (None, vk::DescriptorSet::null())
        };

        let (ubo_buffer, ubo_memory, ubo_mapped) = ubo_data.0.unwrap_or((
            vk::Buffer::null(),
            vk::DeviceMemory::null(),
            std::ptr::null_mut(),
        ));
        let bindless_set = ubo_data.1;

        let geometry_pool =
            GeometryPool::new(&vulkan, 4_000_000, 12_000_000, 400_000)
                .expect("Failed to create GeometryPool");

        let mut ui_backend = UiBackend::new(&vulkan, swapchain.format.format, &asset_manager.vfs);
        ui_backend.set_font(&vulkan, ui_font);

        let max_instances = 100_000usize;
        let mut instance_bufs = Vec::with_capacity(2);
        let mut instance_maps = Vec::with_capacity(2);
        let mut indirect_bufs = Vec::with_capacity(2);
        let mut draw_count_bufs = Vec::with_capacity(2);

        for _ in 0..2 {
            let ib = Buffer::new(
                &vulkan,
                (max_instances * std::mem::size_of::<crate::renderer::vulkan::pipeline::InstanceData>()) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ).expect("Failed to create instance buffer");
            let mapped = unsafe {
                vulkan
                    .device
                    .map_memory(ib.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .expect("Failed to map instance buffer")
            };
            instance_bufs.push(ib);
            instance_maps.push(mapped);

            let idb = Buffer::new(
                &vulkan,
                (max_instances * std::mem::size_of::<vk::DrawIndexedIndirectCommand>()) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ).expect("Failed to create indirect buffer");
            indirect_bufs.push(idb);

            let dcb = Buffer::new(
                &vulkan,
                4,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                // HOST_VISIBLE | HOST_COHERENT: required so vkCmdFillBuffer can
                // zero the atomic counter each frame without a CPU readback stall.
                // DEVICE_LOCAL is intentionally omitted — counter is written by
                // the GPU compute shader and must be visible to the host barrier.
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ).expect("Failed to create draw count buffer");
            draw_count_bufs.push(dcb);
        }

        let instance_buffers = [instance_bufs.remove(0), instance_bufs.remove(0)];
        let instance_mapped = [instance_maps.remove(0), instance_maps.remove(0)];
        let indirect_buffers = [indirect_bufs.remove(0), indirect_bufs.remove(0)];
        let draw_count_buffers = [draw_count_bufs.remove(0), draw_count_bufs.remove(0)];

        // Bind instance buffers into descriptor set 3
        for i in 0..2 {
            let instance_info = vk::DescriptorBufferInfo::default()
                .buffer(instance_buffers[i].handle)
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let write_desc = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set[i])
                .dst_binding(3)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&instance_info));
            unsafe { vulkan.device.update_descriptor_sets(&[write_desc], &[]) };
        }

        if let Some(compute) = &mut compute_pipeline {
            let layouts = [compute.descriptor_set_layout, compute.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(compute_descriptor_pool)
                .set_layouts(&layouts);
            compute.descriptor_sets =
                unsafe { vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap() };
            for i in 0..2 {
                compute.update_descriptor_set(
                    &vulkan,
                    ubo_buffer,
                    indirect_buffers[i].handle,
                    instance_buffers[i].handle,
                    draw_count_buffers[i].handle,
                    compute.descriptor_sets[i],
                );
            }
        }

        let cloth_pipeline = ComputeClothPipeline::new(&vulkan, &asset_manager.vfs);
        let shadow_pass = ShadowPass::new(&vulkan, &asset_manager.vfs);

        if let Some(shadow) = &shadow_pass {
            for set in descriptor_set.iter() {
                let shadow_image_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(shadow.shadow_view)
                    .sampler(shadow.shadow_sampler);
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&shadow_image_info));
                unsafe { vulkan.device.update_descriptor_sets(&[write], &[]) };
            }
        }

        let blur_target = OffscreenTarget::new(
            &vulkan,
            window.width,
            window.height,
            vk::Format::R16G16B16A16_SFLOAT,
        )
        .unwrap();

        let mut backend = Self {
            vulkan,
            swapchain,
            blur_target,
            bloom_target,
            sdr_target,
            offscreen_target,
            geometry_pool,
            draw_count_buffers,
            indirect_buffers,
            instance_buffers,
            instance_mapped,
            ubo_buffer,
            ubo_memory,
            ubo_mapped,
            skinning_pipeline,
            skinning_instances: Vec::new(),
            skinning_descriptor_pool,
            compute_pipeline,
            compute_descriptor_pool,
            cloth_pipeline,
            cloth_instances: Vec::with_capacity(MAX_CLOTH_INSTANCES),
            descriptor_pool,
            descriptor_sets: descriptor_set,
            global_texture_descriptor_sets: [bindless_set, bindless_set],
            post_process_descriptor_pool,
            tonemap_descriptor_set,
            bloom_descriptor_sets,
            blur_descriptor_sets: Vec::new(),
            ui_backend,
            pipeline,
            post_process,
            shadow_pass,
            compute_descriptor_sets: Vec::new(),
            material_descriptor_sets: Vec::new(),
            offscreen_texture_id: 0,
            current_frame: 0,
            bloom_threshold: 1.0,
            resource_tracker: ResourceTracker::new(),
            instance_data_buffer: Vec::with_capacity(max_instances),
            debug_cull: false,
        };
        backend.update_post_process_descriptors();
        backend.ui_backend.update_user_texture(
            &backend.vulkan,
            backend.offscreen_texture_id,
            backend.sdr_target.color_view,
            backend.sdr_target.sampler,
        );
        Some(backend)
    }

    /// Update bindless texture descriptor array with all loaded textures.
    pub fn update_texture_descriptors(&self, asset_manager: &crate::asset_manager::AssetManager) {
        if asset_manager.next_texture_index == 0 { return; }

        let mut image_infos = Vec::new();
        let fallback = asset_manager.get_texture("fallback").unwrap();
        let fallback_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(fallback.view)
            .sampler(fallback.sampler);

        for i in 0..asset_manager.next_texture_index {
            let tex_name = asset_manager.texture_indices.iter().find(|(_, &idx)| idx == i).map(|(name, _)| name);
            if let Some(name) = tex_name {
                if let Some(tex) = asset_manager.get_texture(name) {
                    image_infos.push(vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(tex.view)
                        .sampler(tex.sampler));
                } else {
                    image_infos.push(fallback_info);
                }
            } else {
                image_infos.push(fallback_info);
            }
        }
        
        if image_infos.is_empty() { return; }

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.global_texture_descriptor_sets[0])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos);

        unsafe {
            self.vulkan.device.update_descriptor_sets(&[write], &[]);
        }
    }

    /// Update post-process descriptor bindings to point at current render targets.
    pub fn update_post_process_descriptors(&self) {
        let mut writes: Vec<vk::WriteDescriptorSet<'_>> = Vec::with_capacity(16);

        let tonemap_color_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(self.offscreen_target.color_view)
            .sampler(self.offscreen_target.sampler)];
        let tonemap_bloom_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(self.bloom_target.full_view)
            .sampler(self.bloom_target.sampler)];
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(self.tonemap_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&tonemap_color_info),
        );
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(self.tonemap_descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&tonemap_bloom_info),
        );

        let mip_count = self.bloom_target.mip_levels as usize;
        let mut bloom_infos = Vec::with_capacity(mip_count + 1);
        for i in 0..=mip_count {
            let view = if i == 0 {
                self.offscreen_target.color_view
            } else {
                self.bloom_target.mip_views[i - 1]
            };
            bloom_infos.push([vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(view)
                .sampler(self.bloom_target.sampler)]);
        }
        for (i, info) in bloom_infos.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.bloom_descriptor_sets[i])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(info),
            );
        }

        unsafe { self.vulkan.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Recreate the swapchain and all dependent render targets.
    pub fn recreate_swapchain(&mut self, window: &mut Window, input: &mut crate::app::input::Input) {
        let mut w = window.width;
        let mut h = window.height;
        while w == 0 || h == 0 {
            window.poll_events(input);
            w = window.width;
            h = window.height;
        }
        unsafe { self.vulkan.device.device_wait_idle().unwrap(); }
        self.vulkan.recreate_frame_buffers();
        self.swapchain.recreate(&self.vulkan, w, h);

        let tw = self.swapchain.extent.width;
        let th = self.swapchain.extent.height;

        self.offscreen_target.shutdown(&self.vulkan);
        self.offscreen_target = OffscreenTarget::new(&self.vulkan, tw, th, vk::Format::R16G16B16A16_SFLOAT).unwrap();
        self.sdr_target.shutdown(&self.vulkan);
        self.sdr_target = OffscreenTarget::new(&self.vulkan, tw, th, vk::Format::R8G8B8A8_UNORM).unwrap();
        self.bloom_target.shutdown(&self.vulkan);
        self.bloom_target = BloomTarget::new(&self.vulkan, tw / 2, th / 2, 6).unwrap();

        self.update_post_process_descriptors();
        self.ui_backend.update_user_texture(
            &self.vulkan,
            self.offscreen_texture_id,
            self.sdr_target.color_view,
            self.sdr_target.sampler,
        );
    }

    /// Render one frame.
    ///
    /// # Safety
    /// The caller must ensure the ECS `world` is not mutated concurrently
    /// during rendering.
    pub fn render_frame(
        &mut self,
        window: &mut Window,
        world: &crate::ecs::World,
        world_matrices: &[(u32, crate::math::mat4::Mat4)],
        asset_manager: &crate::asset_manager::AssetManager,
        ui_ctx: &crate::ui::UiContext,
        ui_font: &crate::ui::font::Font,
        input: &mut crate::app::input::Input,
    ) {
        let _pipeline = match &self.pipeline {
            Some(p) => p,
            _ => return,
        };

        // Extract Camera
        let mut view_proj = crate::math::mat4::Mat4::identity();
        let mut camera_pos = [0.0f32; 4];
        {
            let cameras = world.get_component_array::<CameraComponent>();
            let transforms = world.get_component_array::<TransformComponent>();
            if let Some(&cam_entity) = cameras.dense_entities_slice().first() {
                if transforms.has(cam_entity) {
                    let cam_transform = unsafe { transforms.get(cam_entity) };
                    let pitch = cam_transform.rotation.x;
                    let yaw = cam_transform.rotation.y;
                    let forward = Vec3::new(
                        yaw.sin() * pitch.cos(),
                        pitch.sin(),
                        yaw.cos() * pitch.cos(),
                    ).normalize();
                    let center = cam_transform.position + forward;
                    let view = crate::math::mat4::Mat4::look_at(
                        cam_transform.position, center, Vec3::new(0.0, 1.0, 0.0),
                    );
                    let aspect = self.offscreen_target.width as f32 / self.offscreen_target.height as f32;
                    let proj = crate::math::mat4::Mat4::perspective(std::f32::consts::FRAC_PI_4, aspect, 0.1, 10000.0);
                    view_proj = proj * view;
                    camera_pos = [cam_transform.position.x, cam_transform.position.y, cam_transform.position.z, 1.0];
                }
            }
        }

        // Extract Light
        let mut light_dir = [0.0f32, -1.0, 0.0, 0.0];
        let mut light_color = [1.0f32, 1.0, 1.0, 0.0];
        let mut point_lights_array = [crate::renderer::vulkan::pipeline::PointLight::default(); 4];
        let mut num_point_lights = 0u32;
        {
            let lights = world.get_component_array::<LightComponent>();
            if let Some(lc) = lights.as_slice().first() {
                light_dir = [lc.direction.x, lc.direction.y, lc.direction.z, 0.0];
                light_color = [lc.color.x, lc.color.y, lc.color.z, 1.0];
            }
            let pl_comps = world.get_component_array::<crate::ecs::components::PointLightComponent>();
            let transforms = world.get_component_array::<TransformComponent>();
            for (i, pl) in pl_comps.as_slice().iter().enumerate() {
                if num_point_lights >= 4 { break; }
                let entity = pl_comps.dense_entities_slice()[i];
                if transforms.has(entity) {
                    let t = unsafe { transforms.get(entity) };
                    point_lights_array[num_point_lights as usize] =
                        crate::renderer::vulkan::pipeline::PointLight {
                            position: [t.position.x, t.position.y, t.position.z, 1.0],
                            color: [pl.color.x, pl.color.y, pl.color.z, pl.intensity],
                        };
                    num_point_lights += 1;
                }
            }
        }

        unsafe {
            self.vulkan.device.wait_for_fences(
                &[self.vulkan.in_flight_fences[self.current_frame]],
                true,
                u64::MAX,
            ).expect("wait_for_fences failed — possible device loss");
        }

        let mut light_space_matrix = crate::math::mat4::Mat4::identity();
        if let Some(shadow) = &mut self.shadow_pass {
            light_space_matrix = ShadowPass::compute_light_space_matrix(
                Vec3::new(light_dir[0], light_dir[1], light_dir[2]),
                Vec3::new(camera_pos[0], 0.0, camera_pos[2]), // center on camera's ground position
                40.0, // half-size of the shadow volume
            );
            shadow.light_space_matrix = light_space_matrix;
        }

        let ubo = crate::renderer::vulkan::pipeline::GlobalUbo {
            view_proj,
            camera_pos,
            light_dir,
            light_color,
            light_space_matrix,
            point_lights: point_lights_array,
            num_point_lights,
            _padding: [0; 3],
        };
        if !self.ubo_mapped.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &ubo as *const _ as *const u8,
                    self.ubo_mapped as *mut u8,
                    std::mem::size_of::<crate::renderer::vulkan::pipeline::GlobalUbo>(),
                );
            }
        }

        // Swapchain out-of-date detection — runs before any CB recording.
        if window.check_and_clear_resized() {
            self.recreate_swapchain(window, input);
            return;
        }

        let (image_index, _) = unsafe {
            match self.swapchain.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                self.vulkan.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            ) {
                Ok(r) => r,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain(window, input);
                    return;
                }
                Err(e) => { eprintln!("Failed to acquire next image: {:?}", e); return; }
            }
        };

        unsafe {
            self.vulkan.device.reset_fences(&[self.vulkan.in_flight_fences[self.current_frame]])
                .expect("reset_fences failed");
        }

        let cmd = self.vulkan.frame_command_buffers[self.current_frame];
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.vulkan.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
            self.vulkan.device.begin_command_buffer(cmd, &begin_info).unwrap();
        }

        // --- Instance Data ---
        // Reuse the pre-allocated buffer — zero heap allocations on the hot path.
        self.instance_data_buffer.clear();
        {
            let renders = world.get_component_array::<RenderComponent>();
            let transforms = world.get_component_array::<TransformComponent>();
            let dense = renders.as_slice();
            let entities = renders.dense_entities_slice();
            for i in 0..dense.len() {
                let r = &dense[i];
                if !r.visible { continue; }
                let entity = entities[i];
                if !transforms.has(entity) { continue; }
                let t = unsafe { transforms.get(entity) };
                let wm = world_matrices.iter().find(|(e, _)| *e == entity).map(|(_, m)| *m).unwrap_or(t.matrix);
                if let Some(mesh) = asset_manager.get_mesh(r.mesh_index) {
                    self.instance_data_buffer.push(crate::renderer::vulkan::pipeline::InstanceData {
                        world: wm,
                        aabb_min: [mesh.aabb_min[0], mesh.aabb_min[1], mesh.aabb_min[2], 1.0],
                        aabb_max: [mesh.aabb_max[0], mesh.aabb_max[1], mesh.aabb_max[2], 1.0],
                        color: [r.r, r.g, r.b, f32::from_bits(mesh.diffuse_texture_idx)],
                        pbr: [r.metallic, r.roughness, f32::from_bits(mesh.normal_texture_idx), f32::from_bits(mesh.mr_texture_idx)],
                        geometry: [mesh.index_count, mesh.index_offset, mesh.vertex_offset, mesh.emissive_texture_idx],
                    });
                }
            }
        }

        let draw_count = self.instance_data_buffer.len() as u32;
        if draw_count > 1 && self.current_frame % 60 == 0 {
            crate::log_info!("[Backend Debug] draw_count: {}", draw_count);
            for i in 0..self.instance_data_buffer.len() {
                let inst = &self.instance_data_buffer[i];
                crate::log_info!("  Inst {}: geom [count={}, idx_off={}, v_off={}], world pos: ({:.1}, {:.1}, {:.1})",
                    i, inst.geometry[0], inst.geometry[1], inst.geometry[2],
                    inst.world.cols[3].x, inst.world.cols[3].y, inst.world.cols[3].z);
            }
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                self.instance_data_buffer.as_ptr(),
                self.instance_mapped[self.current_frame] as *mut _,
                self.instance_data_buffer.len(),
            );
        }

        let dcb = self.draw_count_buffers[self.current_frame].handle;
        unsafe {
            self.vulkan.device.cmd_fill_buffer(cmd, dcb, 0, 4, 0);
            let mem_bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            self.vulkan.device.cmd_pipeline_barrier(
                cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(), std::slice::from_ref(&mem_bar), &[], &[],
            );
        }

        if let Some(compute) = &self.compute_pipeline {
            if draw_count > 0 {
                unsafe {
                    self.vulkan.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, compute.pipeline);
                    self.vulkan.device.cmd_bind_descriptor_sets(
                        cmd, vk::PipelineBindPoint::COMPUTE, compute.layout, 0,
                        std::slice::from_ref(&compute.descriptor_sets[self.current_frame]), &[],
                    );
                    #[repr(C)] struct PC { total_instances: u32, pad0: u32, pad1: u32, pad2: u32 }
                    let pc = PC { 
                        total_instances: draw_count,
                        pad0: if self.debug_cull { 1 } else { 0 },
                        pad1: 0,
                        pad2: 0,
                    };
                    self.vulkan.device.cmd_push_constants(
                        cmd, compute.layout, vk::ShaderStageFlags::COMPUTE, 0,
                        std::slice::from_raw_parts(&pc as *const _ as *const u8, 16),
                    );
                    self.vulkan.device.cmd_dispatch(cmd, (draw_count + 63) / 64, 1, 1);
                    let draw_bar = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);
                    self.vulkan.device.cmd_pipeline_barrier(
                        cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::DRAW_INDIRECT,
                        vk::DependencyFlags::empty(), std::slice::from_ref(&draw_bar), &[], &[],
                    );
                }
            }
        }

        // Cloth simulation dispatch — runs after GPU culling, before the scene pass.
        // Uses the pre-recorded ClothGpuInstances list maintained by the Application.
        if let Some(cloth) = &self.cloth_pipeline {
            cloth.dispatch_all(&self.vulkan, cmd, &self.cloth_instances, 1.0 / 60.0);
        }

        // Scene pass
        let clear_values = [
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.05, 0.05, 0.05, 1.0] } },
            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
        ];

        if let Some(shadow) = &self.shadow_pass {
            shadow.record_shadow_pass(cmd, world, world_matrices, asset_manager, &self.vulkan, &self.geometry_pool);
        }

        let mut graph = RenderGraph::new();
        graph.add_pass("Scene", vec![
            crate::renderer::vulkan::render_graph::PassResource {
                handle: ResourceHandle(self.offscreen_target.color_image),
                state: ResourceState { layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE, aspect_mask: vk::ImageAspectFlags::COLOR },
            },
            crate::renderer::vulkan::render_graph::PassResource {
                handle: ResourceHandle(self.offscreen_target.depth_image),
                state: ResourceState { layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS, access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE, aspect_mask: vk::ImageAspectFlags::DEPTH },
            },
        ], |cb| unsafe {
            let color_att = vk::RenderingAttachmentInfoKHR::default()
                .image_view(self.offscreen_target.color_view).image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::STORE).clear_value(clear_values[0]);
            let depth_att = vk::RenderingAttachmentInfoKHR::default()
                .image_view(self.offscreen_target.depth_view).image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::STORE).clear_value(clear_values[1]);
            let ri = vk::RenderingInfoKHR::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: self.offscreen_target.width, height: self.offscreen_target.height } })
                .layer_count(1).color_attachments(std::slice::from_ref(&color_att)).depth_attachment(&depth_att);
            self.vulkan.device.cmd_begin_rendering(cb, &ri);
            if let Some(p) = &self.pipeline {
                self.vulkan.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, p.handle);
                if self.descriptor_sets[self.current_frame] != vk::DescriptorSet::null() {
                    self.vulkan.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, p.layout, 0, std::slice::from_ref(&self.descriptor_sets[self.current_frame]), &[]);
                }
            }
            let vp = vk::Viewport { x: 0.0, y: 0.0, width: self.offscreen_target.width as f32, height: self.offscreen_target.height as f32, min_depth: 0.0, max_depth: 1.0 };
            self.vulkan.device.cmd_set_viewport(cb, 0, std::slice::from_ref(&vp));
            let sc = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: self.offscreen_target.width, height: self.offscreen_target.height } };
            self.vulkan.device.cmd_set_scissor(cb, 0, std::slice::from_ref(&sc));
            self.vulkan.device.cmd_bind_vertex_buffers(cb, 0, &[self.geometry_pool.vertex_buffer.handle], &[0]);
            self.vulkan.device.cmd_bind_index_buffer(cb, self.geometry_pool.index_buffer.handle, 0, vk::IndexType::UINT32);
            if self.global_texture_descriptor_sets[0] != vk::DescriptorSet::null() {
                self.vulkan.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.as_ref().unwrap().layout, 1, std::slice::from_ref(&self.global_texture_descriptor_sets[0]), &[]);
            }
            self.vulkan.device.cmd_draw_indexed_indirect_count(
                cb, self.indirect_buffers[self.current_frame].handle, 0,
                self.draw_count_buffers[self.current_frame].handle, 0, 100_000,
                std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
            self.vulkan.device.cmd_end_rendering(cb);
        });

        self.post_process.add_passes(
            &mut graph, &self.vulkan, &self.offscreen_target, &self.sdr_target,
            &self.blur_target, &self.bloom_target, self.tonemap_descriptor_set,
            &self.bloom_descriptor_sets, &self.blur_descriptor_sets, self.bloom_threshold,
        );

        // UI pass
        let swapchain_image = self.swapchain.images[image_index as usize];
        graph.add_pass("UI", vec![
            crate::renderer::vulkan::render_graph::PassResource {
                handle: ResourceHandle(self.sdr_target.color_image),
                state: ResourceState { layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER, access_mask: vk::AccessFlags::SHADER_READ, aspect_mask: vk::ImageAspectFlags::COLOR },
            },
            crate::renderer::vulkan::render_graph::PassResource {
                handle: ResourceHandle(swapchain_image),
                state: ResourceState { layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE, aspect_mask: vk::ImageAspectFlags::COLOR },
            },
        ], |cb| unsafe {
            let ui_clear = [vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }];
            let ui_att = vk::RenderingAttachmentInfoKHR::default()
                .image_view(self.swapchain.image_views[image_index as usize]).image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::STORE).clear_value(ui_clear[0]);
            let ui_ri = vk::RenderingInfoKHR::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: self.swapchain.extent })
                .layer_count(1).color_attachments(std::slice::from_ref(&ui_att));
            self.vulkan.device.cmd_begin_rendering(cb, &ui_ri);
            self.ui_backend.draw(&self.vulkan, cb, self.swapchain.extent.width, self.swapchain.extent.height, ui_ctx, ui_font, self.current_frame);
            self.vulkan.device.cmd_end_rendering(cb);
        });

        self.resource_tracker.clear();
        // Register swapchain image as UNDEFINED so the render graph knows to
        // emit a proper initial layout transition into COLOR_ATTACHMENT_OPTIMAL.
        self.resource_tracker.insert(
            ResourceHandle(swapchain_image),
            ResourceState {
                layout: vk::ImageLayout::UNDEFINED,
                stage_mask: vk::PipelineStageFlags::TOP_OF_PIPE,
                access_mask: vk::AccessFlags::NONE,
                aspect_mask: vk::ImageAspectFlags::COLOR,
            },
        );
        graph.execute(&self.vulkan, cmd, &mut self.resource_tracker);

        // Present barrier
        unsafe {
            let present_bar = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE).dst_access_mask(vk::AccessFlags::NONE)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL).new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(swapchain_image).subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
            self.vulkan.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&present_bar));
            self.resource_tracker.insert(ResourceHandle(swapchain_image), ResourceState { layout: vk::ImageLayout::PRESENT_SRC_KHR, stage_mask: vk::PipelineStageFlags::BOTTOM_OF_PIPE, access_mask: vk::AccessFlags::NONE, aspect_mask: vk::ImageAspectFlags::COLOR });
            self.vulkan.device.end_command_buffer(cmd).unwrap();
        }

        let wait_sems = [self.vulkan.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let cmd_bufs = [cmd];
        let signal_sems = [self.vulkan.render_finished_semaphores[image_index as usize]];
        let si = vk::SubmitInfo::default()
            .wait_semaphores(&wait_sems).wait_dst_stage_mask(&wait_stages)
            .command_buffers(&cmd_bufs).signal_semaphores(&signal_sems);
        unsafe {
            if let Err(e) = self.vulkan.device.queue_submit(self.vulkan.graphics_queue, std::slice::from_ref(&si), self.vulkan.in_flight_fences[self.current_frame]) {
                eprintln!("QUEUE SUBMIT FAILED: {:?}", e); return;
            }
        }

        let scs = [self.swapchain.swapchain];
        let iis = [image_index];
        let pi = vk::PresentInfoKHR::default().wait_semaphores(&signal_sems).swapchains(&scs).image_indices(&iis);
        // Resize is handled at the top of render_frame() after fence wait.
        // Never call recreate_swapchain here — the just-submitted CBs are
        // still in "pending" state and destroying their pool would trigger
        // VUID-vkResetCommandBuffer-commandBuffer-00045.
        let _r = unsafe { self.swapchain.swapchain_loader.queue_present(self.vulkan.graphics_queue, &pi) };
    }
}

impl Drop for RenderBackend {
    fn drop(&mut self) {
        unsafe {
            let _ = self.vulkan.device.device_wait_idle();
            self.vulkan.device.destroy_command_pool(self.vulkan.command_pool, None);
            self.vulkan.command_pool = vk::CommandPool::null();
            self.vulkan.device.destroy_command_pool(self.vulkan.transfer_command_pool, None);
            self.vulkan.transfer_command_pool = vk::CommandPool::null();

            // Render targets
            self.sdr_target.shutdown(&self.vulkan);
            self.bloom_target.shutdown(&self.vulkan);
            self.offscreen_target.shutdown(&self.vulkan);

            // Pipelines & UI
            self.post_process.destroy(&self.vulkan);
            if let Some(mut p) = self.pipeline.take() { p.shutdown(&self.vulkan); }
            self.ui_backend.shutdown(&self.vulkan);

            // Descriptor pools
            if self.post_process_descriptor_pool != vk::DescriptorPool::null() { self.vulkan.device.destroy_descriptor_pool(self.post_process_descriptor_pool, None); }
            if self.descriptor_pool != vk::DescriptorPool::null() { self.vulkan.device.destroy_descriptor_pool(self.descriptor_pool, None); }
            if self.compute_descriptor_pool != vk::DescriptorPool::null() { self.vulkan.device.destroy_descriptor_pool(self.compute_descriptor_pool, None); }
            if self.skinning_descriptor_pool != vk::DescriptorPool::null() { self.vulkan.device.destroy_descriptor_pool(self.skinning_descriptor_pool, None); }
            for mut inst in self.skinning_instances.drain(..) { inst.shutdown(&self.vulkan); }
            if let Some(mut cp) = self.compute_pipeline.take() { cp.shutdown(&self.vulkan); }
            if let Some(mut sp) = self.skinning_pipeline.take() { sp.shutdown(&self.vulkan); }
            // Cloth — shut down the GPU buffers for each instance, then the pipeline.
            for mut inst in self.cloth_instances.drain(..) {
                inst.velocity_buffer.shutdown(&self.vulkan);
                inst.collider_buffer.shutdown(&self.vulkan);
            }
            if let Some(cloth) = self.cloth_pipeline.take() { cloth.shutdown(&self.vulkan); }

            // Buffers
            if self.ubo_buffer != vk::Buffer::null() { self.vulkan.device.destroy_buffer(self.ubo_buffer, None); }
            if self.ubo_memory != vk::DeviceMemory::null() { self.vulkan.device.free_memory(self.ubo_memory, None); }
            for i in 0..2 {
                self.instance_buffers[i].shutdown(&self.vulkan);
                self.indirect_buffers[i].shutdown(&self.vulkan);
                self.draw_count_buffers[i].shutdown(&self.vulkan);
            }

            // Geometry
            self.geometry_pool.shutdown(&self.vulkan);
            self.blur_target.shutdown(&self.vulkan);

            // Core
            self.swapchain.shutdown(&self.vulkan);
            self.vulkan.shutdown();
        }
    }
}