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
    hzb::HzbTarget,
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
    pub animation_pool: crate::renderer::vulkan::animation_pool::AnimationPool,

    // === Lifetime 4: multi-buffered GPU buffers ===
    pub draw_count_buffers: [Buffer; 2],
    pub draw_count_mapped: [*mut u32; 2],
    pub indirect_buffers: [Buffer; 2],
    pub instance_buffers: [Buffer; 2],
    pub instance_mapped: [*mut c_void; 2],
    pub prefix_sum_buffers: [Buffer; 2],
    pub prefix_sum_mapped: [*mut u32; 2],
    pub anim_data_buffers: [Buffer; 2],
    pub anim_data_mapped: [*mut std::ffi::c_void; 2],
    pub anim_bone_matrices_buffer: Buffer,

    // === Lifetime 5: uniform buffer ===
    pub ubo_buffer: vk::Buffer,
    pub ubo_memory: vk::DeviceMemory,
    pub ubo_mapped: *mut c_void,

    pub anim_pipeline: Option<crate::renderer::vulkan::compute_anim::ComputeAnimPipeline>,
    pub anim_descriptor_pool: vk::DescriptorPool,
    pub anim_descriptor_sets: [vk::DescriptorSet; 2],
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
    pub instance_data_buffer: Vec<crate::renderer::vulkan::pipeline::InstanceData>,
    pub anim_instance_data_buffer: Vec<crate::renderer::vulkan::animation::InstanceAnimData>,
    pub prefix_sum_data_buffer: Vec<u32>,
    pub last_visible_meshlets: u32,

    // === HZB occlusion culling (Phase E/F) ===
    pub hzb_target: Option<HzbTarget>,
    /// Sampler used to read the depth buffer R32_SFLOAT for HZB generation.
    pub depth_copy_sampler: vk::Sampler,
    /// Number of completed frames — used to enable HZB after the first frame.
    pub hzb_frame_count: u32,
    /// Min projected screen-radius in pixels to render a meshlet (Phase G LOD).
    pub lod_bias: f32,

    // === Debug Toggles ===
    pub debug_cull: bool,
    pub debug_meshlets: bool,
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
            GeometryPool::new(&vulkan, 10_000_000, 30_000_000, 1_000_000)
                .expect("Failed to create GeometryPool");
        let animation_pool =
            crate::renderer::vulkan::animation_pool::AnimationPool::new(&vulkan, 1024, 4096, 1_000_000)
                .expect("Failed to create AnimationPool");

        let mut ui_backend = UiBackend::new(&vulkan, swapchain.format.format, &asset_manager.vfs);
        ui_backend.set_font(&vulkan, ui_font);

        let max_instances = 100_000usize;
        let mut instance_bufs = Vec::with_capacity(2);
        let mut instance_maps = Vec::with_capacity(2);
        let mut indirect_bufs = Vec::with_capacity(2);
        let mut draw_count_bufs = Vec::with_capacity(2);
        let mut draw_count_maps = Vec::with_capacity(2);

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
            let dcb_mapped = unsafe {
                vulkan
                    .device
                    .map_memory(dcb.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .expect("Failed to map draw count buffer")
            };
            draw_count_bufs.push(dcb);
            draw_count_maps.push(dcb_mapped as *mut u32);
        }

        let anim_data_size = (10_000 * std::mem::size_of::<crate::renderer::vulkan::animation::InstanceAnimData>()) as u64;
        let mut anim_data_bufs = Vec::with_capacity(2);
        let mut anim_data_maps = Vec::with_capacity(2);
        for _ in 0..2 {
            let ab = Buffer::new(
                &vulkan,
                anim_data_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ).expect("Failed to create anim data buffer");
            let mapped = unsafe {
                vulkan
                    .device
                    .map_memory(ab.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .expect("Failed to map anim data buffer")
            };
            anim_data_bufs.push(ab);
            anim_data_maps.push(mapped);
        }

        let mut prefix_sum_bufs = Vec::new();
        let mut prefix_sum_maps = Vec::new();
        for _ in 0..2 {
            let pb = Buffer::new(
                &vulkan,
                // ~500k meshlets => 500,000 * 4 bytes = ~2MB
                2_000_000,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ).expect("Failed to create prefix sum buffer");
            let mapped = unsafe {
                vulkan
                    .device
                    .map_memory(pb.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .expect("Failed to create prefix sum buffer")
            };
            prefix_sum_maps.push(mapped as *mut u32);
            prefix_sum_bufs.push(pb);
        }

        let instance_buffers = [instance_bufs.remove(0), instance_bufs.remove(0)];
        let instance_mapped = [instance_maps.remove(0), instance_maps.remove(0)];
        let indirect_buffers = [indirect_bufs.remove(0), indirect_bufs.remove(0)];
        let draw_count_buffers = [draw_count_bufs.remove(0), draw_count_bufs.remove(0)];
        let draw_count_mapped = [draw_count_maps.remove(0), draw_count_maps.remove(0)];
        let prefix_sum_buffers = [prefix_sum_bufs.remove(0), prefix_sum_bufs.remove(0)];
        let prefix_sum_mapped = [prefix_sum_maps.remove(0), prefix_sum_maps.remove(0)];
        let anim_data_buffers = [anim_data_bufs.remove(0), anim_data_bufs.remove(0)];
        let anim_data_mapped = [anim_data_maps.remove(0), anim_data_maps.remove(0)];
        
        let anim_bone_matrices_size = (max_instances * 128 * std::mem::size_of::<crate::math::mat4::Mat4>()) as u64;
        let anim_bone_matrices_buffer = Buffer::new(
            &vulkan,
            anim_bone_matrices_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ).expect("Failed to create anim bone matrices buffer");

        let mut anim_pipeline = crate::renderer::vulkan::compute_anim::ComputeAnimPipeline::new(&vulkan, &asset_manager.vfs);
        let anim_descriptor_pool = unsafe {
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(10)];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(2);
            vulkan.device.create_descriptor_pool(&pool_info, None).unwrap()
        };
        let anim_descriptor_sets = if let Some(anim) = &anim_pipeline {
            let layouts = [anim.descriptor_set_layout, anim.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(anim_descriptor_pool)
                .set_layouts(&layouts);
            unsafe { vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap().try_into().unwrap() }
        } else {
            [vk::DescriptorSet::null(), vk::DescriptorSet::null()]
        };

        if let Some(anim) = &anim_pipeline {
            for i in 0..2 {
                let inst_info = vk::DescriptorBufferInfo::default()
                    .buffer(anim_data_buffers[i].handle)
                    .offset(0).range(vk::WHOLE_SIZE);
                let skel_info = vk::DescriptorBufferInfo::default()
                    .buffer(animation_pool.skeleton_buffer.handle)
                    .offset(0).range(vk::WHOLE_SIZE);
                let clip_info = vk::DescriptorBufferInfo::default()
                    .buffer(animation_pool.clip_buffer.handle)
                    .offset(0).range(vk::WHOLE_SIZE);
                let kf_info = vk::DescriptorBufferInfo::default()
                    .buffer(animation_pool.keyframe_buffer.handle)
                    .offset(0).range(vk::WHOLE_SIZE);
                let out_info = vk::DescriptorBufferInfo::default()
                    .buffer(anim_bone_matrices_buffer.handle)
                    .offset(0).range(vk::WHOLE_SIZE);
                    
                let writes = [
                    vk::WriteDescriptorSet::default().dst_set(anim_descriptor_sets[i]).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&inst_info)),
                    vk::WriteDescriptorSet::default().dst_set(anim_descriptor_sets[i]).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&skel_info)),
                    vk::WriteDescriptorSet::default().dst_set(anim_descriptor_sets[i]).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&clip_info)),
                    vk::WriteDescriptorSet::default().dst_set(anim_descriptor_sets[i]).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&kf_info)),
                    vk::WriteDescriptorSet::default().dst_set(anim_descriptor_sets[i]).dst_binding(4).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&out_info)),
                ];
                unsafe { vulkan.device.update_descriptor_sets(&writes, &[]); }
            }
        }

        // Bind instance buffers into descriptor set 3 and anim_bone_matrices_buffer into binding 7
        for i in 0..2 {
            let instance_info = vk::DescriptorBufferInfo::default()
                .buffer(instance_buffers[i].handle)
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let anim_info = vk::DescriptorBufferInfo::default()
                .buffer(anim_bone_matrices_buffer.handle)
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set[i])
                    .dst_binding(3)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&instance_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set[i])
                    .dst_binding(7)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&anim_info)),
            ];
            unsafe { vulkan.device.update_descriptor_sets(&writes, &[]) };
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
                    geometry_pool.meshlet_buffer.handle,
                    prefix_sum_buffers[i].handle,
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
            animation_pool,
            draw_count_buffers,
            draw_count_mapped,
            indirect_buffers,
            instance_buffers,
            instance_mapped,
            prefix_sum_buffers,
            prefix_sum_mapped,
            anim_data_buffers,
            anim_data_mapped,
            anim_bone_matrices_buffer,
            ubo_buffer,
            ubo_memory,
            ubo_mapped,
            anim_pipeline,
            anim_descriptor_pool,
            anim_descriptor_sets,
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
            anim_instance_data_buffer: Vec::with_capacity(10_000),
            prefix_sum_data_buffer: Vec::with_capacity(max_instances),
            last_visible_meshlets: 0,
            debug_cull: false,
            debug_meshlets: false,
            hzb_target: None,
            depth_copy_sampler: vk::Sampler::null(),
            hzb_frame_count: 0,
            lod_bias: 2.0,
        };
        backend.update_post_process_descriptors();
        backend.ui_backend.update_user_texture(
            &backend.vulkan,
            backend.offscreen_texture_id,
            backend.sdr_target.color_view,
            backend.sdr_target.sampler,
        );
        // Create HZB target (done here after backend is fully built so we can access vulkan + vfs)
        let hzb_w = backend.swapchain.extent.width;
        let hzb_h = backend.swapchain.extent.height;
        backend.hzb_target = HzbTarget::new(&backend.vulkan, hzb_w, hzb_h, &asset_manager.vfs);
        // Create depth sampler for HZB generation
        {
            let si = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST).min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .anisotropy_enable(false);
            backend.depth_copy_sampler = unsafe { backend.vulkan.device.create_sampler(&si, None).unwrap_or(vk::Sampler::null()) };
        }
        // Do the initial HZB layout transition using the setup command buffer
        if let Some(hzb) = &backend.hzb_target {
            let cmd = unsafe {
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(backend.vulkan.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                backend.vulkan.device.allocate_command_buffers(&alloc_info).unwrap()[0]
            };
            let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                backend.vulkan.device.begin_command_buffer(cmd, &begin).unwrap();
                hzb.initial_transition(&backend.vulkan, cmd);
                backend.vulkan.device.end_command_buffer(cmd).unwrap();
                let si = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
                backend.vulkan.device.queue_submit(backend.vulkan.graphics_queue, std::slice::from_ref(&si), vk::Fence::null()).unwrap();
                backend.vulkan.device.queue_wait_idle(backend.vulkan.graphics_queue).unwrap();
                backend.vulkan.device.free_command_buffers(backend.vulkan.command_pool, &[cmd]);
            }
        }
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
        // Recreate HZB at new resolution
        if let Some(mut old_hzb) = self.hzb_target.take() { old_hzb.shutdown(&self.vulkan); }
        self.hzb_target = HzbTarget::new(&self.vulkan, tw, th, &crate::vfs::Vfs::new("."));
        self.hzb_frame_count = 0;
        if let Some(hzb) = &self.hzb_target {
            let cmd = unsafe {
                let ai = vk::CommandBufferAllocateInfo::default().command_pool(self.vulkan.command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
                self.vulkan.device.allocate_command_buffers(&ai).unwrap()[0]
            };
            let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                self.vulkan.device.begin_command_buffer(cmd, &begin).unwrap();
                hzb.initial_transition(&self.vulkan, cmd);
                self.vulkan.device.end_command_buffer(cmd).unwrap();
                let si = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
                self.vulkan.device.queue_submit(self.vulkan.graphics_queue, std::slice::from_ref(&si), vk::Fence::null()).unwrap();
                self.vulkan.device.queue_wait_idle(self.vulkan.graphics_queue).unwrap();
                self.vulkan.device.free_command_buffers(self.vulkan.command_pool, &[cmd]);
            }
        }
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
        world_matrices: &std::collections::HashMap<crate::ecs::EntityId, crate::math::mat4::Mat4>,
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
        let mut view = crate::math::mat4::Mat4::identity();
        let mut proj = crate::math::mat4::Mat4::identity();
        let mut inverse_proj = crate::math::mat4::Mat4::identity();
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
                    view = crate::math::mat4::Mat4::look_at(
                        cam_transform.position, center, Vec3::new(0.0, 1.0, 0.0),
                    );
                    let aspect = self.offscreen_target.width as f32 / self.offscreen_target.height as f32;
                    proj = crate::math::mat4::Mat4::perspective(std::f32::consts::FRAC_PI_4, aspect, 0.1, 10000.0);
                    view_proj = proj * view;
                    inverse_proj = proj.try_inverse().unwrap_or(crate::math::mat4::Mat4::identity());
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
            
            // Read back visible meshlets count from the previous frame's GPU execution
            self.last_visible_meshlets = std::ptr::read_volatile(self.draw_count_mapped[self.current_frame]);
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
            view,
            proj,
            inverse_proj,
            camera_pos,
            light_dir,
            light_color,
            light_space_matrix,
            screen_size: [self.offscreen_target.width as f32, self.offscreen_target.height as f32],
            z_near: 0.1,
            z_far: 10000.0,
            num_point_lights,
            debug_meshlets: if self.debug_meshlets { 1 } else { 0 },
            _padding: [0; 2],
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
        // Reuse the pre-allocated buffer — zero heap allocations on the hot path.
        self.instance_data_buffer.clear();
        self.anim_instance_data_buffer.clear();
        self.prefix_sum_data_buffer.clear();
        self.prefix_sum_data_buffer.push(0); // Exclusive scan starts with 0

        {
            let renders = world.get_component_array::<RenderComponent>();
            let transforms = world.get_component_array::<TransformComponent>();
            let animators = world.get_component_array::<crate::ecs::components::AnimatorComponent>();
            let skeletons = world.get_component_array::<crate::ecs::components::SkeletonComponent>();
            let dense = renders.as_slice();
            let entities = renders.dense_entities_slice();
            for i in 0..dense.len() {
                let r = &dense[i];
                if !r.visible { continue; }
                let entity = entities[i];
                if !transforms.has(entity) { continue; }
                let t = unsafe { transforms.get(entity) };
                let wm = world_matrices.get(&entity).copied().unwrap_or(t.matrix);
                if let Some(mesh) = asset_manager.get_mesh(r.mesh_index) {
                    let mut anim_instance_id = 0xffffffff;
                    
                    if animators.has(entity) && skeletons.has(entity) {
                        let animator = unsafe { animators.get(entity) };
                        let skeleton_comp = unsafe { skeletons.get(entity) };
                        if let Some(skeleton) = asset_manager.get_skeleton(&skeleton_comp.skeleton_name) {
                            anim_instance_id = self.anim_instance_data_buffer.len() as u32;
                            
                            let mut anim_data = crate::renderer::vulkan::animation::InstanceAnimData::default();
                            anim_data.skeleton_index = skeleton.gpu_index;
                            anim_data.current_time = animator.current_time;
                            anim_data.crossfade_weight = if animator.crossfade_duration > 0.0 {
                                (animator.crossfade_current / animator.crossfade_duration).clamp(0.0, 1.0)
                            } else {
                                0.0
                            };
                            
                            let mut write_state = |state: &crate::ecs::components::AnimationState, is_target: bool| {
                                let (state_type, ca, cb, cc, cd, wab, wcd) = match state {
                                    crate::ecs::components::AnimationState::Clip { clip_handle } => {
                                        (0, asset_manager.animation_clips[*clip_handle as usize].gpu_index, 0, 0, 0, [1.0, 0.0], [0.0, 0.0])
                                    }
                                    crate::ecs::components::AnimationState::Blend1D { clip_a, clip_b, weight } => {
                                        (1, asset_manager.animation_clips[*clip_a as usize].gpu_index, asset_manager.animation_clips[*clip_b as usize].gpu_index, 0, 0, [*weight, 1.0 - *weight], [0.0, 0.0])
                                    }
                                    crate::ecs::components::AnimationState::Blend2D { clip_bl, clip_br, clip_tl, clip_tr, param_x, param_y } => {
                                        (2, asset_manager.animation_clips[*clip_bl as usize].gpu_index, asset_manager.animation_clips[*clip_br as usize].gpu_index, asset_manager.animation_clips[*clip_tl as usize].gpu_index, asset_manager.animation_clips[*clip_tr as usize].gpu_index, [*param_x, *param_y], [0.0, 0.0]) // Note: weights mapping might need adjusting in compute shader based on actual formula
                                    }
                                };
                                
                                if !is_target {
                                    anim_data.state_type = state_type;
                                    anim_data.clip_a = ca;
                                    anim_data.clip_b = cb;
                                    anim_data.clip_c = cc;
                                    anim_data.clip_d = cd;
                                    anim_data.weights_ab = wab;
                                    anim_data.weights_cd = wcd;
                                } else {
                                    anim_data.prev_state_type = state_type;
                                    anim_data.prev_clip_a = ca;
                                    anim_data.prev_clip_b = cb;
                                    anim_data.prev_clip_c = cc;
                                    anim_data.prev_clip_d = cd;
                                    anim_data.prev_weights_ab = wab;
                                    anim_data.prev_weights_cd = wcd;
                                    anim_data.prev_time = animator.transition_time;
                                }
                            };
                            
                            write_state(&animator.state, false);
                            if let Some(target) = &animator.target_state {
                                write_state(target, true);
                            }
                            
                            self.anim_instance_data_buffer.push(anim_data);
                        }
                    }
                
                    self.instance_data_buffer.push(crate::renderer::vulkan::pipeline::InstanceData {
                        world: wm,
                        aabb_min: [mesh.aabb_min[0], mesh.aabb_min[1], mesh.aabb_min[2], 1.0],
                        aabb_max: [mesh.aabb_max[0], mesh.aabb_max[1], mesh.aabb_max[2], 1.0],
                        color: [r.r, r.g, r.b, f32::from_bits(mesh.diffuse_texture_idx)],
                        pbr: [r.metallic, r.roughness, f32::from_bits(mesh.normal_texture_idx), f32::from_bits(mesh.mr_texture_idx)],
                        geometry: [mesh.index_count, mesh.index_offset, mesh.vertex_offset, mesh.emissive_texture_idx],
                        geometry2: [mesh.meshlet_offset, mesh.meshlet_count, anim_instance_id, 0],
                    });
                    
                    let last_prefix = *self.prefix_sum_data_buffer.last().unwrap();
                    self.prefix_sum_data_buffer.push(last_prefix + mesh.meshlet_count);
                }
            }
        }

        let draw_count = self.instance_data_buffer.len() as u32;

        unsafe {
            std::ptr::copy_nonoverlapping(
                self.instance_data_buffer.as_ptr(),
                self.instance_mapped[self.current_frame] as *mut _,
                self.instance_data_buffer.len(),
            );
            if !self.anim_instance_data_buffer.is_empty() {
                std::ptr::copy_nonoverlapping(
                    self.anim_instance_data_buffer.as_ptr(),
                    self.anim_data_mapped[self.current_frame] as *mut _,
                    self.anim_instance_data_buffer.len(),
                );
            }
            std::ptr::copy_nonoverlapping(
                self.prefix_sum_data_buffer.as_ptr(),
                self.prefix_sum_mapped[self.current_frame] as *mut _,
                self.prefix_sum_data_buffer.len(),
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
        
        let anim_count = self.anim_instance_data_buffer.len() as u32;
        if anim_count > 0 {
            if let Some(anim) = &self.anim_pipeline {
                unsafe {
                    self.vulkan.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, anim.pipeline);
                    self.vulkan.device.cmd_bind_descriptor_sets(
                        cmd, vk::PipelineBindPoint::COMPUTE, anim.layout, 0,
                        std::slice::from_ref(&self.anim_descriptor_sets[self.current_frame]), &[],
                    );
                    self.vulkan.device.cmd_push_constants(
                        cmd, anim.layout, vk::ShaderStageFlags::COMPUTE, 0,
                        &anim_count.to_ne_bytes(),
                    );
                    self.vulkan.device.cmd_dispatch(cmd, (anim_count + 63) / 64, 1, 1);
                    
                    let mem_bar = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ);
                    self.vulkan.device.cmd_pipeline_barrier(
                        cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(), std::slice::from_ref(&mem_bar), &[], &[],
                    );
                }
            }
        }

        if let Some(compute) = &self.compute_pipeline {
            if draw_count > 0 {
                unsafe {
                    self.vulkan.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, compute.pipeline);
                    self.vulkan.device.cmd_bind_descriptor_sets(
                        cmd, vk::PipelineBindPoint::COMPUTE, compute.layout, 0,
                        std::slice::from_ref(&compute.descriptor_sets[self.current_frame]), &[],
                    );
                    let total_meshlets = *self.prefix_sum_data_buffer.last().unwrap_or(&0);
                    let sw = self.offscreen_target.width as f32;
                    let sh = self.offscreen_target.height as f32;
                    let hzb_enabled: f32 = if self.hzb_frame_count > 0 && self.hzb_target.is_some() { 1.0 } else { 0.0 };
                    // Update HZB descriptor for this frame if available
                    if let Some(hzb) = &self.hzb_target {
                        compute.update_hzb_descriptor(&self.vulkan, hzb.full_view, hzb.sampler, compute.descriptor_sets[self.current_frame]);
                    }
                    #[repr(C)] struct PC { total_meshlets: u32, total_instances: u32, debug_cull: u32, pad2: u32, screen_width: f32, screen_height: f32, lod_bias: f32, hzb_enabled: f32 }
                    let pc = PC {
                        total_meshlets,
                        total_instances: draw_count,
                        debug_cull: if self.debug_cull { 1 } else { 0 },
                        pad2: 0,
                        screen_width: sw,
                        screen_height: sh,
                        lod_bias: self.lod_bias,
                        hzb_enabled,
                    };
                    self.vulkan.device.cmd_push_constants(
                        cmd, compute.layout, vk::ShaderStageFlags::COMPUTE, 0,
                        std::slice::from_raw_parts(&pc as *const _ as *const u8, 32),
                    );
                    if total_meshlets > 0 {
                        self.vulkan.device.cmd_dispatch(cmd, (total_meshlets + 63) / 64, 1, 1);
                    }
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

        // ── Phase E: Generate HZB from this frame's depth buffer ─────────────
        // The depth image is still in DEPTH_STENCIL_ATTACHMENT_OPTIMAL after the
        // scene pass.  Transition it to SHADER_READ_ONLY so the HZB compute can
        // sample it, then generate the full mip chain.
        if let Some(hzb) = &self.hzb_target {
            // Transition depth → SHADER_READ_ONLY
            let depth_to_read = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(self.offscreen_target.depth_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                });
            unsafe {
                self.vulkan.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(), &[], &[],
                    std::slice::from_ref(&depth_to_read),
                );
            }

            hzb.generate(&self.vulkan, cmd, self.offscreen_target.depth_view, self.depth_copy_sampler);

            // Transition depth back to DEPTH_STENCIL_ATTACHMENT_OPTIMAL for next frame
            let depth_to_attach = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .image(self.offscreen_target.depth_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                });
            unsafe {
                self.vulkan.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    vk::DependencyFlags::empty(), &[], &[],
                    std::slice::from_ref(&depth_to_attach),
                );
            }

            self.hzb_frame_count = self.hzb_frame_count.saturating_add(1);
        }

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
            if self.anim_descriptor_pool != vk::DescriptorPool::null() { self.vulkan.device.destroy_descriptor_pool(self.anim_descriptor_pool, None); }
            if let Some(mut cp) = self.compute_pipeline.take() { cp.shutdown(&self.vulkan); }
            if let Some(mut ap) = self.anim_pipeline.take() { ap.shutdown(&self.vulkan); }
            // Cloth — shut down the GPU buffers for each instance, then the pipeline.
            for mut inst in self.cloth_instances.drain(..) {
                inst.velocity_buffer.shutdown(&self.vulkan);
                inst.collider_buffer.shutdown(&self.vulkan);
            }
            if let Some(cloth) = self.cloth_pipeline.take() { cloth.shutdown(&self.vulkan); }

            // Buffers
            if self.ubo_buffer != vk::Buffer::null() { self.vulkan.device.destroy_buffer(self.ubo_buffer, None); }
            if self.ubo_memory != vk::DeviceMemory::null() { self.vulkan.device.free_memory(self.ubo_memory, None); }
            self.anim_bone_matrices_buffer.shutdown(&self.vulkan);
            for i in 0..2 {
        for b in &mut self.instance_buffers { b.shutdown(&self.vulkan); }
        for b in &mut self.prefix_sum_buffers { b.shutdown(&self.vulkan); }
        for b in &mut self.anim_data_buffers { b.shutdown(&self.vulkan); }
                self.draw_count_buffers[i].shutdown(&self.vulkan);
                self.prefix_sum_buffers[i].shutdown(&self.vulkan);
            }

            // HZB
            if self.depth_copy_sampler != vk::Sampler::null() { self.vulkan.device.destroy_sampler(self.depth_copy_sampler, None); }
            if let Some(mut hzb) = self.hzb_target.take() { hzb.shutdown(&self.vulkan); }

            // Geometry
            self.geometry_pool.shutdown(&self.vulkan);
        self.animation_pool.shutdown(&self.vulkan);
            self.blur_target.shutdown(&self.vulkan);

            // Core
            self.swapchain.shutdown(&self.vulkan);
            self.vulkan.shutdown();
        }
    }
}