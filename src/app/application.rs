//! The core game loop and application state.

use crate::ecs::World;
use crate::ecs::{
    CameraComponent, HierarchyComponent, LightComponent, RenderComponent, TransformComponent,
};
use crate::math::vec::Vec3;
use crate::memory::{MemoryConfig, MemorySubsystem};
use crate::platform::{win32, Timer, Window};
use crate::renderer::vulkan::{Pipeline, Swapchain, VulkanDevice};
use crate::renderer::RenderDevice;
use ash::vk;

/// High-level engine coordinator.
/// Note: Field declaration order dictates drop order.
/// `world` must drop before `memory`. `swapchain` must drop before `vulkan`.
pub struct Application {
    pub world: World,
    pub pipeline: Option<Pipeline>,
    pub asset_manager: crate::asset_manager::AssetManager,
    /// Pre-allocated flat map of entity-index → world matrix.
    /// Cleared and rebuilt every frame. Capacity reserved at startup to avoid
    /// heap allocations on the hot path.
    pub world_matrices: Vec<(u32, crate::math::mat4::Mat4)>,
    pub ubo_buffer: vk::Buffer,
    pub ubo_memory: vk::DeviceMemory,
    pub instance_buffers: [crate::renderer::vulkan::buffer::Buffer; 2],
    pub instance_mapped: [*mut std::ffi::c_void; 2],
    pub indirect_buffers: [crate::renderer::vulkan::buffer::Buffer; 2],
    pub draw_count_buffers: [crate::renderer::vulkan::buffer::Buffer; 2],
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: [vk::DescriptorSet; 2],
    pub global_texture_descriptor_sets: [vk::DescriptorSet; 2],
    pub offscreen_target: crate::renderer::vulkan::OffscreenTarget,
    pub sdr_target: crate::renderer::vulkan::OffscreenTarget,
    pub blur_target: crate::renderer::vulkan::OffscreenTarget,
    pub bloom_target: crate::renderer::vulkan::bloom::BloomTarget,
    pub post_process: crate::renderer::vulkan::PostProcessPipeline,
    pub post_process_descriptor_pool: vk::DescriptorPool,
    pub tonemap_descriptor_set: vk::DescriptorSet,
    pub bloom_descriptor_sets: Vec<vk::DescriptorSet>,
    pub blur_descriptor_sets: Vec<vk::DescriptorSet>,
    pub geometry_pool: crate::renderer::vulkan::GeometryPool,
    pub offscreen_texture_id: u32,
    pub swapchain: Swapchain,
    pub vulkan: VulkanDevice,
    pub window: Window,
    pub input: crate::app::input::Input,
    pub timer: Timer,
    pub memory: MemorySubsystem,
    pub ui_ctx: crate::ui::UiContext,
    pub ui_font: crate::ui::font::Font,
    pub ui_backend: crate::renderer::vulkan::UiBackend,
    pub physics: crate::physics::PhysicsSystem,
    pub selected_entity: Option<crate::ecs::EntityId>,
    pub current_frame: usize,
    pub bloom_threshold: f32,
    /// Zero-allocation flat map tracking Vulkan image states across render
    /// graph passes. Cleared and rebuilt each frame with no heap traffic.
    pub resource_tracker: crate::renderer::vulkan::render_graph::ResourceTracker,
    pub editor: crate::app::editor::Editor,
    pub hot_reloader: Option<crate::app::hot_reload::HotReloader>,
    pub compute_pipeline: Option<crate::renderer::vulkan::compute_cull::ComputeCullPipeline>,
    pub compute_descriptor_pool: vk::DescriptorPool,
    /// Pre-allocated flat array of per-mesh compute descriptor sets.
    /// Indexed by mesh index. Grown once when new meshes are loaded,
    /// never during the frame loop.
    pub compute_descriptor_sets: Vec<Option<vk::DescriptorSet>>,
    pub material_descriptor_sets: Vec<Option<vk::DescriptorSet>>,
    pub skinning_pipeline:
        Option<crate::renderer::vulkan::compute_skinning::ComputeSkinningPipeline>,
    pub skinning_descriptor_pool: vk::DescriptorPool,
    pub skinning_instances: Vec<crate::renderer::vulkan::compute_skinning::SkinningInstance>,
    pub audio_subsystem: Option<crate::audio::AudioSubsystem>,
    pub audio_system: crate::audio::AudioSystem,
    pub script_engine: crate::scripting::engine::ScriptEngine,
    pub is_playing: bool,
}

impl Application {
    /// Initialize the engine subsystems.
    pub fn new(title: &str, width: i32, height: i32) -> Option<Self> {
        crate::utils::shader_compiler::compile_all_shaders();

        // 1. Initialize Memory
        let mut memory = MemorySubsystem::default();
        if !memory.init(MemoryConfig::default()) {
            return None;
        }

        // 2. Initialize Platform Window & Input
        let window = Window::new(title, width, height);
        let input = crate::app::input::Input::new();
        let timer = Timer::new();

        // 3. Initialize Renderer (Vulkan)
        let vulkan = VulkanDevice::new()?;
        let swapchain = Swapchain::new(&vulkan, &window, width as u32, height as u32)?;

        let target_width = swapchain.extent.width;
        let target_height = swapchain.extent.height;

        let offscreen_target = crate::renderer::vulkan::OffscreenTarget::new(
            &vulkan,
            target_width,
            target_height,
            vk::Format::R16G16B16A16_SFLOAT,
        )?;
        let sdr_target = crate::renderer::vulkan::OffscreenTarget::new(
            &vulkan,
            target_width,
            target_height,
            vk::Format::R8G8B8A8_UNORM,
        )?;
        let mip_levels = 6;
        let bloom_target = crate::renderer::vulkan::bloom::BloomTarget::new(
            &vulkan,
            target_width / 2,
            target_height / 2,
            mip_levels,
        )?;
        // Initialize Asset Manager
        let mut asset_manager = crate::asset_manager::AssetManager::new();
        asset_manager.load_checkerboard(&vulkan, "default");
        asset_manager.load_checkerboard(&vulkan, "fallback");
        // Load a procedural studio environment map for gorgeous reflections
        asset_manager.load_procedural_env(&vulkan, "env_default");
        asset_manager.load_solid_color(&vulkan, "shadow_default", 255, 255, 255, 255); // White shadow map (no shadows)
        let post_process = crate::renderer::vulkan::PostProcessPipeline::new(
            &vulkan,
            vk::Format::R8G8B8A8_UNORM,
            &asset_manager.vfs,
        )?;

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(20), // give plenty of space
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(20);
        let post_process_descriptor_pool = unsafe {
            match vulkan.device.create_descriptor_pool(&pool_info, None) {
                Ok(pool) => pool,
                Err(e) => {
                    eprintln!("Failed to create descriptor pool: {:?}", e);
                    return None;
                }
            }
        };

        let tonemap_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(post_process_descriptor_pool)
            .set_layouts(std::slice::from_ref(
                &post_process.tonemap_descriptor_set_layout,
            ));
        let tonemap_descriptor_set = unsafe {
            match vulkan.device.allocate_descriptor_sets(&tonemap_alloc_info) {
                Ok(sets) => sets[0],
                Err(e) => {
                    eprintln!("Failed to allocate tonemap descriptor sets: {:?}", e);
                    return None;
                }
            }
        };

        let mut bloom_layouts = Vec::new();
        for _ in 0..=mip_levels {
            bloom_layouts.push(post_process.bloom_descriptor_set_layout);
        }
        let bloom_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(post_process_descriptor_pool)
            .set_layouts(&bloom_layouts);
        let bloom_descriptor_sets = unsafe {
            match vulkan.device.allocate_descriptor_sets(&bloom_alloc_info) {
                Ok(sets) => sets,
                Err(e) => {
                    eprintln!("Failed to allocate bloom descriptor sets: {:?}", e);
                    return None;
                }
            }
        };

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
                .descriptor_count(2000), // 2 storage buffers per mesh (meshlet + indirect)
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

        let mut compute_pipeline = crate::renderer::vulkan::compute_cull::ComputeCullPipeline::new(
            &vulkan,
            &asset_manager.vfs,
        );

        let skinning_pipeline =
            crate::renderer::vulkan::compute_skinning::ComputeSkinningPipeline::new(
                &vulkan,
                &asset_manager.vfs,
            );

        let skinning_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1000), // Max skinning instances * 3 SSBOs
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

        let ubo_data = if let Some(pipe) = &pipeline {
            if asset_manager
                .load_texture(&vulkan, "default", "test_image.png")
                .is_none()
            {
                asset_manager.load_checkerboard(&vulkan, "fallback");
            }
            let tex = asset_manager
                .get_texture("default")
                .or_else(|| asset_manager.get_texture("fallback"));
            if let Some(_tex) = tex {
                // 1. Create Descriptor Pool
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

                descriptor_pool = unsafe {
                    vulkan
                        .device
                        .create_descriptor_pool(&pool_info, None)
                        .unwrap()
                };

                // 2. Allocate Descriptor Sets
                let layouts = [pipe.descriptor_set_layout, pipe.descriptor_set_layout];
                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts);

                let sets = unsafe { vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap() };
                descriptor_set = [sets[0], sets[1]];

                // Create UBO buffer
                let ubo_size =
                    std::mem::size_of::<crate::renderer::vulkan::pipeline::GlobalUbo>() as u64;
                let ubo_buffer_info = vk::BufferCreateInfo::default()
                    .size(ubo_size)
                    .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);

                let ubo_buffer =
                    unsafe { vulkan.device.create_buffer(&ubo_buffer_info, None).unwrap() };
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

                // 3. Update Descriptor Sets
                for (_i, set) in descriptor_set.iter().enumerate() {
                    let ubo_info = vk::DescriptorBufferInfo::default()
                        .buffer(ubo_buffer)
                        .offset(0)
                        .range(ubo_size);

                    let write_desc_ubo = vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(std::slice::from_ref(&ubo_info));

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

                    let write_desc_env = vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(std::slice::from_ref(&env_image_info));

                    let write_desc_shadow = vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(std::slice::from_ref(&shadow_image_info));

                    // Notice: binding 3 (instance data) is updated in render_frame because the buffer is multi-buffered! Wait, the buffers are ALREADY multi-buffered, so we can update them here if we had access to instance_buffers.
                    // Actually, instance_buffers are created later! So we MUST update binding 3 later.

                    unsafe {
                        vulkan.device.update_descriptor_sets(
                            &[write_desc_ubo, write_desc_env, write_desc_shadow],
                            &[],
                        )
                    };
                }

                // Allocate a single bindless texture set
                let counts = [1000];
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

                // Bind fallback texture to index 0 of the bindless set
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

                unsafe {
                    vulkan.device.update_descriptor_sets(&[write_desc], &[]);
                }

                (Some((ubo_buffer, ubo_memory)), bindless_set)
            } else {
                (None, vk::DescriptorSet::null())
            }
        } else {
            (None, vk::DescriptorSet::null())
        };

        let (ubo_buffer, ubo_memory) = ubo_data
            .0
            .unwrap_or((vk::Buffer::null(), vk::DeviceMemory::null()));
        let bindless_set = ubo_data.1;

        let mut geometry_pool =
            crate::renderer::vulkan::GeometryPool::new(&vulkan, 1_000_000, 3_000_000, 100_000)
                .expect("Failed to create GeometryPool");

        let cube_model_indices = asset_manager
            .load_model(&vulkan, &mut geometry_pool, "cube.obj")
            .unwrap_or(&[])
            .to_vec();
        if cube_model_indices.is_empty() {
            crate::log_info!("Failed to load cube.obj. Rendering will fail.");
        }

        if pipeline.is_none() {
            crate::log_info!("Shaders not found or failed to compile. Rendering will be skipped.");
        }

        // 4. Initialize ECS
        let mut world = unsafe { World::new(memory.persistent_arena()) };

        // Register rendering components
        unsafe {
            world.register_component::<TransformComponent>(1000);
            world.register_component::<RenderComponent>(1000);
            world.register_component::<CameraComponent>(1);
            world.register_component::<LightComponent>(1);
            world.register_component::<crate::ecs::components::PointLightComponent>(10);
            world.register_component::<HierarchyComponent>(1000);
            world.register_component::<crate::ecs::components::RigidBodyComponent>(1000);
            world.register_component::<crate::ecs::components::ColliderComponent>(1000);
            world.register_component::<crate::ecs::components::AudioListenerComponent>(10);
            world.register_component::<crate::ecs::components::AudioEmitterComponent>(100);
            world.register_component::<crate::ecs::components::SkeletonComponent>(100);
            world.register_component::<crate::ecs::components::AnimatorComponent>(100);
            world.register_component::<crate::ecs::components::ScriptBehaviorComponent>(100);
            world.register_component::<crate::ecs::components::SoftBodyComponent>(100);
        }

        let physics = crate::physics::PhysicsSystem::new();

        // Spawn a camera
        let camera_entity = world.create_entity();
        let mut cam_transform = TransformComponent::default();
        cam_transform.position = Vec3::new(0.0, 3.0, -8.0);
        cam_transform.rotation.x = -std::f32::consts::FRAC_PI_8; // Look slightly down
        unsafe {
            world.add_component(camera_entity, cam_transform);
            let mut cam_comp = CameraComponent::default();
            cam_comp.proj = crate::math::mat4::Mat4::perspective(
                std::f32::consts::FRAC_PI_4,
                width as f32 / height as f32,
                0.1,
                100.0,
            );
            world.add_component(camera_entity, cam_comp);
            world.add_component(
                camera_entity,
                crate::ecs::components::AudioListenerComponent,
            );
        }

        // Spawn a directional light
        let light_entity = world.create_entity();
        unsafe {
            world.add_component(
                light_entity,
                LightComponent {
                    // Pointing down and towards the front-right of the car
                    direction: Vec3::new(0.5, -1.0, -0.5).normalize(),
                    color: Vec3::new(3.0, 3.0, 3.0),
                },
            );
        }

        let ui_ctx = crate::ui::UiContext::new();
        let font_bytes = asset_manager
            .vfs
            .read_bytes("assets/fonts/Roboto-Regular.ttf")
            .unwrap_or_else(|_| vec![]);
        let ui_font =
            crate::ui::font::Font::load_ascii(&font_bytes, 64.0).unwrap_or(crate::ui::font::Font {
                texture_data: vec![255; 1024 * 1024 * 4],
                width: 1024,
                height: 1024,
                glyphs: [crate::ui::font::GlyphInfo::default(); 128],
                line_height: 64.0,
            });
        let mut ui_backend = crate::renderer::vulkan::UiBackend::new(
            &vulkan,
            swapchain.format.format,
            &asset_manager.vfs,
        );
        ui_backend.set_font(&vulkan, &ui_font);
        let offscreen_texture_id = 0;

        let audio_subsystem = crate::audio::AudioSubsystem::new();
        let audio_system =
            crate::audio::AudioSystem::new(audio_subsystem.as_ref(), &asset_manager.vfs);
        let script_engine = crate::scripting::engine::ScriptEngine::new();

        let max_instances = 100_000;
        let mut instance_buffers = Vec::new();
        let mut instance_mapped = Vec::new();
        let mut indirect_buffers = Vec::new();
        let mut draw_count_buffers = Vec::new();

        for _ in 0..2 {
            let instance_buffer = crate::renderer::vulkan::buffer::Buffer::new(
                &vulkan,
                max_instances
                    * std::mem::size_of::<crate::renderer::vulkan::pipeline::InstanceData>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .expect("Failed to create instance buffer");
            let mapped = unsafe {
                vulkan
                    .device
                    .map_memory(
                        instance_buffer.memory,
                        0,
                        vk::WHOLE_SIZE,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Failed to map instance buffer memory")
            };
            instance_buffers.push(instance_buffer);
            instance_mapped.push(mapped);

            let indirect_buffer = crate::renderer::vulkan::buffer::Buffer::new(
                &vulkan,
                max_instances * std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Failed to create indirect buffer");
            indirect_buffers.push(indirect_buffer);

            let draw_count_buffer = crate::renderer::vulkan::buffer::Buffer::new(
                &vulkan,
                4,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Failed to create draw count buffer");
            draw_count_buffers.push(draw_count_buffer);
        }

        let instance_buffers = [instance_buffers.remove(0), instance_buffers.remove(0)];
        let instance_mapped = [instance_mapped.remove(0), instance_mapped.remove(0)];
        let indirect_buffers = [indirect_buffers.remove(0), indirect_buffers.remove(0)];
        let draw_count_buffers = [draw_count_buffers.remove(0), draw_count_buffers.remove(0)];

        let blur_target = crate::renderer::vulkan::OffscreenTarget::new(
            &vulkan,
            window.width,
            window.height,
            vk::Format::R16G16B16A16_SFLOAT,
        )
        .unwrap();
        let blur_descriptor_sets = Vec::new();

        // Update descriptor sets with instance buffers (binding 3)
        for i in 0..2 {
            let instance_info = vk::DescriptorBufferInfo::default()
                .buffer(instance_buffers[i].handle)
                .offset(0)
                .range(vk::WHOLE_SIZE);

            let write_desc_instance = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set[i])
                .dst_binding(3)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&instance_info));

            unsafe {
                vulkan
                    .device
                    .update_descriptor_sets(&[write_desc_instance], &[]);
            }
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

        let mut app = Self {
            world,
            pipeline,
            asset_manager,
            world_matrices: Vec::with_capacity(256),
            ubo_buffer,
            ubo_memory,
            instance_buffers,
            instance_mapped,
            indirect_buffers,
            draw_count_buffers,
            descriptor_pool,
            descriptor_sets: descriptor_set,
            global_texture_descriptor_sets: [bindless_set, bindless_set],
            offscreen_target,
            sdr_target,
            blur_target,
            bloom_target,
            post_process,
            post_process_descriptor_pool,
            tonemap_descriptor_set,
            bloom_descriptor_sets,
            blur_descriptor_sets,
            geometry_pool,
            offscreen_texture_id,
            swapchain,
            vulkan,
            window,
            input,
            timer,
            memory,
            ui_ctx,
            ui_font,
            ui_backend,
            physics,
            selected_entity: None,
            current_frame: 0,
            bloom_threshold: 1.0,
            resource_tracker: crate::renderer::vulkan::render_graph::ResourceTracker::new(),
            editor: crate::app::editor::Editor::new(),
            hot_reloader: None,
            compute_pipeline,
            compute_descriptor_pool,
            compute_descriptor_sets: Vec::new(),
            material_descriptor_sets: Vec::new(),
            skinning_pipeline,
            skinning_descriptor_pool,
            skinning_instances: Vec::new(),
            audio_subsystem,
            audio_system,
            script_engine,
            is_playing: false,
        };
        app.update_post_process_descriptors();

        app.ui_backend.update_user_texture(
            &app.vulkan,
            app.offscreen_texture_id,
            app.sdr_target.color_view,
            app.sdr_target.sampler,
        );

        Some(app)
    }

    pub fn update_post_process_descriptors(&self) {
        let mut writes = Vec::new();

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

        let mut bloom_infos = Vec::new();
        for i in 0..=self.bloom_target.mip_levels as usize {
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

        unsafe {
            self.vulkan.device.update_descriptor_sets(&writes, &[]);
        }
    }

    pub fn recreate_swapchain(&mut self) {
        let mut width = self.window.width;
        let mut height = self.window.height;

        while width == 0 || height == 0 {
            self.window.poll_events(&mut self.input);
            width = self.window.width;
            height = self.window.height;
        }

        unsafe {
            self.vulkan.device.device_wait_idle().unwrap();
        }

        self.swapchain.recreate(&self.vulkan, width, height);

        let target_width = self.swapchain.extent.width;
        let target_height = self.swapchain.extent.height;

        self.offscreen_target.shutdown(&self.vulkan);
        self.offscreen_target = crate::renderer::vulkan::OffscreenTarget::new(
            &self.vulkan,
            target_width,
            target_height,
            vk::Format::R16G16B16A16_SFLOAT,
        )
        .unwrap();

        self.sdr_target.shutdown(&self.vulkan);
        self.sdr_target = crate::renderer::vulkan::OffscreenTarget::new(
            &self.vulkan,
            target_width,
            target_height,
            vk::Format::R8G8B8A8_UNORM,
        )
        .unwrap();

        self.bloom_target.shutdown(&self.vulkan);
        self.bloom_target = crate::renderer::vulkan::bloom::BloomTarget::new(
            &self.vulkan,
            target_width / 2,
            target_height / 2,
            6,
        )
        .unwrap();

        self.update_post_process_descriptors();

        self.ui_backend.update_user_texture(
            &self.vulkan,
            self.offscreen_texture_id,
            self.sdr_target.color_view,
            self.sdr_target.sampler,
        );
    }

    /// The canonical game loop.
    pub fn run(&mut self) {
        crate::log_info!("Application started.");

        // Reset timer before loop starts to avoid a massive first dt
        let _ = self.timer.tick();

        while self.window.poll_events(&mut self.input) {
            let dt = self.timer.tick();

            // Auto-scale the UI based on window height (assume 720p is baseline 1.0)
            let ppp = (self.window.height as f32 / 720.0).max(0.5);
            self.input.ui_scale = ppp;

            self.ui_ctx.begin_frame(
                crate::math::vec::Vec2::new(self.input.mouse_x as f32, self.input.mouse_y as f32),
                self.input.keys[0x01],
                self.input.mouse_scroll_y,
            );

            if self.is_playing {
                if let Some(reloader) = &mut self.hot_reloader {
                    reloader.update();
                    reloader.call_game_update(&mut self.world, &mut self.physics, dt as f32);
                }

                use crate::ecs::System;
                self.audio_system.update(dt as f32, &self.world);
                crate::ecs::animation_system::process_animations(
                    &self.world,
                    &self.asset_manager,
                    dt as f32,
                );
                crate::ecs::scripting_system::process_scripts(
                    &self.world,
                    &self.asset_manager,
                    &self.script_engine,
                    &self.physics,
                    dt as f32,
                );
            }

            let mut view_mat = crate::math::mat4::Mat4::identity();
            let mut proj_mat = crate::math::mat4::Mat4::identity();

            let cam_entity = {
                let cameras = self.world.get_component_array::<CameraComponent>();
                cameras.dense_entities_slice().first().copied()
            };
            if let Some(cam_entity) = cam_entity {
                let transforms = self.world.get_component_array::<TransformComponent>();
                if transforms.has(cam_entity) {
                    let transform = unsafe { transforms.get(cam_entity) };
                    let pitch = transform.rotation.x;
                    let yaw = transform.rotation.y;
                    let forward = crate::math::vec::Vec3::new(
                        yaw.sin() * pitch.cos(),
                        pitch.sin(),
                        yaw.cos() * pitch.cos(),
                    )
                    .normalize();
                    let center = transform.position + forward;
                    view_mat = crate::math::mat4::Mat4::look_at(
                        transform.position,
                        center,
                        crate::math::vec::Vec3::new(0.0, 1.0, 0.0),
                    );

                    let aspect_ratio =
                        self.offscreen_target.width as f32 / self.offscreen_target.height as f32;
                    proj_mat = crate::math::mat4::Mat4::perspective(
                        std::f32::consts::FRAC_PI_4,
                        aspect_ratio,
                        0.1,
                        100.0,
                    );
                }
            }

            let (actions, new_viewport_size, raycast_request, viewport_hovered) = self.editor.draw(
                &mut self.ui_ctx,
                &mut self.world,
                &mut self.physics,
                &mut self.selected_entity,
                &mut self.bloom_threshold,
                1.0 / dt as f32,
                self.is_playing,
                self.window.width as f32,
                self.window.height as f32,
                view_mat,
                proj_mat,
            );

            for action in actions {
                match action {
                    crate::app::editor::EditorAction::Play => {
                        self.is_playing = true;
                    }
                    crate::app::editor::EditorAction::Pause => {
                        self.is_playing = false;
                    }
                    crate::app::editor::EditorAction::SpawnEntity => {
                        let new_entity = self.world.create_entity();
                        unsafe {
                            self.world.add_component(
                                new_entity,
                                crate::ecs::TransformComponent::default(),
                            );
                            self.world.add_component(
                                new_entity,
                                crate::ecs::components::HierarchyComponent::default(),
                            );
                        }
                        self.selected_entity = Some(new_entity);
                    }
                    crate::app::editor::EditorAction::TranslateSelected(pos) => {
                        if let Some(entity) = self.selected_entity {
                            let transforms =
                                self.world.get_component_array_mut::<TransformComponent>();
                            if transforms.has(entity) {
                                let transform = unsafe { transforms.get_mut(entity) };
                                transform.position = pos;

                                // Update physics body if needed
                                let physics_comps = self.world.get_component_array::<crate::ecs::components::RigidBodyComponent>();
                                if physics_comps.has(entity) {
                                    let pc = unsafe { physics_comps.get(entity) };
                                    let handle = pc.handle;
                                    if true {
                                        if let Some(body) =
                                            self.physics.rigid_body_set.get_mut(handle)
                                        {
                                            let mut tr = body.position().clone();
                                            tr.translation = rapier3d::math::Isometry::translation(
                                                pos.x, pos.y, pos.z,
                                            )
                                            .translation;
                                            body.set_position(tr, true);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    crate::app::editor::EditorAction::DeleteEntity(e) => {
                        let mut to_delete = vec![e];
                        {
                            let hierarchies = self
                                .world
                                .get_component_array::<crate::ecs::components::HierarchyComponent>(
                                );
                            let mut i = 0;
                            while i < to_delete.len() {
                                let current = to_delete[i];
                                let dense_hier = hierarchies.as_slice();
                                let entities = hierarchies.dense_entities_slice();
                                for (j, hier) in dense_hier.iter().enumerate() {
                                    if hier.parent == Some(current) {
                                        let child_entity = entities[j];
                                        if !to_delete.contains(&child_entity) {
                                            to_delete.push(child_entity);
                                        }
                                    }
                                }
                                i += 1;
                            }
                        }

                        {
                            let colliders = self
                                .world
                                .get_component_array::<crate::ecs::components::ColliderComponent>();
                            let rigid_bodies = self
                                .world
                                .get_component_array::<crate::ecs::components::RigidBodyComponent>(
                                );
                            for &entity in &to_delete {
                                if colliders.has(entity) {
                                    let handle = unsafe { colliders.get(entity) }.handle;
                                    self.physics.collider_set.remove(
                                        handle,
                                        &mut self.physics.island_manager,
                                        &mut self.physics.rigid_body_set,
                                        true,
                                    );
                                }
                                if rigid_bodies.has(entity) {
                                    let handle = unsafe { rigid_bodies.get(entity) }.handle;
                                    self.physics.rigid_body_set.remove(
                                        handle,
                                        &mut self.physics.island_manager,
                                        &mut self.physics.collider_set,
                                        &mut self.physics.impulse_joint_set,
                                        &mut self.physics.multibody_joint_set,
                                        true,
                                    );
                                }
                            }
                        }

                        for &entity in &to_delete {
                            self.world.destroy_entity(entity);
                            if self.selected_entity == Some(entity) {
                                self.selected_entity = None;
                            }
                        }
                    }
                    crate::app::editor::EditorAction::AddComponent(e, name) => {
                        self.editor.registry.add_component(
                            &name,
                            e,
                            &mut self.world,
                            &mut self.physics,
                        );
                    }
                    crate::app::editor::EditorAction::SpawnModel(path) => {
                        let new_entity = self.world.create_entity();

                        // Load model using AssetManager
                        let lower = path.to_lowercase();
                        let mesh_indices = if lower.ends_with(".obj") {
                            self.asset_manager
                                .load_model(&self.vulkan, &mut self.geometry_pool, &path)
                                .map(|m| m.to_vec())
                        } else if lower.ends_with(".gltf") || lower.ends_with(".glb") {
                            self.asset_manager
                                .load_gltf(&self.vulkan, &mut self.geometry_pool, &path, &path)
                                .map(|m| m.to_vec())
                        } else {
                            None
                        };

                        unsafe {
                            self.world.add_component(
                                new_entity,
                                crate::ecs::TransformComponent {
                                    position: crate::math::vec::Vec3::new(0.0, 0.0, 0.0),
                                    rotation: crate::math::vec::Vec3::new(0.0, 0.0, 0.0),
                                    scale: crate::math::vec::Vec3::new(1.0, 1.0, 1.0),
                                    matrix: crate::math::mat4::Mat4::identity(),
                                },
                            );
                            self.world.add_component(
                                new_entity,
                                crate::ecs::components::HierarchyComponent {
                                    parent: None,
                                    local_matrix: crate::math::mat4::Mat4::identity(),
                                },
                            );
                        }

                        if let Some(indices) = mesh_indices {
                            for &mesh_index in &indices {
                                let child_entity = self.world.create_entity();
                                unsafe {
                                    self.world.add_component(
                                        child_entity,
                                        crate::ecs::TransformComponent {
                                            position: crate::math::vec::Vec3::new(0.0, 0.0, 0.0),
                                            rotation: crate::math::vec::Vec3::new(0.0, 0.0, 0.0),
                                            scale: crate::math::vec::Vec3::new(1.0, 1.0, 1.0),
                                            matrix: crate::math::mat4::Mat4::identity(),
                                        },
                                    );

                                    let mut metallic = 0.0;
                                    let mut roughness = 0.8;
                                    let mut r = 1.0;
                                    let mut g = 1.0;
                                    let mut b = 1.0;

                                    if let Some(mesh) = self.asset_manager.get_mesh(mesh_index) {
                                        metallic = mesh.metallic;
                                        roughness = mesh.roughness;
                                        r = mesh.default_color[0];
                                        g = mesh.default_color[1];
                                        b = mesh.default_color[2];

                                        let hx = (mesh.aabb_max[0] - mesh.aabb_min[0]).abs() * 0.5;
                                        let hy = (mesh.aabb_max[1] - mesh.aabb_min[1]).abs() * 0.5;
                                        let hz = (mesh.aabb_max[2] - mesh.aabb_min[2]).abs() * 0.5;

                                        // Avoid zero-thickness colliders
                                        let hx = hx.max(0.01);
                                        let hy = hy.max(0.01);
                                        let hz = hz.max(0.01);

                                        let cx = mesh.aabb_min[0] + hx;
                                        let cy = mesh.aabb_min[1] + hy;
                                        let cz = mesh.aabb_min[2] + hz;

                                        let collider =
                                            rapier3d::prelude::ColliderBuilder::cuboid(hx, hy, hz)
                                                .translation(rapier3d::math::Vector::new(
                                                    cx, cy, cz,
                                                ))
                                                .build();

                                        let handle = self.physics.collider_set.insert(collider);
                                        self.world.add_component(
                                            child_entity,
                                            crate::ecs::components::ColliderComponent { handle },
                                        );
                                    }

                                    self.world.add_component(
                                        child_entity,
                                        crate::ecs::RenderComponent {
                                            visible: true,
                                            mesh_index,
                                            metallic,
                                            roughness,
                                            r,
                                            g,
                                            b,
                                        },
                                    );

                                    self.world.add_component(
                                        child_entity,
                                        crate::ecs::components::HierarchyComponent {
                                            parent: Some(new_entity),
                                            local_matrix: crate::math::mat4::Mat4::identity(),
                                        },
                                    );
                                }
                            }
                        }
                    }
                }
            }

            let asset_events = self.asset_manager.poll_changes(&self.vulkan);
            let mut shader_changed = false;
            let mut texture_changed = false;

            for event in asset_events {
                match event {
                    crate::asset_manager::AssetEvent::ShaderChanged => shader_changed = true,
                    crate::asset_manager::AssetEvent::TextureChanged(_) => texture_changed = true,
                    crate::asset_manager::AssetEvent::ModelChanged(_) => {}
                }
            }

            if shader_changed || texture_changed {
                unsafe {
                    self.vulkan.device.device_wait_idle().unwrap();
                }
                if let Some(mut old_pipe) = self.pipeline.take() {
                    old_pipe.shutdown(&self.vulkan);
                }
                println!("[Hot-Reload] Recreating Vulkan Pipeline...");
                self.pipeline = crate::renderer::vulkan::Pipeline::new(
                    &self.vulkan,
                    vk::Format::R16G16B16A16_SFLOAT,
                    &self.asset_manager.vfs,
                    "shaders/vert.spv",
                    "shaders/frag.spv",
                );

                // Re-allocate descriptor set
                if let Some(pipe) = &self.pipeline {
                    unsafe {
                        self.vulkan
                            .device
                            .reset_descriptor_pool(
                                self.descriptor_pool,
                                vk::DescriptorPoolResetFlags::empty(),
                            )
                            .unwrap();
                    }
                    let layouts = [pipe.descriptor_set_layout, pipe.descriptor_set_layout];
                    let alloc_info = vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(&layouts);

                    if let Ok(sets) =
                        unsafe { self.vulkan.device.allocate_descriptor_sets(&alloc_info) }
                    {
                        self.descriptor_sets = [sets[0], sets[1]];
                        // Update descriptor sets
                        for i in 0..2 {
                            let ubo_info = vk::DescriptorBufferInfo::default()
                                .buffer(self.ubo_buffer)
                                .offset(0)
                                .range(std::mem::size_of::<
                                    crate::renderer::vulkan::pipeline::GlobalUbo,
                                >() as u64);

                            let env_tex = self.asset_manager.get_texture("env_default").unwrap();
                            let shadow_tex =
                                self.asset_manager.get_texture("shadow_default").unwrap();

                            let env_image_info = vk::DescriptorImageInfo::default()
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(env_tex.view)
                                .sampler(env_tex.sampler);

                            let shadow_image_info = vk::DescriptorImageInfo::default()
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(shadow_tex.view)
                                .sampler(shadow_tex.sampler);

                            let instance_info = vk::DescriptorBufferInfo::default()
                                .buffer(self.instance_buffers[i].handle)
                                .offset(0)
                                .range(vk::WHOLE_SIZE);

                            let writes = [
                                vk::WriteDescriptorSet::default()
                                    .dst_set(self.descriptor_sets[i])
                                    .dst_binding(0)
                                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                    .buffer_info(std::slice::from_ref(&ubo_info)),
                                vk::WriteDescriptorSet::default()
                                    .dst_set(self.descriptor_sets[i])
                                    .dst_binding(1)
                                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                    .image_info(std::slice::from_ref(&env_image_info)),
                                vk::WriteDescriptorSet::default()
                                    .dst_set(self.descriptor_sets[i])
                                    .dst_binding(2)
                                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                    .image_info(std::slice::from_ref(&shadow_image_info)),
                                vk::WriteDescriptorSet::default()
                                    .dst_set(self.descriptor_sets[i])
                                    .dst_binding(3)
                                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                    .buffer_info(std::slice::from_ref(&instance_info)),
                            ];

                            unsafe {
                                self.vulkan.device.update_descriptor_sets(&writes, &[]);
                            }
                        }
                    }

                    // Reset material descriptor sets since the pool was reset
                    for set in self.material_descriptor_sets.iter_mut() {
                        *set = None;
                    }
                }
            }

            // Ensure material descriptor sets vector has enough capacity
            if self.material_descriptor_sets.len() < self.asset_manager.meshes.len() {
                self.material_descriptor_sets
                    .resize(self.asset_manager.meshes.len(), None);
            }

            if let Some((w, h)) = new_viewport_size {
                if w != self.offscreen_target.width || h != self.offscreen_target.height {
                    unsafe {
                        self.vulkan.device.device_wait_idle().unwrap();
                    }
                    self.offscreen_target.shutdown(&self.vulkan);
                    let _ = std::mem::replace(
                        &mut self.offscreen_target,
                        crate::renderer::vulkan::OffscreenTarget::new(
                            &self.vulkan,
                            w,
                            h,
                            vk::Format::R16G16B16A16_SFLOAT,
                        )
                        .unwrap(),
                    );
                    self.sdr_target.shutdown(&self.vulkan);
                    self.sdr_target = crate::renderer::vulkan::OffscreenTarget::new(
                        &self.vulkan,
                        w,
                        h,
                        vk::Format::R8G8B8A8_UNORM,
                    )
                    .unwrap();

                    self.bloom_target.shutdown(&self.vulkan);
                    self.bloom_target = crate::renderer::vulkan::bloom::BloomTarget::new(
                        &self.vulkan,
                        (w / 2).max(1),
                        (h / 2).max(1),
                        6,
                    )
                    .unwrap();

                    self.update_post_process_descriptors();

                    self.ui_backend.update_user_texture(
                        &self.vulkan,
                        self.offscreen_texture_id,
                        self.sdr_target.color_view,
                        self.sdr_target.sampler,
                    );
                }
            }

            if let Some((ndc_x, ndc_y)) = raycast_request {
                let cam_entity = {
                    let cameras = self.world.get_component_array::<CameraComponent>();
                    cameras.dense_entities_slice().first().copied()
                };

                if let Some(cam_entity) = cam_entity {
                    let transforms = self.world.get_component_array::<TransformComponent>();
                    let cameras = self.world.get_component_array::<CameraComponent>();
                    if transforms.has(cam_entity) && cameras.has(cam_entity) {
                        let transform = unsafe { transforms.get(cam_entity) };
                        let _camera = unsafe { cameras.get(cam_entity) };

                        let pitch = transform.rotation.x;
                        let yaw = transform.rotation.y;
                        let forward = crate::math::vec::Vec3::new(
                            yaw.sin() * pitch.cos(),
                            pitch.sin(),
                            yaw.cos() * pitch.cos(),
                        )
                        .normalize();
                        let center = transform.position + forward;
                        let view = crate::math::mat4::Mat4::look_at(
                            transform.position,
                            center,
                            crate::math::vec::Vec3::new(0.0, 1.0, 0.0),
                        );
                        let aspect_ratio = self.offscreen_target.width as f32
                            / self.offscreen_target.height as f32;
                        let proj = crate::math::mat4::Mat4::perspective(
                            std::f32::consts::FRAC_PI_4,
                            aspect_ratio,
                            0.1,
                            10000.0,
                        );

                        if let (Some(inv_proj), Some(inv_view)) =
                            (proj.try_inverse(), view.try_inverse())
                        {
                            let mut target =
                                inv_proj * crate::math::vec::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
                            if target.w != 0.0 {
                                target.x /= target.w;
                                target.y /= target.w;
                                target.z /= target.w;
                                target.w = 1.0;
                            }

                            let world_target = inv_view * target;
                            let world_dir = crate::math::vec::Vec3::new(
                                world_target.x - transform.position.x,
                                world_target.y - transform.position.y,
                                world_target.z - transform.position.z,
                            )
                            .normalize();

                            let ray = rapier3d::prelude::Ray::new(
                                rapier3d::math::Point::new(
                                    transform.position.x,
                                    transform.position.y,
                                    transform.position.z,
                                ),
                                rapier3d::math::Vector::new(world_dir.x, world_dir.y, world_dir.z),
                            );

                            if let Some((handle, _toi)) = self.physics.query_pipeline.cast_ray(
                                &self.physics.rigid_body_set,
                                &self.physics.collider_set,
                                &ray,
                                10000.0,
                                true,
                                rapier3d::prelude::QueryFilter::default(),
                            ) {
                                let colliders = self.world.get_component_array::<crate::ecs::components::ColliderComponent>();
                                let dense_colliders = colliders.as_slice();
                                let entities = colliders.dense_entities_slice();
                                self.selected_entity = None;
                                for (i, col) in dense_colliders.iter().enumerate() {
                                    if col.handle == handle {
                                        let mut selected = entities[i];
                                        let hierarchies = self.world.get_component_array::<crate::ecs::components::HierarchyComponent>();
                                        while hierarchies.has(selected) {
                                            if let Some(parent) =
                                                unsafe { hierarchies.get(selected) }.parent
                                            {
                                                selected = parent;
                                            } else {
                                                break;
                                            }
                                        }
                                        self.selected_entity = Some(selected);
                                        break;
                                    }
                                }
                            } else {
                                // Fallback raycast for non-physics visual entities
                                let mut best_dist = f32::MAX;
                                let mut best_entity = None;

                                let renders = self.world.get_component_array::<RenderComponent>();
                                for entity in renders.dense_entities_slice().iter().copied() {
                                    if let Some(matrix) = self
                                        .world_matrices
                                        .iter()
                                        .find(|(e, _)| *e == entity)
                                        .map(|(_, m)| m)
                                    {
                                        let center = crate::math::vec::Vec3::new(
                                            matrix.cols[3].x,
                                            matrix.cols[3].y,
                                            matrix.cols[3].z,
                                        );

                                        let scale_x = crate::math::vec::Vec3::new(
                                            matrix.cols[0].x,
                                            matrix.cols[0].y,
                                            matrix.cols[0].z,
                                        )
                                        .length();
                                        let scale_y = crate::math::vec::Vec3::new(
                                            matrix.cols[1].x,
                                            matrix.cols[1].y,
                                            matrix.cols[1].z,
                                        )
                                        .length();
                                        let scale_z = crate::math::vec::Vec3::new(
                                            matrix.cols[2].x,
                                            matrix.cols[2].y,
                                            matrix.cols[2].z,
                                        )
                                        .length();

                                        // Assume base mesh fits roughly inside a unit sphere
                                        let radius = scale_x.max(scale_y).max(scale_z) * 1.5;

                                        let l = center - transform.position;
                                        let tca = l.dot(world_dir);

                                        if tca >= 0.0 {
                                            let d2 = l.length_sq() - tca * tca;
                                            let r2 = radius * radius;
                                            if d2 <= r2 {
                                                let thc = (r2 - d2).sqrt();
                                                let t = tca - thc;
                                                if t >= 0.0 && t < best_dist {
                                                    best_dist = t;
                                                    best_entity = Some(entity);
                                                }
                                            }
                                        }
                                    }
                                }

                                if let Some(mut selected) = best_entity {
                                    let hierarchies = self.world.get_component_array::<crate::ecs::components::HierarchyComponent>();
                                    while hierarchies.has(selected) {
                                        if let Some(parent) =
                                            unsafe { hierarchies.get(selected) }.parent
                                        {
                                            selected = parent;
                                        } else {
                                            break;
                                        }
                                    }
                                    self.selected_entity = Some(selected);
                                } else {
                                    self.selected_entity = None;
                                }
                            }
                        }
                    }
                }
            }

            // Exit on ESC
            if self.input.is_key_pressed(win32::VK_ESCAPE) {
                break;
            }

            // Save Scene on F5
            if self.input.is_key_pressed(win32::VK_F5) {
                crate::ecs::serialization::save_scene(
                    &self.world,
                    &self.editor.registry,
                    "scene.json",
                );
            }

            // Load Scene on F9
            if self.input.is_key_pressed(win32::VK_F9) {
                crate::ecs::serialization::load_scene(
                    &mut self.world,
                    &self.editor.registry,
                    "scene.json",
                );
            }

            // 1. Update Game State (ECS)
            // Handled by Hot Reloader's game_update call which invokes the Job System.

            // Camera Interactive Update
            {
                let cam_entity = {
                    let cameras = self.world.get_component_array::<CameraComponent>();
                    cameras.dense_entities_slice().first().copied()
                };

                if let Some(cam_entity) = cam_entity {
                    let transforms = self.world.get_component_array_mut::<TransformComponent>();
                    if transforms.has(cam_entity) {
                        let transform = unsafe { transforms.get_mut(cam_entity) };

                        // Update pitch and yaw from mouse input
                        if viewport_hovered && self.input.is_key_down(win32::VK_RBUTTON) {
                            let sensitivity = 0.001;
                            transform.rotation.y += self.input.mouse_dx as f32 * sensitivity;
                            transform.rotation.x -= self.input.mouse_dy as f32 * sensitivity;

                            // Clamp pitch to avoid gimbal lock
                            let max_pitch = std::f32::consts::FRAC_PI_2 - 0.01;
                            if transform.rotation.x > max_pitch {
                                transform.rotation.x = max_pitch;
                            }
                            if transform.rotation.x < -max_pitch {
                                transform.rotation.x = -max_pitch;
                            }
                        }

                        let pitch = transform.rotation.x;
                        let yaw = transform.rotation.y;

                        // Calculate forward and right vectors
                        let forward = Vec3::new(
                            yaw.sin() * pitch.cos(),
                            pitch.sin(),
                            yaw.cos() * pitch.cos(),
                        )
                        .normalize();

                        let right = forward.cross(Vec3::new(0.0, 1.0, 0.0)).normalize();

                        let speed = 2.0 * dt as f32;

                        if viewport_hovered {
                            if self.input.is_key_down(win32::VK_W) {
                                transform.position += forward * speed;
                            }
                            if self.input.is_key_down(win32::VK_S) {
                                transform.position -= forward * speed;
                            }
                            if self.input.is_key_down(win32::VK_A) {
                                transform.position -= right * speed;
                            }
                            if self.input.is_key_down(win32::VK_D) {
                                transform.position += right * speed;
                            }
                            if self.input.is_key_down(win32::VK_TAB) {
                                transform.position.y += speed;
                            }
                            if self.input.is_key_down(win32::VK_SHIFT) {
                                transform.position.y -= speed;
                            }
                        }
                    }
                }
            }

            // Compute World Matrices
            self.world_matrices.clear(); // len → 0, capacity retained

            let transforms = self.world.get_component_array_mut::<TransformComponent>();
            let entities = transforms.dense_entities();

            for (i, transform) in transforms.as_mut_slice().iter_mut().enumerate() {
                let entity = unsafe { *entities.add(i) };

                // Build local matrix: Translation * RotationY * Scale
                let mut rot_y = crate::math::mat4::Mat4::identity();
                let c = transform.rotation.y.cos();
                let s = transform.rotation.y.sin();
                rot_y.cols[0].x = c;
                rot_y.cols[0].z = -s;
                rot_y.cols[2].x = s;
                rot_y.cols[2].z = c;

                let mut t = crate::math::mat4::Mat4::identity();
                t.cols[3].x = transform.position.x;
                t.cols[3].y = transform.position.y;
                t.cols[3].z = transform.position.z;

                let mut sc = crate::math::mat4::Mat4::identity();
                sc.cols[0].x = transform.scale.x;
                sc.cols[1].y = transform.scale.y;
                sc.cols[2].z = transform.scale.z;

                transform.matrix = t * rot_y * sc;

                self.world_matrices.push((entity, transform.matrix));
            }

            // Resolve hierarchy: multiply child local by parent world
            let hierarchies = self.world.get_component_array::<HierarchyComponent>();
            for (i, hier) in hierarchies.as_slice().iter().enumerate() {
                let entity = hierarchies.dense_entities_slice()[i];
                if let Some(parent) = hier.parent {
                    let parent_index = crate::ecs::types::get_entity_index(parent);
                    if let Some(parent_world) = self
                        .world_matrices
                        .iter()
                        .find(|(e, _)| *e == parent_index)
                        .map(|(_, m)| *m)
                    {
                        if let Some(child_idx) =
                            self.world_matrices.iter().position(|(e, _)| *e == entity)
                        {
                            let child_local = self.world_matrices[child_idx].1;
                            self.world_matrices[child_idx].1 = parent_world * child_local;
                        }
                    }
                }
            }

            self.ui_ctx.end_frame();

            // 2. Render Frame
            self.render_frame();

            // 3. Cleanup Frame Allocator
            self.memory.frame_arena().reset(false);
        }

        crate::log_info!("Application shutting down.");

        self.vulkan.wait_idle();

        self.vulkan.wait_idle();
        // All cleanup is now strictly handled by `impl Drop for Application`
        // to prevent double-free crashes during application exit.
    }

    fn render_frame(&mut self) {
        // Only draw if we successfully compiled shaders and uploaded vertices
        let _pipeline = match &self.pipeline {
            Some(p) => p,
            _ => return, // Skip rendering if no pipeline (avoids deadlock)
        };

        // Extract Camera
        let mut view_proj = crate::math::mat4::Mat4::identity();
        let mut camera_pos = [0.0; 4];
        {
            let cameras = self.world.get_component_array::<CameraComponent>();
            let transforms = self.world.get_component_array::<TransformComponent>();
            let dense_cams = cameras.as_slice();
            let cam_entities = cameras.dense_entities_slice();

            if let Some(&cam_entity) = cam_entities.first() {
                let _cam = dense_cams[0];
                if transforms.has(cam_entity) {
                    let cam_transform = unsafe { transforms.get(cam_entity) };
                    let pitch = cam_transform.rotation.x;
                    let yaw = cam_transform.rotation.y;
                    let forward = Vec3::new(
                        yaw.sin() * pitch.cos(),
                        pitch.sin(),
                        yaw.cos() * pitch.cos(),
                    )
                    .normalize();
                    let center = cam_transform.position + forward;
                    let view = crate::math::mat4::Mat4::look_at(
                        cam_transform.position,
                        center,
                        Vec3::new(0.0, 1.0, 0.0),
                    );

                    let aspect_ratio =
                        self.offscreen_target.width as f32 / self.offscreen_target.height as f32;
                    let proj = crate::math::mat4::Mat4::perspective(
                        std::f32::consts::FRAC_PI_4,
                        aspect_ratio,
                        0.1,
                        10000.0,
                    );

                    view_proj = proj * view;
                    camera_pos = [
                        cam_transform.position.x,
                        cam_transform.position.y,
                        cam_transform.position.z,
                        1.0,
                    ];
                }
            }
        }

        // Extract Light
        let mut light_dir = [0.0, -1.0, 0.0, 0.0];
        let mut light_color = [1.0, 1.0, 1.0, 0.0];
        let mut point_lights_array = [crate::renderer::vulkan::pipeline::PointLight::default(); 4];
        let mut num_point_lights = 0;
        {
            let lights = self.world.get_component_array::<LightComponent>();
            let dense_lights = lights.as_slice();
            if let Some(light_comp) = dense_lights.first() {
                light_dir = [
                    light_comp.direction.x,
                    light_comp.direction.y,
                    light_comp.direction.z,
                    0.0,
                ];
                light_color = [
                    light_comp.color.x,
                    light_comp.color.y,
                    light_comp.color.z,
                    1.0,
                ];
            }

            let point_light_components = self
                .world
                .get_component_array::<crate::ecs::components::PointLightComponent>();
            let transforms = self.world.get_component_array::<TransformComponent>();
            let point_lights = point_light_components.as_slice();
            let point_light_entities = point_light_components.dense_entities_slice();

            for (i, pl) in point_lights.iter().enumerate() {
                if num_point_lights >= 4 {
                    break;
                }
                let entity = point_light_entities[i];
                if transforms.has(entity) {
                    let transform = unsafe { transforms.get(entity) };
                    point_lights_array[num_point_lights as usize] =
                        crate::renderer::vulkan::pipeline::PointLight {
                            position: [
                                transform.position.x,
                                transform.position.y,
                                transform.position.z,
                                1.0,
                            ],
                            color: [pl.color.x, pl.color.y, pl.color.z, pl.intensity],
                        };
                    num_point_lights += 1;
                }
            }
        }

        // Update GlobalUbo
        let ubo = crate::renderer::vulkan::pipeline::GlobalUbo {
            view_proj,
            camera_pos,
            light_dir,
            light_color,
            light_space_matrix: crate::math::mat4::Mat4::identity(),
            point_lights: point_lights_array,
            num_point_lights,
            _padding: [0; 3],
        };
        let ubo_size = std::mem::size_of::<crate::renderer::vulkan::pipeline::GlobalUbo>() as u64;
        unsafe {
            let data_ptr = self
                .vulkan
                .device
                .map_memory(self.ubo_memory, 0, ubo_size, vk::MemoryMapFlags::empty())
                .unwrap();
            std::ptr::copy_nonoverlapping(
                &ubo as *const _ as *const u8,
                data_ptr as *mut u8,
                ubo_size as usize,
            );
            self.vulkan.device.unmap_memory(self.ubo_memory);
        }

        // Wait for previous frame to finish
        unsafe {
            self.vulkan
                .device
                .wait_for_fences(
                    &[self.vulkan.in_flight_fences[self.current_frame]],
                    true,
                    u64::MAX,
                )
                .unwrap();
        }

        if self.window.check_and_clear_resized() {
            self.recreate_swapchain();
            return;
        }

        // Acquire next image
        let (image_index, _is_suboptimal) = unsafe {
            match self.swapchain.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                self.vulkan.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain();
                    return;
                }
                Err(e) => {
                    eprintln!("Failed to acquire next image: {:?}", e);
                    return;
                }
            }
        };

        unsafe {
            self.vulkan
                .device
                .reset_fences(&[self.vulkan.in_flight_fences[self.current_frame]])
                .unwrap();
        }

        let command_buffer = self.vulkan.frame_command_buffers[self.current_frame];

        // Begin recording
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.vulkan
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
            self.vulkan
                .device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
        }

        // --- Compute Skinning Pre-pass Setup ---
        let mut skinning_dispatches = Vec::new();
        if let Some(skinning_pipeline) = &self.skinning_pipeline {
            let skeletons = self
                .world
                .get_component_array::<crate::ecs::components::SkeletonComponent>();
            let renders = self
                .world
                .get_component_array::<crate::ecs::RenderComponent>();
            let entities = skeletons.dense_entities_slice();

            for (i, skeleton_comp) in skeletons.as_slice().iter().enumerate() {
                let entity = entities[i];
                if !renders.has(entity) {
                    continue;
                }
                let render = unsafe { renders.get(entity) };
                if !render.visible {
                    continue;
                }

                let mesh_index = render.mesh_index;
                if let Some(mesh) = self.asset_manager.get_mesh(mesh_index) {
                    let mut instance_idx = skeleton_comp.skinning_instance_index;

                    if instance_idx.is_none() {
                        if let Some(new_instance) =
                            crate::renderer::vulkan::compute_skinning::SkinningInstance::new(
                                &self.vulkan,
                                skinning_pipeline,
                                self.descriptor_pool,
                                self.geometry_pool.vertex_buffer.handle,
                                mesh.vertex_offset as u64,
                                mesh.vertex_count,
                            )
                        {
                            let idx = self.skinning_instances.len();
                            self.skinning_instances.push(new_instance);
                            instance_idx = Some(idx);

                            let skeletons_mut = unsafe {
                                &mut *self.world.get_component_array_mut_ptr::<crate::ecs::components::SkeletonComponent>()
                            };
                            let sk_mut = unsafe { skeletons_mut.get_mut(entity) };
                            sk_mut.skinning_instance_index = Some(idx);
                        }
                    }

                    if let Some(idx) = instance_idx {
                        if idx < self.skinning_instances.len() {
                            let instance = &mut self.skinning_instances[idx];
                            if !skeleton_comp.computed_matrices.is_empty() {
                                instance.upload_bone_matrices(
                                    &self.vulkan,
                                    &skeleton_comp.computed_matrices,
                                );
                            }

                            skinning_dispatches.push((idx, mesh.vertex_count));
                        }
                    }
                }
            }
        }

        // --- Compute Culling Pre-pass Setup ---
        let mut instance_data = Vec::new();
        let renders = self
            .world
            .get_component_array::<crate::ecs::RenderComponent>();
        let transforms = self
            .world
            .get_component_array::<crate::ecs::TransformComponent>();
        let dense_renders = renders.as_slice();
        let entities = renders.dense_entities_slice();

        for i in 0..dense_renders.len() {
            let render = &dense_renders[i];
            if render.visible {
                let entity = entities[i];
                if transforms.has(entity) {
                    let transform = unsafe { transforms.get(entity) };
                    let world_matrix = self
                        .world_matrices
                        .iter()
                        .find(|(e, _)| *e == entity)
                        .map(|(_, m)| *m)
                        .unwrap_or(transform.matrix);

                    if let Some(mesh) = self.asset_manager.get_mesh(render.mesh_index) {
                        let albedo_idx: f32 = f32::from_bits(0);
                        let normal_idx: f32 = f32::from_bits(0);
                        let mr_idx: f32 = f32::from_bits(0);
                        let emissive_idx: u32 = 0;

                        instance_data.push(crate::renderer::vulkan::pipeline::InstanceData {
                            world: world_matrix,
                            aabb_min: [-1.0, -1.0, -1.0, 1.0], // Default AABB min for cubes
                            aabb_max: [1.0, 1.0, 1.0, 1.0],    // Default AABB max for cubes
                            color: [render.r, render.g, render.b, albedo_idx],
                            pbr: [render.metallic, render.roughness, normal_idx, mr_idx],
                            geometry: [
                                mesh.index_count,
                                mesh.index_offset as u32,
                                mesh.vertex_offset as u32,
                                emissive_idx,
                            ],
                        });
                    }
                }
            }
        }

        let draw_count = instance_data.len() as u32;

        unsafe {
            let data_ptr = self.instance_mapped[self.current_frame];
            std::ptr::copy_nonoverlapping(
                instance_data.as_ptr(),
                data_ptr as *mut _,
                instance_data.len(),
            );
        }

        let current_draw_count = self.draw_count_buffers[self.current_frame].handle;

        unsafe {
            // Fill draw count with 0
            self.vulkan
                .device
                .cmd_fill_buffer(command_buffer, current_draw_count, 0, 4, 0);

            // Barrier after fill
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);

            self.vulkan.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&memory_barrier),
                &[],
                &[],
            );
        }

        if let Some(compute) = &self.compute_pipeline {
            if draw_count > 0 {
                unsafe {
                    self.vulkan.device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        compute.pipeline,
                    );

                    self.vulkan.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        compute.layout,
                        0,
                        std::slice::from_ref(&compute.descriptor_sets[self.current_frame]),
                        &[],
                    );

                    #[repr(C)]
                    struct CullPushConstants {
                        total_instances: u32,
                    }

                    let pc = CullPushConstants {
                        total_instances: draw_count,
                    };

                    self.vulkan.device.cmd_push_constants(
                        command_buffer,
                        compute.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        std::slice::from_raw_parts(&pc as *const _ as *const u8, 4),
                    );

                    self.vulkan
                        .device
                        .cmd_dispatch(command_buffer, (draw_count + 63) / 64, 1, 1);

                    // Barrier before drawing
                    let draw_barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);

                    self.vulkan.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::DRAW_INDIRECT,
                        vk::DependencyFlags::empty(),
                        std::slice::from_ref(&draw_barrier),
                        &[],
                        &[],
                    );
                }
            }
        }

        // 1. Scene Render Pass
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.05, 0.05, 0.05, 1.0], // Dark studio grey background
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let mut render_graph = crate::renderer::vulkan::render_graph::RenderGraph::new();

        render_graph.add_pass(
            "Scene",
            vec![
                crate::renderer::vulkan::render_graph::PassResource {
                    handle: crate::renderer::vulkan::render_graph::ResourceHandle(
                        self.offscreen_target.color_image,
                    ),
                    state: crate::renderer::vulkan::render_graph::ResourceState {
                        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                    },
                },
                crate::renderer::vulkan::render_graph::PassResource {
                    handle: crate::renderer::vulkan::render_graph::ResourceHandle(
                        self.offscreen_target.depth_image,
                    ),
                    state: crate::renderer::vulkan::render_graph::ResourceState {
                        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                    },
                },
            ],
            |command_buffer| {
                let color_attachment = vk::RenderingAttachmentInfoKHR::default()
                    .image_view(self.offscreen_target.color_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(clear_values[0]);

                let depth_attachment = vk::RenderingAttachmentInfoKHR::default()
                    .image_view(self.offscreen_target.depth_view)
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(clear_values[1]);

                let rendering_info = vk::RenderingInfoKHR::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: self.offscreen_target.width,
                            height: self.offscreen_target.height,
                        },
                    })
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment))
                    .depth_attachment(&depth_attachment);

                unsafe {
                    self.vulkan
                        .device
                        .cmd_begin_rendering(command_buffer, &rendering_info);

                    if let Some(pipeline) = &self.pipeline {
                        self.vulkan.device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.handle,
                        );

                        if self.descriptor_sets[self.current_frame] != vk::DescriptorSet::null() {
                            self.vulkan.device.cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipeline.layout,
                                0,
                                std::slice::from_ref(&self.descriptor_sets[self.current_frame]),
                                &[],
                            );
                        }
                    }

                    let viewport = vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: self.offscreen_target.width as f32,
                        height: self.offscreen_target.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    };
                    self.vulkan.device.cmd_set_viewport(
                        command_buffer,
                        0,
                        std::slice::from_ref(&viewport),
                    );

                    let scissor = vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: self.offscreen_target.width,
                            height: self.offscreen_target.height,
                        },
                    };
                    self.vulkan.device.cmd_set_scissor(
                        command_buffer,
                        0,
                        std::slice::from_ref(&scissor),
                    );

                    // Unified GPU-driven draw call
                    self.vulkan.device.cmd_bind_vertex_buffers(
                        command_buffer,
                        0,
                        &[self.geometry_pool.vertex_buffer.handle],
                        &[0],
                    );
                    self.vulkan.device.cmd_bind_index_buffer(
                        command_buffer,
                        self.geometry_pool.index_buffer.handle,
                        0,
                        vk::IndexType::UINT32,
                    );

                    let max_draws = 100_000;

                    if self.global_texture_descriptor_sets[0] != vk::DescriptorSet::null() {
                        self.vulkan.device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline.as_ref().unwrap().layout,
                            1,
                            std::slice::from_ref(&self.global_texture_descriptor_sets[0]),
                            &[],
                        );
                    }

                    self.vulkan.device.cmd_draw_indexed_indirect_count(
                        command_buffer,
                        self.indirect_buffers[self.current_frame].handle,
                        0, // indirect offset
                        self.draw_count_buffers[self.current_frame].handle,
                        0, // count buffer offset
                        max_draws,
                        std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                    );

                    self.vulkan.device.cmd_end_rendering(command_buffer);
                }
            },
        );

        self.post_process.add_passes(
            &mut render_graph,
            &self.vulkan,
            &self.offscreen_target,
            &self.sdr_target,
            &self.blur_target,
            &self.bloom_target,
            self.tonemap_descriptor_set,
            &self.bloom_descriptor_sets,
            &self.blur_descriptor_sets,
            self.bloom_threshold,
        );

        // 3. UI Render Pass
        let swapchain_image = self.swapchain.images[image_index as usize];
        render_graph.add_pass(
            "UI",
            vec![
                crate::renderer::vulkan::render_graph::PassResource {
                    handle: crate::renderer::vulkan::render_graph::ResourceHandle(
                        self.sdr_target.color_image,
                    ),
                    state: crate::renderer::vulkan::render_graph::ResourceState {
                        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                        access_mask: vk::AccessFlags::SHADER_READ,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                    },
                },
                crate::renderer::vulkan::render_graph::PassResource {
                    handle: crate::renderer::vulkan::render_graph::ResourceHandle(swapchain_image),
                    state: crate::renderer::vulkan::render_graph::ResourceState {
                        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                    },
                },
            ],
            |command_buffer| {
                let ui_clear_values = [vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }];
                let ui_color_attachment = vk::RenderingAttachmentInfoKHR::default()
                    .image_view(self.swapchain.image_views[image_index as usize])
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(ui_clear_values[0]);

                let ui_rendering_info = vk::RenderingInfoKHR::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.swapchain.extent,
                    })
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&ui_color_attachment));

                unsafe {
                    self.vulkan
                        .device
                        .cmd_begin_rendering(command_buffer, &ui_rendering_info);

                    self.ui_backend.draw(
                        &self.vulkan,
                        command_buffer,
                        self.swapchain.extent.width,
                        self.swapchain.extent.height,
                        &self.ui_ctx,
                        &self.ui_font,
                    );

                    self.vulkan.device.cmd_end_rendering(command_buffer);
                }
            },
        );

        self.resource_tracker.clear();
        render_graph.execute(&self.vulkan, command_buffer, &mut self.resource_tracker);

        // 3. Transition swapchain for present
        let present_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::NONE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(swapchain_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        unsafe {
            self.vulkan.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&present_barrier),
            );

            self.resource_tracker.insert(
                crate::renderer::vulkan::render_graph::ResourceHandle(swapchain_image),
                crate::renderer::vulkan::render_graph::ResourceState {
                    layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    stage_mask: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    access_mask: vk::AccessFlags::NONE,
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                },
            );

            self.vulkan
                .device
                .end_command_buffer(command_buffer)
                .unwrap();
        }
        // resource_tracker was updated in-place — no reassignment needed.

        // Submit
        let wait_semaphores = [self.vulkan.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [command_buffer];
        let signal_semaphores =
            [self.vulkan.render_finished_semaphores[(image_index as usize) % 8]];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            if let Err(e) = self.vulkan.device.queue_submit(
                self.vulkan.graphics_queue,
                std::slice::from_ref(&submit_info),
                self.vulkan.in_flight_fences[self.current_frame],
            ) {
                eprintln!("QUEUE SUBMIT FAILED: {:?}", e);
                return;
            }
        }

        // Present
        let swapchains = [self.swapchain.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let result = unsafe {
            self.swapchain
                .swapchain_loader
                .queue_present(self.vulkan.graphics_queue, &present_info)
        };

        match result {
            Ok(suboptimal) => {
                if suboptimal || self.window.check_and_clear_resized() {
                    self.recreate_swapchain();
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain();
            }
            Err(e) => panic!("Failed to present image: {:?}", e),
        }

        // Keep the frame command buffer allocated; it is reset at the start of
        // the next frame after its fence has signaled.
    }
}

impl Drop for Application {
    fn drop(&mut self) {
        unsafe {
            let _ = self.vulkan.device.device_wait_idle();

            // 1. Game Resources
            self.asset_manager.shutdown(&self.vulkan);

            // 2. Render Targets
            self.sdr_target.shutdown(&self.vulkan);
            self.bloom_target.shutdown(&self.vulkan);
            self.offscreen_target.shutdown(&self.vulkan);

            // 3. Pipelines & UI
            self.post_process.destroy(&self.vulkan);
            if let Some(mut p) = self.pipeline.take() {
                p.shutdown(&self.vulkan);
            }
            self.ui_backend.shutdown(&self.vulkan);

            // 4. Descriptor Pools & Buffers
            if self.post_process_descriptor_pool != vk::DescriptorPool::null() {
                self.vulkan
                    .device
                    .destroy_descriptor_pool(self.post_process_descriptor_pool, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.vulkan
                    .device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.compute_descriptor_pool != vk::DescriptorPool::null() {
                self.vulkan
                    .device
                    .destroy_descriptor_pool(self.compute_descriptor_pool, None);
            }
            if self.skinning_descriptor_pool != vk::DescriptorPool::null() {
                self.vulkan
                    .device
                    .destroy_descriptor_pool(self.skinning_descriptor_pool, None);
            }
            for mut instance in self.skinning_instances.drain(..) {
                instance.shutdown(&self.vulkan);
            }
            if let Some(mut cp) = self.compute_pipeline.take() {
                cp.shutdown(&self.vulkan);
            }
            if let Some(mut sp) = self.skinning_pipeline.take() {
                sp.shutdown(&self.vulkan);
            }
            if self.ubo_buffer != vk::Buffer::null() {
                self.vulkan.device.destroy_buffer(self.ubo_buffer, None);
            }
            if self.ubo_memory != vk::DeviceMemory::null() {
                self.vulkan.device.free_memory(self.ubo_memory, None);
            }

            // 5. Core Infrastructure
            self.swapchain.shutdown(&self.vulkan);
            self.window.shutdown();
            self.vulkan.shutdown();
        }
    }
}
