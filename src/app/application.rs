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
    pub world_matrices: std::collections::HashMap<u32, crate::math::mat4::Mat4>,
    pub ubo_buffer: vk::Buffer,
    pub ubo_memory: vk::DeviceMemory,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub offscreen_target: crate::renderer::vulkan::OffscreenTarget,
    pub sdr_target: crate::renderer::vulkan::OffscreenTarget,
    pub bloom_target: crate::renderer::vulkan::bloom::BloomTarget,
    pub post_process: crate::renderer::vulkan::PostProcessPipeline,
    pub post_process_descriptor_pool: vk::DescriptorPool,
    pub tonemap_descriptor_set: vk::DescriptorSet,
    pub bloom_descriptor_sets: Vec<vk::DescriptorSet>,
    pub offscreen_texture_id: egui::TextureId,
    pub swapchain: Swapchain,
    pub vulkan: VulkanDevice,
    pub window: Window,
    pub input: crate::app::input::Input,
    pub timer: Timer,
    pub memory: MemorySubsystem,
    pub egui_ctx: egui::Context,
    pub egui_backend: crate::renderer::vulkan::EguiBackend,
    pub physics: crate::physics::PhysicsSystem,
    pub selected_entity: Option<crate::ecs::EntityId>,
    pub current_frame: usize,
    pub bloom_threshold: f32,
    pub resource_tracker: std::collections::HashMap<
        crate::renderer::vulkan::render_graph::ResourceHandle,
        crate::renderer::vulkan::render_graph::ResourceState,
    >,
    pub editor: crate::app::editor::Editor,
    pub hot_reloader: Option<crate::app::hot_reload::HotReloader>,
    pub compute_pipeline: Option<crate::renderer::vulkan::compute_cull::ComputeCullPipeline>,
    pub compute_descriptor_pool: vk::DescriptorPool,
    pub compute_descriptor_sets: std::collections::HashMap<usize, vk::DescriptorSet>,
    pub audio_subsystem: Option<crate::audio::AudioSubsystem>,
    pub audio_system: crate::audio::AudioSystem,
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

        let offscreen_target =
            crate::renderer::vulkan::OffscreenTarget::new(&vulkan, target_width, target_height, vk::Format::R16G16B16A16_SFLOAT)?;
        let sdr_target =
            crate::renderer::vulkan::OffscreenTarget::new(&vulkan, target_width, target_height, vk::Format::B8G8R8A8_UNORM)?;
        let mip_levels = 6;
        let bloom_target = crate::renderer::vulkan::bloom::BloomTarget::new(&vulkan, target_width / 2, target_height / 2, mip_levels)?;
        let post_process = crate::renderer::vulkan::PostProcessPipeline::new(&vulkan, vk::Format::B8G8R8A8_UNORM)?;

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
            .set_layouts(std::slice::from_ref(&post_process.tonemap_descriptor_set_layout));
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

        let pipeline = Pipeline::new(&vulkan, vk::Format::R16G16B16A16_SFLOAT);

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
            vulkan.device.create_descriptor_pool(&compute_pool_info, None).unwrap()
        };

        // We can pass a dummy max_meshlets to new() since it no longer allocates an indirect buffer
        let compute_pipeline =
            crate::renderer::vulkan::compute_cull::ComputeCullPipeline::new(&vulkan);

        let mut asset_manager = crate::asset_manager::AssetManager::new();
        let mut descriptor_pool = vk::DescriptorPool::null();
        let mut descriptor_set = vk::DescriptorSet::null();

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
            if let Some(tex) = tex {
                // 1. Create Descriptor Pool
                let pool_sizes = [
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(10),
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(10),
                ];

                let pool_info = vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(1);

                descriptor_pool = unsafe {
                    vulkan
                        .device
                        .create_descriptor_pool(&pool_info, None)
                        .unwrap()
                };

                // 2. Allocate Descriptor Set
                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(std::slice::from_ref(&pipe.descriptor_set_layout));

                descriptor_set =
                    unsafe { vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap()[0] };

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

                // 3. Update Descriptor Set
                let ubo_info = vk::DescriptorBufferInfo::default()
                    .buffer(ubo_buffer)
                    .offset(0)
                    .range(ubo_size);

                let write_desc_ubo = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&ubo_info));

                let image_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(tex.view)
                    .sampler(tex.sampler);

                let write_desc_sampler = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&image_info));

                let write_desc_env = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&image_info));

                let write_desc_shadow = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(3)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&image_info));

                unsafe {
                    vulkan.device.update_descriptor_sets(
                        &[
                            write_desc_ubo,
                            write_desc_sampler,
                            write_desc_env,
                            write_desc_shadow,
                        ],
                        &[],
                    )
                };

                Some((ubo_buffer, ubo_memory))
            } else {
                None
            }
        } else {
            None
        };

        let (ubo_buffer, ubo_memory) =
            ubo_data.unwrap_or((vk::Buffer::null(), vk::DeviceMemory::null()));

        let cube_model_indices = asset_manager
            .load_model(&vulkan, "cube.obj")
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
        }

        let mut physics = crate::physics::PhysicsSystem::new();

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
            world.add_component(camera_entity, crate::ecs::components::AudioListenerComponent::default());
        }

        // Spawn a directional light
        let light_entity = world.create_entity();
        unsafe {
            world.add_component(
                light_entity,
                LightComponent {
                    // Pointing down and to the left/forward
                    direction: Vec3::new(-1.0, -1.0, 1.0).normalize(),
                    color: Vec3::new(1.0, 1.0, 1.0),
                },
            );
        }

        // Spawn a few entities in a hierarchy
        // 1. Planet
        let planet = world.create_entity();
        unsafe {
            world.add_component(
                planet,
                TransformComponent {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    scale: Vec3::new(1.0, 1.0, 1.0),
                    ..Default::default()
                },
            );
            if let Some(&mesh_index) = cube_model_indices.first() {
                world.add_component(
                    planet,
                    RenderComponent {
                        visible: true,
                        mesh_index,
                        metallic: 0.1,
                        roughness: 0.8,
                    },
                );
            }
            world.add_component(planet, crate::ecs::components::AudioEmitterComponent::default());
        }

        // 2. Moon (Child of Planet)
        let moon = world.create_entity();
        unsafe {
            world.add_component(
                moon,
                TransformComponent {
                    position: Vec3::new(2.5, 0.0, 0.0),
                    scale: Vec3::new(0.4, 0.4, 0.4),
                    ..Default::default()
                },
            );
            if let Some(&mesh_index) = cube_model_indices.first() {
                world.add_component(
                    moon,
                    RenderComponent {
                        visible: true,
                        mesh_index,
                        metallic: 0.9,
                        roughness: 0.2,
                    },
                );
            }
            world.add_component(
                moon,
                HierarchyComponent {
                    parent: Some(planet),
                    ..Default::default()
                },
            );
        }

        // 3. Satellite (Child of Moon)
        let satellite = world.create_entity();
        unsafe {
            world.add_component(
                satellite,
                TransformComponent {
                    position: Vec3::new(1.5, 0.0, 0.0),
                    scale: Vec3::new(0.2, 0.2, 0.2),
                    ..Default::default()
                },
            );
            if let Some(&mesh_index) = cube_model_indices.first() {
                world.add_component(
                    satellite,
                    RenderComponent {
                        visible: true,
                        mesh_index,
                        metallic: 1.0,
                        roughness: 0.1,
                    },
                );
            }
            world.add_component(
                satellite,
                HierarchyComponent {
                    parent: Some(moon),
                    ..Default::default()
                },
            );
        }

        // Add a Point Light
        let light_entity = world.create_entity();
        unsafe {
            world.add_component(
                light_entity,
                TransformComponent {
                    position: Vec3::new(3.0, 3.0, 3.0),
                    ..Default::default()
                },
            );
            world.add_component(
                light_entity,
                crate::ecs::components::PointLightComponent {
                    color: Vec3::new(1.0, 0.5, 0.2),
                    intensity: 100.0,
                },
            );
        }

        // Add Physics Entities (Floor and Falling Cube)
        // 1. Static Floor
        let floor = world.create_entity();
        let floor_rb = rapier3d::prelude::RigidBodyBuilder::fixed()
            .translation(rapier3d::math::Vector::new(0.0, -5.0, 0.0))
            .build();
        let floor_handle = physics.rigid_body_set.insert(floor_rb);
        let floor_collider = rapier3d::prelude::ColliderBuilder::cuboid(10.0, 0.5, 10.0).build();
        let floor_col_handle = physics.collider_set.insert_with_parent(
            floor_collider,
            floor_handle,
            &mut physics.rigid_body_set,
        );

        unsafe {
            world.add_component(
                floor,
                TransformComponent {
                    position: Vec3::new(0.0, -5.0, 0.0),
                    scale: Vec3::new(10.0, 0.5, 10.0),
                    ..Default::default()
                },
            );
            if let Some(&mesh_index) = cube_model_indices.first() {
                world.add_component(
                    floor,
                    RenderComponent {
                        visible: true,
                        mesh_index,
                        metallic: 0.1,
                        roughness: 0.9,
                    },
                );
            }
            world.add_component(
                floor,
                crate::ecs::components::RigidBodyComponent {
                    handle: floor_handle,
                },
            );
            world.add_component(
                floor,
                crate::ecs::components::ColliderComponent {
                    handle: floor_col_handle,
                },
            );
        }

        // 2. Dynamic Falling Cube
        let falling_cube = world.create_entity();
        let falling_rb = rapier3d::prelude::RigidBodyBuilder::dynamic()
            .translation(rapier3d::math::Vector::new(0.0, 10.0, 0.0))
            .build();
        let falling_handle = physics.rigid_body_set.insert(falling_rb);
        let falling_collider = rapier3d::prelude::ColliderBuilder::cuboid(1.0, 1.0, 1.0).build();
        let falling_col_handle = physics.collider_set.insert_with_parent(
            falling_collider,
            falling_handle,
            &mut physics.rigid_body_set,
        );

        unsafe {
            world.add_component(
                falling_cube,
                TransformComponent {
                    position: Vec3::new(0.0, 10.0, 0.0),
                    scale: Vec3::new(1.0, 1.0, 1.0),
                    ..Default::default()
                },
            );
            if let Some(&mesh_index) = cube_model_indices.first() {
                world.add_component(
                    falling_cube,
                    RenderComponent {
                        visible: true,
                        mesh_index,
                        metallic: 0.5,
                        roughness: 0.5,
                    },
                );
            }
            world.add_component(
                falling_cube,
                crate::ecs::components::RigidBodyComponent {
                    handle: falling_handle,
                },
            );
            world.add_component(
                falling_cube,
                crate::ecs::components::ColliderComponent {
                    handle: falling_col_handle,
                },
            );
        }

        // 3. Bugatti Model
        if let Some(bugatti_indices) = asset_manager.load_model(&vulkan, "bugatti.obj") {
            let bugatti_root = world.create_entity();
            unsafe {
                world.add_component(
                    bugatti_root,
                    TransformComponent {
                        position: Vec3::new(0.0, 0.0, 0.0),
                        scale: Vec3::new(1.0, 1.0, 1.0),
                        ..Default::default()
                    },
                );
            }

            for &mesh_index in bugatti_indices {
                let bugatti_part = world.create_entity();
                unsafe {
                    world.add_component(
                        bugatti_part,
                        TransformComponent {
                            position: Vec3::new(0.0, 0.0, 0.0),
                            scale: Vec3::new(1.0, 1.0, 1.0),
                            ..Default::default()
                        },
                    );
                    world.add_component(
                        bugatti_part,
                        RenderComponent {
                            visible: true,
                            mesh_index,
                            metallic: 0.8,
                            roughness: 0.2,
                        },
                    );
                    world.add_component(
                        bugatti_part,
                        HierarchyComponent {
                            parent: Some(bugatti_root),
                            ..Default::default()
                        },
                    );
                }
            }
        }

        let egui_ctx = egui::Context::default();
        let mut egui_backend =
            crate::renderer::vulkan::EguiBackend::new(&vulkan, swapchain.format.format);
        let offscreen_texture_id = egui_backend.register_user_texture(
            &vulkan,
            sdr_target.color_view,
            sdr_target.sampler,
        );

        let audio_subsystem = crate::audio::AudioSubsystem::new();
        let audio_system = crate::audio::AudioSystem::new(audio_subsystem.as_ref());

        let app = Self {
            world,
            pipeline,
            asset_manager,
            world_matrices: std::collections::HashMap::new(),
            ubo_buffer,
            ubo_memory,
            descriptor_pool,
            descriptor_set,
            offscreen_target,
            sdr_target,
            bloom_target,
            post_process,
            post_process_descriptor_pool,
            tonemap_descriptor_set,
            bloom_descriptor_sets,
            offscreen_texture_id,
            swapchain,
            vulkan,
            window,
            input,
            timer,
            memory,
            egui_ctx,
            egui_backend,
            physics,
            selected_entity: None,
            current_frame: 0,
            bloom_threshold: 1.0,
            resource_tracker: std::collections::HashMap::new(),
            editor: crate::app::editor::Editor::new(),
            hot_reloader: crate::app::hot_reload::HotReloader::new("target/debug/game.dll"),
            compute_pipeline,
            compute_descriptor_pool,
            compute_descriptor_sets: std::collections::HashMap::new(),
            audio_subsystem,
            audio_system,
        };
        app.update_post_process_descriptors();
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
            
        writes.push(vk::WriteDescriptorSet::default()
            .dst_set(self.tonemap_descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&tonemap_color_info));
        writes.push(vk::WriteDescriptorSet::default()
            .dst_set(self.tonemap_descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&tonemap_bloom_info));

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
            writes.push(vk::WriteDescriptorSet::default()
                .dst_set(self.bloom_descriptor_sets[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(info));
        }

        unsafe { self.vulkan.device.update_descriptor_sets(&writes, &[]); }
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
            vk::Format::B8G8R8A8_UNORM,
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

        self.egui_backend.update_user_texture(
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
            
            self.input.egui_input.screen_rect = Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(
                    self.window.width as f32 / ppp,
                    self.window.height as f32 / ppp,
                ),
            ));

            let raw_input = self.input.egui_input.take();
            self.egui_ctx.begin_frame(raw_input);
            self.egui_ctx.set_zoom_factor(ppp);

            if let Some(reloader) = &mut self.hot_reloader {
                reloader.update();
                reloader.call_game_update(&mut self.world, &mut self.physics, dt as f32);
            }
            
            use crate::ecs::System;
            self.audio_system.update(dt as f32, &self.world);

            let mut new_viewport_size = None;
            let mut raycast_request = None;
            let mut viewport_hovered = false;

            self.editor.draw(&self.egui_ctx, &mut self.world, &mut self.physics, &mut self.selected_entity, &mut self.bloom_threshold, 1.0 / dt as f32);

            egui::CentralPanel::default().show(&self.egui_ctx, |ui| {
                let size = ui.available_size();
                new_viewport_size = Some((size.x.max(1.0) as u32, size.y.max(1.0) as u32));
                let image = egui::Image::new(egui::load::SizedTexture::new(
                    self.offscreen_texture_id,
                    size,
                ))
                .sense(egui::Sense::click() | egui::Sense::drag());
                
                let response = ui.add(image);
                viewport_hovered = response.hovered() || response.dragged();
                
                if response.clicked() {
                    self.selected_entity = None;

                    if let Some(pos) = response.interact_pointer_pos() {
                        let local_pos = pos - response.rect.min;
                        let ndc_x = (local_pos.x / response.rect.width()) * 2.0 - 1.0;
                        let ndc_y = (local_pos.y / response.rect.height()) * 2.0 - 1.0;
                        raycast_request = Some((ndc_x, ndc_y));
                    }
                }
            });

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

                    let alloc_info = vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(std::slice::from_ref(&pipe.descriptor_set_layout));

                    if let Ok(sets) =
                        unsafe { self.vulkan.device.allocate_descriptor_sets(&alloc_info) }
                    {
                        self.descriptor_set = sets[0];
                        // Update descriptor set
                        let ubo_info = vk::DescriptorBufferInfo::default()
                            .buffer(self.ubo_buffer)
                            .offset(0)
                            .range(
                                std::mem::size_of::<crate::renderer::vulkan::pipeline::GlobalUbo>()
                                    as u64,
                            );

                        let tex = self
                            .asset_manager
                            .get_texture("default")
                            .or_else(|| self.asset_manager.get_texture("fallback"));
                        if let Some(tex) = tex {
                            let image_info = vk::DescriptorImageInfo::default()
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(tex.view)
                                .sampler(tex.sampler);

                            let writes = [
                                vk::WriteDescriptorSet::default()
                                    .dst_set(self.descriptor_set)
                                    .dst_binding(0)
                                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                    .buffer_info(std::slice::from_ref(&ubo_info)),
                                vk::WriteDescriptorSet::default()
                                    .dst_set(self.descriptor_set)
                                    .dst_binding(1)
                                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                    .image_info(std::slice::from_ref(&image_info)),
                                vk::WriteDescriptorSet::default()
                                    .dst_set(self.descriptor_set)
                                    .dst_binding(2)
                                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                    .image_info(std::slice::from_ref(&image_info)),
                                vk::WriteDescriptorSet::default()
                                    .dst_set(self.descriptor_set)
                                    .dst_binding(3)
                                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                    .image_info(std::slice::from_ref(&image_info)),
                            ];
                            unsafe { self.vulkan.device.update_descriptor_sets(&writes, &[]) };
                        }
                    }
                }
            }

            if let Some((w, h)) = new_viewport_size {
                if w != self.offscreen_target.width || h != self.offscreen_target.height {
                    unsafe {
                        self.vulkan.device.device_wait_idle().unwrap();
                    }
                    self.offscreen_target.shutdown(&self.vulkan);
                    let _ = std::mem::replace(
                        &mut self.offscreen_target,
                        crate::renderer::vulkan::OffscreenTarget::new(&self.vulkan, w, h, vk::Format::R16G16B16A16_SFLOAT).unwrap(),
                    );
                    self.sdr_target.shutdown(&self.vulkan);
                    self.sdr_target = crate::renderer::vulkan::OffscreenTarget::new(&self.vulkan, w, h, vk::Format::B8G8R8A8_UNORM).unwrap();
                    
                    self.bloom_target.shutdown(&self.vulkan);
                    self.bloom_target = crate::renderer::vulkan::bloom::BloomTarget::new(&self.vulkan, (w / 2).max(1), (h / 2).max(1), 6).unwrap();

                    self.update_post_process_descriptors();

                    self.egui_backend.update_user_texture(
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
                            100.0,
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
                                100.0,
                                true,
                                rapier3d::prelude::QueryFilter::default(),
                            ) {
                                let colliders = self.world.get_component_array::<crate::ecs::components::ColliderComponent>();
                                let dense_colliders = colliders.as_slice();
                                let entities = colliders.dense_entities_slice();
                                self.selected_entity = None;
                                for (i, col) in dense_colliders.iter().enumerate() {
                                    if col.handle == handle {
                                        self.selected_entity = Some(entities[i]);
                                        break;
                                    }
                                }
                            } else {
                                // Fallback raycast for non-physics visual entities
                                let mut best_dist = std::f32::MAX;
                                let mut best_entity = None;

                                let renders = self.world.get_component_array::<RenderComponent>();
                                for entity in renders.dense_entities_slice().iter().copied() {
                                    if let Some(matrix) = self.world_matrices.get(&entity) {
                                        let center = crate::math::vec::Vec3::new(matrix.cols[3].x, matrix.cols[3].y, matrix.cols[3].z);
                                        
                                        let scale_x = crate::math::vec::Vec3::new(matrix.cols[0].x, matrix.cols[0].y, matrix.cols[0].z).length();
                                        let scale_y = crate::math::vec::Vec3::new(matrix.cols[1].x, matrix.cols[1].y, matrix.cols[1].z).length();
                                        let scale_z = crate::math::vec::Vec3::new(matrix.cols[2].x, matrix.cols[2].y, matrix.cols[2].z).length();
                                        
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
                                self.selected_entity = best_entity;
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
                crate::ecs::serialization::save_scene(&self.world, "scene.json");
            }

            // Load Scene on F9
            if self.input.is_key_pressed(win32::VK_F9) {
                crate::ecs::serialization::load_scene(&mut self.world, "scene.json");
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
                                transform.position = transform.position + forward * speed;
                            }
                            if self.input.is_key_down(win32::VK_S) {
                                transform.position = transform.position - forward * speed;
                            }
                            if self.input.is_key_down(win32::VK_A) {
                                transform.position = transform.position - right * speed;
                            }
                            if self.input.is_key_down(win32::VK_D) {
                                transform.position = transform.position + right * speed;
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
            let mut world_matrices = std::collections::HashMap::new();

            let transforms = self.world.get_component_array_mut::<TransformComponent>();
            let entities = transforms.dense_entities_slice().to_vec();

            for (i, transform) in transforms.as_mut_slice().iter_mut().enumerate() {
                let entity = entities[i];

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

                world_matrices.insert(entity, transform.matrix);
            }

            // Resolve hierarchy: multiply child local by parent world
            let hierarchies = self.world.get_component_array::<HierarchyComponent>();
            for (i, hier) in hierarchies.as_slice().iter().enumerate() {
                let entity = hierarchies.dense_entities_slice()[i];
                if let Some(parent) = hier.parent {
                    if let Some(&parent_world) = world_matrices.get(&parent) {
                        if let Some(child_local) = world_matrices.get(&entity).copied() {
                            world_matrices.insert(entity, parent_world * child_local);
                        }
                    }
                }
            }

            self.world_matrices = world_matrices;

            let full_output = self.egui_ctx.end_frame();
            for (_id, delta) in full_output.textures_delta.set {
                self.egui_backend
                    .update_texture(&self.vulkan, egui::TextureId::Managed(0), &delta);
            }
            let clipped_primitives = self
                .egui_ctx
                .tessellate(full_output.shapes, full_output.pixels_per_point);

            // 2. Render Frame
            self.render_frame(&clipped_primitives, full_output.pixels_per_point);

            // 3. Cleanup Frame Allocator
            self.memory.frame_arena().reset(false);
        }

        crate::log_info!("Application shutting down.");

        self.vulkan.wait_idle();

        self.vulkan.wait_idle();
        // All cleanup is now strictly handled by `impl Drop for Application`
        // to prevent double-free crashes during application exit.
    }

    fn render_frame(
        &mut self,
        clipped_primitives: &[egui::ClippedPrimitive],
        pixels_per_point: f32,
    ) {
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
                        100.0,
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
            let _ = self
                .vulkan
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
            let _ = self
                .vulkan
                .device
                .reset_fences(&[self.vulkan.in_flight_fences[self.current_frame]])
                .unwrap();
        }

        // Allocate a command buffer for this frame
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.vulkan.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            self.vulkan
                .device
                .allocate_command_buffers(&alloc_info)
                .unwrap()[0]
        };

        // Begin recording
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.vulkan
                .device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
        }

        // --- Compute Culling Pre-pass Setup ---
        let mut compute_dispatches = Vec::new();
        if let Some(compute) = &self.compute_pipeline {
            let renders = self.world.get_component_array::<crate::ecs::RenderComponent>();
            let transforms = self.world.get_component_array::<crate::ecs::TransformComponent>();
            let dense_renders = renders.as_slice();
            let entities = renders.dense_entities_slice();

            for i in 0..dense_renders.len() {
                let render = &dense_renders[i];
                if render.visible {
                    let entity_index = entities[i];
                    if transforms.has(entity_index) {
                        let transform = unsafe { transforms.get(entity_index) };
                        let world_matrix = *self
                            .world_matrices
                            .get(&entity_index)
                            .unwrap_or(&transform.matrix);
                        let mesh_index = render.mesh_index;

                        if let Some(mesh) = self.asset_manager.get_mesh(mesh_index) {
                            if !self.compute_descriptor_sets.contains_key(&mesh_index) {
                                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                                    .descriptor_pool(self.compute_descriptor_pool)
                                    .set_layouts(std::slice::from_ref(&compute.descriptor_set_layout));
                                let set = unsafe { self.vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap()[0] };
                                
                                compute.update_descriptor_set(
                                    &self.vulkan, 
                                    self.ubo_buffer, 
                                    mesh.meshlet_buffer.handle, 
                                    mesh.indirect_buffer.handle, 
                                    set
                                );
                                self.compute_descriptor_sets.insert(mesh_index, set);
                            }
                            
                            let set = self.compute_descriptor_sets[&mesh_index];
                            compute_dispatches.push((mesh_index, mesh.meshlet_count, world_matrix, set));
                        }
                    }
                }
            }

            // Dispatch compute for each entity's mesh
            if !compute_dispatches.is_empty() {
                unsafe {
                    self.vulkan.device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        compute.pipeline,
                    );
                }

                for (_mesh_index, meshlet_count, world_matrix, set) in compute_dispatches {
                    unsafe {
                        self.vulkan.device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            compute.layout,
                            0,
                            std::slice::from_ref(&set),
                            &[],
                        );
                        
                        #[repr(C)]
                        struct PushConstants {
                            total_meshlets: u32,
                            _pad: [u32; 3], // 12 bytes padding for 16-byte alignment of Mat4
                            world: crate::math::mat4::Mat4,
                        }
                        
                        let pc = PushConstants {
                            total_meshlets: meshlet_count,
                            _pad: [0; 3],
                            world: world_matrix,
                        };
                        
                        let pc_bytes = std::slice::from_raw_parts(
                            &pc as *const _ as *const u8,
                            std::mem::size_of::<PushConstants>(),
                        );
                        
                        self.vulkan.device.cmd_push_constants(
                            command_buffer,
                            compute.layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            pc_bytes,
                        );

                        // 64 threads per group (match local_size_x in cull.comp)
                        let group_count_x = (meshlet_count + 63) / 64;
                        self.vulkan.device.cmd_dispatch(command_buffer, group_count_x, 1, 1);
                    }
                }

                // Barrier to ensure compute shader writes are visible to indirect draw
                unsafe {
                    let barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);
                        
                    self.vulkan.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::DRAW_INDIRECT,
                        vk::DependencyFlags::empty(),
                        std::slice::from_ref(&barrier),
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
                    float32: [0.1, 0.1, 0.1, 1.0],
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

                        if self.descriptor_set != vk::DescriptorSet::null() {
                            self.vulkan.device.cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipeline.layout,
                                0,
                                std::slice::from_ref(&self.descriptor_set),
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

                    let renders = self.world.get_component_array::<RenderComponent>();
                    let transforms = self.world.get_component_array::<TransformComponent>();

                    let mut _draw_count = 0;
                    let dense_renders = renders.as_slice();
                    let entities = renders.dense_entities_slice();

                    for i in 0..dense_renders.len() {
                        let render = &dense_renders[i];
                        if render.visible {
                            let entity_index = entities[i];
                            if transforms.has(entity_index) {
                                let transform = transforms.get(entity_index);
                                let world_matrix = *self
                                    .world_matrices
                                    .get(&entity_index)
                                    .unwrap_or(&transform.matrix);
                                let push_constants =
                                    crate::renderer::vulkan::pipeline::PushConstants {
                                        world: world_matrix,
                                        metallic: render.metallic,
                                        roughness: render.roughness,
                                        _padding: [0.0; 2],
                                    };
                                let constants_ptr = &push_constants as *const _ as *const u8;
                                let constants_slice = std::slice::from_raw_parts(
                                    constants_ptr,
                                    std::mem::size_of::<
                                        crate::renderer::vulkan::pipeline::PushConstants,
                                    >(),
                                );

                                self.vulkan.device.cmd_push_constants(
                                    command_buffer,
                                    self.pipeline.as_ref().unwrap().layout,
                                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                    0,
                                    constants_slice,
                                );

                                let mesh = self.asset_manager.get_mesh(render.mesh_index).unwrap();
                                self.vulkan.device.cmd_bind_vertex_buffers(
                                    command_buffer,
                                    0,
                                    &[mesh.vertex_buffer.handle],
                                    &[0],
                                );
                                self.vulkan.device.cmd_bind_index_buffer(
                                    command_buffer,
                                    mesh.index_buffer.handle,
                                    0,
                                    vk::IndexType::UINT32,
                                );

                                if self.compute_pipeline.is_some() {
                                    self.vulkan.device.cmd_draw_indexed_indirect(
                                        command_buffer,
                                        mesh.indirect_buffer.handle,
                                        0,
                                        mesh.meshlet_count,
                                        std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                                    );
                                } else {
                                    self.vulkan.device.cmd_draw_indexed(
                                        command_buffer,
                                        mesh.index_count,
                                        1,
                                        0,
                                        0,
                                        0,
                                    );
                                }
                                _draw_count += 1;
                            }
                        }
                    }

                    self.vulkan.device.cmd_end_rendering(command_buffer);
                }
            },
        );

        self.post_process.add_passes(
            &mut render_graph,
            &self.vulkan,
            &self.offscreen_target,
            &self.sdr_target,
            &self.bloom_target,
            self.tonemap_descriptor_set,
            &self.bloom_descriptor_sets,
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

                    self.egui_backend.draw(
                        &self.vulkan,
                        command_buffer,
                        clipped_primitives,
                        pixels_per_point,
                        [
                            self.swapchain.extent.width as f32 / pixels_per_point,
                            self.swapchain.extent.height as f32 / pixels_per_point,
                        ],
                    );

                    self.vulkan.device.cmd_end_rendering(command_buffer);
                }
            },
        );

        let mut tracker = std::mem::take(&mut self.resource_tracker);
        render_graph.execute(&self.vulkan, command_buffer, &mut tracker);

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

            tracker.insert(
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
        self.resource_tracker = tracker;

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

        // Clean up command buffer (in a real engine we'd reuse them per-frame in an array)
        unsafe {
            self.vulkan
                .device
                .wait_for_fences(
                    &[self.vulkan.in_flight_fences[self.current_frame]],
                    true,
                    u64::MAX,
                )
                .unwrap();
            self.vulkan
                .device
                .reset_command_pool(self.vulkan.command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap();
            self.vulkan
                .device
                .free_command_buffers(self.vulkan.command_pool, &command_buffers);
        }
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
            self.egui_backend.shutdown(&self.vulkan);
            
            // 4. Descriptor Pools & Buffers
            if self.post_process_descriptor_pool != vk::DescriptorPool::null() {
                self.vulkan.device.destroy_descriptor_pool(self.post_process_descriptor_pool, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.vulkan.device.destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.compute_descriptor_pool != vk::DescriptorPool::null() {
                self.vulkan.device.destroy_descriptor_pool(self.compute_descriptor_pool, None);
            }
            if let Some(mut cp) = self.compute_pipeline.take() {
                cp.shutdown(&self.vulkan);
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
