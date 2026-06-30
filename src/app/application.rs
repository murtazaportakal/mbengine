//! The core game loop and application state.

use crate::ecs::World;
use crate::ecs::{TransformComponent, RenderComponent, CameraComponent, LightComponent, HierarchyComponent};
use crate::memory::{MemoryConfig, MemorySubsystem};
use crate::platform::{Timer, Window, win32};
use crate::renderer::vulkan::{Pipeline, RenderPass, Swapchain, VulkanDevice};
use crate::renderer::RenderDevice;
use crate::math::vec::Vec3;
use ash::vk;
use crate::ecs::system::System;

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
    pub scene_render_pass: RenderPass,
    pub ui_render_pass: RenderPass,
    pub offscreen_target: crate::renderer::vulkan::OffscreenTarget,
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
    pub shader_rx: std::sync::mpsc::Receiver<notify::Result<notify::Event>>,
    pub _shader_watcher: notify::RecommendedWatcher,
}

impl Application {
    /// Initialize the engine subsystems.
    pub fn new(title: &str, width: i32, height: i32) -> Option<Self> {
        crate::utils::shader_compiler::compile_all_shaders();
        
        use notify::{Watcher, RecursiveMode};
        let (tx, shader_rx) = std::sync::mpsc::channel();
        let mut _shader_watcher = notify::recommended_watcher(tx).unwrap();
        let _ = _shader_watcher.watch(std::path::Path::new("src/shaders"), RecursiveMode::Recursive);

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
        let mut swapchain = Swapchain::new(&vulkan, &window, width as u32, height as u32)?;
        
        let scene_render_pass = RenderPass::new_scene(&vulkan, vk::Format::B8G8R8A8_SRGB)?;
        let ui_render_pass = RenderPass::new_ui(&vulkan, swapchain.format.format)?;
        swapchain.create_framebuffers(&vulkan, ui_render_pass.handle);
        
        let offscreen_target = crate::renderer::vulkan::OffscreenTarget::new(&vulkan, scene_render_pass.handle, width as u32, height as u32)?;
        let pipeline = Pipeline::new(&vulkan, scene_render_pass.handle, swapchain.extent);

        let mut asset_manager = crate::asset_manager::AssetManager::new();
        let mut descriptor_pool = vk::DescriptorPool::null();
        let mut descriptor_set = vk::DescriptorSet::null();

        let ubo_data = if let Some(pipe) = &pipeline {
            if asset_manager.load_texture(&vulkan, "default", "test_image.png").is_none() {
                asset_manager.load_checkerboard(&vulkan, "fallback");
            }
            let tex = asset_manager.get_texture("default").or_else(|| asset_manager.get_texture("fallback"));
            if let Some(tex) = tex {
                // 1. Create Descriptor Pool
                let pool_sizes = [
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1),
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                ];
                
                let pool_info = vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(1);
                
                descriptor_pool = unsafe { vulkan.device.create_descriptor_pool(&pool_info, None).unwrap() };

                // 2. Allocate Descriptor Set
                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(std::slice::from_ref(&pipe.descriptor_set_layout));

                descriptor_set = unsafe { vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap()[0] };

                // Create UBO buffer
                let ubo_size = std::mem::size_of::<crate::renderer::vulkan::pipeline::GlobalUbo>() as u64;
                let ubo_buffer_info = vk::BufferCreateInfo::default()
                    .size(ubo_size)
                    .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                
                let ubo_buffer = unsafe { vulkan.device.create_buffer(&ubo_buffer_info, None).unwrap() };
                let mem_req = unsafe { vulkan.device.get_buffer_memory_requirements(ubo_buffer) };
                let mem_type_index = vulkan.find_memory_type(mem_req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).unwrap();
                
                let alloc_info = vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_req.size)
                    .memory_type_index(mem_type_index);
                    
                let ubo_memory = unsafe { vulkan.device.allocate_memory(&alloc_info, None).unwrap() };
                unsafe { vulkan.device.bind_buffer_memory(ubo_buffer, ubo_memory, 0).unwrap() };

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
                
                unsafe { vulkan.device.update_descriptor_sets(&[write_desc_ubo, write_desc_sampler], &[]) };
                
                Some((ubo_buffer, ubo_memory))
            } else { None }
        } else { None };

        let (ubo_buffer, ubo_memory) = ubo_data.unwrap_or((vk::Buffer::null(), vk::DeviceMemory::null()));

        let cube_model_indices = asset_manager.load_model(&vulkan, "cube.obj").unwrap_or(&[]).to_vec();
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
            cam_comp.proj = crate::math::mat4::Mat4::perspective(std::f32::consts::FRAC_PI_4, width as f32 / height as f32, 0.1, 100.0);
            world.add_component(camera_entity, cam_comp);
        }

        // Spawn a directional light
        let light_entity = world.create_entity();
        unsafe {
            world.add_component(light_entity, LightComponent {
                // Pointing down and to the left/forward
                direction: Vec3::new(-1.0, -1.0, 1.0).normalize(),
                color: Vec3::new(1.0, 1.0, 1.0),
            });
        }

        // Spawn a few entities in a hierarchy
        // 1. Planet
        let planet = world.create_entity();
        unsafe {
            world.add_component(planet, TransformComponent {
                position: Vec3::new(0.0, 0.0, 0.0),
                scale: Vec3::new(1.0, 1.0, 1.0),
                ..Default::default()
            });
            if let Some(&mesh_index) = cube_model_indices.first() {
                world.add_component(planet, RenderComponent {
                    visible: true,
                    mesh_index,
                    metallic: 0.1,
                    roughness: 0.8,
                });
            }
        }

        // 2. Moon (Child of Planet)
        let moon = world.create_entity();
        unsafe {
            world.add_component(moon, TransformComponent {
                position: Vec3::new(2.5, 0.0, 0.0),
                scale: Vec3::new(0.4, 0.4, 0.4),
                ..Default::default()
            });
            if let Some(&mesh_index) = cube_model_indices.first() {
                world.add_component(moon, RenderComponent {
                    visible: true,
                    mesh_index,
                    metallic: 0.9,
                    roughness: 0.2,
                });
            }
            world.add_component(moon, HierarchyComponent { parent: Some(planet), ..Default::default() });
        }

        // 3. Satellite (Child of Moon)
        let satellite = world.create_entity();
        unsafe {
            world.add_component(satellite, TransformComponent {
                position: Vec3::new(1.5, 0.0, 0.0),
                scale: Vec3::new(0.2, 0.2, 0.2),
                ..Default::default()
            });
            if let Some(&mesh_index) = cube_model_indices.first() {
                world.add_component(satellite, RenderComponent {
                    visible: true,
                    mesh_index,
                    metallic: 1.0,
                    roughness: 0.1,
                });
            }
            world.add_component(satellite, HierarchyComponent { parent: Some(moon), ..Default::default() });
        }

        // Add a Point Light
        let light_entity = world.create_entity();
        unsafe {
            world.add_component(light_entity, TransformComponent {
                position: Vec3::new(3.0, 3.0, 3.0),
                ..Default::default()
            });
            world.add_component(light_entity, crate::ecs::components::PointLightComponent {
                color: Vec3::new(1.0, 0.5, 0.2),
                intensity: 100.0,
            });
        }

        // Add Physics Entities (Floor and Falling Cube)
        // 1. Static Floor
        let floor = world.create_entity();
        let floor_rb = rapier3d::prelude::RigidBodyBuilder::fixed().translation(rapier3d::math::Vector::new(0.0, -5.0, 0.0)).build();
        let floor_handle = physics.rigid_body_set.insert(floor_rb);
        let floor_collider = rapier3d::prelude::ColliderBuilder::cuboid(10.0, 0.5, 10.0).build();
        let floor_col_handle = physics.collider_set.insert_with_parent(floor_collider, floor_handle, &mut physics.rigid_body_set);
        
        unsafe {
            world.add_component(floor, TransformComponent {
                position: Vec3::new(0.0, -5.0, 0.0),
                scale: Vec3::new(10.0, 0.5, 10.0),
                ..Default::default()
            });
            if let Some(&mesh_index) = cube_model_indices.first() {
                world.add_component(floor, RenderComponent {
                    visible: true,
                    mesh_index,
                    metallic: 0.1,
                    roughness: 0.9,
                });
            }
            world.add_component(floor, crate::ecs::components::RigidBodyComponent { handle: floor_handle });
            world.add_component(floor, crate::ecs::components::ColliderComponent { handle: floor_col_handle });
        }

        // 2. Dynamic Falling Cube
        let falling_cube = world.create_entity();
        let falling_rb = rapier3d::prelude::RigidBodyBuilder::dynamic().translation(rapier3d::math::Vector::new(0.0, 10.0, 0.0)).build();
        let falling_handle = physics.rigid_body_set.insert(falling_rb);
        let falling_collider = rapier3d::prelude::ColliderBuilder::cuboid(1.0, 1.0, 1.0).build();
        let falling_col_handle = physics.collider_set.insert_with_parent(falling_collider, falling_handle, &mut physics.rigid_body_set);
        
        unsafe {
            world.add_component(falling_cube, TransformComponent {
                position: Vec3::new(0.0, 10.0, 0.0),
                scale: Vec3::new(1.0, 1.0, 1.0),
                ..Default::default()
            });
            if let Some(&mesh_index) = cube_model_indices.first() {
                world.add_component(falling_cube, RenderComponent {
                    visible: true,
                    mesh_index,
                    metallic: 0.5,
                    roughness: 0.5,
                });
            }
            world.add_component(falling_cube, crate::ecs::components::RigidBodyComponent { handle: falling_handle });
            world.add_component(falling_cube, crate::ecs::components::ColliderComponent { handle: falling_col_handle });
        }

        let egui_ctx = egui::Context::default();
        let mut egui_backend = crate::renderer::vulkan::EguiBackend::new(&vulkan, ui_render_pass.handle);
        let offscreen_texture_id = egui_backend.register_user_texture(&vulkan, offscreen_target.color_view, offscreen_target.sampler);
        let mut offscreen_target = offscreen_target;
        offscreen_target.descriptor_set = *egui_backend.user_textures.get(&match offscreen_texture_id { egui::TextureId::User(id) => id, _ => 0 }).unwrap();

        Some(Self {
            world,
            pipeline,
            asset_manager,
            world_matrices: std::collections::HashMap::new(),
            ubo_buffer,
            ubo_memory,
            descriptor_pool,
            descriptor_set,
            scene_render_pass,
            ui_render_pass,
            offscreen_target,
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
            shader_rx,
            _shader_watcher,
        })
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
        self.swapchain.create_framebuffers(&self.vulkan, self.ui_render_pass.handle);
        
        // Also recreate offscreen target
        self.offscreen_target.shutdown(&self.vulkan);
        self.offscreen_target = crate::renderer::vulkan::OffscreenTarget::new(&self.vulkan, self.scene_render_pass.handle, width, height).unwrap();
        self.offscreen_texture_id = self.egui_backend.register_user_texture(&self.vulkan, self.offscreen_target.color_view, self.offscreen_target.sampler);
        self.offscreen_target.descriptor_set = *self.egui_backend.user_textures.get(&match self.offscreen_texture_id { egui::TextureId::User(id) => id, _ => 0 }).unwrap();
    }

    /// The canonical game loop.
    pub fn run(&mut self) {
        crate::log_info!("Application started.");
        
        // Reset timer before loop starts to avoid a massive first dt
        let _ = self.timer.tick();
        
        while self.window.poll_events(&mut self.input) {
            let dt = self.timer.tick();

            self.input.egui_input.screen_rect = Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(self.window.width as f32 / 2.0, self.window.height as f32 / 2.0)
            ));
            
            let raw_input = self.input.egui_input.take();
            self.egui_ctx.begin_frame(raw_input);
            self.egui_ctx.set_zoom_factor(2.0);

            egui::Window::new("Engine Inspector").show(&self.egui_ctx, |ui| {
                ui.label(format!("FPS: {:.1}", 1.0 / dt));
                ui.label(format!("Entities: {}", self.world.get_component_array::<TransformComponent>().as_slice().len()));
                
                ui.separator();
                if let Some(entity_id) = self.selected_entity {
                    ui.label(format!("Selected Entity: {}", entity_id));
                    
                    let mut transform_changed = false;
                    let mut new_position = crate::math::vec::Vec3::new(0.0, 0.0, 0.0);
                    let mut new_rotation = crate::math::vec::Vec3::new(0.0, 0.0, 0.0);
                    
                    let transforms = self.world.get_component_array_mut::<TransformComponent>();
                    if transforms.has(entity_id) {
                        let transform = unsafe { transforms.get_mut(entity_id) };
                        
                        ui.horizontal(|ui| {
                            ui.label("Position:");
                            transform_changed |= ui.add(egui::DragValue::new(&mut transform.position.x).speed(0.1)).changed();
                            transform_changed |= ui.add(egui::DragValue::new(&mut transform.position.y).speed(0.1)).changed();
                            transform_changed |= ui.add(egui::DragValue::new(&mut transform.position.z).speed(0.1)).changed();
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Rotation:");
                            transform_changed |= ui.add(egui::DragValue::new(&mut transform.rotation.x).speed(0.05)).changed();
                            transform_changed |= ui.add(egui::DragValue::new(&mut transform.rotation.y).speed(0.05)).changed();
                            transform_changed |= ui.add(egui::DragValue::new(&mut transform.rotation.z).speed(0.05)).changed();
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Scale:");
                            ui.add(egui::DragValue::new(&mut transform.scale.x).speed(0.1));
                            ui.add(egui::DragValue::new(&mut transform.scale.y).speed(0.1));
                            ui.add(egui::DragValue::new(&mut transform.scale.z).speed(0.1));
                        });
                        
                        new_position = transform.position;
                        new_rotation = transform.rotation;
                    }
                    
                    if transform_changed {
                        let rb_components = self.world.get_component_array::<crate::ecs::components::RigidBodyComponent>();
                        if rb_components.has(entity_id) {
                            let rb_comp = unsafe { rb_components.get(entity_id) };
                            if let Some(rb) = self.physics.rigid_body_set.get_mut(rb_comp.handle) {
                                rb.set_translation(rapier3d::math::Vector::new(new_position.x, new_position.y, new_position.z), true);
                                // For rotation, Rapier expects a UnitQuaternion. We'll construct it from Euler angles.
                                let quat = rapier3d::math::Rotation::from_euler_angles(new_rotation.x, new_rotation.y, new_rotation.z);
                                rb.set_rotation(quat, true);
                            }
                        }
                    }
                    
                    ui.separator();
                    if ui.button("Deselect").clicked() {
                        self.selected_entity = None;
                    }
                } else {
                    ui.label("No Entity Selected.");
                }
            });

            let mut new_viewport_size = None;
            let mut raycast_request = None;
            
            egui::Window::new("Viewport").show(&self.egui_ctx, |ui| {
                let size = ui.available_size();
                new_viewport_size = Some((size.x.max(1.0) as u32, size.y.max(1.0) as u32));
                // We use texture size to preserve aspect ratio, but scale to fit available size
                let image = egui::Image::new(egui::load::SizedTexture::new(self.offscreen_texture_id, size)).sense(egui::Sense::click());
                let response = ui.add(image);
                
                if response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let local_pos = pos - response.rect.min;
                        let ndc_x = (local_pos.x / size.x) * 2.0 - 1.0;
                        let ndc_y = (local_pos.y / size.y) * 2.0 - 1.0;
                        raycast_request = Some((ndc_x, ndc_y));
                    }
                }
            });

            let mut shader_changed = false;
            while let Ok(event_res) = self.shader_rx.try_recv() {
                if let Ok(event) = event_res {
                    if let notify::EventKind::Modify(_) = event.kind {
                        for path in event.paths {
                            if let Some(ext) = path.extension() {
                                if ext == "vert" || ext == "frag" {
                                    if crate::utils::shader_compiler::compile_shader(&path).is_ok() {
                                        shader_changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if shader_changed {
                unsafe { self.vulkan.device.device_wait_idle().unwrap(); }
                if let Some(mut old_pipe) = self.pipeline.take() {
                    old_pipe.shutdown(&self.vulkan);
                }
                println!("[Hot-Reload] Recreating Vulkan Pipeline...");
                self.pipeline = crate::renderer::vulkan::Pipeline::new(&self.vulkan, self.scene_render_pass.handle, self.swapchain.extent);
                
                // Re-allocate descriptor set
                if let Some(pipe) = &self.pipeline {
                    unsafe { self.vulkan.device.reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty()).unwrap(); }
                    
                    let alloc_info = vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(std::slice::from_ref(&pipe.descriptor_set_layout));
                    
                    if let Ok(sets) = unsafe { self.vulkan.device.allocate_descriptor_sets(&alloc_info) } {
                        self.descriptor_set = sets[0];
                        // Update descriptor set
                        let ubo_info = vk::DescriptorBufferInfo::default()
                            .buffer(self.ubo_buffer)
                            .offset(0)
                            .range(std::mem::size_of::<crate::renderer::vulkan::pipeline::GlobalUbo>() as u64);
                        
                        let tex = self.asset_manager.get_texture("default").or_else(|| self.asset_manager.get_texture("fallback"));
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
                            ];
                            unsafe { self.vulkan.device.update_descriptor_sets(&writes, &[]) };
                        }
                    }
                }
            }

            if let Some((w, h)) = new_viewport_size {
                if w != self.offscreen_target.width || h != self.offscreen_target.height {
                    unsafe { self.vulkan.device.device_wait_idle().unwrap(); }
                    self.offscreen_target.shutdown(&self.vulkan);
                    self.offscreen_target = crate::renderer::vulkan::OffscreenTarget::new(&self.vulkan, self.scene_render_pass.handle, w, h).unwrap();
                    self.egui_backend.update_user_texture(&self.vulkan, self.offscreen_texture_id, self.offscreen_target.color_view, self.offscreen_target.sampler);
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
                        let camera = unsafe { cameras.get(cam_entity) };
                        
                        let pitch = transform.rotation.x;
                        let yaw = transform.rotation.y;
                        let forward = crate::math::vec::Vec3::new(
                            yaw.sin() * pitch.cos(),
                            pitch.sin(),
                            yaw.cos() * pitch.cos(),
                        ).normalize();
                        let center = transform.position + forward;
                        let view = crate::math::mat4::Mat4::look_at(transform.position, center, crate::math::vec::Vec3::new(0.0, 1.0, 0.0));
                        let aspect_ratio = self.offscreen_target.width as f32 / self.offscreen_target.height as f32;
                        let proj = crate::math::mat4::Mat4::perspective(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.1, 100.0);
                        
                        if let (Some(inv_proj), Some(inv_view)) = (proj.try_inverse(), view.try_inverse()) {
                            let mut target = inv_proj * crate::math::vec::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
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
                            ).normalize();
                            
                            let ray = rapier3d::prelude::Ray::new(
                                rapier3d::math::Point::new(transform.position.x, transform.position.y, transform.position.z),
                                rapier3d::math::Vector::new(world_dir.x, world_dir.y, world_dir.z)
                            );
                            
                            if let Some((handle, _toi)) = self.physics.query_pipeline.cast_ray(
                                &self.physics.rigid_body_set,
                                &self.physics.collider_set,
                                &ray,
                                100.0,
                                true,
                                rapier3d::prelude::QueryFilter::default()
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
                                self.selected_entity = None;
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

            // 1. Update Game State (ECS & Physics)
            self.world.update_systems(dt as f32);
            self.physics.update(dt as f32, &mut self.world);

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
                        if self.input.is_key_down(win32::VK_RBUTTON) {
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
                        ).normalize();
                        
                        let right = forward.cross(Vec3::new(0.0, 1.0, 0.0)).normalize();
                        
                        let speed = 2.0 * dt as f32;
                        
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

            // Compute World Matrices
            let mut world_matrices = std::collections::HashMap::new();

            // Collect entity metadata before mutably borrowing transforms
            let render_entities: std::collections::HashSet<u32> = {
                let renders = self.world.get_component_array::<RenderComponent>();
                renders.dense_entities_slice().iter().copied().collect()
            };
            let hierarchy_roots: std::collections::HashSet<u32> = {
                let hier = self.world.get_component_array::<HierarchyComponent>();
                hier.dense_entities_slice().iter().copied().collect()
            };

            let transforms = self.world.get_component_array_mut::<TransformComponent>();
            let entities = transforms.dense_entities_slice().to_vec();

            for (i, transform) in transforms.as_mut_slice().iter_mut().enumerate() {
                let entity = entities[i];

                // Spin renderable entities (not camera/light)
                if render_entities.contains(&entity) {
                    if !hierarchy_roots.contains(&entity) {
                        // Root entity (planet): slow spin
                        transform.rotation.y += 1.0 * dt as f32;
                    } else {
                        // Child entities: spin faster
                        transform.rotation.y += 2.5 * dt as f32;
                    }
                }

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
                self.egui_backend.update_font_texture(&self.vulkan, &delta.image);
            }
            let clipped_primitives = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

            // 2. Render Frame
            self.render_frame(&clipped_primitives, full_output.pixels_per_point);

            // 3. Cleanup Frame Allocator
            self.memory.frame_arena().reset(false);
        }

        crate::log_info!("Application shutting down.");
        
        self.vulkan.wait_idle();
        
        self.asset_manager.shutdown(&self.vulkan);
        if self.descriptor_pool != vk::DescriptorPool::null() {
            unsafe { self.vulkan.device.destroy_descriptor_pool(self.descriptor_pool, None) };
        }
        if let Some(mut p) = self.pipeline.take() {
            p.shutdown(&self.vulkan);
        }
        self.egui_backend.shutdown(&self.vulkan);
        self.offscreen_target.shutdown(&self.vulkan);
        self.scene_render_pass.shutdown(&self.vulkan);
        self.ui_render_pass.shutdown(&self.vulkan);
        self.swapchain.shutdown(&self.vulkan);
        self.vulkan.shutdown();
    }
    
    fn render_frame(&mut self, clipped_primitives: &[egui::ClippedPrimitive], pixels_per_point: f32) {
        // Only draw if we successfully compiled shaders and uploaded vertices
        let pipeline = match &self.pipeline {
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
                    ).normalize();
                    let center = cam_transform.position + forward;
                    let view = crate::math::mat4::Mat4::look_at(cam_transform.position, center, Vec3::new(0.0, 1.0, 0.0));
                    
                    let aspect_ratio = self.offscreen_target.width as f32 / self.offscreen_target.height as f32;
                    let proj = crate::math::mat4::Mat4::perspective(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.1, 100.0);
                    
                    view_proj = proj * view;
                    camera_pos = [cam_transform.position.x, cam_transform.position.y, cam_transform.position.z, 1.0];
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
            if let Some(light) = dense_lights.first() {
                light_dir = [light.direction.x, light.direction.y, light.direction.z, 0.0];
                light_color = [light.color.x, light.color.y, light.color.z, 1.0];
            }

            let point_light_components = self.world.get_component_array::<crate::ecs::components::PointLightComponent>();
            let transforms = self.world.get_component_array::<TransformComponent>();
            let point_lights = point_light_components.as_slice();
            let point_light_entities = point_light_components.dense_entities_slice();

            for (i, pl) in point_lights.iter().enumerate() {
                if num_point_lights >= 4 { break; }
                let entity = point_light_entities[i];
                if transforms.has(entity) {
                    let transform = unsafe { transforms.get(entity) };
                    point_lights_array[num_point_lights as usize] = crate::renderer::vulkan::pipeline::PointLight {
                        position: [transform.position.x, transform.position.y, transform.position.z, 1.0],
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
            point_lights: point_lights_array,
            num_point_lights,
            _padding: [0; 3],
        };
        let ubo_size = std::mem::size_of::<crate::renderer::vulkan::pipeline::GlobalUbo>() as u64;
        unsafe {
            let data_ptr = self.vulkan.device.map_memory(self.ubo_memory, 0, ubo_size, vk::MemoryMapFlags::empty()).unwrap();
            std::ptr::copy_nonoverlapping(&ubo as *const _ as *const u8, data_ptr as *mut u8, ubo_size as usize);
            self.vulkan.device.unmap_memory(self.ubo_memory);
        }

        // Wait for previous frame to finish
        unsafe {
            let _ = self.vulkan.device.wait_for_fences(std::slice::from_ref(&self.vulkan.in_flight_fence), true, u64::MAX);
        }

        // Acquire next image
        let result = unsafe {
            self.swapchain.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                self.vulkan.image_available_semaphore,
                vk::Fence::null(),
            )
        };

        let image_index = match result {
            Ok((idx, suboptimal)) => {
                if suboptimal || self.window.check_and_clear_resized() {
                    self.recreate_swapchain();
                    return;
                }
                idx
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain();
                return;
            }
            Err(e) => panic!("Failed to acquire image: {:?}", e),
        };

        unsafe {
            let _ = self.vulkan.device.reset_fences(std::slice::from_ref(&self.vulkan.in_flight_fence));
        }

        // Allocate a command buffer for this frame
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.vulkan.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        
        let command_buffer = unsafe {
            self.vulkan.device.allocate_command_buffers(&alloc_info).unwrap()[0]
        };

            // Begin recording
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                
            unsafe {
                self.vulkan.device.begin_command_buffer(command_buffer, &begin_info).unwrap();
            }

            // 1. Begin Scene Render Pass (Offscreen)
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.1, 0.1, 0.1, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
                },
            ];
            let scene_render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.scene_render_pass.handle)
                .framebuffer(self.offscreen_target.framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: self.offscreen_target.width, height: self.offscreen_target.height },
                })
                .clear_values(&clear_values);

            unsafe {
                self.vulkan.device.cmd_begin_render_pass(
                    command_buffer,
                    &scene_render_pass_info,
                    vk::SubpassContents::INLINE,
                );

                if let Some(pipeline) = &self.pipeline {
                    self.vulkan.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
                    
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
                self.vulkan.device.cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));

                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: self.offscreen_target.width, height: self.offscreen_target.height },
                };
                self.vulkan.device.cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));

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
                            // Push the world matrix (64 bytes)
                            let world_matrix = *self.world_matrices.get(&entity_index).unwrap_or(&transform.matrix);
                            let push_constants = crate::renderer::vulkan::pipeline::PushConstants {
                                world: world_matrix,
                                metallic: render.metallic,
                                roughness: render.roughness,
                                _padding: [0.0; 2],
                            };
                            let constants_ptr = &push_constants as *const _ as *const u8;
                            let constants_slice = std::slice::from_raw_parts(constants_ptr, std::mem::size_of::<crate::renderer::vulkan::pipeline::PushConstants>());

                            self.vulkan.device.cmd_push_constants(
                                command_buffer,
                                pipeline.layout,
                                vk::ShaderStageFlags::VERTEX,
                                0,
                                constants_slice,
                            );

                            let mesh = self.asset_manager.get_mesh(render.mesh_index).unwrap();
                            self.vulkan.device.cmd_bind_vertex_buffers(
                                command_buffer, 0, &[mesh.vertex_buffer.handle], &[0]
                            );
                            self.vulkan.device.cmd_bind_index_buffer(
                                command_buffer, mesh.index_buffer.handle, 0, vk::IndexType::UINT32
                            );
                            
                            // Draw indexed
                            self.vulkan.device.cmd_draw_indexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
                            _draw_count += 1;
                        }
                    }
                }

                // End Scene Render Pass
                self.vulkan.device.cmd_end_render_pass(command_buffer);
                
                // 2. Begin UI Render Pass (Swapchain)
                let ui_clear_values = [
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0], // Background of the window
                        },
                    },
                ];
                let ui_render_pass_info = vk::RenderPassBeginInfo::default()
                    .render_pass(self.ui_render_pass.handle)
                    .framebuffer(self.swapchain.framebuffers[image_index as usize])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.swapchain.extent,
                    })
                    .clear_values(&ui_clear_values);

                self.vulkan.device.cmd_begin_render_pass(
                    command_buffer,
                    &ui_render_pass_info,
                    vk::SubpassContents::INLINE,
                );

                self.egui_backend.draw(
                    &self.vulkan,
                    command_buffer,
                    clipped_primitives,
                    pixels_per_point,
                    [self.swapchain.extent.width as f32, self.swapchain.extent.height as f32],
                );

                self.vulkan.device.cmd_end_render_pass(command_buffer);
                self.vulkan.device.end_command_buffer(command_buffer).unwrap();
            }

            // Submit
            let wait_semaphores = [self.vulkan.image_available_semaphore];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [command_buffer];
            let signal_semaphores = [self.vulkan.render_finished_semaphore];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            unsafe {
                self.vulkan.device.queue_submit(
                    self.vulkan.graphics_queue,
                    std::slice::from_ref(&submit_info),
                    self.vulkan.in_flight_fence,
                ).unwrap();
            }

            // Present
            let swapchains = [self.swapchain.swapchain];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let result = unsafe {
                self.swapchain.swapchain_loader.queue_present(self.vulkan.graphics_queue, &present_info)
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
                self.vulkan.device.wait_for_fences(std::slice::from_ref(&self.vulkan.in_flight_fence), true, u64::MAX).unwrap();
                self.vulkan.device.reset_command_pool(self.vulkan.command_pool, vk::CommandPoolResetFlags::empty()).unwrap();
                self.vulkan.device.free_command_buffers(self.vulkan.command_pool, &command_buffers);
            }
    }
}
