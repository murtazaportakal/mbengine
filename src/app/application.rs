//! The core game loop and application state.

use crate::ecs::World;
use crate::ecs::{TransformComponent, RenderComponent, CameraComponent, LightComponent, HierarchyComponent};
use crate::memory::{MemoryConfig, MemorySubsystem};
use crate::platform::{Timer, Window, win32};
use crate::renderer::vulkan::{Pipeline, RenderPass, Swapchain, VulkanDevice};
use crate::renderer::RenderDevice;
use crate::math::vec::Vec3;
use ash::vk;

/// High-level engine coordinator.
/// Note: Field declaration order dictates drop order.
/// `world` must drop before `memory`. `swapchain` must drop before `vulkan`.
pub struct Application {
    pub world: World,
    pub pipeline: Option<Pipeline>,
    pub texture: Option<crate::renderer::vulkan::Texture>,
    pub meshes: Vec<crate::renderer::vulkan::Mesh>,
    pub world_matrices: std::collections::HashMap<u32, crate::math::mat4::Mat4>,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub render_pass: RenderPass,
    pub swapchain: Swapchain,
    pub vulkan: VulkanDevice,
    pub window: Window,
    pub input: crate::app::input::Input,
    pub timer: Timer,
    pub memory: MemorySubsystem,
}

impl Application {
    /// Initialize the engine subsystems.
    pub fn new(title: &str, width: i32, height: i32) -> Option<Self> {
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
        
        let render_pass = RenderPass::new(&vulkan, swapchain.format.format)?;
        swapchain.create_framebuffers(&vulkan, render_pass.handle);
        
        let pipeline = Pipeline::new(&vulkan, render_pass.handle, swapchain.extent);

        let mut texture = None;
        let mut descriptor_pool = vk::DescriptorPool::null();
        let mut descriptor_set = vk::DescriptorSet::null();

        if let Some(pipe) = &pipeline {
            if let Some(tex) = crate::renderer::vulkan::Texture::new_checkerboard(&vulkan) {
                // 1. Create Descriptor Pool
                let pool_size = vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1);
                
                let pool_info = vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(std::slice::from_ref(&pool_size))
                    .max_sets(1);
                
                descriptor_pool = unsafe { vulkan.device.create_descriptor_pool(&pool_info, None).unwrap() };

                // 2. Allocate Descriptor Set
                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(std::slice::from_ref(&pipe.descriptor_set_layout));

                descriptor_set = unsafe { vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap()[0] };

                // 3. Update Descriptor Set
                let image_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(tex.view)
                    .sampler(tex.sampler);
                
                let write_desc = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&image_info));
                
                unsafe { vulkan.device.update_descriptor_sets(std::slice::from_ref(&write_desc), &[]) };
                
                texture = Some(tex);
            }
        }

        let mut meshes = Vec::new();
        if let Some(cube_mesh) = crate::renderer::vulkan::Mesh::load_obj("cube.obj", &vulkan) {
            meshes.push(cube_mesh);
        } else {
            crate::log_info!("Failed to load cube.obj. Rendering will fail.");
        }

        let world_matrices = std::collections::HashMap::new();

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
            world.register_component::<HierarchyComponent>(1000);
        }

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
            world.add_component(planet, RenderComponent {
                visible: true,
                mesh_index: 0,
            });
        }

        // 2. Moon (Child of Planet)
        let moon = world.create_entity();
        unsafe {
            world.add_component(moon, TransformComponent {
                position: Vec3::new(2.5, 0.0, 0.0),
                scale: Vec3::new(0.4, 0.4, 0.4),
                ..Default::default()
            });
            world.add_component(moon, RenderComponent {
                visible: true,
                mesh_index: 0,
            });
            world.add_component(moon, HierarchyComponent { parent: Some(planet) });
        }

        // 3. Satellite (Child of Moon)
        let satellite = world.create_entity();
        unsafe {
            world.add_component(satellite, TransformComponent {
                position: Vec3::new(1.5, 0.0, 0.0),
                scale: Vec3::new(0.2, 0.2, 0.2),
                ..Default::default()
            });
            world.add_component(satellite, RenderComponent {
                visible: true,
                mesh_index: 0,
            });
            world.add_component(satellite, HierarchyComponent { parent: Some(moon) });
        }

        Some(Self {
            world,
            pipeline,
            texture,
            meshes,
            world_matrices,
            descriptor_pool,
            descriptor_set,
            render_pass,
            swapchain,
            vulkan,
            window,
            input,
            timer,
            memory,
        })
    }

    /// The canonical game loop.
    pub fn run(&mut self) {
        crate::log_info!("Application started.");
        
        // Reset timer before loop starts to avoid a massive first dt
        let _ = self.timer.tick();
        
        while self.window.poll_events(&mut self.input) {
            let dt = self.timer.tick();

            // Exit on ESC
            if self.input.is_key_pressed(win32::VK_ESCAPE) {
                break;
            }

            // 1. Update Game State (ECS)
            self.world.update_systems(dt as f32);

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
                        if self.input.is_key_down(win32::VK_E) {
                            transform.position.y += speed;
                        }
                        if self.input.is_key_down(win32::VK_Q) {
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

            // 2. Render Frame
            self.render_frame();

            // 3. Cleanup Frame Allocator
            self.memory.frame_arena().reset(false);
        }

        crate::log_info!("Application shutting down.");
        
        self.vulkan.wait_idle();
        
        if let Some(mut tex) = self.texture.take() {
            tex.shutdown(&self.vulkan);
        }
        if self.descriptor_pool != vk::DescriptorPool::null() {
            unsafe { self.vulkan.device.destroy_descriptor_pool(self.descriptor_pool, None) };
        }
        if let Some(mut p) = self.pipeline.take() {
            p.shutdown(&self.vulkan);
        }
        for mesh in &mut self.meshes {
            mesh.shutdown(&self.vulkan);
        }
        self.render_pass.shutdown(&self.vulkan);
        self.swapchain.shutdown(&self.vulkan);
        self.vulkan.shutdown();
    }
    
    fn render_frame(&mut self) {
        // Only draw if we successfully compiled shaders and uploaded vertices
        let pipeline = match &self.pipeline {
            Some(p) => p,
            _ => return, // Skip rendering if no pipeline (avoids deadlock)
        };

        // Wait for previous frame to finish
        unsafe {
            let _ = self.vulkan.device.wait_for_fences(std::slice::from_ref(&self.vulkan.in_flight_fence), true, u64::MAX);
        }

        // Acquire next image
        let (image_index, _is_suboptimal) = unsafe {
            let result = self.swapchain.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                self.vulkan.image_available_semaphore,
                vk::Fence::null(),
            );
            match result {
                Ok(r) => r,
                Err(_) => return, // Handle recreation later
            }
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

            // Begin Render Pass
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
            let render_pass_begin_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass.handle)
                .framebuffer(self.swapchain.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain.extent,
                })
                .clear_values(&clear_values);

            unsafe {
                self.vulkan.device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
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

                // Extract Light
                let mut light_dir = [0.0, -1.0, 0.0, 0.0];
                let mut light_color = [1.0, 1.0, 1.0, 0.0];
                {
                    let lights = self.world.get_component_array::<LightComponent>();
                    let dense_lights = lights.as_slice();
                    if let Some(light) = dense_lights.first() {
                        // We negate the direction because the shader expects a vector POINTING to the light
                        light_dir = [-light.direction.x, -light.direction.y, -light.direction.z, 0.0];
                        light_color = [light.color.x, light.color.y, light.color.z, 0.0];
                    }
                }

                // Extract Camera
                let mut view_proj = crate::math::mat4::Mat4::identity();
                {
                    let cameras = self.world.get_component_array::<CameraComponent>();
                    let transforms = self.world.get_component_array::<TransformComponent>();
                    let dense_cams = cameras.as_slice();
                    let cam_entities = cameras.dense_entities_slice();
                    
                    if let Some(&cam_entity) = cam_entities.first() {
                        let cam = dense_cams[0];
                        if transforms.has(cam_entity) {
                            let cam_transform = transforms.get(cam_entity);
                            
                            let pitch = cam_transform.rotation.x;
                            let yaw = cam_transform.rotation.y;
                            let forward = Vec3::new(
                                yaw.sin() * pitch.cos(),
                                pitch.sin(),
                                yaw.cos() * pitch.cos(),
                            ).normalize();
                            
                            let center = cam_transform.position + forward;
                            let view = crate::math::mat4::Mat4::look_at(cam_transform.position, center, Vec3::new(0.0, 1.0, 0.0));
                            view_proj = cam.proj * view;
                        }
                    }
                }

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
                            
                            // Push the mvp matrix (64 bytes) + light data (32 bytes)
                            let world_matrix = *self.world_matrices.get(&entity_index).unwrap_or(&transform.matrix);
                            let mvp = view_proj * world_matrix;
                            let push_constants = crate::renderer::vulkan::pipeline::PushConstants {
                                mvp,
                                light_dir,
                                light_color,
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

                            let mesh = &self.meshes[render.mesh_index];
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

            unsafe {
                let _ = self.swapchain.swapchain_loader.queue_present(self.vulkan.graphics_queue, &present_info);
            }
            
            // Clean up command buffer (in a real engine we'd reuse them per-frame in an array)
            unsafe {
                self.vulkan.device.wait_for_fences(std::slice::from_ref(&self.vulkan.in_flight_fence), true, u64::MAX).unwrap();
                self.vulkan.device.free_command_buffers(self.vulkan.command_pool, &command_buffers);
            }
    }
}
