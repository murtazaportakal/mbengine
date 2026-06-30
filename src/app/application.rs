//! The core game loop and application state.

use crate::ecs::World;
use crate::ecs::{TransformComponent, RenderComponent};
use crate::memory::{MemoryConfig, MemorySubsystem};
use crate::platform::{Timer, Window, win32};
use crate::renderer::vulkan::{Pipeline, RenderPass, Swapchain, VulkanDevice, Buffer};
use crate::renderer::vulkan::pipeline::Vertex;
use crate::renderer::RenderDevice;
use crate::math::vec::Vec3;
use ash::vk;

/// High-level engine coordinator.
/// Note: Field declaration order dictates drop order.
/// `world` must drop before `memory`. `swapchain` must drop before `vulkan`.
pub struct Application {
    pub world: World,
    pub pipeline: Option<Pipeline>,
    pub vertex_buffer: Option<Buffer>,
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

        let vertices = [
            Vertex { pos: [0.0, -0.5, 0.0], color: [1.0, 0.0, 0.0] },
            Vertex { pos: [0.5, 0.5, 0.0], color: [0.0, 1.0, 0.0] },
            Vertex { pos: [-0.5, 0.5, 0.0], color: [0.0, 0.0, 1.0] },
        ];

        let vertex_buffer = Buffer::new_device_local(
            &vulkan,
            &vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        );

        if pipeline.is_none() {
            crate::log_info!("Shaders not found or failed to compile. Rendering will be skipped.");
        }

        // 4. Initialize ECS
        let mut world = unsafe { World::new(memory.persistent_arena()) };

        // Register rendering components
        unsafe {
            world.register_component::<TransformComponent>(1000);
            world.register_component::<RenderComponent>(1000);
        }

        // Spawn a few entities to render
        for i in 0..3 {
            let entity = world.create_entity();
            let mut transform = TransformComponent::default();
            transform.position = Vec3::new(-0.5 + (i as f32 * 0.5), 0.0, 0.0);
            transform.scale = Vec3::new(0.5, 0.5, 1.0);
            transform.update_matrix(); // Precompute the matrix for the shader

            unsafe {
                world.add_component(entity, transform);
                world.add_component(entity, RenderComponent { visible: true });
            }
        }

        Some(Self {
            world,
            pipeline,
            vertex_buffer,
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

            // Update entity transforms dynamically (for demonstration)
            let transforms = self.world.get_component_array_mut::<TransformComponent>();
            for transform in transforms.as_mut_slice() {
                // Let's spin them around slightly
                // We don't have proper Quaternions applied to the matrix yet, 
                // so we just bounce the scale or position for fun.
                transform.rotation.z += 1.0 * dt as f32; // Just mutating a property
                transform.update_matrix(); // Recompute
            }

            // 2. Render Frame
            self.render_frame();

            // 3. Cleanup Frame Allocator
            self.memory.frame_arena().reset(false);
        }

        crate::log_info!("Application shutting down.");
        
        self.vulkan.wait_idle();
        
        if let Some(mut p) = self.pipeline.take() {
            p.shutdown(&self.vulkan);
        }
        if let Some(mut vb) = self.vertex_buffer.take() {
            vb.shutdown(&self.vulkan);
        }
        self.render_pass.shutdown(&self.vulkan);
        self.swapchain.shutdown(&self.vulkan);
        self.vulkan.shutdown();
    }
    
    fn render_frame(&mut self) {
        // Only draw if we successfully compiled shaders and uploaded vertices
        let (pipeline, vertex_buffer) = match (&self.pipeline, &self.vertex_buffer) {
            (Some(p), Some(vb)) => (p, vb),
            _ => return, // Skip rendering if no pipeline/buffer (avoids deadlock)
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

                self.vulkan.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.handle,
                );
                
                // Bind Vertex Buffer
                self.vulkan.device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    std::slice::from_ref(&vertex_buffer.handle),
                    &[0],
                );

                // ECS Iteration for rendering
                let renders = self.world.get_component_array::<RenderComponent>();
                let transforms = self.world.get_component_array::<TransformComponent>();
                
                let mut _draw_count = 0;
                let dense_renders = renders.as_slice();
                let entities = renders.dense_entities_slice();

                for i in 0..dense_renders.len() {
                    if dense_renders[i].visible {
                        let entity_index = entities[i];
                        if transforms.has(entity_index) {
                            let transform = transforms.get(entity_index);
                            
                            // Push the model matrix (64 bytes)
                            let constants_ptr = &transform.matrix as *const _ as *const u8;
                            let constants_slice = std::slice::from_raw_parts(constants_ptr, std::mem::size_of::<crate::math::mat4::Mat4>());

                            self.vulkan.device.cmd_push_constants(
                                command_buffer,
                                pipeline.layout,
                                vk::ShaderStageFlags::VERTEX,
                                0,
                                constants_slice,
                            );

                            // Draw 3 vertices for the triangle
                            self.vulkan.device.cmd_draw(command_buffer, 3, 1, 0, 0);
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
