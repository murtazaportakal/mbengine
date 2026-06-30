use ash::vk;
use std::ffi::CString;

use crate::renderer::vulkan::VulkanDevice;
use crate::renderer::vulkan::buffer::Buffer;
use crate::renderer::vulkan::texture::Texture;

pub struct EguiBackend {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    
    font_texture: Option<Texture>,
    pub font_descriptor_set: vk::DescriptorSet,

    vertex_buffer: Option<Buffer>,
    index_buffer: Option<Buffer>,
    
    
    vertex_capacity: usize,
    index_capacity: usize,
    
    pub user_textures: std::collections::HashMap<u64, vk::DescriptorSet>,
    next_user_texture_id: u64,
}

impl EguiBackend {
    pub fn new(vulkan: &VulkanDevice, render_pass: vk::RenderPass) -> Self {
        unsafe {
            // Descriptor set layout for font texture
            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ];

            let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            let descriptor_set_layout = vulkan.device.create_descriptor_set_layout(&layout_info, None).unwrap();

            // Descriptor pool
            let pool_sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(10), // Room for font + user textures
            ];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(10)
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
            let descriptor_pool = vulkan.device.create_descriptor_pool(&pool_info, None).unwrap();

            // Push constants
            let push_constant_ranges = [
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::VERTEX)
                    .offset(0)
                    .size(std::mem::size_of::<[f32; 2]>() as u32), // Screen size
            ];

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                .push_constant_ranges(&push_constant_ranges);
            let pipeline_layout = vulkan.device.create_pipeline_layout(&pipeline_layout_info, None).unwrap();

            // Shaders
            let vert_code = std::fs::read("src/renderer/shaders/egui_vert.spv").unwrap();
            let frag_code = std::fs::read("src/renderer/shaders/egui_frag.spv").unwrap();

            let vert_module = create_shader_module(&vulkan.device, &vert_code);
            let frag_module = create_shader_module(&vulkan.device, &frag_code);
            let main_function_name = CString::new("main").unwrap();

            let shader_stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vert_module)
                    .name(&main_function_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag_module)
                    .name(&main_function_name),
            ];

            // Vertex input
            let binding_descriptions = [
                vk::VertexInputBindingDescription::default()
                    .binding(0)
                    .stride(std::mem::size_of::<egui::epaint::Vertex>() as u32)
                    .input_rate(vk::VertexInputRate::VERTEX),
            ];
            
            // egui::epaint::Vertex has: pos: [f32; 2], uv: [f32; 2], color: Color32 ([u8; 4])
            let attribute_descriptions = [
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(0)
                    .format(vk::Format::R32G32_SFLOAT) // pos
                    .offset(0),
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(1)
                    .format(vk::Format::R32G32_SFLOAT) // uv
                    .offset(8),
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(2)
                    .format(vk::Format::R8G8B8A8_UNORM) // color
                    .offset(16),
            ];

            let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&binding_descriptions)
                .vertex_attribute_descriptions(&attribute_descriptions);

            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);

            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1); // dynamic

            let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false);

            let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(false)
                .depth_write_enable(false)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false);

            // Egui needs pre-multiplied alpha blending
            let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_DST_ALPHA)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE)
                .alpha_blend_op(vk::BlendOp::ADD);

            let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op_enable(false)
                .attachments(std::slice::from_ref(&color_blend_attachment));

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&dynamic_states);

            let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&shader_stages)
                .vertex_input_state(&vertex_input_info)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterizer)
                .multisample_state(&multisampling)
                .depth_stencil_state(&depth_stencil)
                .color_blend_state(&color_blending)
                .dynamic_state(&dynamic_state_info)
                .layout(pipeline_layout)
                .render_pass(render_pass)
                .subpass(0);

            let pipeline = vulkan.device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None).unwrap()[0];

            vulkan.device.destroy_shader_module(vert_module, None);
            vulkan.device.destroy_shader_module(frag_module, None);

            Self {
                pipeline_layout,
                pipeline,
                descriptor_set_layout,
                descriptor_pool,
                font_texture: None,
                font_descriptor_set: vk::DescriptorSet::null(),
                vertex_buffer: None,
                index_buffer: None,
                vertex_capacity: 0,
                index_capacity: 0,
                user_textures: std::collections::HashMap::new(),
                next_user_texture_id: 1, // 0 is reserved for font
            }
        }
    }

    pub fn update_font_texture(&mut self, vulkan: &VulkanDevice, image_data: &egui::ImageData) {
        let (width, height, pixels) = match image_data {
            egui::ImageData::Color(image) => {
                let p: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_array()).collect();
                (image.width() as u32, image.height() as u32, p)
            }
            egui::ImageData::Font(image) => {
                let p: Vec<u8> = image.srgba_pixels(None).flat_map(|c| c.to_array()).collect();
                (image.width() as u32, image.height() as u32, p)
            }
        };
        
        let texture = Texture::from_rgba8(vulkan, width, height, &pixels).unwrap();

        unsafe {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));
            
            let descriptor_set = vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap()[0];

            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(texture.view)
                .sampler(texture.sampler);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info));

            vulkan.device.update_descriptor_sets(std::slice::from_ref(&write), &[]);

            self.font_texture = Some(texture);
            self.font_descriptor_set = descriptor_set;
        }
    }
    
    pub fn register_user_texture(&mut self, vulkan: &VulkanDevice, view: vk::ImageView, sampler: vk::Sampler) -> egui::TextureId {
        unsafe {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));
            
            let descriptor_set = vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap()[0];

            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(view)
                .sampler(sampler);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info));

            vulkan.device.update_descriptor_sets(std::slice::from_ref(&write), &[]);

            let id = self.next_user_texture_id;
            self.next_user_texture_id += 1;
            self.user_textures.insert(id, descriptor_set);
            
            egui::TextureId::User(id)
        }
    }
    
    pub fn update_user_texture(&mut self, vulkan: &VulkanDevice, texture_id: egui::TextureId, view: vk::ImageView, sampler: vk::Sampler) {
        if let egui::TextureId::User(id) = texture_id {
            if let Some(&descriptor_set) = self.user_textures.get(&id) {
                unsafe {
                    let image_info = vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(view)
                        .sampler(sampler);

                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(std::slice::from_ref(&image_info));

                    vulkan.device.update_descriptor_sets(std::slice::from_ref(&write), &[]);
                }
            }
        }
    }
    
    pub fn draw(
        &mut self,
        vulkan: &VulkanDevice,
        command_buffer: vk::CommandBuffer,
        clipped_primitives: &[egui::ClippedPrimitive],
        pixels_per_point: f32,
        screen_size: [f32; 2],
    ) {
        if clipped_primitives.is_empty() { return; }
        
        // Count total vertices and indices
        let mut vertex_count = 0;
        let mut index_count = 0;
        for p in clipped_primitives {
            if let egui::epaint::Primitive::Mesh(mesh) = &p.primitive {
                vertex_count += mesh.vertices.len();
                index_count += mesh.indices.len();
            }
        }

        if vertex_count == 0 || index_count == 0 { return; }

        // Ensure buffer capacities
        if self.vertex_capacity < vertex_count {
            if let Some(mut buf) = self.vertex_buffer.take() { buf.shutdown(vulkan); }
            self.vertex_capacity = vertex_count.next_power_of_two().max(1024);
            self.vertex_buffer = Some(Buffer::new(
                vulkan,
                (self.vertex_capacity * std::mem::size_of::<egui::epaint::Vertex>()) as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ).unwrap());
        }

        if self.index_capacity < index_count {
            if let Some(mut buf) = self.index_buffer.take() { buf.shutdown(vulkan); }
            self.index_capacity = index_count.next_power_of_two().max(1024);
            self.index_buffer = Some(Buffer::new(
                vulkan,
                (self.index_capacity * std::mem::size_of::<u32>()) as u64,
                vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ).unwrap());
        }

        let vb = self.vertex_buffer.as_ref().unwrap();
        let ib = self.index_buffer.as_ref().unwrap();

        // Copy data
        unsafe {
            let vb_ptr = vulkan.device.map_memory(vb.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap() as *mut egui::epaint::Vertex;
            let ib_ptr = vulkan.device.map_memory(ib.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap() as *mut u32;

            let mut vb_offset = 0;
            let mut ib_offset = 0;

            for p in clipped_primitives {
                if let egui::epaint::Primitive::Mesh(mesh) = &p.primitive {
                    std::ptr::copy_nonoverlapping(mesh.vertices.as_ptr(), vb_ptr.add(vb_offset), mesh.vertices.len());
                    std::ptr::copy_nonoverlapping(mesh.indices.as_ptr(), ib_ptr.add(ib_offset), mesh.indices.len());
                    vb_offset += mesh.vertices.len();
                    ib_offset += mesh.indices.len();
                }
            }

            vulkan.device.unmap_memory(vb.memory);
            vulkan.device.unmap_memory(ib.memory);

            // Draw
            vulkan.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            // Default to font descriptor set. We will dynamically re-bind in the loop.
            let mut current_texture_id = egui::TextureId::Managed(0);
            vulkan.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.font_descriptor_set),
                &[],
            );
            
            vulkan.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_raw_parts(&screen_size as *const _ as *const u8, std::mem::size_of::<[f32; 2]>()),
            );

            vulkan.device.cmd_bind_vertex_buffers(command_buffer, 0, std::slice::from_ref(&vb.handle), &[0]);
            vulkan.device.cmd_bind_index_buffer(command_buffer, ib.handle, 0, vk::IndexType::UINT32);

            let mut vb_base = 0;
            let mut ib_base = 0;

            for p in clipped_primitives {
                if let egui::epaint::Primitive::Mesh(mesh) = &p.primitive {
                    if mesh.texture_id != current_texture_id {
                        let desc_set = match mesh.texture_id {
                            egui::TextureId::Managed(_) => self.font_descriptor_set,
                            egui::TextureId::User(id) => *self.user_textures.get(&id).unwrap_or(&self.font_descriptor_set),
                        };
                        vulkan.device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline_layout,
                            0,
                            std::slice::from_ref(&desc_set),
                            &[],
                        );
                        current_texture_id = mesh.texture_id;
                    }

                    // Apply scissor
                    let clip_rect = p.clip_rect;
                    let min_x = (clip_rect.min.x * pixels_per_point).round() as i32;
                    let min_y = (clip_rect.min.y * pixels_per_point).round() as i32;
                    let max_x = (clip_rect.max.x * pixels_per_point).round() as i32;
                    let max_y = (clip_rect.max.y * pixels_per_point).round() as i32;

                    let scissor = vk::Rect2D {
                        offset: vk::Offset2D { x: min_x.max(0), y: min_y.max(0) },
                        extent: vk::Extent2D {
                            width: (max_x - min_x).max(0) as u32,
                            height: (max_y - min_y).max(0) as u32,
                        },
                    };

                    vulkan.device.cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));

                    vulkan.device.cmd_draw_indexed(
                        command_buffer,
                        mesh.indices.len() as u32,
                        1,
                        ib_base,
                        vb_base,
                        0,
                    );

                    vb_base += mesh.vertices.len() as i32;
                    ib_base += mesh.indices.len() as u32;
                }
            }
        }
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            if let Some(mut tex) = self.font_texture.take() {
                tex.shutdown(vulkan);
            }
            if let Some(mut buf) = self.vertex_buffer.take() {
                buf.shutdown(vulkan);
            }
            if let Some(mut buf) = self.index_buffer.take() {
                buf.shutdown(vulkan);
            }
            vulkan.device.destroy_descriptor_pool(self.descriptor_pool, None);
            vulkan.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            vulkan.device.destroy_pipeline(self.pipeline, None);
            vulkan.device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

unsafe fn create_shader_module(device: &ash::Device, code: &[u8]) -> vk::ShaderModule {
    let mut create_info = vk::ShaderModuleCreateInfo::default();
    create_info.code_size = code.len();
    create_info.p_code = code.as_ptr() as *const u32;
    device.create_shader_module(&create_info, None).unwrap()
}
