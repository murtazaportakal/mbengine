use ash::vk;
use std::ffi::CString;

use crate::renderer::vulkan::buffer::Buffer;
use crate::renderer::vulkan::texture::Texture;
use crate::renderer::vulkan::VulkanDevice;
use crate::ui::context::{DrawCommand, UiContext};
use crate::ui::font::Font;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct UiVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

pub struct UiBackend {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,

    font_texture: Option<Texture>,
    pub font_descriptor_set: vk::DescriptorSet,

    /// Double-buffered vertex buffers (one per in-flight frame).
    vertex_buffers: [Option<Buffer>; 2],
    /// Double-buffered index buffers (one per in-flight frame).
    index_buffers: [Option<Buffer>; 2],

    vertex_capacity: usize,
    index_capacity: usize,

    white_texture: Option<Texture>,
    pub white_descriptor_set: vk::DescriptorSet,
    pub user_descriptor_sets: [Option<vk::DescriptorSet>; 4],
}

impl UiBackend {
    pub fn new(vulkan: &VulkanDevice, color_format: vk::Format, vfs: &crate::vfs::Vfs) -> Self {
        unsafe {
            // Descriptor set layout for font/user textures
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

            let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            let descriptor_set_layout = vulkan
                .device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap();

            // Descriptor pool
            let pool_sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(10), // Room for font + white + user textures
            ];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(10)
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
            let descriptor_pool = vulkan
                .device
                .create_descriptor_pool(&pool_info, None)
                .unwrap();

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
            let pipeline_layout = vulkan
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap();

            let (vert_module, frag_module) = {
                let vert_code = vfs.read_bytes("src/renderer/shaders/ui_vert.spv").unwrap();
                let frag_code = vfs.read_bytes("src/renderer/shaders/ui_frag.spv").unwrap();

                let vert_module = create_shader_module(&vulkan.device, &vert_code);
                let frag_module = create_shader_module(&vulkan.device, &frag_code);
                (vert_module, frag_module)
            };
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
            let binding_descriptions = [vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(std::mem::size_of::<UiVertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)];

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
                    .format(vk::Format::R32G32B32A32_SFLOAT) // color
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

            // Pre-multiplied alpha blending
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
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

            let color_attachment_formats = [color_format];
            let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfoKHR::default()
                .color_attachment_formats(&color_attachment_formats);

            let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&shader_stages)
                .vertex_input_state(&vertex_input_info)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterizer)
                .multisample_state(&multisampling)
                .color_blend_state(&color_blending)
                .depth_stencil_state(&depth_stencil)
                .dynamic_state(&dynamic_state_info)
                .layout(pipeline_layout)
                .push_next(&mut pipeline_rendering_info);

            let pipeline = vulkan
                .device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .unwrap()[0];

            vulkan.device.destroy_shader_module(vert_module, None);
            vulkan.device.destroy_shader_module(frag_module, None);

            // Create white texture
            let white_pixels = vec![255, 255, 255, 255];
            let white_texture = Texture::from_rgba8(vulkan, 1, 1, &white_pixels).unwrap();

            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));

            let white_descriptor_set =
                vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap()[0];

            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(white_texture.view)
                .sampler(white_texture.sampler);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(white_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info));
            vulkan.device.update_descriptor_sets(&[write], &[]);

            // Pre-allocate UI geometry buffers for both in-flight frames.
            // 64K vertices + 64K indices per frame covers full editor UI +
            // gizmos without growing on the hot path.
            let ui_vertex_capacity = 64 * 1024;
            let ui_index_capacity = 64 * 1024;

            let mut vertex_buffers: [Option<Buffer>; 2] = [None::<Buffer>, None::<Buffer>].map(|_| None);
            let mut index_buffers: [Option<Buffer>; 2] = [None::<Buffer>, None::<Buffer>].map(|_| None);

            for frame_idx in 0..2 {
                let vb = Buffer::new(
                    vulkan,
                    (ui_vertex_capacity * std::mem::size_of::<UiVertex>()) as u64,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .unwrap();
                let ib = Buffer::new(
                    vulkan,
                    (ui_index_capacity * std::mem::size_of::<u32>()) as u64,
                    vk::BufferUsageFlags::INDEX_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .unwrap();
                vertex_buffers[frame_idx] = Some(vb);
                index_buffers[frame_idx] = Some(ib);
            }

            Self {
                pipeline_layout,
                pipeline,
                descriptor_set_layout,
                descriptor_pool,
                font_texture: None,
                font_descriptor_set: vk::DescriptorSet::null(),
                vertex_buffers,
                index_buffers,
                vertex_capacity: ui_vertex_capacity,
                index_capacity: ui_index_capacity,
                white_texture: Some(white_texture),
                white_descriptor_set,
                user_descriptor_sets: [None; 4],
            }
        }
    }

    pub fn set_font(&mut self, vulkan: &VulkanDevice, font: &Font) {
        let texture =
            Texture::from_rgba8(vulkan, font.width, font.height, &font.texture_data).unwrap();

        unsafe {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));

            let font_descriptor_set =
                vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap()[0];

            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(texture.view)
                .sampler(texture.sampler);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(font_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info));

            vulkan.device.update_descriptor_sets(&[write], &[]);

            if let Some(mut old) = self.font_texture.take() {
                old.shutdown(vulkan);
                vulkan
                    .device
                    .free_descriptor_sets(self.descriptor_pool, &[self.font_descriptor_set])
                    .unwrap();
            }

            self.font_texture = Some(texture);
            self.font_descriptor_set = font_descriptor_set;
        }
    }

    pub fn update_user_texture(
        &mut self,
        vulkan: &VulkanDevice,
        id: u32,
        view: vk::ImageView,
        sampler: vk::Sampler,
    ) {
        if id >= 4 {
            return;
        }
        unsafe {
            if self.user_descriptor_sets[id as usize].is_none() {
                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));
                self.user_descriptor_sets[id as usize] =
                    Some(vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap()[0]);
            }

            let set = self.user_descriptor_sets[id as usize].unwrap();
            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(view)
                .sampler(sampler);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info));

            vulkan.device.update_descriptor_sets(&[write], &[]);
        }
    }

    /// Draw the UI overlay.
    ///
    /// `frame_index` selects the double-buffered geometry slot (0 or 1).
    /// Must match the in-flight frame index so the CPU writes into a
    /// buffer the GPU is guaranteed not to be reading from.
    pub fn draw(
        &mut self,
        vulkan: &VulkanDevice,
        command_buffer: vk::CommandBuffer,
        window_width: u32,
        window_height: u32,
        ui_ctx: &UiContext,
        font: &Font,
        frame_index: usize,
    ) {
        if ui_ctx.draw_commands.is_empty() {
            return;
        }

        // Compute vertex and index count
        let mut num_vertices = 0;
        let mut num_indices = 0;
        for cmd in &ui_ctx.draw_commands {
            match cmd {
                DrawCommand::Quad { .. } => {
                    num_vertices += 4;
                    num_indices += 6;
                }
                DrawCommand::Line { .. } => {
                    num_vertices += 4;
                    num_indices += 6;
                }
                DrawCommand::Image { .. } => {
                    num_vertices += 4;
                    num_indices += 6;
                }
                DrawCommand::Text { text, .. } => {
                    num_vertices += text.len() * 4;
                    num_indices += text.len() * 6;
                }
                DrawCommand::SetScissor { .. } => {}
            }
        }

        if num_vertices == 0 {
            return;
        }

        // Pre-allocated buffers are sized for full editor UI + gizmos.
        // overflowing means the initial capacity was too low — increase it.
        assert!(
            num_vertices <= self.vertex_capacity,
            "UI vertex buffer overflow: {} > {}",
            num_vertices,
            self.vertex_capacity,
        );
        assert!(
            num_indices <= self.index_capacity,
            "UI index buffer overflow: {} > {}",
            num_indices,
            self.index_capacity,
        );

        let frame = frame_index % 2;
        let vertex_buffer = self.vertex_buffers[frame].as_ref().unwrap();
        let index_buffer = self.index_buffers[frame].as_ref().unwrap();

        unsafe {
            // Build vertices and indices
            let v_mem = vertex_buffer.memory;
            let v_ptr = vulkan
                .device
                .map_memory(v_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .unwrap() as *mut UiVertex;
            let i_mem = index_buffer.memory;
            let i_ptr = vulkan
                .device
                .map_memory(i_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .unwrap() as *mut u32;

            let mut v_offset = 0;
            let mut i_offset = 0;

            #[derive(Clone, Copy)]
            struct DrawCall {
                index_count: u32,
                first_index: u32,
                descriptor: vk::DescriptorSet,
                scissor: Option<crate::ui::context::UiRect>,
            }
            let mut draw_calls = Vec::new();
            let mut current_draw_call = DrawCall {
                index_count: 0,
                first_index: 0,
                descriptor: self.white_descriptor_set,
                scissor: None,
            };

            for cmd in &ui_ctx.draw_commands {
                match cmd {
                    DrawCommand::Quad { rect, color, .. } => {
                        if current_draw_call.descriptor != self.white_descriptor_set
                            && current_draw_call.index_count > 0
                        {
                            draw_calls.push(current_draw_call);
                            current_draw_call = DrawCall {
                                index_count: 0,
                                first_index: i_offset as u32,
                                descriptor: self.white_descriptor_set,
                                scissor: current_draw_call.scissor,
                            };
                        } else if current_draw_call.index_count == 0 {
                            current_draw_call.descriptor = self.white_descriptor_set;
                            current_draw_call.first_index = i_offset as u32;
                        }

                        let base_vertex = v_offset as u32;
                        let r = color.r as f32 / 255.0;
                        let g = color.g as f32 / 255.0;
                        let b = color.b as f32 / 255.0;
                        let a = color.a as f32 / 255.0;
                        let c = [r * a, g * a, b * a, a];

                        v_ptr.add(v_offset).write(UiVertex {
                            pos: [rect.x, rect.y],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        v_ptr.add(v_offset + 1).write(UiVertex {
                            pos: [rect.x + rect.w, rect.y],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        v_ptr.add(v_offset + 2).write(UiVertex {
                            pos: [rect.x + rect.w, rect.y + rect.h],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        v_ptr.add(v_offset + 3).write(UiVertex {
                            pos: [rect.x, rect.y + rect.h],
                            uv: [0.0, 0.0],
                            color: c,
                        });

                        i_ptr.add(i_offset).write(base_vertex);
                        i_ptr.add(i_offset + 1).write(base_vertex + 1);
                        i_ptr.add(i_offset + 2).write(base_vertex + 2);
                        i_ptr.add(i_offset + 3).write(base_vertex);
                        i_ptr.add(i_offset + 4).write(base_vertex + 2);
                        i_ptr.add(i_offset + 5).write(base_vertex + 3);

                        v_offset += 4;
                        i_offset += 6;
                        current_draw_call.index_count += 6;
                    }
                    DrawCommand::Line {
                        p1,
                        p2,
                        color,
                        thickness,
                    } => {
                        if current_draw_call.descriptor != self.white_descriptor_set
                            && current_draw_call.index_count > 0
                        {
                            draw_calls.push(current_draw_call);
                            current_draw_call = DrawCall {
                                index_count: 0,
                                first_index: i_offset as u32,
                                descriptor: self.white_descriptor_set,
                                scissor: current_draw_call.scissor,
                            };
                        } else if current_draw_call.index_count == 0 {
                            current_draw_call.descriptor = self.white_descriptor_set;
                            current_draw_call.first_index = i_offset as u32;
                        }

                        let base_vertex = v_offset as u32;
                        let r = color.r as f32 / 255.0;
                        let g = color.g as f32 / 255.0;
                        let b = color.b as f32 / 255.0;
                        let a = color.a as f32 / 255.0;
                        let c = [r * a, g * a, b * a, a];

                        let dir = [p2.x - p1.x, p2.y - p1.y];
                        let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
                        let norm = if len > 0.0 {
                            [dir[0] / len, dir[1] / len]
                        } else {
                            [1.0, 0.0]
                        };
                        let perp = [-norm[1], norm[0]];
                        let t = *thickness * 0.5;

                        v_ptr.add(v_offset).write(UiVertex {
                            pos: [p1.x + perp[0] * t, p1.y + perp[1] * t],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        v_ptr.add(v_offset + 1).write(UiVertex {
                            pos: [p2.x + perp[0] * t, p2.y + perp[1] * t],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        v_ptr.add(v_offset + 2).write(UiVertex {
                            pos: [p2.x - perp[0] * t, p2.y - perp[1] * t],
                            uv: [0.0, 0.0],
                            color: c,
                        });
                        v_ptr.add(v_offset + 3).write(UiVertex {
                            pos: [p1.x - perp[0] * t, p1.y - perp[1] * t],
                            uv: [0.0, 0.0],
                            color: c,
                        });

                        i_ptr.add(i_offset).write(base_vertex);
                        i_ptr.add(i_offset + 1).write(base_vertex + 1);
                        i_ptr.add(i_offset + 2).write(base_vertex + 2);
                        i_ptr.add(i_offset + 3).write(base_vertex);
                        i_ptr.add(i_offset + 4).write(base_vertex + 2);
                        i_ptr.add(i_offset + 5).write(base_vertex + 3);

                        v_offset += 4;
                        i_offset += 6;
                        current_draw_call.index_count += 6;
                    }
                    DrawCommand::Image {
                        rect,
                        uv_min,
                        uv_max,
                        color,
                        texture_id,
                    } => {
                        let desc = self
                            .user_descriptor_sets
                            .get(*texture_id as usize)
                            .and_then(|x| *x)
                            .unwrap_or(self.white_descriptor_set);
                        if current_draw_call.descriptor != desc && current_draw_call.index_count > 0
                        {
                            draw_calls.push(current_draw_call);
                            current_draw_call = DrawCall {
                                index_count: 0,
                                first_index: i_offset as u32,
                                descriptor: desc,
                                scissor: current_draw_call.scissor,
                            };
                        } else if current_draw_call.index_count == 0 {
                            current_draw_call.descriptor = desc;
                            current_draw_call.first_index = i_offset as u32;
                        }

                        let base_vertex = v_offset as u32;
                        let r = color.r as f32 / 255.0;
                        let g = color.g as f32 / 255.0;
                        let b = color.b as f32 / 255.0;
                        let a = color.a as f32 / 255.0;
                        let c = [r * a, g * a, b * a, a];

                        v_ptr.add(v_offset).write(UiVertex {
                            pos: [rect.x, rect.y],
                            uv: [uv_min.x, uv_min.y],
                            color: c,
                        });
                        v_ptr.add(v_offset + 1).write(UiVertex {
                            pos: [rect.x + rect.w, rect.y],
                            uv: [uv_max.x, uv_min.y],
                            color: c,
                        });
                        v_ptr.add(v_offset + 2).write(UiVertex {
                            pos: [rect.x + rect.w, rect.y + rect.h],
                            uv: [uv_max.x, uv_max.y],
                            color: c,
                        });
                        v_ptr.add(v_offset + 3).write(UiVertex {
                            pos: [rect.x, rect.y + rect.h],
                            uv: [uv_min.x, uv_max.y],
                            color: c,
                        });

                        i_ptr.add(i_offset).write(base_vertex);
                        i_ptr.add(i_offset + 1).write(base_vertex + 1);
                        i_ptr.add(i_offset + 2).write(base_vertex + 2);
                        i_ptr.add(i_offset + 3).write(base_vertex);
                        i_ptr.add(i_offset + 4).write(base_vertex + 2);
                        i_ptr.add(i_offset + 5).write(base_vertex + 3);

                        v_offset += 4;
                        i_offset += 6;
                        current_draw_call.index_count += 6;
                    }
                    DrawCommand::Text {
                        pos,
                        text,
                        color,
                        font_size,
                    } => {
                        if current_draw_call.descriptor != self.font_descriptor_set
                            && current_draw_call.index_count > 0
                        {
                            draw_calls.push(current_draw_call);
                            current_draw_call = DrawCall {
                                index_count: 0,
                                first_index: i_offset as u32,
                                descriptor: self.font_descriptor_set,
                                scissor: current_draw_call.scissor,
                            };
                        } else if current_draw_call.index_count == 0 {
                            current_draw_call.descriptor = self.font_descriptor_set;
                            current_draw_call.first_index = i_offset as u32;
                        }

                        let r = color.r as f32 / 255.0;
                        let g = color.g as f32 / 255.0;
                        let b = color.b as f32 / 255.0;
                        let a = color.a as f32 / 255.0;
                        let c = [r * a, g * a, b * a, a];

                        let scale = *font_size / font.line_height;
                        let mut cur_x = pos.x;
                        let cur_y = pos.y;

                        for ch in text.chars() {
                            let idx = if ch as usize >= 32 && (ch as usize) < 127 {
                                ch as usize
                            } else {
                                32
                            };
                            let gi = &font.glyphs[idx];

                            let px = cur_x + gi.offset_x * scale;
                            let py = cur_y + gi.offset_y * scale;
                            let pw = gi.size_x * scale;
                            let ph = gi.size_y * scale;

                            let base_vertex = v_offset as u32;
                            v_ptr.add(v_offset).write(UiVertex {
                                pos: [px, py],
                                uv: [gi.u_min, gi.v_min],
                                color: c,
                            });
                            v_ptr.add(v_offset + 1).write(UiVertex {
                                pos: [px + pw, py],
                                uv: [gi.u_max, gi.v_min],
                                color: c,
                            });
                            v_ptr.add(v_offset + 2).write(UiVertex {
                                pos: [px + pw, py + ph],
                                uv: [gi.u_max, gi.v_max],
                                color: c,
                            });
                            v_ptr.add(v_offset + 3).write(UiVertex {
                                pos: [px, py + ph],
                                uv: [gi.u_min, gi.v_max],
                                color: c,
                            });

                            i_ptr.add(i_offset).write(base_vertex);
                            i_ptr.add(i_offset + 1).write(base_vertex + 1);
                            i_ptr.add(i_offset + 2).write(base_vertex + 2);
                            i_ptr.add(i_offset + 3).write(base_vertex);
                            i_ptr.add(i_offset + 4).write(base_vertex + 2);
                            i_ptr.add(i_offset + 5).write(base_vertex + 3);

                            v_offset += 4;
                            i_offset += 6;
                            current_draw_call.index_count += 6;

                            cur_x += gi.advance * scale;
                        }
                    }
                    DrawCommand::SetScissor { rect } => {
                        if current_draw_call.index_count > 0 {
                            draw_calls.push(current_draw_call);
                            current_draw_call = DrawCall {
                                index_count: 0,
                                first_index: i_offset as u32,
                                descriptor: current_draw_call.descriptor,
                                scissor: *rect,
                            };
                        } else {
                            current_draw_call.scissor = *rect;
                        }
                    }
                }
            }
            if current_draw_call.index_count > 0 {
                draw_calls.push(current_draw_call);
            }
            vulkan.device.unmap_memory(v_mem);
            vulkan.device.unmap_memory(i_mem);

            vulkan.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            vulkan.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[vertex_buffer.handle],
                &[0],
            );
            vulkan.device.cmd_bind_index_buffer(
                command_buffer,
                index_buffer.handle,
                0,
                vk::IndexType::UINT32,
            );

            let pc = [window_width as f32, window_height as f32];
            let pc_bytes: &[u8] =
                std::slice::from_raw_parts(pc.as_ptr() as *const u8, std::mem::size_of_val(&pc));
            vulkan.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                pc_bytes,
            );

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: window_width as f32,
                height: window_height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            vulkan
                .device
                .cmd_set_viewport(command_buffer, 0, &[viewport]);

            for call in draw_calls {
                let scissor = if let Some(rect) = call.scissor {
                    vk::Rect2D {
                        offset: vk::Offset2D { x: rect.x as i32, y: rect.y as i32 },
                        extent: vk::Extent2D { width: rect.w as u32, height: rect.h as u32 },
                    }
                } else {
                    vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: window_width,
                            height: window_height,
                        },
                    }
                };
                vulkan.device.cmd_set_scissor(command_buffer, 0, &[scissor]);

                vulkan.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[call.descriptor],
                    &[],
                );
                vulkan.device.cmd_draw_indexed(
                    command_buffer,
                    call.index_count,
                    1,
                    call.first_index,
                    0,
                    0,
                );
            }
        }
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            for frame_idx in 0..2 {
                if let Some(mut b) = self.vertex_buffers[frame_idx].take() {
                    b.shutdown(vulkan);
                }
                if let Some(mut b) = self.index_buffers[frame_idx].take() {
                    b.shutdown(vulkan);
                }
            }
            if let Some(mut t) = self.font_texture.take() {
                t.shutdown(vulkan);
            }
            if let Some(mut t) = self.white_texture.take() {
                t.shutdown(vulkan);
            }
            vulkan
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            vulkan
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            vulkan.device.destroy_pipeline(self.pipeline, None);
            vulkan
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

fn create_shader_module(device: &ash::Device, code: &[u8]) -> vk::ShaderModule {
    let mut aligned = vec![0u32; code.len() / 4];
    unsafe {
        std::ptr::copy_nonoverlapping(code.as_ptr(), aligned.as_mut_ptr() as *mut u8, code.len());
    }
    let info = vk::ShaderModuleCreateInfo::default().code(&aligned);
    unsafe { device.create_shader_module(&info, None).unwrap() }
}