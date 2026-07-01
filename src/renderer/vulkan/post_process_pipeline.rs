use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

pub struct PostProcessPipeline {
    pub tonemap_layout: vk::PipelineLayout,
    pub tonemap_pipeline: vk::Pipeline,
    pub tonemap_descriptor_set_layout: vk::DescriptorSetLayout,

    pub bloom_layout: vk::PipelineLayout,
    pub bloom_downsample_pipeline: vk::Pipeline,
    pub bloom_upsample_pipeline: vk::Pipeline,
    pub bloom_descriptor_set_layout: vk::DescriptorSetLayout,
}

impl PostProcessPipeline {
    pub fn new(vulkan: &VulkanDevice, surface_format: vk::Format) -> Option<Self> {
        let vert_code = std::fs::read("shaders/post_process_vert.spv").ok()?;
        let tonemap_frag_code = std::fs::read("shaders/post_process_frag.spv").ok()?;
        let downsample_frag_code = std::fs::read("shaders/bloom_downsample_frag.spv").ok()?;
        let upsample_frag_code = std::fs::read("shaders/bloom_upsample_frag.spv").ok()?;

        let vert_module = crate::renderer::vulkan::Pipeline::create_shader_module(vulkan, &vert_code)?;
        let tonemap_frag_module = crate::renderer::vulkan::Pipeline::create_shader_module(vulkan, &tonemap_frag_code)?;
        let downsample_frag_module = crate::renderer::vulkan::Pipeline::create_shader_module(vulkan, &downsample_frag_code)?;
        let upsample_frag_module = crate::renderer::vulkan::Pipeline::create_shader_module(vulkan, &upsample_frag_code)?;

        let entry_name = c"main";

        let vert_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(entry_name);

        // --- Common State ---
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));
        
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // --- Tonemap Pipeline ---
        let tonemap_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let tonemap_dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&tonemap_bindings);
        let tonemap_descriptor_set_layout = unsafe { vulkan.device.create_descriptor_set_layout(&tonemap_dsl_info, None).ok()? };

        let tonemap_pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&tonemap_descriptor_set_layout));
        let tonemap_layout = unsafe { vulkan.device.create_pipeline_layout(&tonemap_pipeline_layout_info, None).ok()? };

        let mut tonemap_rendering_info = vk::PipelineRenderingCreateInfoKHR::default()
            .color_attachment_formats(std::slice::from_ref(&surface_format));

        let tonemap_frag_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(tonemap_frag_module)
            .name(entry_name);
        let tonemap_stages = [vert_stage, tonemap_frag_stage];

        let tonemap_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&tonemap_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state_info)
            .layout(tonemap_layout)
            .push_next(&mut tonemap_rendering_info);

        let tonemap_pipeline = unsafe {
            vulkan.device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&tonemap_pipeline_info), None).map_err(|e| e.1).ok()?[0]
        };

        // --- Bloom Pipelines ---
        let bloom_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let bloom_dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bloom_bindings);
        let bloom_descriptor_set_layout = unsafe { vulkan.device.create_descriptor_set_layout(&bloom_dsl_info, None).ok()? };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(16);

        let bloom_pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&bloom_descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let bloom_layout = unsafe { vulkan.device.create_pipeline_layout(&bloom_pipeline_layout_info, None).ok()? };

        let mut bloom_rendering_info = vk::PipelineRenderingCreateInfoKHR::default()
            .color_attachment_formats(std::slice::from_ref(&vk::Format::R16G16B16A16_SFLOAT)); // Bloom targets are HDR

        // Downsample
        let downsample_frag_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(downsample_frag_module)
            .name(entry_name);
        let downsample_stages = [vert_stage, downsample_frag_stage];

        let downsample_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&downsample_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state_info)
            .layout(bloom_layout)
            .push_next(&mut bloom_rendering_info);

        let bloom_downsample_pipeline = unsafe {
            vulkan.device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&downsample_pipeline_info), None).map_err(|e| e.1).ok()?[0]
        };

        // Upsample
        let upsample_frag_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(upsample_frag_module)
            .name(entry_name);
        let upsample_stages = [vert_stage, upsample_frag_stage];

        // Upsample uses additive blending!
        let mut upsample_color_blend_attachment = color_blend_attachment.clone();
        upsample_color_blend_attachment.blend_enable = 1;
        upsample_color_blend_attachment.src_color_blend_factor = vk::BlendFactor::ONE;
        upsample_color_blend_attachment.dst_color_blend_factor = vk::BlendFactor::ONE;
        upsample_color_blend_attachment.color_blend_op = vk::BlendOp::ADD;
        upsample_color_blend_attachment.src_alpha_blend_factor = vk::BlendFactor::ONE;
        upsample_color_blend_attachment.dst_alpha_blend_factor = vk::BlendFactor::ZERO;
        upsample_color_blend_attachment.alpha_blend_op = vk::BlendOp::ADD;

        let upsample_color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&upsample_color_blend_attachment));

        let upsample_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&upsample_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&upsample_color_blending) // ADDITIVE BLENDING
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state_info)
            .layout(bloom_layout)
            .push_next(&mut bloom_rendering_info);

        let bloom_upsample_pipeline = unsafe {
            vulkan.device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&upsample_pipeline_info), None).map_err(|e| e.1).ok()?[0]
        };

        unsafe {
            vulkan.device.destroy_shader_module(vert_module, None);
            vulkan.device.destroy_shader_module(tonemap_frag_module, None);
            vulkan.device.destroy_shader_module(downsample_frag_module, None);
            vulkan.device.destroy_shader_module(upsample_frag_module, None);
        }

        Some(Self {
            tonemap_layout,
            tonemap_pipeline,
            tonemap_descriptor_set_layout,
            bloom_layout,
            bloom_downsample_pipeline,
            bloom_upsample_pipeline,
            bloom_descriptor_set_layout,
        })
    }

    pub fn add_passes<'a>(
        &'a self,
        graph: &mut crate::renderer::vulkan::render_graph::RenderGraph<'a>,
        vulkan: &'a VulkanDevice,
        offscreen_target: &'a crate::renderer::vulkan::OffscreenTarget,
        sdr_target: &'a crate::renderer::vulkan::OffscreenTarget,
        bloom_target: &'a crate::renderer::vulkan::bloom::BloomTarget,
        tonemap_descriptor_set: vk::DescriptorSet,
        bloom_descriptor_sets: &'a [vk::DescriptorSet],
        bloom_threshold: f32,
    ) {
        #[repr(C)]
        struct BloomPushConstants {
            inv_resolution: [f32; 2],
            threshold: f32,
            is_first_pass: f32,
        }

        graph.add_pass(
            "PostProcess",
            vec![
                crate::renderer::vulkan::render_graph::PassResource {
                    handle: crate::renderer::vulkan::render_graph::ResourceHandle(offscreen_target.color_image),
                    state: crate::renderer::vulkan::render_graph::ResourceState {
                        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                        access_mask: vk::AccessFlags::SHADER_READ,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                    },
                },
                crate::renderer::vulkan::render_graph::PassResource {
                    handle: crate::renderer::vulkan::render_graph::ResourceHandle(sdr_target.color_image),
                    state: crate::renderer::vulkan::render_graph::ResourceState {
                        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                    },
                },
            ],
            move |command_buffer| {
                unsafe {
                    // --- Bloom Pass ---
                    // Transition entire bloom image to SHADER_READ_ONLY_OPTIMAL first
                    let initial_barrier = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::NONE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image(bloom_target.image)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: bloom_target.mip_levels,
                            base_array_layer: 0,
                            layer_count: 1,
                        });
                    vulkan.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        std::slice::from_ref(&initial_barrier),
                    );

                    vulkan.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.bloom_downsample_pipeline);

                    // Downsample
                    for i in 0..bloom_target.mip_levels as usize {
                        let mip_width = (bloom_target.width >> i).max(1);
                        let mip_height = (bloom_target.height >> i).max(1);

                        // Transition mip `i` to COLOR_ATTACHMENT_OPTIMAL
                        let barrier = vk::ImageMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_READ)
                            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .image(bloom_target.image)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: i as u32,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            });
                        vulkan.device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&barrier));

                        let color_attachment = vk::RenderingAttachmentInfoKHR::default()
                            .image_view(bloom_target.mip_views[i])
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .store_op(vk::AttachmentStoreOp::STORE);
                        
                        let rendering_info = vk::RenderingInfoKHR::default()
                            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: mip_width, height: mip_height } })
                            .layer_count(1)
                            .color_attachments(std::slice::from_ref(&color_attachment));
                        
                        vulkan.device.cmd_begin_rendering(command_buffer, &rendering_info);

                        let viewport = vk::Viewport { x: 0.0, y: 0.0, width: mip_width as f32, height: mip_height as f32, min_depth: 0.0, max_depth: 1.0 };
                        vulkan.device.cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));
                        let scissor = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: mip_width, height: mip_height } };
                        vulkan.device.cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));

                        vulkan.device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.bloom_layout, 0, std::slice::from_ref(&bloom_descriptor_sets[i]), &[]);

                        let src_width = if i == 0 { offscreen_target.width } else { (bloom_target.width >> (i - 1)).max(1) };
                        let src_height = if i == 0 { offscreen_target.height } else { (bloom_target.height >> (i - 1)).max(1) };
                        
                        let pc = BloomPushConstants {
                            inv_resolution: [1.0 / src_width as f32, 1.0 / src_height as f32],
                            threshold: bloom_threshold,
                            is_first_pass: if i == 0 { 1.0 } else { 0.0 },
                        };
                        
                        let pc_bytes = std::slice::from_raw_parts(&pc as *const _ as *const u8, std::mem::size_of::<BloomPushConstants>());
                        vulkan.device.cmd_push_constants(command_buffer, self.bloom_layout, vk::ShaderStageFlags::FRAGMENT, 0, pc_bytes);

                        vulkan.device.cmd_draw(command_buffer, 3, 1, 0, 0);
                        vulkan.device.cmd_end_rendering(command_buffer);

                        // Transition mip `i` back to SHADER_READ_ONLY_OPTIMAL
                        let barrier = vk::ImageMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                            .dst_access_mask(vk::AccessFlags::SHADER_READ)
                            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image(bloom_target.image)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: i as u32,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            });
                        vulkan.device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&barrier));
                    }

                    // Upsample
                    vulkan.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.bloom_upsample_pipeline);
                    for i in (0..bloom_target.mip_levels as usize - 1).rev() {
                        let mip_width = (bloom_target.width >> i).max(1);
                        let mip_height = (bloom_target.height >> i).max(1);

                        let barrier = vk::ImageMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_READ)
                            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .image(bloom_target.image)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: i as u32,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            });
                        vulkan.device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&barrier));

                        let color_attachment = vk::RenderingAttachmentInfoKHR::default()
                            .image_view(bloom_target.mip_views[i])
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE);
                        
                        let rendering_info = vk::RenderingInfoKHR::default()
                            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: mip_width, height: mip_height } })
                            .layer_count(1)
                            .color_attachments(std::slice::from_ref(&color_attachment));
                        
                        vulkan.device.cmd_begin_rendering(command_buffer, &rendering_info);

                        let viewport = vk::Viewport { x: 0.0, y: 0.0, width: mip_width as f32, height: mip_height as f32, min_depth: 0.0, max_depth: 1.0 };
                        vulkan.device.cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));
                        let scissor = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: mip_width, height: mip_height } };
                        vulkan.device.cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));

                        vulkan.device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.bloom_layout, 0, std::slice::from_ref(&bloom_descriptor_sets[i + 2]), &[]);

                        let pc = BloomPushConstants {
                            inv_resolution: [1.0 / mip_width as f32, 1.0 / mip_height as f32],
                            threshold: 0.0,
                            is_first_pass: 0.0,
                        };
                        
                        let pc_bytes = std::slice::from_raw_parts(&pc as *const _ as *const u8, std::mem::size_of::<BloomPushConstants>());
                        vulkan.device.cmd_push_constants(command_buffer, self.bloom_layout, vk::ShaderStageFlags::FRAGMENT, 0, pc_bytes);

                        vulkan.device.cmd_draw(command_buffer, 3, 1, 0, 0);
                        vulkan.device.cmd_end_rendering(command_buffer);

                        let barrier = vk::ImageMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                            .dst_access_mask(vk::AccessFlags::SHADER_READ)
                            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image(bloom_target.image)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: i as u32,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            });
                        vulkan.device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&barrier));
                    }

                    // --- Tonemap Pass ---
                    vulkan.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.tonemap_pipeline);
                    
                    let color_attachment = vk::RenderingAttachmentInfoKHR::default()
                        .image_view(sdr_target.color_view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .store_op(vk::AttachmentStoreOp::STORE);
                    
                    let rendering_info = vk::RenderingInfoKHR::default()
                        .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: sdr_target.width, height: sdr_target.height } })
                        .layer_count(1)
                        .color_attachments(std::slice::from_ref(&color_attachment));
                    
                    vulkan.device.cmd_begin_rendering(command_buffer, &rendering_info);

                    let viewport = vk::Viewport { x: 0.0, y: 0.0, width: sdr_target.width as f32, height: sdr_target.height as f32, min_depth: 0.0, max_depth: 1.0 };
                    vulkan.device.cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));
                    let scissor = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: sdr_target.width, height: sdr_target.height } };
                    vulkan.device.cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));

                    vulkan.device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.tonemap_layout, 0, std::slice::from_ref(&tonemap_descriptor_set), &[]);

                    vulkan.device.cmd_draw(command_buffer, 3, 1, 0, 0);
                    vulkan.device.cmd_end_rendering(command_buffer);
                }
            }
        );
    }

    pub fn destroy(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan.device.destroy_descriptor_set_layout(self.tonemap_descriptor_set_layout, None);
            vulkan.device.destroy_pipeline(self.tonemap_pipeline, None);
            vulkan.device.destroy_pipeline_layout(self.tonemap_layout, None);

            vulkan.device.destroy_descriptor_set_layout(self.bloom_descriptor_set_layout, None);
            vulkan.device.destroy_pipeline(self.bloom_downsample_pipeline, None);
            vulkan.device.destroy_pipeline(self.bloom_upsample_pipeline, None);
            vulkan.device.destroy_pipeline_layout(self.bloom_layout, None);
        }
    }
}
