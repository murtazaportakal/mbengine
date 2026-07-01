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
