use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

pub struct PostProcessPipeline {
    pub layout: vk::PipelineLayout,
    pub handle: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

impl PostProcessPipeline {
    pub fn new(vulkan: &VulkanDevice, surface_format: vk::Format) -> Option<Self> {
        let vert_code = std::fs::read("shaders/post_process_vert.spv").ok()?;
        let frag_code = std::fs::read("shaders/post_process_frag.spv").ok()?;

        let vert_module =
            crate::renderer::vulkan::Pipeline::create_shader_module(vulkan, &vert_code)?;
        let frag_module =
            crate::renderer::vulkan::Pipeline::create_shader_module(vulkan, &frag_code)?;

        let entry_name = c"main";

        let vert_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(entry_name);

        let frag_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(entry_name);

        let shader_stages = [vert_stage, frag_stage];

        // Empty vertex input (we use gl_VertexIndex)
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

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

        // Descriptor Set Layout for the HDR offscreen texture
        let sampler_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = [sampler_layout_binding];
        let descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            vulkan
                .device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
                .ok()?
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        let layout = unsafe {
            vulkan
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .ok()?
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let mut rendering_info = vk::PipelineRenderingCreateInfoKHR::default()
            .color_attachment_formats(std::slice::from_ref(&surface_format));

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
            .layout(layout)
            .push_next(&mut rendering_info);

        let handle = unsafe {
            vulkan
                .device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .map_err(|e| e.1)
                .ok()?[0]
        };

        unsafe {
            vulkan.device.destroy_shader_module(vert_module, None);
            vulkan.device.destroy_shader_module(frag_module, None);
        }

        Some(Self {
            layout,
            handle,
            descriptor_set_layout,
        })
    }

    pub fn destroy(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            vulkan.device.destroy_pipeline(self.handle, None);
            vulkan.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}
