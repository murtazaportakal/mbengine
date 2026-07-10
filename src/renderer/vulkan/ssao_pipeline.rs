use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

pub struct SsaoPipeline {
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

impl SsaoPipeline {
    pub fn new(vulkan: &VulkanDevice, vfs: &crate::vfs::Vfs) -> Option<Self> {
        let vert_code = vfs
            .read_bytes("shaders/post_process_vert.spv")
            .unwrap_or_else(|e| {
                println!("Failed to read post_process_vert.spv: {:?}", e);
                vec![]
            });
        if vert_code.is_empty() {
            return None;
        }

        let frag_code = vfs.read_bytes("shaders/ssao_frag.spv").unwrap_or_else(|e| {
            println!("Failed to read ssao_frag.spv: {:?}", e);
            vec![]
        });
        if frag_code.is_empty() {
            return None;
        }

        let vert_module =
            crate::renderer::vulkan::Pipeline::create_shader_module(vulkan, &vert_code)
                .unwrap_or_else(|| {
                    println!("Failed to create vert shader module for SSAO");
                    vk::ShaderModule::null()
                });
        if vert_module == vk::ShaderModule::null() {
            return None;
        }

        let frag_module =
            crate::renderer::vulkan::Pipeline::create_shader_module(vulkan, &frag_code)
                .unwrap_or_else(|| {
                    println!("Failed to create frag shader module for SSAO");
                    vk::ShaderModule::null()
                });
        if frag_module == vk::ShaderModule::null() {
            return None;
        }

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
            .color_write_mask(vk::ColorComponentFlags::R)
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let bindings = [
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
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            vulkan
                .device
                .create_descriptor_set_layout(&dsl_info, None)
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

        let color_attachment_formats = [vk::Format::R8_UNORM];
        let mut rendering_info = vk::PipelineRenderingCreateInfoKHR::default()
            .color_attachment_formats(&color_attachment_formats);

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
            .layout(layout)
            .push_next(&mut rendering_info);

        let pipeline = unsafe {
            vulkan
                .device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .unwrap_or_else(|e| {
                    println!("Failed to create graphics pipeline for SSAO: {:?}", e);
                    vec![vk::Pipeline::null()]
                })[0]
        };
        if pipeline == vk::Pipeline::null() {
            return None;
        }

        unsafe {
            vulkan.device.destroy_shader_module(vert_module, None);
            vulkan.device.destroy_shader_module(frag_module, None);
        }

        Some(Self {
            layout,
            pipeline,
            descriptor_set_layout,
        })
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan.device.destroy_pipeline(self.pipeline, None);
            vulkan.device.destroy_pipeline_layout(self.layout, None);
            vulkan
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
