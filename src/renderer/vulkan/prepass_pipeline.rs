use crate::renderer::vulkan::pipeline::Vertex;
use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

pub struct PrepassPipeline {
    pub layout: vk::PipelineLayout,
    pub handle: vk::Pipeline,
}

impl PrepassPipeline {
    pub fn new(
        vulkan: &VulkanDevice,
        vfs: &crate::vfs::Vfs,
        pipeline_layout: vk::PipelineLayout,
        depth_format: vk::Format,
    ) -> Option<Self> {
        let vert_code = vfs.read_bytes("shaders/prepass_vert.spv").ok()?;
        let frag_code = vfs.read_bytes("shaders/prepass_frag.spv").ok()?;

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

        let binding_description = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);

        let attribute_descriptions = [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(memoffset::offset_of!(Vertex, pos) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(memoffset::offset_of!(Vertex, normal) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(memoffset::offset_of!(Vertex, uv) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(3)
                .format(vk::Format::R32G32B32A32_UINT)
                .offset(memoffset::offset_of!(Vertex, joint_ids) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(4)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(memoffset::offset_of!(Vertex, joint_weights) as u32),
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_description))
            .vertex_attribute_descriptions(&attribute_descriptions);

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
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_attachment_formats = [vk::Format::R16G16B16A16_SFLOAT];
        let mut rendering_info = vk::PipelineRenderingCreateInfoKHR::default()
            .color_attachment_formats(&color_attachment_formats)
            .depth_attachment_format(depth_format);

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
            .push_next(&mut rendering_info);

        let handle = unsafe {
            vulkan
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .unwrap_or_else(|e| {
                    println!("Failed to create graphics pipeline for Prepass: {:?}", e);
                    vec![vk::Pipeline::null()]
                })[0]
        };
        if handle == vk::Pipeline::null() {
            return None;
        }

        unsafe {
            vulkan.device.destroy_shader_module(vert_module, None);
            vulkan.device.destroy_shader_module(frag_module, None);
        }

        Some(Self {
            layout: pipeline_layout,
            handle,
        })
    }
}
