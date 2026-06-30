use ash::vk;
use crate::renderer::vulkan::VulkanDevice;


#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}

pub struct Pipeline {
    pub layout: vk::PipelineLayout,
    pub handle: vk::Pipeline,
}

impl Pipeline {
    pub fn new(vulkan: &VulkanDevice, render_pass: vk::RenderPass, extent: vk::Extent2D) -> Option<Self> {
        // Attempt to load shaders from disk. If missing, return None gracefully.
        let vert_code = std::fs::read("shaders/vert.spv").ok()?;
        let frag_code = std::fs::read("shaders/frag.spv").ok()?;

        let vert_module = Self::create_shader_module(vulkan, &vert_code)?;
        let frag_module = Self::create_shader_module(vulkan, &frag_code)?;

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
                .offset(memoffset::offset_of!(Vertex, color) as u32),
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_description))
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<crate::math::mat4::Mat4>() as u32);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let layout = unsafe { vulkan.device.create_pipeline_layout(&pipeline_layout_info, None).ok()? };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);

        let handle = unsafe {
            vulkan.device
                .create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None)
                .map_err(|e| e.1)
                .ok()?[0]
        };

        unsafe {
            vulkan.device.destroy_shader_module(vert_module, None);
            vulkan.device.destroy_shader_module(frag_module, None);
        }

        Some(Self { layout, handle })
    }

    fn create_shader_module(vulkan: &VulkanDevice, code: &[u8]) -> Option<vk::ShaderModule> {
        let (prefix, code_u32, suffix) = unsafe { code.align_to::<u32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return None;
        }

        let create_info = vk::ShaderModuleCreateInfo::default().code(code_u32);
        unsafe { vulkan.device.create_shader_module(&create_info, None).ok() }
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan.device.destroy_pipeline(self.handle, None);
            vulkan.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}
