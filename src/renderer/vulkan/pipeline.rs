use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub joint_ids: [u32; 4],     // Bone indices (up to 4 influences)
    pub joint_weights: [f32; 4], // Per-bone blend weights
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PointLight {
    pub position: [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlobalUbo {
    pub view_proj: crate::math::mat4::Mat4,
    pub light_space_matrix: crate::math::mat4::Mat4,
    pub camera_pos: [f32; 4],
    pub light_dir: [f32; 4],
    pub light_color: [f32; 4],
    pub point_lights: [PointLight; 4],
    pub num_point_lights: u32,
    pub _padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PushConstants {
    pub world: crate::math::mat4::Mat4,
    pub metallic: f32,
    pub roughness: f32,
    pub padding: [f32; 2],
    pub color: [f32; 4],
}

/// GPU instance data for Multi-Draw Indirect.
///
/// MUST be `#[repr(C, align(16))]` so the Rust layout exactly matches the
/// GLSL `std430` storage-buffer layout used by `cull.comp`, `shader.vert`,
/// and `prepass.vert`. Without `align(16)`, the GPU may read misaligned
/// `mat4` columns, producing garbage transforms or Vulkan validation errors.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct InstanceData {
    pub world: crate::math::mat4::Mat4, // 64 bytes (offset 0)
    pub aabb_min: [f32; 4],             // 16 bytes (offset 64)
    pub aabb_max: [f32; 4],             // 16 bytes (offset 80)
    pub color: [f32; 4],                // 16 bytes (offset 96)  (r, g, b, albedo_idx_bitcast)
    pub pbr: [f32; 4], // 16 bytes (offset 112) (metallic, roughness, normal_idx_bitcast, mr_idx_bitcast)
    pub geometry: [u32; 4], // 16 bytes (offset 128) (index_count, index_offset, vertex_offset, emissive_idx)
    pub geometry2: [u32; 4], // 16 bytes (offset 144) (meshlet_offset, meshlet_count, pad0, pad1)
}

pub struct Pipeline {
    pub layout: vk::PipelineLayout,
    pub handle: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub material_descriptor_set_layout: vk::DescriptorSetLayout,
}

impl Pipeline {
    pub fn new(
        vulkan: &VulkanDevice,
        color_format: vk::Format,
        vfs: &crate::vfs::Vfs,
        vert_shader: &str,
        frag_shader: &str,
    ) -> Option<Self> {
        // Attempt to load shaders from disk. If missing, return None gracefully.
        let vert_code = vfs.read_bytes(vert_shader).ok()?;
        let frag_code = vfs.read_bytes(frag_shader).ok()?;

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

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32);

        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

        let sampler_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1024) // Maximum unbounded array for bindless texturing
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let env_map_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let shadow_map_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let instance_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

        let global_bindings = [
            ubo_layout_binding,
            env_map_layout_binding,
            shadow_map_layout_binding,
            instance_layout_binding,
        ];

        let material_bindings = [sampler_layout_binding];

        let descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&global_bindings);

        let descriptor_set_layout = unsafe {
            vulkan
                .device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
                .ok()?
        };

        let binding_flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];

        let mut binding_flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let material_descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&material_bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .push_next(&mut binding_flags_info);

        let material_descriptor_set_layout = unsafe {
            vulkan
                .device
                .create_descriptor_set_layout(&material_descriptor_set_layout_info, None)
                .ok()?
        };

        let set_layouts = [descriptor_set_layout, material_descriptor_set_layout];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(std::slice::from_ref(&push_constant_range))
            .set_layouts(&set_layouts);

        let layout = unsafe {
            vulkan
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .ok()?
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_attachment_formats = [color_format];
        let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfoKHR::default()
            .color_attachment_formats(&color_attachment_formats)
            .depth_attachment_format(vk::Format::D32_SFLOAT);

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
            .push_next(&mut pipeline_rendering_info);

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
            material_descriptor_set_layout,
        })
    }

    pub fn create_shader_module(vulkan: &VulkanDevice, code: &[u8]) -> Option<vk::ShaderModule> {
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
            vulkan
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            vulkan
                .device
                .destroy_descriptor_set_layout(self.material_descriptor_set_layout, None);
        }
    }
}
