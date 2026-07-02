use crate::renderer::vulkan::pipeline::GlobalUbo;
use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

pub struct ComputeCullPipeline {
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

impl ComputeCullPipeline {
    pub fn new(
        vulkan: &VulkanDevice,
    ) -> Option<Self> {
        let comp_code = std::fs::read("shaders/cull.spv").ok()?;
        let comp_module = crate::renderer::vulkan::pipeline::Pipeline::create_shader_module(vulkan, &comp_code)?;

        let entry_name = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(comp_module)
            .name(entry_name);

        let bindings = [
            // UBO
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // MeshletBuffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // IndirectDrawBuffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            vulkan.device.create_descriptor_set_layout(&layout_info, None).ok()?
        };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(80); // totalMeshlets (4) + pad (12) + world_matrix (64)

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let layout = unsafe {
            vulkan.device.create_pipeline_layout(&pipeline_layout_info, None).ok()?
        };

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);

        let pipeline = unsafe {
            vulkan.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            ).map_err(|e| e.1).ok()?[0]
        };

        unsafe {
            vulkan.device.destroy_shader_module(comp_module, None);
        }

        Some(Self {
            layout,
            pipeline,
            descriptor_set_layout,
        })
    }

    pub fn update_descriptor_set(
        &self,
        vulkan: &VulkanDevice,
        ubo_buffer: vk::Buffer,
        meshlet_buffer: vk::Buffer,
        indirect_buffer: vk::Buffer,
        descriptor_set: vk::DescriptorSet,
    ) {
        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(ubo_buffer)
            .offset(0)
            .range(std::mem::size_of::<GlobalUbo>() as vk::DeviceSize);

        let meshlet_info = vk::DescriptorBufferInfo::default()
            .buffer(meshlet_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);

        let indirect_info = vk::DescriptorBufferInfo::default()
            .buffer(indirect_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);

        let write_ubo = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&ubo_info));

        let write_meshlet = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&meshlet_info));

        let write_indirect = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&indirect_info));

        unsafe {
            vulkan.device.update_descriptor_sets(
                &[write_ubo, write_meshlet, write_indirect],
                &[],
            );
        }
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan.device.destroy_pipeline(self.pipeline, None);
            vulkan.device.destroy_pipeline_layout(self.layout, None);
            vulkan.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
