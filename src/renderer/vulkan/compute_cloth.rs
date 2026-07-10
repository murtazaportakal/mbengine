//! Vulkan compute pipeline for GPU cloth simulation (soft bodies).

use crate::renderer::vulkan::buffer::Buffer;
use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SphereCollider {
    pub pos: [f32; 3],
    pub radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ClothPushConstants {
    pub grid_width: u32,
    pub grid_height: u32,
    pub dt: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub num_colliders: u32,
}

pub struct ClothInstance {
    pub vertex_buffer: Buffer,
    pub velocity_buffer: Buffer,
    pub grid_width: u32,
    pub grid_height: u32,
    pub descriptor_set: vk::DescriptorSet,
    pub solve_descriptor_set: vk::DescriptorSet,
}

impl ClothInstance {
    pub fn new(
        vulkan: &VulkanDevice,
        pipeline: &ComputeClothPipeline,
        descriptor_pool: vk::DescriptorPool,
        _source_vertex_buffer: vk::Buffer,
        vertex_count: u32,
        grid_width: u32,
        grid_height: u32,
    ) -> Option<Self> {
        let vertex_size = std::mem::size_of::<crate::renderer::vulkan::pipeline::Vertex>();
        let vertex_buffer_size = (vertex_count as usize * vertex_size) as u64;

        // Create the read-write vertex buffer and copy the initial vertices into it
        let vertex_buffer = Buffer::new(
            vulkan,
            vertex_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Velocity buffer: vec4 (xyz=vel, w=inv_mass)
        let velocity_buffer_size = (vertex_count as usize * std::mem::size_of::<[f32; 4]>()) as u64;
        let velocity_buffer = Buffer::new(
            vulkan,
            velocity_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Allocate descriptor sets
        let layouts = [
            pipeline.descriptor_set_layout,
            pipeline.solve_descriptor_set_layout,
        ];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let sets = unsafe { vulkan.device.allocate_descriptor_sets(&alloc_info).ok()? };
        let descriptor_set = sets[0];
        let solve_descriptor_set = sets[1];

        let vertex_info = vk::DescriptorBufferInfo::default()
            .buffer(vertex_buffer.handle)
            .offset(0)
            .range(vertex_buffer_size);

        let velocity_info = vk::DescriptorBufferInfo::default()
            .buffer(velocity_buffer.handle)
            .offset(0)
            .range(velocity_buffer_size);

        let colliders_info = vk::DescriptorBufferInfo::default()
            .buffer(pipeline.colliders_buffer.handle)
            .offset(0)
            .range(vk::WHOLE_SIZE);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&vertex_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&velocity_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&colliders_info)),
            // Solve set
            vk::WriteDescriptorSet::default()
                .dst_set(solve_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&vertex_info)),
        ];

        unsafe {
            vulkan.device.update_descriptor_sets(&writes, &[]);
        }

        Some(Self {
            vertex_buffer,
            velocity_buffer,
            grid_width,
            grid_height,
            descriptor_set,
            solve_descriptor_set,
        })
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        self.vertex_buffer.shutdown(vulkan);
        self.velocity_buffer.shutdown(vulkan);
    }
}

pub struct ComputeClothPipeline {
    pub integrate_pipeline: vk::Pipeline,
    pub solve_pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub solve_pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub solve_descriptor_set_layout: vk::DescriptorSetLayout,
    pub colliders_buffer: Buffer,
}

impl ComputeClothPipeline {
    pub fn new(vulkan: &VulkanDevice, vfs: &crate::vfs::Vfs) -> Option<Self> {
        let colliders_size = 64 * std::mem::size_of::<SphereCollider>() as u64;
        let colliders_buffer = Buffer::new(
            vulkan,
            colliders_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            vulkan
                .device
                .create_descriptor_set_layout(&layout_info, None)
                .ok()?
        };

        let solve_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let solve_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&solve_bindings);
        let solve_descriptor_set_layout = unsafe {
            vulkan
                .device
                .create_descriptor_set_layout(&solve_layout_info, None)
                .ok()?
        };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ClothPushConstants>() as u32);

        let solve_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size((2 * std::mem::size_of::<u32>()) as u32);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout = unsafe {
            vulkan
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .ok()?
        };

        let solve_pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&solve_descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&solve_push_constant_range));
        let solve_pipeline_layout = unsafe {
            vulkan
                .device
                .create_pipeline_layout(&solve_pipeline_layout_info, None)
                .ok()?
        };

        let int_code = vfs.read_bytes("shaders/cloth_integrate.spv").ok()?;
        let solve_code = vfs.read_bytes("shaders/cloth_solve.spv").ok()?;

        let int_module =
            crate::renderer::vulkan::pipeline::Pipeline::create_shader_module(vulkan, &int_code)?;
        let solve_module =
            crate::renderer::vulkan::pipeline::Pipeline::create_shader_module(vulkan, &solve_code)?;
        let entry_name = c"main";

        let int_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(int_module)
            .name(entry_name);

        let solve_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(solve_module)
            .name(entry_name);

        let int_info = vk::ComputePipelineCreateInfo::default()
            .stage(int_stage)
            .layout(pipeline_layout);

        let solve_info = vk::ComputePipelineCreateInfo::default()
            .stage(solve_stage)
            .layout(solve_pipeline_layout);

        let pipelines = unsafe {
            vulkan
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[int_info, solve_info], None)
                .ok()?
        };

        unsafe {
            vulkan.device.destroy_shader_module(int_module, None);
            vulkan.device.destroy_shader_module(solve_module, None);
        }

        Some(Self {
            integrate_pipeline: pipelines[0],
            solve_pipeline: pipelines[1],
            pipeline_layout,
            solve_pipeline_layout,
            descriptor_set_layout,
            solve_descriptor_set_layout,
            colliders_buffer,
        })
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan
                .device
                .destroy_pipeline(self.integrate_pipeline, None);
            vulkan.device.destroy_pipeline(self.solve_pipeline, None);
            vulkan
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            vulkan
                .device
                .destroy_pipeline_layout(self.solve_pipeline_layout, None);
            vulkan
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            vulkan
                .device
                .destroy_descriptor_set_layout(self.solve_descriptor_set_layout, None);
        }
        self.colliders_buffer.shutdown(vulkan);
    }
}
