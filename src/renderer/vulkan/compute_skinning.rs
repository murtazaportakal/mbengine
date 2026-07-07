//! Vulkan compute pipeline for GPU skeletal vertex skinning.
//!
//! Dispatches a compute shader that reads original vertex data + bone matrices,
//! applies weighted bone transforms, and writes deformed vertices to an output buffer.
//! The output buffer is then bound as the vertex buffer during rendering.

use crate::math::mat4::Mat4;
use crate::renderer::vulkan::buffer::Buffer;
use crate::renderer::vulkan::skeleton::MAX_BONES;
use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

/// GPU resources for a single skinned mesh instance.
pub struct SkinningInstance {
    /// SSBO containing the bone matrices (MAX_BONES * Mat4).
    pub bone_buffer: Buffer,
    /// SSBO containing the deformed vertex data (output of compute).
    pub skinned_vertex_buffer: Buffer,
    /// Number of vertices in the mesh.
    pub vertex_count: u32,
    /// Descriptor set bound to this instance's buffers.
    pub descriptor_set: vk::DescriptorSet,
}

impl SkinningInstance {
    /// Create a new skinning instance for a mesh with the given vertex count.
    ///
    /// # Safety
    /// `source_vertex_buffer` must be a valid Vulkan buffer handle.
    pub fn new(
        vulkan: &VulkanDevice,
        pipeline: &ComputeSkinningPipeline,
        descriptor_pool: vk::DescriptorPool,
        source_vertex_buffer: vk::Buffer,
        vertex_count: u32,
    ) -> Option<Self> {
        // Bone matrix buffer (MAX_BONES * 64 bytes per Mat4)
        let bone_buffer_size = (MAX_BONES * std::mem::size_of::<Mat4>()) as u64;
        let bone_buffer = Buffer::new(
            vulkan,
            bone_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Initialize with identity matrices
        let identity_matrices = vec![Mat4::identity(); MAX_BONES];
        bone_buffer.upload(vulkan, &identity_matrices);

        // Skinned vertex output buffer (same size as input)
        let vertex_size = std::mem::size_of::<crate::renderer::vulkan::pipeline::Vertex>();
        let skinned_buffer_size = (vertex_count as usize * vertex_size) as u64;
        let skinned_vertex_buffer = Buffer::new(
            vulkan,
            skinned_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Allocate descriptor set
        let set_layouts = [pipeline.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let descriptor_set =
            unsafe { vulkan.device.allocate_descriptor_sets(&alloc_info).ok()?[0] };

        // Update descriptor set
        let bone_info = vk::DescriptorBufferInfo::default()
            .buffer(bone_buffer.handle)
            .offset(0)
            .range(bone_buffer_size);

        let input_info = vk::DescriptorBufferInfo::default()
            .buffer(source_vertex_buffer)
            .offset(0)
            .range(skinned_buffer_size);

        let output_info = vk::DescriptorBufferInfo::default()
            .buffer(skinned_vertex_buffer.handle)
            .offset(0)
            .range(skinned_buffer_size);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&bone_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&input_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&output_info)),
        ];

        unsafe {
            vulkan.device.update_descriptor_sets(&writes, &[]);
        }

        Some(Self {
            bone_buffer,
            skinned_vertex_buffer,
            vertex_count,
            descriptor_set,
        })
    }

    /// Upload bone matrices to the GPU.
    pub fn upload_bone_matrices(&self, vulkan: &VulkanDevice, matrices: &[Mat4]) {
        // Pad to MAX_BONES with identity
        let mut padded = vec![Mat4::identity(); MAX_BONES];
        let count = matrices.len().min(MAX_BONES);
        padded[..count].copy_from_slice(&matrices[..count]);
        self.bone_buffer.upload(vulkan, &padded);
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        self.bone_buffer.shutdown(vulkan);
        self.skinned_vertex_buffer.shutdown(vulkan);
    }
}

/// The Vulkan compute pipeline for skeletal skinning.
pub struct ComputeSkinningPipeline {
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

impl ComputeSkinningPipeline {
    /// Create the skinning compute pipeline.
    pub fn new(vulkan: &VulkanDevice, vfs: &crate::vfs::Vfs) -> Option<Self> {
        let comp_code = vfs.read_bytes("shaders/skinning.spv").ok()?;
        let comp_module =
            crate::renderer::vulkan::Pipeline::create_shader_module(vulkan, &comp_code)?;

        let entry_name = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(comp_module)
            .name(entry_name);

        // Descriptor set layout: 3 SSBOs
        let bindings = [
            // Bone matrices
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Input vertices
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Output (skinned) vertices
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

        // Push constant: vertex_count (u32)
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(4); // sizeof(u32)

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let layout = unsafe {
            vulkan
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .ok()?
        };

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);

        let pipeline = unsafe {
            vulkan
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .map_err(|e| e.1)
                .ok()?[0]
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

    /// Record compute dispatch commands for a skinning instance.
    ///
    /// Must be called before the render pass that uses the skinned vertex buffer.
    /// A pipeline barrier is inserted after the dispatch to ensure the compute
    /// writes are visible to the vertex shader.
    ///
    /// # Safety
    /// `command_buffer` must be in a recording state and not inside a render pass.
    pub unsafe fn dispatch(
        &self,
        vulkan: &VulkanDevice,
        command_buffer: vk::CommandBuffer,
        instance: &SkinningInstance,
    ) {
        vulkan.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline,
        );

        vulkan.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.layout,
            0,
            &[instance.descriptor_set],
            &[],
        );

        let vertex_count = instance.vertex_count;
        vulkan.device.cmd_push_constants(
            command_buffer,
            self.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &vertex_count.to_ne_bytes(),
        );

        // Dispatch: ceil(vertex_count / 256)
        let group_count = vertex_count.div_ceil(256);
        vulkan
            .device
            .cmd_dispatch(command_buffer, group_count, 1, 1);

        // Memory barrier: compute write → vertex read
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::VERTEX_ATTRIBUTE_READ);

        vulkan.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::VERTEX_INPUT,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&barrier),
            &[],
            &[],
        );
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
