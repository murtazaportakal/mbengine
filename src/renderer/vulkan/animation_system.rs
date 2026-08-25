use ash::vk;
use crate::renderer::vulkan::{VulkanDevice, compute_anim::ComputeAnimPipeline};
use crate::renderer::vulkan::buffer::Buffer;

pub struct AnimationSystem {
    pub pipeline: ComputeAnimPipeline,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: [vk::DescriptorSet; 2],
    
    pub anim_data_buffers: [Buffer; 2],
    pub anim_data_mapped: [*mut std::ffi::c_void; 2],
    pub anim_bone_matrices_buffer: Buffer,
    pub anim_instance_data_buffer: Vec<crate::renderer::vulkan::animation::InstanceAnimData>,
}

impl AnimationSystem {
    pub fn new(vulkan: &VulkanDevice, vfs: &crate::vfs::Vfs) -> Option<Self> {
        let pipeline = ComputeAnimPipeline::new(vulkan, vfs)?;
        
        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 10 },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(2);
        let descriptor_pool = unsafe { vulkan.device.create_descriptor_pool(&pool_info, None).unwrap() };
        
        let layouts = [pipeline.descriptor_set_layout, pipeline.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets: [vk::DescriptorSet; 2] = unsafe { 
            vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap().try_into().unwrap() 
        };

        let anim_data_size = (10_000 * std::mem::size_of::<crate::renderer::vulkan::animation::InstanceAnimData>()) as u64;
        let mut anim_data_bufs = Vec::with_capacity(2);
        let mut anim_data_maps = Vec::with_capacity(2);
        for _ in 0..2 {
            let ab = Buffer::new(
                vulkan,
                anim_data_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ).expect("Failed to create anim data buffer");
            let mapped = unsafe {
                vulkan
                    .device
                    .map_memory(ab.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .expect("Failed to map anim data buffer")
            };
            anim_data_bufs.push(ab);
            anim_data_maps.push(mapped);
        }
        let anim_data_buffers = [anim_data_bufs.remove(0), anim_data_bufs.remove(0)];
        let anim_data_mapped = [anim_data_maps.remove(0), anim_data_maps.remove(0)];
        
        let anim_bone_matrices_buffer = Buffer::new(
            vulkan, 
            (std::mem::size_of::<crate::math::mat4::Mat4>() * 10_000 * 128) as u64, 
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS, 
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        ).unwrap();
        
        Some(Self {
            pipeline,
            descriptor_pool,
            descriptor_sets,
            anim_data_buffers,
            anim_data_mapped,
            anim_bone_matrices_buffer,
            anim_instance_data_buffer: Vec::with_capacity(10_000),
        })
    }

    pub fn update_descriptors(&self, vulkan: &VulkanDevice, animation_pool: &crate::renderer::vulkan::animation_pool::AnimationPool) {
        for i in 0..2 {
            let inst_info = vk::DescriptorBufferInfo::default()
                .buffer(self.anim_data_buffers[i].handle)
                .offset(0).range(vk::WHOLE_SIZE);
            let skel_info = vk::DescriptorBufferInfo::default()
                .buffer(animation_pool.skeleton_buffer.handle)
                .offset(0).range(vk::WHOLE_SIZE);
            let clip_info = vk::DescriptorBufferInfo::default()
                .buffer(animation_pool.clip_buffer.handle)
                .offset(0).range(vk::WHOLE_SIZE);
            let kf_info = vk::DescriptorBufferInfo::default()
                .buffer(animation_pool.keyframe_buffer.handle)
                .offset(0).range(vk::WHOLE_SIZE);
            let out_info = vk::DescriptorBufferInfo::default()
                .buffer(self.anim_bone_matrices_buffer.handle)
                .offset(0).range(vk::WHOLE_SIZE);
                
            let writes = [
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&inst_info)),
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&skel_info)),
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&clip_info)),
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&kf_info)),
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(4).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&out_info)),
            ];
            unsafe { vulkan.device.update_descriptor_sets(&writes, &[]); }
        }
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        if self.descriptor_pool != vk::DescriptorPool::null() {
            unsafe { vulkan.device.destroy_descriptor_pool(self.descriptor_pool, None); }
        }
        for i in 0..2 {
            self.anim_data_buffers[i].shutdown(vulkan);
        }
        self.anim_bone_matrices_buffer.shutdown(vulkan);
    }
}
