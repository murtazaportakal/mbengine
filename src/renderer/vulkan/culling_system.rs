use ash::vk;
use crate::renderer::vulkan::{VulkanDevice, compute_cull::ComputeCullPipeline, hzb::HzbTarget};
use crate::renderer::vulkan::buffer::Buffer;

pub struct CullingSystem {
    pub pipeline: ComputeCullPipeline,
    pub descriptor_pool: vk::DescriptorPool,
    
    // Multi-buffered GPU buffers
    pub draw_count_buffers: [Buffer; 2],
    pub draw_count_mapped: [*mut u32; 2],
    pub indirect_buffers: [Buffer; 2],
    
    pub prefix_sum_buffers: [Buffer; 2],
    pub prefix_sum_mapped: [*mut u32; 2],
    pub prefix_sum_data_buffer: Vec<u32>,
    
    pub occluded_meshlets_buffers: [Buffer; 2],
    pub occluded_count_buffers: [Buffer; 2],
    pub indirect_buffers_phase2: [Buffer; 2],
    pub draw_count_buffers_phase2: [Buffer; 2],
    
    pub hzb_target: Option<HzbTarget>,
    pub depth_copy_sampler: vk::Sampler,
    pub hzb_frame_count: u32,
}

impl CullingSystem {
    pub fn new(vulkan: &VulkanDevice, vfs: &crate::vfs::Vfs) -> Self {
        let max_instances = 100_000usize;
        let max_meshlets = 10_000_000usize; // Max visible meshlets globally across all instances
        
        // --- Pipeline & Descriptors ---
        let pipeline = ComputeCullPipeline::new(vulkan, vfs).unwrap();
        
        let compute_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1000),
        ];
        let compute_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&compute_pool_sizes)
            .max_sets(1000);
        let descriptor_pool = unsafe {
            vulkan.device.create_descriptor_pool(&compute_pool_info, None).unwrap()
        };
        
        // --- Buffers ---
        let mut indirect_bufs = Vec::with_capacity(2);
        let mut draw_count_bufs = Vec::with_capacity(2);
        let mut draw_count_maps = Vec::with_capacity(2);

        let mut occluded_meshlets_bufs = Vec::with_capacity(2);
        let mut occluded_count_bufs = Vec::with_capacity(2);
        let mut indirect_bufs_phase2 = Vec::with_capacity(2);
        let mut draw_count_bufs_phase2 = Vec::with_capacity(2);

        for _ in 0..2 {
            let idb = Buffer::new(
                vulkan,
                (max_meshlets * std::mem::size_of::<vk::DrawIndexedIndirectCommand>()) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ).expect("Failed to create indirect buffer");
            indirect_bufs.push(idb);

            let dcb = Buffer::new(
                vulkan,
                4,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ).expect("Failed to create draw count buffer");
            let dcb_mapped = unsafe {
                vulkan.device.map_memory(dcb.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .expect("Failed to map draw count buffer")
            };
            draw_count_bufs.push(dcb);
            draw_count_maps.push(dcb_mapped as *mut u32);

            let ocb = Buffer::new(
                vulkan,
                4,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ).expect("Failed to create occluded count buffer");
            
            let omb = Buffer::new(
                vulkan,
                (max_meshlets * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ).expect("Failed to create occluded meshlets buffer");

            let idb2 = Buffer::new(
                vulkan,
                (max_meshlets * std::mem::size_of::<vk::DrawIndexedIndirectCommand>()) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ).expect("Failed to create indirect buffer 2");

            let dcb2 = Buffer::new(
                vulkan,
                4,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ).expect("Failed to create draw count buffer 2");

            occluded_count_bufs.push(ocb);
            occluded_meshlets_bufs.push(omb);
            indirect_bufs_phase2.push(idb2);
            draw_count_bufs_phase2.push(dcb2);
        }

        let mut prefix_sum_bufs = Vec::with_capacity(2);
        let mut prefix_sum_maps = Vec::with_capacity(2);
        for _ in 0..2 {
            let pb = Buffer::new(
                vulkan,
                (max_instances * 4) as u64, // max_instances is correct here because prefix sum is PER-INSTANCE
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ).expect("Failed to create prefix sum buffer");
            let mapped = unsafe {
                vulkan.device.map_memory(pb.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .expect("Failed to map prefix sum buffer")
            };
            prefix_sum_bufs.push(pb);
            prefix_sum_maps.push(mapped as *mut u32);
        }

        Self {
            pipeline,
            descriptor_pool,
            draw_count_buffers: [draw_count_bufs.remove(0), draw_count_bufs.remove(0)],
            draw_count_mapped: [draw_count_maps.remove(0), draw_count_maps.remove(0)],
            indirect_buffers: [indirect_bufs.remove(0), indirect_bufs.remove(0)],
            prefix_sum_buffers: [prefix_sum_bufs.remove(0), prefix_sum_bufs.remove(0)],
            prefix_sum_mapped: [prefix_sum_maps.remove(0), prefix_sum_maps.remove(0)],
            prefix_sum_data_buffer: Vec::with_capacity(max_instances),
            occluded_meshlets_buffers: [occluded_meshlets_bufs.remove(0), occluded_meshlets_bufs.remove(0)],
            occluded_count_buffers: [occluded_count_bufs.remove(0), occluded_count_bufs.remove(0)],
            indirect_buffers_phase2: [indirect_bufs_phase2.remove(0), indirect_bufs_phase2.remove(0)],
            draw_count_buffers_phase2: [draw_count_bufs_phase2.remove(0), draw_count_bufs_phase2.remove(0)],
            hzb_target: None,
            depth_copy_sampler: vk::Sampler::null(),
            hzb_frame_count: 0,
        }
    }
    
    pub fn update_descriptor_sets(
        &mut self,
        vulkan: &VulkanDevice,
        ubo_buffers: &[vk::Buffer; 2],
        instance_buffers: &[Buffer; 2],
        meshlet_buffer: vk::Buffer,
    ) {
        let layouts = [
            self.pipeline.descriptor_set_layout,
            self.pipeline.descriptor_set_layout,
            self.pipeline.descriptor_set_layout,
            self.pipeline.descriptor_set_layout,
        ];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        self.pipeline.descriptor_sets = unsafe {
            vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap()
        };

        for i in 0..2 {
            // Phase 1 descriptor set
            self.pipeline.update_descriptor_set(
                vulkan,
                ubo_buffers[i],
                self.indirect_buffers[i].handle,
                instance_buffers[i].handle,
                self.draw_count_buffers[i].handle,
                meshlet_buffer,
                self.prefix_sum_buffers[i].handle,
                self.occluded_meshlets_buffers[i].handle,
                self.occluded_count_buffers[i].handle,
                self.pipeline.descriptor_sets[i * 2],
            );
            // Phase 2 descriptor set
            self.pipeline.update_descriptor_set(
                vulkan,
                ubo_buffers[i],
                self.indirect_buffers_phase2[i].handle,
                instance_buffers[i].handle,
                self.draw_count_buffers_phase2[i].handle,
                meshlet_buffer,
                self.prefix_sum_buffers[i].handle,
                self.occluded_meshlets_buffers[i].handle,
                self.occluded_count_buffers[i].handle,
                self.pipeline.descriptor_sets[i * 2 + 1],
            );
        }
    }

    pub fn init_hzb(&mut self, vulkan: &VulkanDevice, width: u32, height: u32, vfs: &crate::vfs::Vfs) {
        if self.depth_copy_sampler == vk::Sampler::null() {
            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .min_lod(0.0)
                .max_lod(vk::LOD_CLAMP_NONE);
            self.depth_copy_sampler = unsafe { vulkan.device.create_sampler(&sampler_info, None).unwrap() };
        }
        if let Some(mut old) = self.hzb_target.take() {
            old.shutdown(vulkan);
        }
        self.hzb_target = HzbTarget::new(vulkan, width, height, vfs, self.depth_copy_sampler);
        self.hzb_frame_count = 0;
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        if self.descriptor_pool != vk::DescriptorPool::null() {
            unsafe { vulkan.device.destroy_descriptor_pool(self.descriptor_pool, None); }
        }
        self.pipeline.shutdown(vulkan);
        for i in 0..2 {
            self.draw_count_buffers[i].shutdown(vulkan);
            self.indirect_buffers[i].shutdown(vulkan);
            self.prefix_sum_buffers[i].shutdown(vulkan);
            self.occluded_meshlets_buffers[i].shutdown(vulkan);
            self.occluded_count_buffers[i].shutdown(vulkan);
            self.indirect_buffers_phase2[i].shutdown(vulkan);
            self.draw_count_buffers_phase2[i].shutdown(vulkan);
        }
        if self.depth_copy_sampler != vk::Sampler::null() {
            unsafe { vulkan.device.destroy_sampler(self.depth_copy_sampler, None); }
        }
        if let Some(mut hzb) = self.hzb_target.take() {
            hzb.shutdown(vulkan);
        }
    }
}


