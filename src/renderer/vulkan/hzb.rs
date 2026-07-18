use ash::vk;
use crate::renderer::vulkan::{VulkanDevice, buffer::Buffer};

pub const HZB_MAX_MIPS: usize = 13;

pub struct HzbTarget {
    pub image: vk::Image,
    pub image_memory: vk::DeviceMemory,
    pub mip_views: Vec<vk::ImageView>,
    pub full_view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub mip_count: u32,
    pub base_width: u32,
    pub base_height: u32,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub descriptor_pool: vk::DescriptorPool,
    pub param_buffers: Vec<Buffer>,
    pub param_mapped: Vec<*mut u8>,
}

unsafe impl Send for HzbTarget {}
unsafe impl Sync for HzbTarget {}

#[repr(C)]
struct HzbParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
}

impl HzbTarget {
    pub fn new(
        vulkan: &VulkanDevice,
        width: u32,
        height: u32,
        vfs: &crate::vfs::Vfs,
    ) -> Option<Self> {
        let mip_count = {
            let mut m = 1u32;
            let mut w = width;
            let mut h = height;
            while (w > 1 || h > 1) && m < HZB_MAX_MIPS as u32 {
                w = (w / 2).max(1);
                h = (h / 2).max(1);
                m += 1;
            }
            m
        };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(mip_count)
            .array_layers(1)
            .format(vk::Format::R32_SFLOAT)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe { vulkan.device.create_image(&image_info, None).ok()? };
        let mem_req = unsafe { vulkan.device.get_image_memory_requirements(image) };
        let mem_type = vulkan.find_memory_type(mem_req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let image_memory = unsafe {
            let ai = vk::MemoryAllocateInfo::default().allocation_size(mem_req.size).memory_type_index(mem_type);
            vulkan.device.allocate_memory(&ai, None).ok()?
        };
        unsafe { vulkan.device.bind_image_memory(image, image_memory, 0).ok()? };

        let mut mip_views = Vec::with_capacity(mip_count as usize);
        for mip in 0..mip_count {
            let vi = vk::ImageViewCreateInfo::default()
                .image(image).view_type(vk::ImageViewType::TYPE_2D).format(vk::Format::R32_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: mip, level_count: 1, base_array_layer: 0, layer_count: 1 });
            mip_views.push(unsafe { vulkan.device.create_image_view(&vi, None).ok()? });
        }

        let full_vi = vk::ImageViewCreateInfo::default()
            .image(image).view_type(vk::ImageViewType::TYPE_2D).format(vk::Format::R32_SFLOAT)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: mip_count, base_array_layer: 0, layer_count: 1 });
        let full_view = unsafe { vulkan.device.create_image_view(&full_vi, None).ok()? };

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST).min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0).max_lod(mip_count as f32).anisotropy_enable(false);
        let sampler = unsafe { vulkan.device.create_sampler(&sampler_info, None).ok()? };

        let shader_code = vfs.read_bytes("shaders/generate_hzb.spv").ok()?;
        let shader_module = crate::renderer::vulkan::pipeline::Pipeline::create_shader_module(vulkan, &shader_code)?;

        let bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let descriptor_set_layout = unsafe { vulkan.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).ok()? };
        let pipeline_layout = unsafe { vulkan.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&descriptor_set_layout)), None).ok()? };

        let entry = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(shader_module).name(entry);
        let pipeline_info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout);
        let pipeline = unsafe { vulkan.device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None).map_err(|e| e.1).ok()?[0] };
        unsafe { vulkan.device.destroy_shader_module(shader_module, None); }

        let pool_sizes = [
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(mip_count * 2),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(mip_count),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(mip_count),
        ];
        let descriptor_pool = unsafe { vulkan.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(mip_count), None).ok()? };
        let layouts: Vec<_> = (0..mip_count).map(|_| descriptor_set_layout).collect();
        let descriptor_sets = unsafe { vulkan.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)).ok()? };

        let mut param_buffers = Vec::with_capacity(mip_count as usize);
        let mut param_mapped = Vec::with_capacity(mip_count as usize);
        for _ in 0..mip_count {
            let buf = Buffer::new(vulkan, std::mem::size_of::<HzbParams>() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).expect("hzb param buf");
            let ptr = unsafe { vulkan.device.map_memory(buf.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).expect("hzb param map") as *mut u8 };
            param_buffers.push(buf);
            param_mapped.push(ptr);
        }

        Some(Self { image, image_memory, mip_views, full_view, sampler, mip_count, base_width: width, base_height: height, pipeline, pipeline_layout, descriptor_set_layout, descriptor_sets, descriptor_pool, param_buffers, param_mapped })
    }

    pub fn initial_transition(&self, vulkan: &VulkanDevice, cmd: vk::CommandBuffer) {
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::NONE).dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED).new_layout(vk::ImageLayout::GENERAL)
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: self.mip_count, base_array_layer: 0, layer_count: 1 });
        unsafe { vulkan.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&barrier)); }
    }

    pub fn generate(&self, vulkan: &VulkanDevice, cmd: vk::CommandBuffer, depth_view: vk::ImageView, depth_sampler: vk::Sampler) {
        // Barrier: ensure previous frame's cull read is done, then let compute write
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ).dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::GENERAL).new_layout(vk::ImageLayout::GENERAL)
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: self.mip_count, base_array_layer: 0, layer_count: 1 });
        unsafe { vulkan.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&barrier)); }
        unsafe { vulkan.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline); }

        let mut src_w = self.base_width;
        let mut src_h = self.base_height;

        for mip in 0..self.mip_count {
            let dst_w = (src_w / 2).max(1);
            let dst_h = (src_h / 2).max(1);

            let params = HzbParams { src_width: src_w, src_height: src_h, dst_width: dst_w, dst_height: dst_h };
            unsafe { std::ptr::copy_nonoverlapping(&params as *const HzbParams as *const u8, self.param_mapped[mip as usize], std::mem::size_of::<HzbParams>()); }

            let (src_view, src_layout) = if mip == 0 {
                (depth_view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            } else {
                (self.mip_views[(mip - 1) as usize], vk::ImageLayout::GENERAL)
            };

            let src_info = vk::DescriptorImageInfo::default().image_view(src_view).image_layout(src_layout).sampler(depth_sampler);
            let dst_info = vk::DescriptorImageInfo::default().image_view(self.mip_views[mip as usize]).image_layout(vk::ImageLayout::GENERAL);
            let param_info = vk::DescriptorBufferInfo::default().buffer(self.param_buffers[mip as usize].handle).offset(0).range(vk::WHOLE_SIZE);

            let writes = [
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[mip as usize]).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(std::slice::from_ref(&src_info)),
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[mip as usize]).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(std::slice::from_ref(&dst_info)),
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[mip as usize]).dst_binding(2).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(std::slice::from_ref(&param_info)),
            ];
            unsafe { vulkan.device.update_descriptor_sets(&writes, &[]); }
            unsafe {
                vulkan.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, std::slice::from_ref(&self.descriptor_sets[mip as usize]), &[]);
                vulkan.device.cmd_dispatch(cmd, (dst_w + 7) / 8, (dst_h + 7) / 8, 1);
            }

            if mip + 1 < self.mip_count {
                let mb = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL).new_layout(vk::ImageLayout::GENERAL)
                    .image(self.image)
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: mip, level_count: 1, base_array_layer: 0, layer_count: 1 });
                unsafe { vulkan.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&mb)); }
            }
            src_w = dst_w;
            src_h = dst_h;
        }

        // Make HZB readable for cull.comp
        let to_read = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL).new_layout(vk::ImageLayout::GENERAL)
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: self.mip_count, base_array_layer: 0, layer_count: 1 });
        unsafe { vulkan.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&to_read)); }
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan.device.destroy_pipeline(self.pipeline, None);
            vulkan.device.destroy_pipeline_layout(self.pipeline_layout, None);
            vulkan.device.destroy_descriptor_pool(self.descriptor_pool, None);
            vulkan.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            vulkan.device.destroy_sampler(self.sampler, None);
            vulkan.device.destroy_image_view(self.full_view, None);
            for v in &self.mip_views { vulkan.device.destroy_image_view(*v, None); }
            for b in &mut self.param_buffers { b.shutdown(vulkan); }
            vulkan.device.destroy_image(self.image, None);
            vulkan.device.free_memory(self.image_memory, None);
        }
    }
}
