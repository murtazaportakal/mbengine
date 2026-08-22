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
    pub copy_pipeline: vk::Pipeline,
    pub depth_staging_buffer: crate::renderer::vulkan::buffer::Buffer,
    pub depth_copy_image: vk::Image,
    pub depth_copy_memory: vk::DeviceMemory,
    pub depth_copy_view: vk::ImageView,
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
        depth_sampler: vk::Sampler,
    ) -> Option<Self> {
        let hzb_w = (width / 2).max(1);
        let hzb_h = (height / 2).max(1);
        let mip_count = {
            let mut m = 1u32;
            let mut w = hzb_w;
            let mut h = hzb_h;
            while (w > 1 || h > 1) && m < HZB_MAX_MIPS as u32 {
                w = (w / 2).max(1);
                h = (h / 2).max(1);
                m += 1;
            }
            m
        };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D { width: hzb_w, height: hzb_h, depth: 1 })
            .mip_levels(mip_count)
            .array_layers(1)
            .format(vk::Format::R32_SFLOAT)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
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
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let descriptor_set_layout = unsafe { vulkan.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).ok()? };
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&descriptor_set_layout));
        let pipeline_layout = unsafe { vulkan.device.create_pipeline_layout(&layout_info, None).ok()? };

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(shader_module).name(c"main"))
            .layout(pipeline_layout);
        let pipeline = unsafe { vulkan.device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None).map_err(|e| e.1).ok()?[0] };
        unsafe { vulkan.device.destroy_shader_module(shader_module, None); }

        // Create Graphics Pipeline for Depth Copy
        let vert_code = std::fs::read(vfs.resolve_path("shaders/post_process_vert.spv")).ok()?;
        let frag_code = std::fs::read(vfs.resolve_path("shaders/copy_depth.spv")).ok()?;
        let vert_module = crate::renderer::vulkan::pipeline::Pipeline::create_shader_module(vulkan, &vert_code)?;
        let frag_module = crate::renderer::vulkan::pipeline::Pipeline::create_shader_module(vulkan, &frag_code)?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert_module).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag_module).name(c"main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default().polygon_mode(vk::PolygonMode::FILL).cull_mode(vk::CullModeFlags::NONE).front_face(vk::FrontFace::COUNTER_CLOCKWISE).line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::R).blend_enable(false);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(std::slice::from_ref(&color_blend_attachment));
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&[vk::Format::R32_SFLOAT]);
        let graphics_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .push_next(&mut rendering_info);

        let copy_pipeline = unsafe { vulkan.device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&graphics_info), None).map_err(|e| e.1).ok()?[0] };
        unsafe { 
            vulkan.device.destroy_shader_module(vert_module, None);
            vulkan.device.destroy_shader_module(frag_module, None);
        }

        let pool_sizes = [
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(mip_count * 2),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(mip_count),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(mip_count),
        ];
        let descriptor_pool = unsafe { vulkan.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(mip_count), None).ok()? };
        let layouts: Vec<_> = (0..mip_count).map(|_| descriptor_set_layout).collect();
        let alloc_info = vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts);
        let descriptor_sets = unsafe { vulkan.device.allocate_descriptor_sets(&alloc_info).unwrap() };

        // Create Depth Copy Image (1920x1080 R32_SFLOAT)
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R32_SFLOAT)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let depth_copy_image = unsafe { vulkan.device.create_image(&image_info, None).unwrap() };
        let mem_reqs2 = unsafe { vulkan.device.get_image_memory_requirements(depth_copy_image) };
        let mem_type2 = vulkan.find_memory_type(mem_reqs2.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL).unwrap();
        let alloc_info2 = vk::MemoryAllocateInfo::default().allocation_size(mem_reqs2.size).memory_type_index(mem_type2);
        let depth_copy_memory = unsafe { vulkan.device.allocate_memory(&alloc_info2, None).unwrap() };
        unsafe { vulkan.device.bind_image_memory(depth_copy_image, depth_copy_memory, 0).unwrap(); }

        let depth_copy_view_info = vk::ImageViewCreateInfo::default()
            .image(depth_copy_image).view_type(vk::ImageViewType::TYPE_2D).format(vk::Format::R32_SFLOAT)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
        let depth_copy_view = unsafe { vulkan.device.create_image_view(&depth_copy_view_info, None).unwrap() };

        let mut param_buffers = Vec::with_capacity(mip_count as usize);
        let mut param_mapped = Vec::with_capacity(mip_count as usize);
        for mip in 0..mip_count {
            let buf = Buffer::new(vulkan, std::mem::size_of::<HzbParams>() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).expect("hzb param buf");
            let ptr = unsafe { vulkan.device.map_memory(buf.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).expect("hzb param map") as *mut u8 };
            param_mapped.push(ptr);

            let (src_view, src_layout) = if mip == 0 {
                (depth_copy_view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            } else {
                (mip_views[(mip - 1) as usize], vk::ImageLayout::GENERAL)
            };

            let src_info = vk::DescriptorImageInfo::default().image_view(src_view).image_layout(src_layout).sampler(depth_sampler);
            let dst_info = vk::DescriptorImageInfo::default().image_view(mip_views[mip as usize]).image_layout(vk::ImageLayout::GENERAL);
            let param_info = vk::DescriptorBufferInfo::default().buffer(buf.handle).offset(0).range(vk::WHOLE_SIZE);

            let writes = [
                vk::WriteDescriptorSet::default().dst_set(descriptor_sets[mip as usize]).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(std::slice::from_ref(&src_info)),
                vk::WriteDescriptorSet::default().dst_set(descriptor_sets[mip as usize]).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(std::slice::from_ref(&dst_info)),
                vk::WriteDescriptorSet::default().dst_set(descriptor_sets[mip as usize]).dst_binding(2).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(std::slice::from_ref(&param_info)),
            ];
            unsafe { vulkan.device.update_descriptor_sets(&writes, &[]); }

            param_buffers.push(buf);
        }

        // Staging buffer for D32 -> R32 copy
        let depth_staging_buffer = crate::renderer::vulkan::buffer::Buffer::new(
            vulkan,
            (width * height * 16) as u64, // (1920x1080 * 4 bytes)
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        Some(Self { image, image_memory, mip_views, full_view, sampler, mip_count, base_width: width, base_height: height, pipeline, copy_pipeline, depth_staging_buffer, depth_copy_image, depth_copy_memory, depth_copy_view, pipeline_layout, descriptor_set_layout, descriptor_sets, descriptor_pool, param_buffers, param_mapped })
    }

    pub fn initial_transition(&self, vulkan: &VulkanDevice, cmd: vk::CommandBuffer) {
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::NONE).dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED).new_layout(vk::ImageLayout::GENERAL)
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: self.mip_count, base_array_layer: 0, layer_count: 1 });
        let barrier2 = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::NONE).dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::UNDEFINED).new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(self.depth_copy_image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
        unsafe { vulkan.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &[barrier, barrier2]); }
    }

    pub fn generate(&self, vulkan: &VulkanDevice, cmd: vk::CommandBuffer, depth_image: vk::Image, depth_w: u32, depth_h: u32) {
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ).dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::GENERAL).new_layout(vk::ImageLayout::GENERAL)
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: self.mip_count, base_array_layer: 0, layer_count: 1 });
        unsafe { vulkan.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&barrier)); }
        
        unsafe { vulkan.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline); }

        let mut src_w = depth_w;
        let mut src_h = depth_h;

        for mip in 0..self.mip_count {
            let dst_w = (src_w / 2).max(1);
            let dst_h = (src_h / 2).max(1);

            let params = HzbParams { src_width: src_w, src_height: src_h, dst_width: dst_w, dst_height: dst_h };
            unsafe { std::ptr::copy_nonoverlapping(&params as *const HzbParams as *const u8, self.param_mapped[mip as usize], std::mem::size_of::<HzbParams>()); }

            if mip == 0 {
                // Use Buffer copy to bypass D32_SFLOAT to R32_SFLOAT validation constraints
                
                // 1. Copy Depth Image to Staging Buffer
                let buffer_copy = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(src_w)
                    .buffer_image_height(src_h)
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::DEPTH, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D { width: src_w, height: src_h, depth: 1 });
                unsafe { vulkan.device.cmd_copy_image_to_buffer(cmd, depth_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, self.depth_staging_buffer.handle, std::slice::from_ref(&buffer_copy)); }

                // 2. Barrier Buffer TRANSFER_WRITE to TRANSFER_READ
                let buffer_barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .buffer(self.depth_staging_buffer.handle)
                    .size(vk::WHOLE_SIZE);
                unsafe { vulkan.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], std::slice::from_ref(&buffer_barrier), &[]); }

                // 3. Barrier depth_copy_image to TRANSFER_DST_OPTIMAL
                let copy_barrier_dst = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_READ).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(self.depth_copy_image)
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
                unsafe { vulkan.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&copy_barrier_dst)); }

                // 4. Copy Staging Buffer to depth_copy_image (which is R32_SFLOAT, 1920x1080)
                let buffer_to_image = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(src_w)
                    .buffer_image_height(src_h)
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D { width: src_w, height: src_h, depth: 1 });
                unsafe { vulkan.device.cmd_copy_buffer_to_image(cmd, self.depth_staging_buffer.handle, self.depth_copy_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, std::slice::from_ref(&buffer_to_image)); }
                
                // 5. Barrier depth_copy_image to SHADER_READ_ONLY_OPTIMAL
                let copy_barrier_read = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL).new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(self.depth_copy_image)
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
                unsafe { vulkan.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&copy_barrier_read)); }
                
                // 6. Use Compute Pipeline to downsample depth_copy_image -> mip_views[0]
                unsafe {
                    vulkan.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
                    vulkan.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, std::slice::from_ref(&self.descriptor_sets[0]), &[]);
                    vulkan.device.cmd_dispatch(cmd, (dst_w + 7) / 8, (dst_h + 7) / 8, 1);
                }

                // 7. Barrier for next mip
                let mip_barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL).new_layout(vk::ImageLayout::GENERAL)
                    .image(self.image)
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
                unsafe { vulkan.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&mip_barrier)); }
            } else {
                unsafe {
                    vulkan.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
                    vulkan.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, std::slice::from_ref(&self.descriptor_sets[mip as usize]), &[]);
                    vulkan.device.cmd_dispatch(cmd, (dst_w + 7) / 8, (dst_h + 7) / 8, 1);
                    
                    let mip_barrier = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(self.image)
                        .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: mip, level_count: 1, base_array_layer: 0, layer_count: 1 });
                    vulkan.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&mip_barrier));
                }
            }

            src_w = dst_w;
            src_h = dst_h;
        }

        let full_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL).new_layout(vk::ImageLayout::GENERAL)
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: self.mip_count, base_array_layer: 0, layer_count: 1 });
        unsafe { vulkan.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&full_barrier)); }
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan.device.destroy_image_view(self.depth_copy_view, None);
            vulkan.device.destroy_image(self.depth_copy_image, None);
            vulkan.device.free_memory(self.depth_copy_memory, None);
            self.depth_staging_buffer.shutdown(vulkan);
            vulkan.device.destroy_pipeline(self.pipeline, None);
            vulkan.device.destroy_pipeline(self.copy_pipeline, None);
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
