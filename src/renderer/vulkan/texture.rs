//! Vulkan Texture and Sampler abstractions.

use ash::vk;
use crate::renderer::vulkan::VulkanDevice;
use crate::renderer::vulkan::buffer::Buffer;

pub struct Texture {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
}

impl Texture {
    pub fn load_from_file(vulkan: &VulkanDevice, path: &str) -> Option<Self> {
        let img = image::open(path).ok()?.into_rgba8();
        let (width, height) = img.dimensions();
        Self::from_rgba8(vulkan, width, height, &img)
    }

    /// Generates a procedural 256x256 checkerboard texture.
    pub fn new_checkerboard(vulkan: &VulkanDevice) -> Option<Self> {
        let width = 256;
        let height = 256;
        let mut pixels = vec![0u8; width * height * 4];

        for y in 0..height {
            for x in 0..width {
                let is_white = ((x / 32) % 2) == ((y / 32) % 2);
                let color = if is_white { 255 } else { 0 };
                let i = (y * width + x) * 4;
                pixels[i] = color;
                pixels[i + 1] = color;
                pixels[i + 2] = color;
                pixels[i + 3] = 255;
            }
        }

        Self::from_rgba8(vulkan, width as u32, height as u32, &pixels)
    }

    pub fn from_rgba8(vulkan: &VulkanDevice, width: u32, height: u32, pixels: &[u8]) -> Option<Self> {
        // 1. Create Staging Buffer
        let buffer_size = pixels.len() as u64;
        let staging_buffer = Buffer::new(
            vulkan,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        staging_buffer.upload(vulkan, pixels);

        // 2. Create Image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D { width: width as u32, height: height as u32, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_SRGB)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe { vulkan.device.create_image(&image_info, None).ok()? };

        let mem_reqs = unsafe { vulkan.device.get_image_memory_requirements(image) };
        let memory_type_index = vulkan.find_memory_type(mem_reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { vulkan.device.allocate_memory(&alloc_info, None).ok()? };
        unsafe { vulkan.device.bind_image_memory(image, memory, 0).ok()? };

        // 3. Transition to TRANSFER_DST and Copy
        let cmd = vulkan.begin_single_time_commands()?;

        Self::transition_image_layout(
            vulkan, cmd, image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D { width: width as u32, height: height as u32, depth: 1 });

        unsafe {
            vulkan.device.cmd_copy_buffer_to_image(
                cmd,
                staging_buffer.handle,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );
        }

        Self::transition_image_layout(
            vulkan, cmd, image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        vulkan.end_single_time_commands(cmd);

        let mut staging = staging_buffer;
        staging.shutdown(vulkan);

        // 4. Create Image View
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let view = unsafe { vulkan.device.create_image_view(&view_info, None).ok()? };

        // 5. Create Sampler
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(false)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR);

        let sampler = unsafe { vulkan.device.create_sampler(&sampler_info, None).ok()? };

        Some(Self { image, memory, view, sampler })
    }

    fn transition_image_layout(
        vulkan: &VulkanDevice,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        _format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let mut barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let source_stage;
        let destination_stage;

        if old_layout == vk::ImageLayout::UNDEFINED && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL {
            barrier.src_access_mask = vk::AccessFlags::empty();
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            destination_stage = vk::PipelineStageFlags::TRANSFER;
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
            source_stage = vk::PipelineStageFlags::TRANSFER;
            destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        } else {
            panic!("Unsupported layout transition!");
        }

        unsafe {
            vulkan.device.cmd_pipeline_barrier(
                cmd,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        }
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan.device.destroy_sampler(self.sampler, None);
            vulkan.device.destroy_image_view(self.view, None);
            vulkan.device.destroy_image(self.image, None);
            vulkan.device.free_memory(self.memory, None);
        }
    }
}
