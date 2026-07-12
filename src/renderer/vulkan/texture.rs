//! Vulkan Texture and Sampler abstractions.

use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

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

    pub fn new_solid_color(vulkan: &VulkanDevice, r: u8, g: u8, b: u8, a: u8) -> Option<Self> {
        let pixels = vec![r, g, b, a];
        Self::from_rgba8(vulkan, 1, 1, &pixels)
    }

    /// Generates a procedural 1024x512 equirectangular studio environment.
    pub fn new_procedural_env(vulkan: &VulkanDevice) -> Option<Self> {
        let width = 1024;
        let height = 512;
        let mut pixels = vec![0u8; width * height * 4];

        for y in 0..height {
            let v = y as f32 / height as f32; // 0.0 top, 1.0 bottom
            for x in 0..width {
                let u = x as f32 / width as f32; // 0.0 left, 1.0 right

                // Base background gradient: mid grey top, dark grey bottom
                let mut r = (128.0 - 64.0 * v) as u8;
                let mut g = (128.0 - 64.0 * v) as u8;
                let mut b = (128.0 - 64.0 * v) as u8;

                // Add studio lights (bright white rectangles)
                // Light 1: Top Left
                if u > 0.1 && u < 0.3 && v > 0.2 && v < 0.4 {
                    r = 255;
                    g = 255;
                    b = 255;
                }
                // Light 2: Top Right
                if u > 0.7 && u < 0.9 && v > 0.2 && v < 0.4 {
                    r = 255;
                    g = 255;
                    b = 255;
                }
                // Light 3: Horizon line glow
                if v > 0.45 && v < 0.5 {
                    r = r.saturating_add(64);
                    g = g.saturating_add(64);
                    b = b.saturating_add(64);
                }

                let i = (y * width + x) * 4;
                pixels[i] = r;
                pixels[i + 1] = g;
                pixels[i + 2] = b;
                pixels[i + 3] = 255;
            }
        }

        Self::from_rgba8(vulkan, width as u32, height as u32, &pixels)
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

    pub fn load_hdr(vulkan: &VulkanDevice, path: &str) -> Option<Self> {
        let img = image::open(path).ok()?.into_rgba32f();
        let (width, height) = img.dimensions();
        let raw_pixels = unsafe {
            std::slice::from_raw_parts(img.as_raw().as_ptr() as *const u8, img.as_raw().len() * 4)
        };
        Self::from_pixels(
            vulkan,
            width,
            height,
            raw_pixels,
            vk::Format::R32G32B32A32_SFLOAT,
        )
    }

    pub fn from_rgba8(
        vulkan: &VulkanDevice,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Option<Self> {
        Self::from_pixels(vulkan, width, height, pixels, vk::Format::R8G8B8A8_SRGB)
    }

    fn from_pixels(
        vulkan: &VulkanDevice,
        width: u32,
        height: u32,
        pixels: &[u8],
        format: vk::Format,
    ) -> Option<Self> {
        let buffer_size = pixels.len() as u64;

        // Upload pixel data into the shared persistent staging buffer
        // (allocated once in VulkanDevice, never destroyed during runtime).
        // This eliminates vkDestroyBuffer validation errors that would
        // occur with transient staging buffers.
        let data_ptr = unsafe {
            vulkan
                .device
                .map_memory(
                    vulkan.staging_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap()
        };
        unsafe {
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), data_ptr as *mut u8, buffer_size as usize);
            vulkan.device.unmap_memory(vulkan.staging_memory);
        }

        // Calculate mip levels
        let mip_levels = (std::cmp::max(width, height) as f32).log2().floor() as u32 + 1;

        // Create Image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe { vulkan.device.create_image(&image_info, None).ok()? };

        let mem_reqs = unsafe { vulkan.device.get_image_memory_requirements(image) };
        let memory_type_index = vulkan.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { vulkan.device.allocate_memory(&alloc_info, None).ok()? };
        unsafe { vulkan.device.bind_image_memory(image, memory, 0).ok()? };

        // Transition to TRANSFER_DST and Copy
        let cmd = vulkan.begin_single_time_commands()?;

        Self::transition_image_layout(
            vulkan,
            cmd,
            image,
            format,
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
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        unsafe {
            vulkan.device.cmd_copy_buffer_to_image(
                cmd,
                vulkan.staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );
        }

        // Generate mipmaps
        let mut mip_width = width as i32;
        let mut mip_height = height as i32;

        for i in 1..mip_levels {
            // Transition source mip (i-1) to TRANSFER_SRC_OPTIMAL
            let barrier = vk::ImageMemoryBarrier::default()
                .image(image)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(i - 1)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

            unsafe {
                vulkan.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&barrier),
                );
            }

            // Transition destination mip (i) from UNDEFINED to TRANSFER_DST_OPTIMAL
            // so the blit can write to it.  This is required because higher mips
            // were never initialised by the initial layout transition.
            let dst_init_barrier = vk::ImageMemoryBarrier::default()
                .image(image)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(i)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::NONE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

            unsafe {
                vulkan.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&dst_init_barrier),
                );
            }

            let blit = vk::ImageBlit::default()
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ])
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(i - 1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: if mip_width > 1 { mip_width / 2 } else { 1 },
                        y: if mip_height > 1 { mip_height / 2 } else { 1 },
                        z: 1,
                    },
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(i)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            unsafe {
                vulkan.device.cmd_blit_image(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    std::slice::from_ref(&blit),
                    vk::Filter::LINEAR,
                );
            }

            let barrier_read = vk::ImageMemoryBarrier::default()
                .image(image)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(i - 1)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);

            unsafe {
                vulkan.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&barrier_read),
                );
            }

            if mip_width > 1 {
                mip_width /= 2;
            }
            if mip_height > 1 {
                mip_height /= 2;
            }
        }

        let barrier_last = vk::ImageMemoryBarrier::default()
            .image(image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(mip_levels - 1)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        unsafe {
            vulkan.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier_last),
            );
        }

        vulkan.end_single_time_commands(cmd);

        // Note: no staging buffer shutdown — the shared staging buffer
        // lives in VulkanDevice and is cleaned up at shutdown.

        // Create Image View
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let view = unsafe { vulkan.device.create_image_view(&view_info, None).ok()? };

        // Create Sampler
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .min_lod(0.0)
            .max_lod(mip_levels as f32)
            .mip_lod_bias(0.0);

        let sampler = unsafe { vulkan.device.create_sampler(&sampler_info, None).ok()? };

        Some(Self {
            image,
            memory,
            view,
            sampler,
        })
    }

    fn transition_image_layout(
        vulkan: &VulkanDevice,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        _format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let (src_access_mask, dst_access_mask, src_stage, dst_stage) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::NONE,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::NONE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_READ,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                vk::AccessFlags::NONE,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ),
            (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::SHADER_READ,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ),
            _ => panic!(
                "Unsupported layout transition: {:?} -> {:?}",
                old_layout, new_layout
            ),
        };

        let barrier = vk::ImageMemoryBarrier::default()
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
            )
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);

        unsafe {
            vulkan.device.cmd_pipeline_barrier(
                cmd,
                src_stage,
                dst_stage,
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