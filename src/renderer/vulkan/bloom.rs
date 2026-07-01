use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

pub struct BloomTarget {
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,

    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub full_view: vk::ImageView,
    pub mip_views: Vec<vk::ImageView>,

    pub sampler: vk::Sampler,
}

impl BloomTarget {
    pub fn new(vulkan: &VulkanDevice, width: u32, height: u32, mip_levels: u32) -> Option<Self> {
        let format = vk::Format::R16G16B16A16_SFLOAT;
        
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
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe { vulkan.device.create_image(&image_info, None).ok()? };
        let mem_req = unsafe { vulkan.device.get_image_memory_requirements(image) };
        let memory_type_index = vulkan.find_memory_type(
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { vulkan.device.allocate_memory(&alloc_info, None).ok()? };
        unsafe { vulkan.device.bind_image_memory(image, memory, 0).ok()? };

        let full_view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        let full_view = unsafe { vulkan.device.create_image_view(&full_view_info, None).ok()? };

        let mut mip_views = Vec::with_capacity(mip_levels as usize);
        for i in 0..mip_levels {
            let mip_view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: i,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let mip_view = unsafe { vulkan.device.create_image_view(&mip_view_info, None).ok()? };
            mip_views.push(mip_view);
        }

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .mip_lod_bias(0.0)
            .max_anisotropy(1.0)
            .min_lod(0.0)
            .max_lod(mip_levels as f32);

        let sampler = unsafe { vulkan.device.create_sampler(&sampler_info, None).ok()? };

        Some(Self {
            width,
            height,
            mip_levels,
            image,
            memory,
            full_view,
            mip_views,
            sampler,
        })
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan.device.destroy_sampler(self.sampler, None);
            for view in &self.mip_views {
                vulkan.device.destroy_image_view(*view, None);
            }
            vulkan.device.destroy_image_view(self.full_view, None);
            vulkan.device.destroy_image(self.image, None);
            vulkan.device.free_memory(self.memory, None);
        }
    }
}
