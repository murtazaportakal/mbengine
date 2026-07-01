use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

pub struct OffscreenTarget {
    pub width: u32,
    pub height: u32,

    pub color_image: vk::Image,
    pub color_memory: vk::DeviceMemory,
    pub color_view: vk::ImageView,

    pub depth_image: vk::Image,
    pub depth_memory: vk::DeviceMemory,
    pub depth_view: vk::ImageView,

    pub framebuffer: vk::Framebuffer,
    pub sampler: vk::Sampler,

    pub descriptor_set: vk::DescriptorSet,
}

impl OffscreenTarget {
    pub fn new(vulkan: &VulkanDevice, width: u32, height: u32, color_format: vk::Format) -> Option<Self> {
        let (color_image, color_memory, color_view) =
            Self::create_color_resources(vulkan, width, height, color_format)?;
        let (depth_image, depth_memory, depth_view) =
            Self::create_depth_resources(vulkan, width, height)?;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR);

        let sampler = unsafe { vulkan.device.create_sampler(&sampler_info, None).ok()? };

        Some(Self {
            width,
            height,
            color_image,
            color_memory,
            color_view,
            depth_image,
            depth_memory,
            depth_view,
            framebuffer: vk::Framebuffer::null(),
            sampler,
            descriptor_set: vk::DescriptorSet::null(),
        })
    }

    fn create_color_resources(
        vulkan: &VulkanDevice,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Option<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
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

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { vulkan.device.create_image_view(&view_info, None).ok()? };

        Some((image, memory, view))
    }

    fn create_depth_resources(
        vulkan: &VulkanDevice,
        width: u32,
        height: u32,
    ) -> Option<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
        let depth_format = vk::Format::D32_SFLOAT;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(depth_format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
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

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(depth_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { vulkan.device.create_image_view(&view_info, None).ok()? };

        Some((image, memory, view))
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan.device.destroy_sampler(self.sampler, None);
            vulkan.device.destroy_framebuffer(self.framebuffer, None);

            vulkan.device.destroy_image_view(self.color_view, None);
            vulkan.device.destroy_image(self.color_image, None);
            vulkan.device.free_memory(self.color_memory, None);

            vulkan.device.destroy_image_view(self.depth_view, None);
            vulkan.device.destroy_image(self.depth_image, None);
            vulkan.device.free_memory(self.depth_memory, None);
        }
    }
}

pub struct BloomMip {
    pub width: u32,
    pub height: u32,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub descriptor_set: vk::DescriptorSet,
}

pub struct BloomTarget {
    pub mips: Vec<BloomMip>,
    pub sampler: vk::Sampler,
}

impl BloomTarget {
    pub fn new(vulkan: &VulkanDevice, base_width: u32, base_height: u32, num_mips: usize) -> Option<Self> {
        let mut mips = Vec::with_capacity(num_mips);
        
        let mut current_width = base_width / 2;
        let mut current_height = base_height / 2;

        for _ in 0..num_mips {
            if current_width == 0 { current_width = 1; }
            if current_height == 0 { current_height = 1; }

            let format = vk::Format::R16G16B16A16_SFLOAT;
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D {
                    width: current_width,
                    height: current_height,
                    depth: 1,
                })
                .mip_levels(1)
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

            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let view = unsafe { vulkan.device.create_image_view(&view_info, None).ok()? };

            mips.push(BloomMip {
                width: current_width,
                height: current_height,
                image,
                memory,
                view,
                descriptor_set: vk::DescriptorSet::null(),
            });

            current_width /= 2;
            current_height /= 2;
        }

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR);

        let sampler = unsafe { vulkan.device.create_sampler(&sampler_info, None).ok()? };

        Some(Self { mips, sampler })
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan.device.destroy_sampler(self.sampler, None);
            for mip in &self.mips {
                vulkan.device.destroy_image_view(mip.view, None);
                vulkan.device.destroy_image(mip.image, None);
                vulkan.device.free_memory(mip.memory, None);
            }
        }
        self.mips.clear();
    }
}
