use ash::vk;
use crate::renderer::vulkan::VulkanDevice;

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
    
    // An egui descriptor set representing this texture. 
    // We update this via `EguiBackend` when recreating the target.
    pub descriptor_set: vk::DescriptorSet, 
}

impl OffscreenTarget {
    pub fn new(
        vulkan: &VulkanDevice, 
        render_pass: vk::RenderPass, 
        width: u32, 
        height: u32
    ) -> Option<Self> {
        let (color_image, color_memory, color_view) = Self::create_color_resources(vulkan, width, height)?;
        let (depth_image, depth_memory, depth_view) = Self::create_depth_resources(vulkan, width, height)?;

        let attachments = [color_view, depth_view];
        let fb_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);

        let framebuffer = unsafe { vulkan.device.create_framebuffer(&fb_info, None).ok()? };

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
            framebuffer,
            sampler,
            descriptor_set: vk::DescriptorSet::null(),
        })
    }

    fn create_color_resources(vulkan: &VulkanDevice, width: u32, height: u32) -> Option<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::B8G8R8A8_SRGB)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe { vulkan.device.create_image(&image_info, None).ok()? };
        let mem_req = unsafe { vulkan.device.get_image_memory_requirements(image) };
        let memory_type_index = vulkan.find_memory_type(mem_req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { vulkan.device.allocate_memory(&alloc_info, None).ok()? };
        unsafe { vulkan.device.bind_image_memory(image, memory, 0).ok()? };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::B8G8R8A8_SRGB)
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

    fn create_depth_resources(vulkan: &VulkanDevice, width: u32, height: u32) -> Option<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
        let depth_format = vk::Format::D32_SFLOAT;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D { width, height, depth: 1 })
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
        let memory_type_index = vulkan.find_memory_type(mem_req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

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
