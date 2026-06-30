//! Swapchain creation and presentation.

use crate::platform::Window;
use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

pub struct Swapchain {
    pub surface_loader: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
}

impl Swapchain {
    pub fn new(vulkan: &VulkanDevice, window: &Window, width: u32, height: u32) -> Option<Self> {
        let win32_surface_loader =
            ash::khr::win32_surface::Instance::new(&vulkan.entry, &vulkan.instance);

        let create_info = vk::Win32SurfaceCreateInfoKHR::default()
            .hinstance(window.hinstance() as isize)
            .hwnd(window.hwnd() as isize);

        let surface = unsafe {
            win32_surface_loader
                .create_win32_surface(&create_info, None)
                .ok()?
        };
        let surface_loader = ash::khr::surface::Instance::new(&vulkan.entry, &vulkan.instance);

        // Check if graphics queue can present to this surface
        let present_support = unsafe {
            surface_loader
                .get_physical_device_surface_support(
                    vulkan.physical_device,
                    vulkan.graphics_queue_family_index,
                    surface,
                )
                .unwrap_or(false)
        };

        if !present_support {
            return None; // Cannot present to this window
        }

        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(vulkan.physical_device, surface)
                .unwrap_or_default()
        };

        let format = formats
            .into_iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_SRGB,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            });

        let caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(vulkan.physical_device, surface)
                .unwrap_or_default()
        };

        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 && image_count > caps.max_image_count {
            image_count = caps.max_image_count;
        }

        let mut actual_extent = vk::Extent2D { width, height };
        if caps.current_extent.width != u32::MAX {
            actual_extent = caps.current_extent;
        } else {
            actual_extent.width = actual_extent
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width);
            actual_extent.height = actual_extent
                .height
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height);
        }
        let extent = actual_extent;

        let swapchain_loader = ash::khr::swapchain::Device::new(&vulkan.instance, &vulkan.device);
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_color_space(format.color_space)
            .image_format(format.format)
            .image_extent(extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .image_array_layers(1);

        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .ok()?
        };

        let images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .unwrap_or_default()
        };

        let mut image_views = Vec::with_capacity(images.len());
        for &img in &images {
            let view_info = vk::ImageViewCreateInfo::default()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(img);

            let view = unsafe { vulkan.device.create_image_view(&view_info, None).ok()? };
            image_views.push(view);
        }

        Some(Self {
            surface_loader,
            surface,
            swapchain_loader,
            swapchain,
            format,
            extent,
            images,
            image_views,
        })
    }

    pub fn cleanup_swapchain(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            for &view in &self.image_views {
                vulkan.device.destroy_image_view(view, None);
            }
            self.image_views.clear();
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }

    pub fn recreate(&mut self, vulkan: &VulkanDevice, width: u32, height: u32) -> bool {
        self.cleanup_swapchain(vulkan);

        let caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(vulkan.physical_device, self.surface)
                .unwrap_or_default()
        };

        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 && image_count > caps.max_image_count {
            image_count = caps.max_image_count;
        }

        let mut final_extent = vk::Extent2D { width, height };
        if caps.current_extent.width != u32::MAX {
            final_extent = caps.current_extent;
        } else {
            final_extent.width = final_extent
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width);
            final_extent.height = final_extent
                .height
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height);
        }
        self.extent = final_extent;

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_color_space(self.format.color_space)
            .image_format(self.format.format)
            .image_extent(self.extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .image_array_layers(1);

        self.swapchain = unsafe {
            match self
                .swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
            {
                Ok(s) => s,
                Err(_) => return false,
            }
        };

        self.images = unsafe {
            self.swapchain_loader
                .get_swapchain_images(self.swapchain)
                .unwrap_or_default()
        };

        for &img in &self.images {
            let view_info = vk::ImageViewCreateInfo::default()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(self.format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(img);

            if let Ok(view) = unsafe { vulkan.device.create_image_view(&view_info, None) } {
                self.image_views.push(view);
            } else {
                return false;
            }
        }

        true
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        self.cleanup_swapchain(vulkan);
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
