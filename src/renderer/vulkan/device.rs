//! Vulkan logical device and instance initialization.

use crate::renderer::RenderDevice;
use ash::vk;


pub struct VulkanDevice {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub graphics_queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    
    // Sync primitives for a single frame
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub in_flight_fence: vk::Fence,
}

impl VulkanDevice {
    /// Attempt to initialize the Vulkan backend.
    /// Returns None if no suitable GPU or driver is found.
    pub fn new() -> Option<Self> {
        let entry = unsafe { ash::Entry::load().ok()? };

        let app_name = c"MBEngine";
        let engine_name = c"MBEngine Core";

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(engine_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 2, 0));

        let extension_names = vec![
            ash::khr::surface::NAME.as_ptr(),
            ash::khr::win32_surface::NAME.as_ptr(),
        ];

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        let instance = unsafe { entry.create_instance(&create_info, None).ok()? };

        let pdevices = unsafe { instance.enumerate_physical_devices().unwrap_or_default() };
        
        let (physical_device, queue_family_index) = pdevices.into_iter().find_map(|pdevice| {
            let props = unsafe { instance.get_physical_device_queue_family_properties(pdevice) };
            
            for (index, info) in props.iter().enumerate() {
                if info.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    return Some((pdevice, index as u32));
                }
            }
            None
        })?;

        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let device_extension_names = vec![
            ash::khr::swapchain::NAME.as_ptr(),
        ];

        let features = vk::PhysicalDeviceFeatures::default();
        
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names)
            .enabled_features(&features);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None).ok()? };
        
        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);
            
        let command_pool = unsafe { device.create_command_pool(&pool_create_info, None).ok()? };

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let image_available_semaphore = unsafe { device.create_semaphore(&semaphore_info, None).ok()? };
        let render_finished_semaphore = unsafe { device.create_semaphore(&semaphore_info, None).ok()? };
        let in_flight_fence = unsafe { device.create_fence(&fence_info, None).ok()? };

        Some(Self {
            entry,
            instance,
            physical_device,
            memory_properties,
            device,
            graphics_queue,
            graphics_queue_family_index: queue_family_index,
            command_pool,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
        })
    }

    /// Find a memory type matching the required properties.
    pub fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> Option<u32> {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && self.memory_properties.memory_types[i as usize].property_flags.contains(properties)
            {
                return Some(i);
            }
        }
        None
    }

    pub fn begin_single_time_commands(&self) -> Option<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool)
            .command_buffer_count(1);

        let command_buffer = unsafe { self.device.allocate_command_buffers(&alloc_info).ok()?[0] };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device.begin_command_buffer(command_buffer, &begin_info).ok()?;
        }

        Some(command_buffer)
    }

    pub fn end_single_time_commands(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.end_command_buffer(command_buffer).unwrap();

            let command_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

            self.device
                .queue_submit(self.graphics_queue, std::slice::from_ref(&submit_info), vk::Fence::null())
                .unwrap();

            self.device.queue_wait_idle(self.graphics_queue).unwrap();
            self.device.free_command_buffers(self.command_pool, &command_buffers);
        }
    }
}

impl RenderDevice for VulkanDevice {
    fn wait_idle(&self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
    }

    fn shutdown(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.render_finished_semaphore, None);
            self.device.destroy_semaphore(self.image_available_semaphore, None);
            self.device.destroy_fence(self.in_flight_fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // Shutdown is expected to be called manually, but if not, 
        // we could call it here. For safety, we assume explicit shutdown.
    }
}
