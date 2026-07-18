//! Vulkan logical device and instance initialization.

use ash::vk;

pub struct VulkanDevice {
    pub entry: std::mem::ManuallyDrop<ash::Entry>,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub graphics_queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub transfer_command_pool: vk::CommandPool,
    pub frame_command_buffers: [vk::CommandBuffer; 2],

    /// Shared persistent staging buffer for texture/model uploads.
    /// Created with raw Vulkan API to avoid circular dependency with
    /// Buffer::new which requires a VulkanDevice reference.  Allocated
    /// once, reused via map/unmap, destroyed at shutdown.
    pub staging_buffer: vk::Buffer,
    pub staging_memory: vk::DeviceMemory,

    // Sync primitives for a single frame
    pub image_available_semaphores: [vk::Semaphore; 2],
    pub render_finished_semaphores: [vk::Semaphore; 8],
    pub in_flight_fences: [vk::Fence; 2],

    // Debugging
    pub debug_utils_loader: Option<ash::ext::debug_utils::Instance>,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl VulkanDevice {
    /// Attempt to initialize the Vulkan backend.
    /// Returns None if no suitable GPU or driver is found.
    pub fn new() -> Option<Self> {
        let entry = std::mem::ManuallyDrop::new(unsafe { ash::Entry::load().ok()? });

        let app_name = c"MBEngine";
        let engine_name = c"MBEngine Core";

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(engine_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let mut extension_names = vec![
            ash::khr::surface::NAME.as_ptr(),
            ash::khr::win32_surface::NAME.as_ptr(),
        ];

        // Add debug utils
        extension_names.push(ash::ext::debug_utils::NAME.as_ptr());

        // Disable validation layers to rule out debug overlay text
        // artifacts ("keyboard symbols" flickering).
        let layer_names_raw: Vec<*const std::ffi::c_char> = Vec::new();

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names_raw);

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Validation layers might not be installed")
        };

        // Debug utils callback
        unsafe extern "system" fn vulkan_debug_callback(
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
            _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
            p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
            _user_data: *mut std::os::raw::c_void,
        ) -> vk::Bool32 {
            let callback_data = *p_callback_data;
            let message_id_name = if callback_data.p_message_id_name.is_null() {
                std::borrow::Cow::from("")
            } else {
                std::ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
            };
            let message = if callback_data.p_message.is_null() {
                std::borrow::Cow::from("")
            } else {
                std::ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
            };
            println!(
                "[Validation] {:?} [{}] : {}",
                message_severity, message_id_name, message
            );
            vk::FALSE
        }

        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
        let debug_messenger = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap_or_else(|_| vk::DebugUtilsMessengerEXT::null())
        };

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

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let device_extension_names = vec![
            ash::khr::swapchain::NAME.as_ptr(),
            ash::khr::dynamic_rendering::NAME.as_ptr(),
        ];

        let mut dynamic_rendering =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

        // Standard features to enable
        let features = vk::PhysicalDeviceFeatures::default()
            .sampler_anisotropy(true)
            .multi_draw_indirect(true)
            .draw_indirect_first_instance(true);

        let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
            .draw_indirect_count(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_sampled_image_update_after_bind(true);

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(features)
            .push_next(&mut dynamic_rendering)
            .push_next(&mut vulkan12_features);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names)
            .push_next(&mut features2);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .ok()?
        };

        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);

        let command_pool = unsafe { device.create_command_pool(&pool_create_info, None).ok()? };

        // Dedicated transfer command pool for single-time uploads (staging,
        // texture copies, etc.).  Using a separate pool with the TRANSIENT
        // flag avoids interference with the per-frame command buffers and
        // eliminates VUID-vkDestroyBuffer-buffer-00922 validation errors.
        let transfer_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family_index);
        let transfer_command_pool =
            unsafe { device.create_command_pool(&transfer_pool_info, None).ok()? };

        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(command_pool)
            .command_buffer_count(2);
        let allocated_command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_alloc_info)
                .ok()?
        };
        let frame_command_buffers = [allocated_command_buffers[0], allocated_command_buffers[1]];

        // Persistent staging buffer for texture uploads — raw API to avoid
        // circular dependency with Buffer::new(VulkanDevice).
        let staging_size: u64 = 256 * 1024 * 1024; // 256 MB
        let staging_buffer = unsafe {
            device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(staging_size)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .ok()?
        };
        let staging_mem_reqs = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let staging_mem_type = (0..memory_properties.memory_type_count)
            .find(|&i| {
                (staging_mem_reqs.memory_type_bits & (1 << i)) != 0
                    && memory_properties.memory_types[i as usize]
                        .property_flags
                        .contains(
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )
            })?;
        let staging_memory = unsafe {
            device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(staging_mem_reqs.size)
                        .memory_type_index(staging_mem_type),
                    None,
                )
                .ok()?
        };
        unsafe {
            device
                .bind_buffer_memory(staging_buffer, staging_memory, 0)
                .ok()?;
        }

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores = [vk::Semaphore::null(); 2];
        let mut render_finished_semaphores = [vk::Semaphore::null(); 8];
        let mut in_flight_fences = [vk::Fence::null(); 2];

        for i in 0..2 {
            image_available_semaphores[i] =
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() };
            in_flight_fences[i] = unsafe { device.create_fence(&fence_info, None).unwrap() };
        }
        for sem in &mut render_finished_semaphores {
            *sem = unsafe { device.create_semaphore(&semaphore_info, None).unwrap() };
        }

        Some(Self {
            entry,
            instance,
            physical_device,
            memory_properties,
            device,
            graphics_queue,
            graphics_queue_family_index: queue_family_index,
            command_pool,
            transfer_command_pool,
            frame_command_buffers,
            staging_buffer,
            staging_memory,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            debug_utils_loader: Some(debug_utils_loader),
            debug_messenger,
        })
    }

    /// Find a memory type matching the required properties.
    pub fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        (0..self.memory_properties.memory_type_count).find(|&i| {
            (type_filter & (1 << i)) != 0
                && self.memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
        })
    }

    pub fn begin_single_time_commands(&self) -> Option<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.transfer_command_pool)
            .command_buffer_count(1);

        let command_buffer = unsafe { self.device.allocate_command_buffers(&alloc_info).ok()?[0] };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .ok()?;
        }

        Some(command_buffer)
    }

    pub fn end_single_time_commands(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.end_command_buffer(command_buffer).unwrap();

            // Use an explicit fence instead of queue_wait_idle so that the
            // validation layer can accurately track the command buffer's
            // lifetime relative to the staging buffers it references.
            let fence_info = vk::FenceCreateInfo::default();
            let fence = self.device.create_fence(&fence_info, None).unwrap();

            let command_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

            self.device
                .queue_submit(
                    self.graphics_queue,
                    std::slice::from_ref(&submit_info),
                    fence,
                )
                .unwrap();

            self.device
                .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                .unwrap();
            self.device.destroy_fence(fence, None);
            self.device
                .free_command_buffers(self.transfer_command_pool, &command_buffers);
        }
    }

    /// Destroy the existing command pool and recreate frame command
    /// buffers from a fresh pool.  This is the only approach that
    /// guarantees validation-layer buffer-reference tracking is
    /// completely cleared.
    ///
    /// # Safety
    /// The caller must ensure the GPU is idle (via `device_wait_idle`)
    /// before calling this.
    pub fn recreate_frame_buffers(&mut self) {
        unsafe {
            self.device
                .destroy_command_pool(self.command_pool, None);

            let new_pool = self
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(self.graphics_queue_family_index),
                    None,
                )
                .unwrap();

            let cb_alloc = vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(new_pool)
                .command_buffer_count(2);

            let new_cbs = self.device.allocate_command_buffers(&cb_alloc).unwrap();
            self.command_pool = new_pool;
            self.frame_command_buffers = [new_cbs[0], new_cbs[1]];
        }
    }

    /// Wait for the device to finish all operations.
    pub fn wait_idle(&self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
    }

    /// Shut down and clean up device resources.
    pub fn shutdown(&mut self) {
        unsafe {
            for semaphore in self.render_finished_semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.image_available_semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            for fence in self.in_flight_fences {
                self.device.destroy_fence(fence, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device
                .destroy_command_pool(self.transfer_command_pool, None);
            self.device.destroy_buffer(self.staging_buffer, None);
            self.device.free_memory(self.staging_memory, None);
            self.device.destroy_device(None);

            if let Some(loader) = &self.debug_utils_loader {
                if self.debug_messenger != vk::DebugUtilsMessengerEXT::null() {
                    loader.destroy_debug_utils_messenger(self.debug_messenger, None);
                }
            }

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