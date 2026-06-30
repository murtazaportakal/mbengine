//! Vulkan Memory Buffer Abstraction.

use ash::vk;
use crate::renderer::vulkan::VulkanDevice;

pub struct Buffer {
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
}

impl Buffer {
    /// Create a new raw buffer and bind memory to it.
    pub fn new(vulkan: &VulkanDevice, size: u64, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags) -> Option<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = unsafe { vulkan.device.create_buffer(&buffer_info, None).ok()? };

        let mem_requirements = unsafe { vulkan.device.get_buffer_memory_requirements(handle) };

        let memory_type_index = vulkan.find_memory_type(mem_requirements.memory_type_bits, properties)?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { vulkan.device.allocate_memory(&alloc_info, None).ok()? };

        unsafe {
            vulkan.device.bind_buffer_memory(handle, memory, 0).ok()?;
        }

        Some(Self { handle, memory, size })
    }

    /// Upload CPU data directly into HOST_VISIBLE buffer memory.
    pub fn upload<T: Copy>(&self, vulkan: &VulkanDevice, data: &[T]) {
        let data_size = (data.len() * std::mem::size_of::<T>()) as u64;
        assert!(data_size <= self.size);

        unsafe {
            let data_ptr = vulkan.device.map_memory(self.memory, 0, data_size, vk::MemoryMapFlags::empty()).unwrap();
            let mut align = ash::util::Align::new(data_ptr, std::mem::align_of::<T>() as u64, data_size);
            align.copy_from_slice(data);
            vulkan.device.unmap_memory(self.memory);
        }
    }

    /// Create a Device Local buffer and stage CPU data into it.
    pub fn new_device_local<T: Copy>(vulkan: &VulkanDevice, data: &[T], usage: vk::BufferUsageFlags) -> Option<Self> {
        let buffer_size = (data.len() * std::mem::size_of::<T>()) as u64;

        // 1. Create Staging Buffer
        let staging_buffer = Self::new(
            vulkan,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // 2. Upload to Staging
        staging_buffer.upload(vulkan, data);

        // 3. Create Device Local Buffer
        let device_buffer = Self::new(
            vulkan,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // 4. Copy Staging to Device Local
        let cmd = vulkan.begin_single_time_commands()?;
        let copy_region = vk::BufferCopy::default().size(buffer_size);
        unsafe {
            vulkan.device.cmd_copy_buffer(cmd, staging_buffer.handle, device_buffer.handle, std::slice::from_ref(&copy_region));
        }
        vulkan.end_single_time_commands(cmd);

        // 5. Cleanup Staging
        let mut staging = staging_buffer;
        staging.shutdown(vulkan);

        Some(device_buffer)
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan.device.destroy_buffer(self.handle, None);
            vulkan.device.free_memory(self.memory, None);
        }
    }
}
