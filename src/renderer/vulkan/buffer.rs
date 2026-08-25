//! Vulkan Memory Buffer Abstraction.

use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

pub struct Buffer {
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
}

impl Buffer {
    /// Create a new raw buffer and bind memory to it.
    pub fn new(
        vulkan: &VulkanDevice,
        size: u64,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = unsafe { vulkan.device.create_buffer(&buffer_info, None).ok()? };

        let mem_requirements = unsafe { vulkan.device.get_buffer_memory_requirements(handle) };

        let memory_type_index =
            vulkan.find_memory_type(mem_requirements.memory_type_bits, properties)?;

        let needs_device_address = usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let mut flags_info = vk::MemoryAllocateFlagsInfo::default()
            .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);

        let mut alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        if needs_device_address {
            alloc_info = alloc_info.push_next(&mut flags_info);
        }

        let memory = unsafe { vulkan.device.allocate_memory(&alloc_info, None).ok()? };

        unsafe {
            vulkan.device.bind_buffer_memory(handle, memory, 0).ok()?;
        }

        Some(Self {
            handle,
            memory,
            size,
        })
    }

    /// Upload CPU data directly into HOST_VISIBLE buffer memory.
    pub fn upload<T: Copy>(&self, vulkan: &VulkanDevice, data: &[T]) {
        let data_size = std::mem::size_of_val(data) as u64;
        assert!(data_size <= self.size);

        unsafe {
            let data_ptr = vulkan
                .device
                .map_memory(self.memory, 0, data_size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align =
                ash::util::Align::new(data_ptr, std::mem::align_of::<T>() as u64, data_size);
            align.copy_from_slice(data);
            vulkan.device.unmap_memory(self.memory);
        }
    }

    /// Create a Device Local buffer and stage CPU data into it.
    pub fn new_device_local<T: Copy>(
        vulkan: &VulkanDevice,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Option<Self> {
        let buffer_size = std::mem::size_of_val(data) as u64;

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
            vulkan.device.cmd_copy_buffer(
                cmd,
                staging_buffer.handle,
                device_buffer.handle,
                std::slice::from_ref(&copy_region),
            );
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

    /// Create a DEVICE_LOCAL buffer from a raw byte slice, using a Vulkan 1.3
    /// Synchronization2 pipeline barrier to make it shader-readable.
    ///
    /// This is the zero-copy upload path: the caller obtains `bytes` from
    /// any source (mmap, VFS read, etc.) and this function handles the
    /// staging → device-local transfer with proper synchronization.
    ///
    /// # Safety
    ///
    /// Uses `copy_nonoverlapping` to transfer from the input slice to
    /// Vulkan-mapped memory.  Safe because:
    /// - The staging buffer is freshly allocated with the exact byte count.
    /// - Source and destination are non-overlapping virtual address ranges.
    pub fn new_device_local_from_bytes(
        vulkan: &VulkanDevice,
        bytes: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Option<Self> {
        let byte_count = bytes.len() as u64;
        if byte_count == 0 {
            return None;
        }

        // 1. Create staging buffer
        let staging = Self::new(
            vulkan,
            byte_count,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // 2. Map staging and blast raw bytes in
        unsafe {
            let dst = vulkan
                .device
                .map_memory(staging.memory, 0, byte_count, vk::MemoryMapFlags::empty())
                .ok()?;

            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst as *mut u8, bytes.len());

            vulkan.device.unmap_memory(staging.memory);
        }

        // 3. Create device-local target
        let device_buffer = Self::new(
            vulkan,
            byte_count,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // 4. Record transfer + Sync2 barrier
        let cmd = vulkan.begin_single_time_commands()?;
        unsafe {
            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: byte_count,
            };
            vulkan.device.cmd_copy_buffer(
                cmd,
                staging.handle,
                device_buffer.handle,
                std::slice::from_ref(&copy_region),
            );

            // Vulkan 1.3 Synchronization2: COPY → VERTEX_SHADER | COMPUTE_SHADER
            let buffer_barrier = vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::VERTEX_SHADER
                        | vk::PipelineStageFlags2::COMPUTE_SHADER,
                )
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(device_buffer.handle)
                .offset(0)
                .size(vk::WHOLE_SIZE);

            let dependency_info = vk::DependencyInfo::default()
                .buffer_memory_barriers(std::slice::from_ref(&buffer_barrier));

            vulkan.device.cmd_pipeline_barrier2(cmd, &dependency_info);
        }
        vulkan.end_single_time_commands(cmd);

        // 5. Cleanup staging
        let mut staging = staging;
        staging.shutdown(vulkan);

        Some(device_buffer)
    }

    /// Retrieve the 64-bit `VkDeviceAddress` for this buffer (BDA).
    ///
    /// # Precondition
    ///
    /// The buffer must have been created with `SHADER_DEVICE_ADDRESS` usage.
    pub fn device_address(&self, vulkan: &VulkanDevice) -> vk::DeviceAddress {
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.handle);
        unsafe { vulkan.device.get_buffer_device_address(&info) }
    }
}
