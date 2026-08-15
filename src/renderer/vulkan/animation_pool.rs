use crate::renderer::vulkan::buffer::Buffer;
use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

pub struct AnimationPool {
    pub skeleton_buffer: Buffer,
    pub clip_buffer: Buffer,
    pub keyframe_buffer: Buffer,

    pub staging_skeleton: Buffer,
    pub staging_clip: Buffer,
    pub staging_keyframe: Buffer,

    pub current_skeleton_count: u32,
    pub current_clip_count: u32,
    pub current_keyframe_count: u32,

    pub max_skeletons: u32,
    pub max_clips: u32,
    pub max_keyframes: u32,
}

impl AnimationPool {
    pub fn new(
        vulkan: &VulkanDevice,
        max_skeletons: u32,
        max_clips: u32,
        max_keyframes: u32,
    ) -> Option<Self> {
        let skeleton_size = (max_skeletons as usize * std::mem::size_of::<crate::renderer::vulkan::animation::GpuSkeleton>()) as u64;
        let clip_size = (max_clips as usize * std::mem::size_of::<crate::renderer::vulkan::animation::GpuClip>()) as u64;
        let keyframe_size = (max_keyframes as usize * std::mem::size_of::<crate::renderer::vulkan::animation::GpuKeyframe>()) as u64;

        let skeleton_buffer = Buffer::new(
            vulkan,
            skeleton_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let clip_buffer = Buffer::new(
            vulkan,
            clip_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let keyframe_buffer = Buffer::new(
            vulkan,
            keyframe_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let staging_skeleton = Buffer::new(
            vulkan,
            skeleton_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let staging_clip = Buffer::new(
            vulkan,
            clip_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let staging_keyframe = Buffer::new(
            vulkan,
            keyframe_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        Some(Self {
            skeleton_buffer,
            clip_buffer,
            keyframe_buffer,
            staging_skeleton,
            staging_clip,
            staging_keyframe,
            current_skeleton_count: 0,
            current_clip_count: 0,
            current_keyframe_count: 0,
            max_skeletons,
            max_clips,
            max_keyframes,
        })
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        self.skeleton_buffer.shutdown(vulkan);
        self.clip_buffer.shutdown(vulkan);
        self.keyframe_buffer.shutdown(vulkan);
        self.staging_skeleton.shutdown(vulkan);
        self.staging_clip.shutdown(vulkan);
        self.staging_keyframe.shutdown(vulkan);
    }

    pub fn append_skeleton(
        &mut self,
        vulkan: &VulkanDevice,
        gpu_skeleton: &crate::renderer::vulkan::animation::GpuSkeleton,
    ) -> Option<u32> {
        if self.current_skeleton_count >= self.max_skeletons {
            crate::log_info!("AnimationPool out of skeleton space!");
            return None;
        }

        let skeleton_index = self.current_skeleton_count;
        let byte_size = std::mem::size_of::<crate::renderer::vulkan::animation::GpuSkeleton>() as u64;

        unsafe {
            let ptr = vulkan
                .device
                .map_memory(self.staging_skeleton.memory, 0, byte_size, vk::MemoryMapFlags::empty())
                .unwrap() as *mut crate::renderer::vulkan::animation::GpuSkeleton;
            
            ptr.write(*gpu_skeleton);
            vulkan.device.unmap_memory(self.staging_skeleton.memory);
        }

        if let Some(cmd) = vulkan.begin_single_time_commands() {
            let copy_region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(skeleton_index as u64 * byte_size)
                .size(byte_size);
            unsafe {
                vulkan.device.cmd_copy_buffer(
                    cmd,
                    self.staging_skeleton.handle,
                    self.skeleton_buffer.handle,
                    std::slice::from_ref(&copy_region),
                );
            }
            vulkan.end_single_time_commands(cmd);
        }

        self.current_skeleton_count += 1;
        Some(skeleton_index)
    }

    pub fn append_keyframes(
        &mut self,
        vulkan: &VulkanDevice,
        keyframes: &[crate::renderer::vulkan::animation::GpuKeyframe],
    ) -> Option<u32> {
        let count = keyframes.len() as u32;
        if count == 0 {
            return Some(self.current_keyframe_count);
        }

        if self.current_keyframe_count + count > self.max_keyframes {
            crate::log_info!("AnimationPool out of keyframe space!");
            return None;
        }

        let start_index = self.current_keyframe_count;
        let byte_size = (count as usize * std::mem::size_of::<crate::renderer::vulkan::animation::GpuKeyframe>()) as u64;

        unsafe {
            let ptr = vulkan
                .device
                .map_memory(self.staging_keyframe.memory, 0, byte_size, vk::MemoryMapFlags::empty())
                .unwrap() as *mut crate::renderer::vulkan::animation::GpuKeyframe;
            
            std::ptr::copy_nonoverlapping(keyframes.as_ptr(), ptr, keyframes.len());
            vulkan.device.unmap_memory(self.staging_keyframe.memory);
        }

        if let Some(cmd) = vulkan.begin_single_time_commands() {
            let copy_region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(start_index as u64 * std::mem::size_of::<crate::renderer::vulkan::animation::GpuKeyframe>() as u64)
                .size(byte_size);
            unsafe {
                vulkan.device.cmd_copy_buffer(
                    cmd,
                    self.staging_keyframe.handle,
                    self.keyframe_buffer.handle,
                    std::slice::from_ref(&copy_region),
                );
            }
            vulkan.end_single_time_commands(cmd);
        }

        self.current_keyframe_count += count;
        Some(start_index)
    }

    pub fn append_clip(
        &mut self,
        vulkan: &VulkanDevice,
        gpu_clip: &crate::renderer::vulkan::animation::GpuClip,
    ) -> Option<u32> {
        if self.current_clip_count >= self.max_clips {
            crate::log_info!("AnimationPool out of clip space!");
            return None;
        }

        let clip_index = self.current_clip_count;
        let byte_size = std::mem::size_of::<crate::renderer::vulkan::animation::GpuClip>() as u64;

        unsafe {
            let ptr = vulkan
                .device
                .map_memory(self.staging_clip.memory, 0, byte_size, vk::MemoryMapFlags::empty())
                .unwrap() as *mut crate::renderer::vulkan::animation::GpuClip;
            
            ptr.write(*gpu_clip);
            vulkan.device.unmap_memory(self.staging_clip.memory);
        }

        if let Some(cmd) = vulkan.begin_single_time_commands() {
            let copy_region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(clip_index as u64 * byte_size)
                .size(byte_size);
            unsafe {
                vulkan.device.cmd_copy_buffer(
                    cmd,
                    self.staging_clip.handle,
                    self.clip_buffer.handle,
                    std::slice::from_ref(&copy_region),
                );
            }
            vulkan.end_single_time_commands(cmd);
        }

        self.current_clip_count += 1;
        Some(clip_index)
    }
}
