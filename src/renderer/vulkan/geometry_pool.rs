use crate::renderer::vulkan::buffer::Buffer;
use crate::renderer::vulkan::mesh::MeshletData;
use crate::renderer::vulkan::pipeline::Vertex;
use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

/// Maximum upload size for a single mesh (256 MB each for vertex/index data
/// and 64 MB for meshlets).  If a single model exceeds this, it won't fit
/// in the persistent staging buffers — but at that point you'd need to
/// rethink the asset pipeline anyway.
const STAGING_VERTEX_SIZE: u64 = 256 * 1024 * 1024; // 256 MB
const STAGING_INDEX_SIZE: u64 = 256 * 1024 * 1024; // 256 MB
const STAGING_MESHLET_SIZE: u64 = 64 * 1024 * 1024; // 64 MB

pub struct GeometryPool {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub meshlet_buffer: Buffer,

    /// Persistent host-visible staging buffers that live for the pool's
    /// entire lifetime.  We map/unmap them each time we need to upload,
    /// but we never destroy them until shutdown().  This completely
    /// eliminates VUID-vkDestroyBuffer-buffer-00922 validation errors
    /// because no staging buffer is ever destroyed while a frame command
    /// buffer might still reference the pool's device-local buffers.
    pub staging_vertex: Buffer,
    pub staging_index: Buffer,
    pub staging_meshlet: Buffer,

    pub current_vertex_count: u32,
    pub current_index_count: u32,
    pub current_meshlet_count: u32,

    pub max_vertices: u32,
    pub max_indices: u32,
    pub max_meshlets: u32,
}

impl GeometryPool {
    pub fn new(
        vulkan: &VulkanDevice,
        max_vertices: u32,
        max_indices: u32,
        max_meshlets: u32,
    ) -> Option<Self> {
        let vertex_size = (max_vertices as usize * std::mem::size_of::<Vertex>()) as u64;
        let index_size = (max_indices as usize * std::mem::size_of::<u32>()) as u64;
        let meshlet_size = (max_meshlets as usize * std::mem::size_of::<MeshletData>()) as u64;

        let vertex_buffer = Buffer::new(
            vulkan,
            vertex_size,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let index_buffer = Buffer::new(
            vulkan,
            index_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let meshlet_buffer = Buffer::new(
            vulkan,
            meshlet_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Persistent staging buffers — allocated once, never destroyed during
        // operation.  Mapped/unmapped per upload.
        let staging_vertex = Buffer::new(
            vulkan,
            STAGING_VERTEX_SIZE,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let staging_index = Buffer::new(
            vulkan,
            STAGING_INDEX_SIZE,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let staging_meshlet = Buffer::new(
            vulkan,
            STAGING_MESHLET_SIZE,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        Some(Self {
            vertex_buffer,
            index_buffer,
            meshlet_buffer,
            staging_vertex,
            staging_index,
            staging_meshlet,
            current_vertex_count: 0,
            current_index_count: 0,
            current_meshlet_count: 0,
            max_vertices,
            max_indices,
            max_meshlets,
        })
    }

    /// Append a mesh to the geometry pool, populated by a callback.
    ///
    /// The callback `fill_data` is provided with raw pointers to the mapped
    /// staging buffers for vertices, indices, and meshlets respectively.
    /// It must write exactly the expected number of bytes.
    pub fn append_with_callback<F>(
        &mut self,
        vulkan: &VulkanDevice,
        v_count: u32,
        i_count: u32,
        m_count: u32,
        mut fill_data: F,
    ) -> Option<(u32, u32, u32)>
    where
        F: FnMut(*mut u8, *mut u8, *mut u8) -> std::io::Result<()>,
    {
        if self.current_vertex_count + v_count > self.max_vertices
            || self.current_index_count + i_count > self.max_indices
            || self.current_meshlet_count + m_count > self.max_meshlets
        {
            crate::log_info!("GeometryPool out of space!");
            return None;
        }

        let v_byte_size = (v_count as usize * std::mem::size_of::<Vertex>()) as u64;
        let i_byte_size = (i_count as usize * std::mem::size_of::<u32>()) as u64;
        let m_byte_size = (m_count as usize * std::mem::size_of::<MeshletData>()) as u64;

        assert!(
            v_byte_size <= STAGING_VERTEX_SIZE
                && i_byte_size <= STAGING_INDEX_SIZE
                && m_byte_size <= STAGING_MESHLET_SIZE,
            "Mesh exceeds persistent staging buffer capacity"
        );

        let v_offset = self.current_vertex_count;
        let i_offset = self.current_index_count;
        let m_offset = self.current_meshlet_count;

        let mut v_ptr = std::ptr::null_mut();
        let mut i_ptr = std::ptr::null_mut();
        let mut m_ptr = std::ptr::null_mut();

        if v_count > 0 {
            v_ptr = unsafe {
                vulkan
                    .device
                    .map_memory(self.staging_vertex.memory, 0, v_byte_size, vk::MemoryMapFlags::empty())
                    .unwrap()
            } as *mut u8;
        }
        if i_count > 0 {
            i_ptr = unsafe {
                vulkan
                    .device
                    .map_memory(self.staging_index.memory, 0, i_byte_size, vk::MemoryMapFlags::empty())
                    .unwrap()
            } as *mut u8;
        }
        if m_count > 0 {
            m_ptr = unsafe {
                vulkan
                    .device
                    .map_memory(self.staging_meshlet.memory, 0, m_byte_size, vk::MemoryMapFlags::empty())
                    .unwrap()
            } as *mut u8;
        }

        if fill_data(v_ptr, i_ptr, m_ptr).is_err() {
            crate::log_info!("Failed to fill mesh data via callback");
            if v_count > 0 { unsafe { vulkan.device.unmap_memory(self.staging_vertex.memory); } }
            if i_count > 0 { unsafe { vulkan.device.unmap_memory(self.staging_index.memory); } }
            if m_count > 0 { unsafe { vulkan.device.unmap_memory(self.staging_meshlet.memory); } }
            return None;
        }

        if v_count > 0 { unsafe { vulkan.device.unmap_memory(self.staging_vertex.memory); } }
        if i_count > 0 { unsafe { vulkan.device.unmap_memory(self.staging_index.memory); } }
        if m_count > 0 { unsafe { vulkan.device.unmap_memory(self.staging_meshlet.memory); } }

        // --- Single transfer command buffer copies all three ---
        if let Some(cmd) = vulkan.begin_single_time_commands() {
            if v_count > 0 {
                let copy_region = vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset((v_offset as usize * std::mem::size_of::<Vertex>()) as u64)
                    .size(v_byte_size);
                unsafe {
                    vulkan.device.cmd_copy_buffer(
                        cmd,
                        self.staging_vertex.handle,
                        self.vertex_buffer.handle,
                        std::slice::from_ref(&copy_region),
                    );
                }
            }

            if i_count > 0 {
                let copy_region = vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset((i_offset as usize * std::mem::size_of::<u32>()) as u64)
                    .size(i_byte_size);
                unsafe {
                    vulkan.device.cmd_copy_buffer(
                        cmd,
                        self.staging_index.handle,
                        self.index_buffer.handle,
                        std::slice::from_ref(&copy_region),
                    );
                }
            }

            if m_count > 0 {
                let copy_region = vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset((m_offset as usize * std::mem::size_of::<MeshletData>()) as u64)
                    .size(m_byte_size);
                unsafe {
                    vulkan.device.cmd_copy_buffer(
                        cmd,
                        self.staging_meshlet.handle,
                        self.meshlet_buffer.handle,
                        std::slice::from_ref(&copy_region),
                    );
                }
            }

            vulkan.end_single_time_commands(cmd);
        }

        self.current_vertex_count += v_count;
        self.current_index_count += i_count;
        self.current_meshlet_count += m_count;

        Some((v_offset, i_offset, m_offset))
    }

    /// Append a mesh to the geometry pool from slices in memory.
    pub fn append_mesh(
        &mut self,
        vulkan: &VulkanDevice,
        vertices: &[Vertex],
        indices: &[u32],
        meshlets: &[MeshletData],
    ) -> Option<(u32, u32, u32)> {
        self.append_with_callback(
            vulkan,
            vertices.len() as u32,
            indices.len() as u32,
            meshlets.len() as u32,
            |v_ptr, i_ptr, m_ptr| {
                if !v_ptr.is_null() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            vertices.as_ptr() as *const u8,
                            v_ptr,
                            vertices.len() * std::mem::size_of::<Vertex>(),
                        );
                    }
                }
                if !i_ptr.is_null() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            indices.as_ptr() as *const u8,
                            i_ptr,
                            indices.len() * std::mem::size_of::<u32>(),
                        );
                    }
                }
                if !m_ptr.is_null() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            meshlets.as_ptr() as *const u8,
                            m_ptr,
                            meshlets.len() * std::mem::size_of::<MeshletData>(),
                        );
                    }
                }
                Ok(())
            },
        )
    }


    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        self.vertex_buffer.shutdown(vulkan);
        self.index_buffer.shutdown(vulkan);
        self.meshlet_buffer.shutdown(vulkan);
        self.staging_vertex.shutdown(vulkan);
        self.staging_index.shutdown(vulkan);
        self.staging_meshlet.shutdown(vulkan);
    }
}