use crate::renderer::vulkan::buffer::Buffer;
use crate::renderer::vulkan::mesh::MeshletData;
use crate::renderer::vulkan::pipeline::Vertex;
use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

pub struct GeometryPool {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub meshlet_buffer: Buffer,

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
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
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

        Some(Self {
            vertex_buffer,
            index_buffer,
            meshlet_buffer,
            current_vertex_count: 0,
            current_index_count: 0,
            current_meshlet_count: 0,
            max_vertices,
            max_indices,
            max_meshlets,
        })
    }

    pub fn append_mesh(
        &mut self,
        vulkan: &VulkanDevice,
        vertices: &[Vertex],
        indices: &[u32],
        meshlets: &[MeshletData],
    ) -> Option<(u32, u32, u32)> {
        let v_count = vertices.len() as u32;
        let i_count = indices.len() as u32;
        let m_count = meshlets.len() as u32;

        if self.current_vertex_count + v_count > self.max_vertices
            || self.current_index_count + i_count > self.max_indices
            || self.current_meshlet_count + m_count > self.max_meshlets
        {
            crate::log_info!("GeometryPool out of space!");
            return None;
        }

        let v_offset = self.current_vertex_count;
        let i_offset = self.current_index_count;
        let m_offset = self.current_meshlet_count;

        // Stage and copy Vertices
        if v_count > 0 {
            let size = (v_count as usize * std::mem::size_of::<Vertex>()) as u64;
            if let Some(staging) = Buffer::new(
                vulkan,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ) {
                staging.upload(vulkan, vertices);
                self.copy_buffer(
                    vulkan,
                    &staging,
                    &self.vertex_buffer,
                    size,
                    0,
                    (v_offset as usize * std::mem::size_of::<Vertex>()) as u64,
                );
                let mut st = staging;
                st.shutdown(vulkan);
            } else {
                crate::log_info!("Failed to create vertex staging buffer!");
                return None;
            }
        }

        // Stage and copy Indices
        if i_count > 0 {
            let size = (i_count as usize * std::mem::size_of::<u32>()) as u64;
            if let Some(staging) = Buffer::new(
                vulkan,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ) {
                staging.upload(vulkan, indices);
                self.copy_buffer(
                    vulkan,
                    &staging,
                    &self.index_buffer,
                    size,
                    0,
                    (i_offset as usize * std::mem::size_of::<u32>()) as u64,
                );
                let mut st = staging;
                st.shutdown(vulkan);
            } else {
                crate::log_info!("Failed to create index staging buffer!");
                return None;
            }
        }

        // Stage and copy Meshlets
        if m_count > 0 {
            let size = (m_count as usize * std::mem::size_of::<MeshletData>()) as u64;
            if let Some(staging) = Buffer::new(
                vulkan,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ) {
                staging.upload(vulkan, meshlets);
                self.copy_buffer(
                    vulkan,
                    &staging,
                    &self.meshlet_buffer,
                    size,
                    0,
                    (m_offset as usize * std::mem::size_of::<MeshletData>()) as u64,
                );
                let mut st = staging;
                st.shutdown(vulkan);
            } else {
                crate::log_info!("Failed to create meshlet staging buffer!");
                return None;
            }
        }

        self.current_vertex_count += v_count;
        self.current_index_count += i_count;
        self.current_meshlet_count += m_count;

        Some((v_offset, i_offset, m_offset))
    }

    fn copy_buffer(
        &self,
        vulkan: &VulkanDevice,
        src: &Buffer,
        dst: &Buffer,
        size: u64,
        src_offset: u64,
        dst_offset: u64,
    ) {
        if let Some(cmd) = vulkan.begin_single_time_commands() {
            let copy_region = vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(size);
            unsafe {
                vulkan.device.cmd_copy_buffer(
                    cmd,
                    src.handle,
                    dst.handle,
                    std::slice::from_ref(&copy_region),
                );
            }
            vulkan.end_single_time_commands(cmd);
        }
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        self.vertex_buffer.shutdown(vulkan);
        self.index_buffer.shutdown(vulkan);
        self.meshlet_buffer.shutdown(vulkan);
    }
}
