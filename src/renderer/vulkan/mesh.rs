//! Mesh and Asset Loading.

use crate::renderer::vulkan::buffer::Buffer;
use crate::renderer::vulkan::pipeline::Vertex;
use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
}

impl Mesh {
    /// Loads an .obj file with potentially multiple sub-meshes using `tobj`.
    pub fn load_models(path: &str, vulkan: &VulkanDevice) -> Option<Vec<Self>> {
        let options = tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ignore_points: true,
            ignore_lines: true,
        };

        let (models, _materials) = tobj::load_obj(path, &options).ok()?;

        let mut loaded_meshes = Vec::new();

        for model in models {
            let mesh = &model.mesh;
            let mut vertices = Vec::new();

            let num_vertices = mesh.positions.len() / 3;
            for i in 0..num_vertices {
                let pos = [
                    mesh.positions[i * 3],
                    mesh.positions[i * 3 + 1],
                    mesh.positions[i * 3 + 2],
                ];

                let normal = if !mesh.normals.is_empty() {
                    [
                        mesh.normals[i * 3],
                        mesh.normals[i * 3 + 1],
                        mesh.normals[i * 3 + 2],
                    ]
                } else {
                    [0.0, 1.0, 0.0]
                };

                let uv = if !mesh.texcoords.is_empty() {
                    [
                        mesh.texcoords[i * 2],
                        // tobj usually gives V as top-down, but Vulkan is also top-down, or we flip it if needed.
                        mesh.texcoords[i * 2 + 1],
                    ]
                } else {
                    [0.0, 0.0]
                };

                vertices.push(Vertex { pos, normal, uv });
            }

            let indices = mesh.indices.clone();

            if vertices.is_empty() || indices.is_empty() {
                continue;
            }

            let vertex_buffer =
                Buffer::new_device_local(vulkan, &vertices, vk::BufferUsageFlags::VERTEX_BUFFER)?;

            let index_buffer =
                Buffer::new_device_local(vulkan, &indices, vk::BufferUsageFlags::INDEX_BUFFER)?;

            loaded_meshes.push(Self {
                vertex_buffer,
                index_buffer,
                index_count: indices.len() as u32,
            });
        }

        if loaded_meshes.is_empty() {
            None
        } else {
            Some(loaded_meshes)
        }
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        self.vertex_buffer.shutdown(vulkan);
        self.index_buffer.shutdown(vulkan);
    }
}
