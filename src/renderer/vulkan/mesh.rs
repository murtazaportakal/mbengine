//! Mesh and Asset Loading.

use crate::renderer::vulkan::buffer::Buffer;
use crate::renderer::vulkan::pipeline::Vertex;
use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MeshletData {
    pub center: [f32; 3],
    pub radius: f32,
    pub index_offset: u32,
    pub triangle_count: u32,
    pub padding: [u32; 2],
}

pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    pub meshlet_buffer: Buffer,
    pub meshlet_count: u32,
    pub indirect_buffer: Buffer,
}

impl Mesh {
    /// Loads an .obj file, triangulates, and generates Meshlets via meshopt.
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

            // --- Meshlet Generation via meshopt ---
            // Max 64 vertices and 124 triangles (divisible by 4) per meshlet.
            let vertices_u8: &[u8] = unsafe { 
                std::slice::from_raw_parts(
                    vertices.as_ptr() as *const u8, 
                    vertices.len() * std::mem::size_of::<Vertex>()
                ) 
            };
            
            let vertex_data = meshopt::VertexDataAdapter::new(
                vertices_u8,
                std::mem::size_of::<Vertex>(),
                0, // pos offset
            ).unwrap();

            let meshlets_raw = meshopt::build_meshlets(&indices, &vertex_data, 64, 124, 0.5);
            
            let mut global_indices = Vec::new();
            let mut meshlet_data_vec = Vec::new();

            for i in 0..meshlets_raw.meshlets.len() {
                let raw_m = &meshlets_raw.meshlets[i];
                let index_offset = global_indices.len() as u32;
                
                // Reconstruct global indices for this meshlet
                for tri_idx in 0..(raw_m.triangle_count * 3) {
                    let local_index = meshlets_raw.triangles[(raw_m.triangle_offset + tri_idx) as usize];
                    let global_vertex_index = meshlets_raw.vertices[(raw_m.vertex_offset + local_index as u32) as usize];
                    global_indices.push(global_vertex_index);
                }

                // Compute bounding sphere
                // meshopt returns it as an iterator of Meshlet<'_>, but let's do a simple AABB/Sphere manually 
                // to avoid the meshopt struct bounds mismatch, or just parse the bounds.
                // We'll compute a simple bounding sphere for the meshlet.
                let mut min = [f32::MAX; 3];
                let mut max = [f32::MIN; 3];
                for idx in index_offset as usize..global_indices.len() {
                    let v = &vertices[global_indices[idx] as usize];
                    for j in 0..3 {
                        if v.pos[j] < min[j] { min[j] = v.pos[j]; }
                        if v.pos[j] > max[j] { max[j] = v.pos[j]; }
                    }
                }
                let center = [
                    (min[0] + max[0]) * 0.5,
                    (min[1] + max[1]) * 0.5,
                    (min[2] + max[2]) * 0.5,
                ];
                let mut radius_sq = 0.0f32;
                for idx in index_offset as usize..global_indices.len() {
                    let v = &vertices[global_indices[idx] as usize];
                    let dist_sq = (v.pos[0] - center[0]).powi(2)
                        + (v.pos[1] - center[1]).powi(2)
                        + (v.pos[2] - center[2]).powi(2);
                    if dist_sq > radius_sq {
                        radius_sq = dist_sq;
                    }
                }

                meshlet_data_vec.push(MeshletData {
                    center,
                    radius: radius_sq.sqrt(),
                    index_offset,
                    triangle_count: raw_m.triangle_count,
                    padding: [0; 2],
                });
            }

            // Upload to GPU
            let vertex_buffer =
                Buffer::new_device_local(vulkan, &vertices, vk::BufferUsageFlags::VERTEX_BUFFER)?;

            let index_buffer =
                Buffer::new_device_local(vulkan, &global_indices, vk::BufferUsageFlags::INDEX_BUFFER)?;

            let meshlet_buffer = Buffer::new_device_local(
                vulkan, 
                &meshlet_data_vec, 
                vk::BufferUsageFlags::STORAGE_BUFFER // Used by Compute Shader
            )?;

            let indirect_buffer = Buffer::new_device_local(
                vulkan,
                &vec![0u8; meshlet_data_vec.len() * 20], // 5 u32s per command
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER
            )?;

            loaded_meshes.push(Self {
                vertex_buffer,
                index_buffer,
                index_count: global_indices.len() as u32,
                meshlet_buffer,
                meshlet_count: meshlet_data_vec.len() as u32,
                indirect_buffer,
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
        self.meshlet_buffer.shutdown(vulkan);
        self.indirect_buffer.shutdown(vulkan);
    }
}
