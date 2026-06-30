//! Mesh and Asset Loading.

use crate::renderer::vulkan::pipeline::Vertex;
use crate::renderer::vulkan::buffer::Buffer;
use crate::renderer::vulkan::VulkanDevice;
use ash::vk;
use std::collections::HashMap;

pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
}

impl Mesh {
    /// Loads a simple .obj file. Currently parses `v` and `f`, generating vertex colors from `vn`.
    pub fn load_obj(path: &str, vulkan: &VulkanDevice) -> Option<Self> {
        let contents = std::fs::read_to_string(path).ok()?;
        
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        
        // Maps a tuple of (pos_idx, uv_idx, norm_idx) to a unique Vertex index.
        let mut unique_vertices: HashMap<(usize, usize, usize), u32> = HashMap::new();

        for line in contents.lines() {
            let mut parts = line.split_whitespace();
            let Some(cmd) = parts.next() else { continue };

            match cmd {
                "v" => {
                    let x = parts.next()?.parse::<f32>().ok()?;
                    let y = parts.next()?.parse::<f32>().ok()?;
                    let z = parts.next()?.parse::<f32>().ok()?;
                    positions.push([x, y, z]);
                }
                "vt" => {
                    let u = parts.next()?.parse::<f32>().ok()?;
                    let v = parts.next()?.parse::<f32>().ok()?;
                    uvs.push([u, v]);
                }
                "vn" => {
                    let nx = parts.next()?.parse::<f32>().ok()?;
                    let ny = parts.next()?.parse::<f32>().ok()?;
                    let nz = parts.next()?.parse::<f32>().ok()?;
                    normals.push([nx, ny, nz]);
                }
                "f" => {
                    // Face format: v/vt/vn. We might not have vt.
                    // Process each vertex in the face (assuming triangles)
                    let mut face_indices = Vec::new();
                    
                    for part in parts {
                        let sub_parts: Vec<&str> = part.split('/').collect();
                        // OBJ is 1-indexed
                        let p_idx = sub_parts[0].parse::<usize>().ok()? - 1;
                        let mut uv_idx = 0;
                        let mut n_idx = 0;
                        
                        if sub_parts.len() >= 2 && !sub_parts[1].is_empty() {
                            uv_idx = sub_parts[1].parse::<usize>().ok()? - 1;
                        }
                        if sub_parts.len() >= 3 && !sub_parts[2].is_empty() {
                            n_idx = sub_parts[2].parse::<usize>().ok()? - 1;
                        }

                        let key = (p_idx, uv_idx, n_idx);
                        
                        let idx = *unique_vertices.entry(key).or_insert_with(|| {
                            let new_idx = vertices.len() as u32;
                            let pos = positions[p_idx];
                            
                            let normal = if n_idx < normals.len() {
                                normals[n_idx]
                            } else {
                                [0.0, 1.0, 0.0]
                            };

                            let uv = if uv_idx < uvs.len() {
                                uvs[uv_idx]
                            } else {
                                [0.0, 0.0]
                            };
                            
                            vertices.push(Vertex { pos, normal, uv });
                            new_idx
                        });
                        
                        face_indices.push(idx);
                    }
                    
                    // Simple triangulation (if face has > 3 vertices, make a fan)
                    for i in 1..(face_indices.len() - 1) {
                        indices.push(face_indices[0]);
                        indices.push(face_indices[i]);
                        indices.push(face_indices[i + 1]);
                    }
                }
                _ => {}
            }
        }

        if vertices.is_empty() || indices.is_empty() {
            return None;
        }

        let vertex_buffer = Buffer::new_device_local(
            vulkan,
            &vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;

        let index_buffer = Buffer::new_device_local(
            vulkan,
            &indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;

        Some(Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        })
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        self.vertex_buffer.shutdown(vulkan);
        self.index_buffer.shutdown(vulkan);
    }
}
