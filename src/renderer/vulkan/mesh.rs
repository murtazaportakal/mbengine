//! Mesh and Asset Loading.

use crate::renderer::vulkan::pipeline::Vertex;
use crate::renderer::vulkan::VulkanDevice;

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
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub meshlet_offset: u32,
    pub index_count: u32,
    pub meshlet_count: u32,
    pub vertex_count: u32,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub default_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub diffuse_texture: Option<String>,
    pub normal_texture: Option<String>,
    pub mr_texture: Option<String>,
    pub emissive_texture: Option<String>,
    pub diffuse_texture_idx: u32,
    pub normal_texture_idx: u32,
    pub mr_texture_idx: u32,
    pub emissive_texture_idx: u32,
}

impl Mesh {
    /// Loads an .obj file, triangulates, and generates Meshlets via meshopt.
    pub fn load_models(
        path: &str,
        vulkan: &VulkanDevice,
        geometry_pool: &mut crate::renderer::vulkan::GeometryPool,
    ) -> Option<Vec<Self>> {
        let options = tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ignore_points: true,
            ignore_lines: true,
        };

        let (models, materials_result) = match tobj::load_obj(path, &options) {
            Ok(result) => result,
            Err(e) => {
                crate::log_info!("[OBJ] tobj::load_obj failed for '{}': {:?}", path, e);
                return None;
            }
        };
        let materials = match materials_result {
            Ok(m) => m,
            Err(e) => {
                crate::log_info!("Failed to load MTL file for {}: {:?}", path, e);
                Vec::new()
            }
        };

        let mut loaded_meshes = Vec::new();

        for model in models {
            let mesh = &model.mesh;
            let mut default_color = [1.0, 1.0, 1.0];
            let mut metallic = 0.0;
            let mut roughness = 0.3;
            let mut diffuse_texture = None;

            if let Some(mat_id) = mesh.material_id {
                if let Some(mat) = materials.get(mat_id) {
                    if let Some(diffuse) = mat.diffuse {
                        default_color = diffuse;
                    }
                    if let Some(tex) = &mat.diffuse_texture {
                        if !tex.is_empty() {
                            diffuse_texture = Some(tex.clone());
                        }
                    }
                    if let Some(shininess) = mat.shininess {
                        if shininess > 0.0 {
                            roughness = (2.0 / (shininess + 2.0)).sqrt();
                        }
                    }
                    if let Some(pr) = mat.unknown_param.get("Pr") {
                        if let Ok(val) = pr.parse::<f32>() {
                            roughness = val;
                        }
                    }
                    if let Some(pm) = mat.unknown_param.get("Pm") {
                        if let Ok(val) = pm.parse::<f32>() {
                            metallic = val;
                        }
                    }
                }
            }

            let mut vertices = Vec::new();

            let mut calculated_normals = Vec::new();
            if mesh.normals.is_empty() {
                calculated_normals.resize(mesh.positions.len(), 0.0_f32);
                for i in (0..mesh.indices.len()).step_by(3) {
                    let i0 = mesh.indices[i] as usize;
                    let i1 = mesh.indices[i + 1] as usize;
                    let i2 = mesh.indices[i + 2] as usize;

                    let v0 = [
                        mesh.positions[i0 * 3],
                        mesh.positions[i0 * 3 + 1],
                        mesh.positions[i0 * 3 + 2],
                    ];
                    let v1 = [
                        mesh.positions[i1 * 3],
                        mesh.positions[i1 * 3 + 1],
                        mesh.positions[i1 * 3 + 2],
                    ];
                    let v2 = [
                        mesh.positions[i2 * 3],
                        mesh.positions[i2 * 3 + 1],
                        mesh.positions[i2 * 3 + 2],
                    ];

                    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

                    let cross = [
                        e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0],
                    ];

                    for &idx in &[i0, i1, i2] {
                        calculated_normals[idx * 3] += cross[0];
                        calculated_normals[idx * 3 + 1] += cross[1];
                        calculated_normals[idx * 3 + 2] += cross[2];
                    }
                }
                for i in (0..calculated_normals.len()).step_by(3) {
                    let n = &mut calculated_normals[i..i + 3];
                    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    if len > 0.0 {
                        n[0] /= len;
                        n[1] /= len;
                        n[2] /= len;
                    } else {
                        n[1] = 1.0;
                    }
                }
            }

            let normals_slice = if !mesh.normals.is_empty() {
                &mesh.normals
            } else {
                &calculated_normals
            };

            let num_vertices = mesh.positions.len() / 3;
            for i in 0..num_vertices {
                let pos = [
                    mesh.positions[i * 3],
                    mesh.positions[i * 3 + 1],
                    mesh.positions[i * 3 + 2],
                ];

                let normal = [
                    normals_slice[i * 3],
                    normals_slice[i * 3 + 1],
                    normals_slice[i * 3 + 2],
                ];

                let uv = if !mesh.texcoords.is_empty() {
                    [mesh.texcoords[i * 2], mesh.texcoords[i * 2 + 1]]
                } else {
                    [0.0, 0.0]
                };

                vertices.push(Vertex {
                    pos,
                    normal,
                    uv,
                    joint_ids: [0; 4],
                    joint_weights: [0.0; 4],
                });
            }

            let indices = mesh.indices.clone();

            if vertices.is_empty() || indices.is_empty() {
                continue;
            }

            let mut aabb_min = [f32::MAX; 3];
            let mut aabb_max = [f32::MIN; 3];
            for v in &vertices {
                for i in 0..3 {
                    aabb_min[i] = aabb_min[i].min(v.pos[i]);
                    aabb_max[i] = aabb_max[i].max(v.pos[i]);
                }
            }

            // --- Meshlet Generation via meshopt ---
            // Max 64 vertices and 124 triangles (divisible by 4) per meshlet.
            let vertices_u8: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    vertices.as_ptr() as *const u8,
                    vertices.len() * std::mem::size_of::<Vertex>(),
                )
            };

            let vertex_data = meshopt::VertexDataAdapter::new(
                vertices_u8,
                std::mem::size_of::<Vertex>(),
                0, // pos offset
            )
            .unwrap();

            let meshlets_raw = meshopt::build_meshlets(&indices, &vertex_data, 64, 124, 0.5);

            let mut global_indices = Vec::new();
            let mut meshlet_data_vec = Vec::new();

            for i in 0..meshlets_raw.meshlets.len() {
                let raw_m = &meshlets_raw.meshlets[i];
                let index_offset = global_indices.len() as u32;

                // Reconstruct global indices for this meshlet
                for tri_idx in 0..(raw_m.triangle_count * 3) {
                    let local_index =
                        meshlets_raw.triangles[(raw_m.triangle_offset + tri_idx) as usize];
                    let global_vertex_index =
                        meshlets_raw.vertices[(raw_m.vertex_offset + local_index as u32) as usize];
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
                        if v.pos[j] < min[j] {
                            min[j] = v.pos[j];
                        }
                        if v.pos[j] > max[j] {
                            max[j] = v.pos[j];
                        }
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

            // Upload to Geometry Pool
            let offsets = match geometry_pool.append_mesh(vulkan, &vertices, &global_indices, &meshlet_data_vec) {
                Some(o) => o,
                None => {
                    crate::log_info!("[OBJ] geometry_pool.append_mesh failed (out of space?) for '{}'", path);
                    return None;
                }
            };

            loaded_meshes.push(Self {
                vertex_offset: offsets.0,
                index_offset: offsets.1,
                meshlet_offset: offsets.2,
                index_count: global_indices.len() as u32,
                meshlet_count: meshlet_data_vec.len() as u32,
                vertex_count: vertices.len() as u32,
                aabb_min,
                aabb_max,
                default_color,
                metallic,
                roughness,
                diffuse_texture,
                normal_texture: None,
                mr_texture: None, // obj usually doesn't have MR
                emissive_texture: None,
                diffuse_texture_idx: 0,
                normal_texture_idx: 0,
                mr_texture_idx: 0,
                emissive_texture_idx: 0,
            });
        }

        Some(loaded_meshes)
    }

    /// Create a Mesh directly from pre-parsed vertex and index data (e.g., from GLTF).
    /// Skips meshlet generation for simplicity — uses a single draw call.
    pub fn from_gltf_data(
        vulkan: &VulkanDevice,
        geometry_pool: &mut crate::renderer::vulkan::GeometryPool,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Option<Self> {
        if vertices.is_empty() || indices.is_empty() {
            return None;
        }

        let mut aabb_min = [f32::MAX; 3];
        let mut aabb_max = [f32::MIN; 3];
        for v in vertices {
            for i in 0..3 {
                aabb_min[i] = aabb_min[i].min(v.pos[i]);
                aabb_max[i] = aabb_max[i].max(v.pos[i]);
            }
        }

        // Single meshlet covering the entire mesh.
        // Compute a proper bounding sphere from the AABB so the GPU culling
        // shader can perform correct frustum checks.
        let meshlet_center = [
            (aabb_min[0] + aabb_max[0]) * 0.5,
            (aabb_min[1] + aabb_max[1]) * 0.5,
            (aabb_min[2] + aabb_max[2]) * 0.5,
        ];
        let mut meshlet_radius_sq = 0.0f32;
        for v in vertices {
            let dx = v.pos[0] - meshlet_center[0];
            let dy = v.pos[1] - meshlet_center[1];
            let dz = v.pos[2] - meshlet_center[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq > meshlet_radius_sq {
                meshlet_radius_sq = dist_sq;
            }
        }
        let meshlet_data = vec![crate::renderer::vulkan::mesh::MeshletData {
            center: meshlet_center,
            radius: meshlet_radius_sq.sqrt().max(0.001),
            index_offset: 0,
            triangle_count: indices.len() as u32 / 3,
            padding: [0; 2],
        }];

        let offsets = geometry_pool.append_mesh(vulkan, vertices, indices, &meshlet_data)?;

        Some(Self {
            vertex_offset: offsets.0,
            index_offset: offsets.1,
            meshlet_offset: offsets.2,
            index_count: indices.len() as u32,
            meshlet_count: 1,
            vertex_count: vertices.len() as u32,
            aabb_min,
            aabb_max,
            default_color: [1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.8,
            diffuse_texture: None,
            normal_texture: None,
            mr_texture: None,
            emissive_texture: None,
            diffuse_texture_idx: 0,
            normal_texture_idx: 0,
            mr_texture_idx: 0,
            emissive_texture_idx: 0,
        })
    }

    pub fn create_grid(
        vulkan: &VulkanDevice,
        geometry_pool: &mut crate::renderer::vulkan::GeometryPool,
        width: u32,
        height: u32,
        spacing: f32,
    ) -> Option<Self> {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let offset_x = (width as f32 * spacing) / 2.0;
        let offset_z = (height as f32 * spacing) / 2.0;

        for y in 0..height {
            for x in 0..width {
                let pos = [
                    (x as f32 * spacing) - offset_x,
                    5.0, // Start high up so it can fall
                    (y as f32 * spacing) - offset_z,
                ];
                let normal = [0.0, 1.0, 0.0];
                let uv = [
                    x as f32 / (width - 1) as f32,
                    y as f32 / (height - 1) as f32,
                ];

                vertices.push(Vertex {
                    pos,
                    normal,
                    uv,
                    joint_ids: [0; 4],
                    joint_weights: [0.0; 4],
                });
            }
        }

        let mut aabb_min = [f32::MAX; 3];
        let mut aabb_max = [f32::MIN; 3];
        for v in &vertices {
            for i in 0..3 {
                aabb_min[i] = aabb_min[i].min(v.pos[i]);
                aabb_max[i] = aabb_max[i].max(v.pos[i]);
            }
        }

        for y in 0..height - 1 {
            for x in 0..width - 1 {
                let idx0 = y * width + x;
                let idx1 = idx0 + 1;
                let idx2 = (y + 1) * width + x;
                let idx3 = idx2 + 1;

                indices.push(idx0);
                indices.push(idx2);
                indices.push(idx1);

                indices.push(idx1);
                indices.push(idx2);
                indices.push(idx3);
            }
        }

        Self::from_gltf_data(vulkan, geometry_pool, &vertices, &indices)
    }

    pub fn shutdown(&mut self, _vulkan: &VulkanDevice) {
        // Buffers are now managed globally by GeometryPool
    }
}
