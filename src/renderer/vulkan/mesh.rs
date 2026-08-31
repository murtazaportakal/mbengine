//! Mesh and Asset Loading.

use crate::renderer::vulkan::pipeline::Vertex;
use crate::renderer::vulkan::VulkanDevice;

// MeshletData lives in gpu_format.rs — the single source of truth.
// Re-exported here so existing `use mesh::MeshletData` imports keep working.
pub use crate::renderer::vulkan::gpu_format::MeshletData;

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
    pub is_skinned: bool,
}

impl Mesh {

    /// Create a Mesh directly from pre-parsed vertex and index data.
    /// Skips meshlet generation for simplicity — uses a single draw call.
    pub fn from_raw_data(
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
            cone_axis: [0.0, 1.0, 0.0],
            cone_cutoff: -1.0,
            index_offset: 0,
            triangle_count: indices.len() as u32 / 3,
            _pad: [0; 2],
        }];

        let offsets = geometry_pool.append_mesh(vulkan, vertices, indices, &meshlet_data)?;

        crate::log_info!("[GLTF Mesh] AABB: [{:.3}, {:.3}, {:.3}] .. [{:.3}, {:.3}, {:.3}]", 
            aabb_min[0], aabb_min[1], aabb_min[2], aabb_max[0], aabb_max[1], aabb_max[2]);

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
            is_skinned: false,
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
                    _pad0: 0,
                    normal,
                    _pad1: 0,
                    uv,
                    _pad2: [0; 2],
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

        Self::from_raw_data(vulkan, geometry_pool, &vertices, &indices)
    }

    pub fn shutdown(&mut self, _vulkan: &VulkanDevice) {
        // Buffers are now managed globally by GeometryPool
    }
}
