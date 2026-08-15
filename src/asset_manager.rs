use crate::renderer::vulkan::skeleton::{AnimationClip, Skeleton};
use crate::renderer::vulkan::{Mesh, Texture, VulkanDevice};
use crate::scripting::engine::ScriptEngine;
use crate::vfs::Vfs;
use bytemuck::{Pod, Zeroable};
use notify::{RecursiveMode, Watcher};
use rhai::AST;
use std::collections::HashMap;
use std::path::Path;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeshHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub flags: u32,
    pub vertex_count: u32,
    pub index_count: u32,
    pub meshlet_count: u32,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MatHeader {
    pub magic: [u8; 4],
    pub version: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MatData {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub albedo_path_offset: u32,
    pub normal_path_offset: u32,
    pub mr_path_offset: u32,
    pub emissive_path_offset: u32,
    pub _pad: [u32; 2],
}

#[derive(Clone, Debug)]
pub struct MaterialAsset {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub albedo_texture: Option<String>,
    pub normal_texture: Option<String>,
    pub mr_texture: Option<String>,
    pub emissive_texture: Option<String>,
}

pub enum AssetEvent {
    ShaderChanged,
    TextureChanged(String), // Name of the texture
    ModelChanged(String),   // Name of the model
}

pub struct AssetManager {
    textures: HashMap<String, Texture>,
    pub texture_indices: HashMap<String, u32>,
    pub next_texture_index: u32,
    pub new_textures_since_last_frame: Vec<String>,

    pub meshes: Vec<Mesh>,
    model_map: HashMap<String, Vec<usize>>,
    pub materials: HashMap<String, MaterialAsset>,

    // File to Name reverse mappings to know which asset changed
    texture_paths: HashMap<String, String>,
    model_paths: HashMap<String, String>,

    pub skeletons: HashMap<String, Skeleton>,
    /// Flat array of loaded animation clips (AnimationClipHandle registry).
    pub animation_clips: Vec<AnimationClip>,

    /// Cached Rhai ASTs.
    scripts: HashMap<String, AST>,
    script_paths: HashMap<String, String>,

    pub vfs: Vfs,
    watcher: Option<notify::RecommendedWatcher>,
    rx: Option<std::sync::mpsc::Receiver<notify::Result<notify::Event>>>,
}

impl Default for AssetManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetManager {
    pub fn new() -> Self {
        let mut manager = Self {
            textures: HashMap::new(),
            texture_indices: HashMap::new(),
            next_texture_index: 0,
            new_textures_since_last_frame: Vec::new(),
            meshes: Vec::new(),
            model_map: HashMap::new(),
            materials: HashMap::new(),
            texture_paths: HashMap::new(),
            model_paths: HashMap::new(),
            skeletons: HashMap::new(),
            animation_clips: Vec::new(),
            scripts: HashMap::new(),
            script_paths: HashMap::new(),
            vfs: Vfs::default(),
            watcher: None,
            rx: None,
        };

        // Initialize file watcher
        let (tx, rx) = std::sync::mpsc::channel();
        if let Ok(mut watcher) = notify::recommended_watcher(tx) {
            let _ = watcher.watch(Path::new("src/shaders"), RecursiveMode::Recursive);
            let _ = watcher.watch(Path::new("assets"), RecursiveMode::Recursive);
            manager.watcher = Some(watcher);
            manager.rx = Some(rx);
        }

        manager
    }

    pub fn get_texture_index(&self, name: &str) -> Option<u32> {
        self.texture_indices.get(name).copied()
    }

    pub fn add_mesh(&mut self, mesh: Mesh) -> usize {
        let index = self.meshes.len();
        self.meshes.push(mesh);
        index
    }

    pub fn load_texture(
        &mut self,
        vulkan: &VulkanDevice,
        name: &str,
        path: &str,
    ) -> Option<&Texture> {
        if !self.textures.contains_key(name) {
            let is_tex = path.ends_with(".tex");
            let tex_opt = if is_tex {
                Self::load_cooked_texture_file(vulkan, path)
            } else {
                Texture::load_from_file(vulkan, path)
            };

            if let Some(tex) = tex_opt {
                let idx = self.next_texture_index;
                self.next_texture_index += 1;
                self.texture_indices.insert(name.to_string(), idx);
                self.new_textures_since_last_frame.push(name.to_string());

                self.textures.insert(name.to_string(), tex);
                self.texture_paths
                    .insert(path.to_string(), name.to_string());
            } else {
                crate::log_info!("Failed to load texture: {}", path);
                return None;
            }
        }
        self.textures.get(name)
    }

    fn load_cooked_texture_file(vulkan: &VulkanDevice, path: &str) -> Option<Texture> {
        let bytes = std::fs::read(path).ok()?;
        if bytes.len() < 32 {
            return None;
        }

        let magic = &bytes[0..4];
        if magic != b"TEXL" {
            return None;
        }

        let width = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let height = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let format = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let data_size = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;

        if bytes.len() < 32 + data_size {
            return None;
        }

        let data = &bytes[32..32 + data_size];
        Texture::new_bc7(vulkan, width, height, format, data)
    }

    pub fn load_hdr_texture(
        &mut self,
        vulkan: &VulkanDevice,
        name: &str,
        path: &str,
    ) -> Option<&Texture> {
        if !self.textures.contains_key(name) {
            if let Some(tex) = Texture::load_hdr(vulkan, path) {
                let idx = self.next_texture_index;
                self.next_texture_index += 1;
                self.texture_indices.insert(name.to_string(), idx);
                self.new_textures_since_last_frame.push(name.to_string());
                self.textures.insert(name.to_string(), tex);
                self.texture_paths
                    .insert(path.to_string(), name.to_string());
            } else {
                crate::log_info!("Failed to load HDR texture: {}", path);
                return None;
            }
        }
        self.textures.get(name)
    }

    pub fn load_solid_color(
        &mut self,
        vulkan: &VulkanDevice,
        name: &str,
        r: u8,
        g: u8,
        b: u8,
        a: u8,
    ) -> Option<&Texture> {
        if !self.textures.contains_key(name) {
            if let Some(tex) = Texture::new_solid_color(vulkan, r, g, b, a) {
                let idx = self.next_texture_index;
                self.next_texture_index += 1;
                self.texture_indices.insert(name.to_string(), idx);
                self.new_textures_since_last_frame.push(name.to_string());
                self.textures.insert(name.to_string(), tex);
            }
        }
        self.textures.get(name)
    }

    pub fn load_procedural_env(&mut self, vulkan: &VulkanDevice, name: &str) -> Option<&Texture> {
        if !self.textures.contains_key(name) {
            if let Some(tex) = Texture::new_procedural_env(vulkan) {
                let idx = self.next_texture_index;
                self.next_texture_index += 1;
                self.texture_indices.insert(name.to_string(), idx);
                self.new_textures_since_last_frame.push(name.to_string());
                self.textures.insert(name.to_string(), tex);
            }
        }
        self.textures.get(name)
    }

    /// Provides a fallback checkerboard texture if requested
    pub fn load_checkerboard(&mut self, vulkan: &VulkanDevice, name: &str) -> Option<&Texture> {
        if !self.textures.contains_key(name) {
            if let Some(tex) = Texture::new_checkerboard(vulkan) {
                let idx = self.next_texture_index;
                self.next_texture_index += 1;
                self.texture_indices.insert(name.to_string(), idx);
                self.new_textures_since_last_frame.push(name.to_string());
                self.textures.insert(name.to_string(), tex);
            } else {
                return None;
            }
        }
        self.textures.get(name)
    }

    pub fn get_texture(&self, name: &str) -> Option<&Texture> {
        self.textures.get(name)
    }

    pub fn load_cooked_mesh(
        &mut self,
        vulkan: &VulkanDevice,
        geometry_pool: &mut crate::renderer::vulkan::GeometryPool,
        path: &str,
    ) -> Option<usize> {
        if let Some(indices) = self.model_map.get(path) {
            return indices.first().copied();
        }

        let mut header = MeshHeader::zeroed();
        let header_slice = bytemuck::bytes_of_mut(&mut header);
        if unsafe { self.vfs.read_chunk_into_ptr(path, 0, header_slice.as_mut_ptr(), header_slice.len()) }.is_err() {
            crate::log_info!("[AssetMgr] Failed to read .mesh header from {}", path);
            return None;
        }

        if &header.magic != b"MESH" {
            crate::log_info!("[AssetMgr] Invalid .mesh magic in {}", path);
            return None;
        }

        let header_size = std::mem::size_of::<MeshHeader>() as u64;
        let v_byte_size = (header.vertex_count as usize * std::mem::size_of::<crate::renderer::vulkan::pipeline::Vertex>()) as u64;
        let i_byte_size = (header.index_count as usize * std::mem::size_of::<u32>()) as u64;
        let m_byte_size = (header.meshlet_count as usize * std::mem::size_of::<crate::renderer::vulkan::mesh::MeshletData>()) as u64;

        let v_offset_in_file = header_size;
        let i_offset_in_file = v_offset_in_file + v_byte_size;
        let m_offset_in_file = i_offset_in_file + i_byte_size;

        let vfs_path = path.to_string();
        let vfs = &self.vfs;

        let offsets = geometry_pool.append_with_callback(
            vulkan,
            header.vertex_count,
            header.index_count,
            header.meshlet_count,
            |v_ptr, i_ptr, m_ptr| {
                if !v_ptr.is_null() {
                    unsafe { vfs.read_chunk_into_ptr(&vfs_path, v_offset_in_file, v_ptr, v_byte_size as usize)? };
                }
                if !i_ptr.is_null() {
                    unsafe { vfs.read_chunk_into_ptr(&vfs_path, i_offset_in_file, i_ptr, i_byte_size as usize)? };
                }
                if !m_ptr.is_null() {
                    unsafe { vfs.read_chunk_into_ptr(&vfs_path, m_offset_in_file, m_ptr, m_byte_size as usize)? };
                }
                Ok(())
            }
        )?;

        let mesh = Mesh {
            vertex_offset: offsets.0,
            index_offset: offsets.1,
            meshlet_offset: offsets.2,
            index_count: header.index_count,
            meshlet_count: header.meshlet_count,
            vertex_count: header.vertex_count,
            aabb_min: header.aabb_min,
            aabb_max: header.aabb_max,
            default_color: [0.8, 0.8, 0.8],
            metallic: 0.0,
            roughness: 0.8,
            diffuse_texture: None,
            normal_texture: None,
            mr_texture: None,
            emissive_texture: None,
            diffuse_texture_idx: self.get_texture_index("default_white").unwrap_or(0) as u32,
            normal_texture_idx: 0,
            mr_texture_idx: 0,
            emissive_texture_idx: 0,

        };

        let index = self.add_mesh(mesh);
        self.model_map.insert(path.to_string(), vec![index]);
        self.model_paths.insert(path.to_string(), path.to_string());

        Some(index)
    }

    pub fn load_cooked_material(&mut self, vulkan: &VulkanDevice, name: &str, path: &str) -> Option<()> {
        if self.materials.contains_key(name) { return Some(()); }
        
        let bytes = self.vfs.read_bytes(path).ok()?;
        if bytes.len() < std::mem::size_of::<MatHeader>() + std::mem::size_of::<MatData>() { return None; }
        
        let header = bytemuck::pod_read_unaligned::<MatHeader>(&bytes[0..std::mem::size_of::<MatHeader>()]);
        if &header.magic != b"MATL" { return None; }
        
        let data = bytemuck::pod_read_unaligned::<MatData>(&bytes[std::mem::size_of::<MatHeader>()..std::mem::size_of::<MatHeader>() + std::mem::size_of::<MatData>()]);
        
        let read_str = |offset: u32| -> Option<String> {
            if offset == 0 { return None; }
            let start = offset as usize;
            if start >= bytes.len() { return None; }
            let mut end = start;
            while end < bytes.len() && bytes[end] != 0 {
                end += 1;
            }
            String::from_utf8(bytes[start..end].to_vec()).ok()
        };
        
        let albedo_texture = read_str(data.albedo_path_offset);
        let normal_texture = read_str(data.normal_path_offset);
        let mr_texture = read_str(data.mr_path_offset);
        let emissive_texture = read_str(data.emissive_path_offset);
        
        let mut load_tex = |tex: &Option<String>| {
            if let Some(t) = tex {
                self.load_texture(vulkan, t, t);
            }
        };
        load_tex(&albedo_texture);
        load_tex(&normal_texture);
        load_tex(&mr_texture);
        load_tex(&emissive_texture);
        
        self.materials.insert(name.to_string(), MaterialAsset {
            base_color_factor: data.base_color_factor,
            metallic_factor: data.metallic_factor,
            roughness_factor: data.roughness_factor,
            albedo_texture,
            normal_texture,
            mr_texture,
            emissive_texture,
        });
        
        Some(())
    }

    pub fn load_model(
        &mut self,
        vulkan: &VulkanDevice,
        geometry_pool: &mut crate::renderer::vulkan::GeometryPool,
        path: &str,
    ) -> Option<&[usize]> {
        if !self.model_map.contains_key(path) {
            crate::log_info!("[AssetMgr] Loading model: {}", path);
            if let Some(loaded_meshes) = Mesh::load_models(path, vulkan, geometry_pool) {
                crate::log_info!("[AssetMgr] Model {} loaded {} meshes", path, loaded_meshes.len());
                let mut indices = Vec::with_capacity(loaded_meshes.len());
                for (mi, mut mesh) in loaded_meshes.into_iter().enumerate() {
                    crate::log_info!("[AssetMgr]   mesh[{}]: {} verts, {} indices, AABB [{:.1},{:.1},{:.1}]..[{:.1},{:.1},{:.1}]",
                        mi, mesh.vertex_count, mesh.index_count,
                        mesh.aabb_min[0], mesh.aabb_min[1], mesh.aabb_min[2],
                        mesh.aabb_max[0], mesh.aabb_max[1], mesh.aabb_max[2]);
                    
                    if mesh.diffuse_texture.is_none() {
                        mesh.diffuse_texture_idx = self.get_texture_index("default_white").unwrap_or(0) as u32;
                    }

                    if let Some(tex_name) = &mesh.diffuse_texture {
                        let tex_path = if std::path::Path::new(tex_name).is_absolute() {
                            tex_name.clone()
                        } else if let Some(parent) = std::path::Path::new(path).parent() {
                            parent.join(tex_name).to_string_lossy().to_string()
                        } else {
                            tex_name.clone()
                        };
                        self.load_texture(vulkan, tex_name, &tex_path);
                        mesh.diffuse_texture_idx = self.get_texture_index(tex_name).unwrap_or(0);
                    }
                    if let Some(tex_name) = &mesh.normal_texture {
                        mesh.normal_texture_idx = self.get_texture_index(tex_name).unwrap_or(0);
                    }
                    if let Some(tex_name) = &mesh.mr_texture {
                        mesh.mr_texture_idx = self.get_texture_index(tex_name).unwrap_or(0);
                    }
                    if let Some(tex_name) = &mesh.emissive_texture {
                        mesh.emissive_texture_idx = self.get_texture_index(tex_name).unwrap_or(0);
                    }
                    indices.push(self.meshes.len());
                    self.meshes.push(mesh);
                }
                self.model_map.insert(path.to_string(), indices);
                self.model_paths.insert(path.to_string(), path.to_string());
            } else {
                crate::log_info!("[AssetMgr] Failed to load model: {}", path);
                return None;
            }
        }
        self.model_map.get(path).map(|v| v.as_slice())
    }

    /// Load a GLTF/GLB file containing meshes, skeleton, and animation clips.
    ///
    /// Returns the mesh indices for the loaded primitives, or None on failure.
    pub fn load_gltf(
        &mut self,
        vulkan: &VulkanDevice,
        geometry_pool: &mut crate::renderer::vulkan::GeometryPool,
        name: &str,
        path: &str,
    ) -> Option<&[usize]> {
        if self.model_map.contains_key(name) {
            return self.model_map.get(name).map(|v| v.as_slice());
        }

        let gltf_data = crate::renderer::vulkan::gltf_loader::load_gltf(path)?;

        // Load images into textures
        let mut gltf_texture_names = Vec::new();
        for (i, img) in gltf_data.images.iter().enumerate() {
            let tex_name = format!("{}_tex_{}", name, i);
            if !self.textures.contains_key(&tex_name) {
                if let Some(tex) = crate::renderer::vulkan::texture::Texture::from_rgba8(
                    vulkan,
                    img.width,
                    img.height,
                    &img.pixels,
                ) {
                    let idx = self.next_texture_index;
                    self.next_texture_index += 1;
                    self.texture_indices.insert(tex_name.clone(), idx);
                    self.new_textures_since_last_frame.push(tex_name.clone());
                    self.textures.insert(tex_name.clone(), tex);
                }
            }
            gltf_texture_names.push(tex_name);
        }

        // Store meshes
        let mut indices = Vec::with_capacity(gltf_data.primitives.len());
        for (vertices, idx_data, mat_idx) in &gltf_data.primitives {
            let mut mesh = Mesh::from_gltf_data(vulkan, geometry_pool, vertices, idx_data)?;

            // Apply material properties if available
            if let Some(m_idx) = mat_idx {
                if let Some(mat) = gltf_data.materials.get(*m_idx) {
                    mesh.default_color = [
                        mat.base_color_factor[0],
                        mat.base_color_factor[1],
                        mat.base_color_factor[2],
                    ];
                    mesh.metallic = mat.metallic_factor;
                    mesh.roughness = mat.roughness_factor;

                    if let Some(tex_idx) = mat.base_color_texture {
                        let name = gltf_texture_names[tex_idx].clone();
                        mesh.diffuse_texture = Some(name.clone());
                        mesh.diffuse_texture_idx = self.get_texture_index(&name).unwrap_or(0);
                    }
                    if let Some(tex_idx) = mat.normal_texture {
                        let name = gltf_texture_names[tex_idx].clone();
                        mesh.normal_texture = Some(name.clone());
                        mesh.normal_texture_idx = self.get_texture_index(&name).unwrap_or(0);
                    }
                    if let Some(tex_idx) = mat.metallic_roughness_texture {
                        let name = gltf_texture_names[tex_idx].clone();
                        mesh.mr_texture = Some(name.clone());
                        mesh.mr_texture_idx = self.get_texture_index(&name).unwrap_or(0);
                    }
                    if let Some(tex_idx) = mat.emissive_texture {
                        let name = gltf_texture_names[tex_idx].clone();
                        mesh.emissive_texture = Some(name.clone());
                        mesh.emissive_texture_idx = self.get_texture_index(&name).unwrap_or(0);
                    }
                }
            }

            indices.push(self.meshes.len());
            self.meshes.push(mesh);
        }
        self.model_map.insert(name.to_string(), indices);
        self.model_paths.insert(path.to_string(), name.to_string());

        // Store skeleton
        if let Some(skeleton) = gltf_data.skeleton {
            self.skeletons.insert(name.to_string(), skeleton);
        }

        // Store animation clips
        if !gltf_data.clips.is_empty() {
            self.animation_clips.extend(gltf_data.clips);
        }

        self.model_map.get(name).map(|v| v.as_slice())
    }

    /// Get a skeleton by name.
    pub fn get_skeleton(&self, name: &str) -> Option<&Skeleton> {
        self.skeletons.get(name)
    }

    /// Find an animation clip by name and return its handle and reference.
    pub fn get_animation_clip_by_name(&self, clip_name: &str) -> Option<(u32, &AnimationClip)> {
        self.animation_clips
            .iter()
            .enumerate()
            .find(|(_, c)| c.name == clip_name)
            .map(|(i, c)| (i as u32, c))
    }

    /// Get an animation clip by its handle.
    pub fn get_animation_clip(&self, handle: u32) -> Option<&AnimationClip> {
        self.animation_clips.get(handle as usize)
    }

    pub fn load_cooked_anim(
        &mut self,
        vulkan: &crate::renderer::vulkan::VulkanDevice,
        animation_pool: &mut crate::renderer::vulkan::AnimationPool,
        name: &str,
        path: &str,
    ) -> Option<()> {
        let bytes = self.vfs.read_bytes(path).ok()?;
        if bytes.len() < 16 { return None; } // AnimHeader size

        let magic = &bytes[0..4];
        if magic != b"ANIM" { return None; }
        
        let _version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let bone_count = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let clip_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap());

        let mut offset = 16;
        let read_str = |len: u32, offset: &mut usize| -> Option<String> {
            let start = *offset;
            let end = start + len as usize;
            if end > bytes.len() { return None; }
            let s = String::from_utf8_lossy(&bytes[start..end]).to_string();
            let pad = (4 - (len % 4)) % 4;
            *offset = end + pad as usize;
            Some(s)
        };

        let mut bones = Vec::new();
        for i in 0..bone_count {
            if offset + 72 > bytes.len() { return None; } // BoneHeader size: 16*4 + 4 + 4 = 72
            let mut ibm = [0f32; 16];
            for j in 0..16 {
                ibm[j] = f32::from_le_bytes(bytes[offset + j*4 .. offset + j*4 + 4].try_into().unwrap());
            }
            let parent_index = i32::from_le_bytes(bytes[offset + 64 .. offset + 68].try_into().unwrap());
            let name_len = u32::from_le_bytes(bytes[offset + 68 .. offset + 72].try_into().unwrap());
            offset += 72;

            let bname = read_str(name_len, &mut offset)?;

            let inverse_bind_matrix = crate::math::mat4::Mat4::new(
                crate::math::vec::Vec4::new(ibm[0], ibm[1], ibm[2], ibm[3]),
                crate::math::vec::Vec4::new(ibm[4], ibm[5], ibm[6], ibm[7]),
                crate::math::vec::Vec4::new(ibm[8], ibm[9], ibm[10], ibm[11]),
                crate::math::vec::Vec4::new(ibm[12], ibm[13], ibm[14], ibm[15]),
            );

            bones.push(crate::renderer::vulkan::skeleton::Bone {
                index: i as usize,
                name: bname,
                inverse_bind_matrix,
                parent: if parent_index >= 0 { Some(parent_index as usize) } else { None },
            });
        }

        let mut gpu_skeleton = crate::renderer::vulkan::animation::GpuSkeleton::default();
        gpu_skeleton.bone_count = bones.len() as u32;
        for (i, b) in bones.iter().enumerate() {
            let ibm = b.inverse_bind_matrix;
            gpu_skeleton.bones[i].inverse_bind_matrix = [
                ibm.cols[0].x, ibm.cols[0].y, ibm.cols[0].z, ibm.cols[0].w,
                ibm.cols[1].x, ibm.cols[1].y, ibm.cols[1].z, ibm.cols[1].w,
                ibm.cols[2].x, ibm.cols[2].y, ibm.cols[2].z, ibm.cols[2].w,
                ibm.cols[3].x, ibm.cols[3].y, ibm.cols[3].z, ibm.cols[3].w,
            ];
            gpu_skeleton.bones[i].parent_index = b.parent.map(|p| p as i32).unwrap_or(-1);
        }
        let skeleton_idx = animation_pool.append_skeleton(vulkan, &gpu_skeleton)?;
        
        let skeleton = Skeleton::new(bones, skeleton_idx);
        self.skeletons.insert(name.to_string(), skeleton);

        for _ in 0..clip_count {
            if offset + 12 > bytes.len() { return None; } // ClipHeader size
            let name_len = u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap());
            let duration = f32::from_le_bytes(bytes[offset+4..offset+8].try_into().unwrap());
            let channel_count = u32::from_le_bytes(bytes[offset+8..offset+12].try_into().unwrap());
            offset += 12;

            let cname = read_str(name_len, &mut offset)?;

            let mut channels = Vec::new();
            for _ in 0..channel_count {
                if offset + 16 > bytes.len() { return None; }
                let bone_index = u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap());
                let t_count = u32::from_le_bytes(bytes[offset+4..offset+8].try_into().unwrap());
                let r_count = u32::from_le_bytes(bytes[offset+8..offset+12].try_into().unwrap());
                let s_count = u32::from_le_bytes(bytes[offset+12..offset+16].try_into().unwrap());
                offset += 16;

                let mut translation_keys = Vec::new();
                for _ in 0..t_count {
                    if offset + 16 > bytes.len() { return None; }
                    let time = f32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap());
                    let x = f32::from_le_bytes(bytes[offset+4..offset+8].try_into().unwrap());
                    let y = f32::from_le_bytes(bytes[offset+8..offset+12].try_into().unwrap());
                    let z = f32::from_le_bytes(bytes[offset+12..offset+16].try_into().unwrap());
                    translation_keys.push((time, crate::math::vec::Vec3::new(x, y, z)));
                    offset += 16;
                }

                let mut rotation_keys = Vec::new();
                for _ in 0..r_count {
                    if offset + 20 > bytes.len() { return None; }
                    let time = f32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap());
                    let x = f32::from_le_bytes(bytes[offset+4..offset+8].try_into().unwrap());
                    let y = f32::from_le_bytes(bytes[offset+8..offset+12].try_into().unwrap());
                    let z = f32::from_le_bytes(bytes[offset+12..offset+16].try_into().unwrap());
                    let w = f32::from_le_bytes(bytes[offset+16..offset+20].try_into().unwrap());
                    rotation_keys.push((time, [x, y, z, w]));
                    offset += 20;
                }

                let mut scale_keys = Vec::new();
                for _ in 0..s_count {
                    if offset + 16 > bytes.len() { return None; }
                    let time = f32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap());
                    let x = f32::from_le_bytes(bytes[offset+4..offset+8].try_into().unwrap());
                    let y = f32::from_le_bytes(bytes[offset+8..offset+12].try_into().unwrap());
                    let z = f32::from_le_bytes(bytes[offset+12..offset+16].try_into().unwrap());
                    scale_keys.push((time, crate::math::vec::Vec3::new(x, y, z)));
                    offset += 16;
                }

                channels.push(crate::renderer::vulkan::skeleton::BoneChannel {
                    bone_index: bone_index as usize,
                    translation_keys,
                    rotation_keys,
                    scale_keys,
                });
            }

            let _clip_index = self.animation_clips.len() as u32;

            let mut gpu_clip = crate::renderer::vulkan::animation::GpuClip::default();
            gpu_clip.duration = duration;
            
            let mut gpu_keyframes = Vec::new();
            
            for ch in &channels {
                let bidx = ch.bone_index;
                let t_start = gpu_keyframes.len() as u32;
                for (t, v) in &ch.translation_keys {
                    gpu_keyframes.push(crate::renderer::vulkan::animation::GpuKeyframe {
                        time: *t,
                        value: [v.x, v.y, v.z, 0.0],
                    });
                }
                let t_count = (gpu_keyframes.len() as u32) - t_start;

                let r_start = gpu_keyframes.len() as u32;
                for (t, q) in &ch.rotation_keys {
                    gpu_keyframes.push(crate::renderer::vulkan::animation::GpuKeyframe {
                        time: *t,
                        value: *q,
                    });
                }
                let r_count = (gpu_keyframes.len() as u32) - r_start;

                let s_start = gpu_keyframes.len() as u32;
                for (t, v) in &ch.scale_keys {
                    gpu_keyframes.push(crate::renderer::vulkan::animation::GpuKeyframe {
                        time: *t,
                        value: [v.x, v.y, v.z, 0.0],
                    });
                }
                let s_count = (gpu_keyframes.len() as u32) - s_start;

                gpu_clip.channels[bidx] = [t_start, t_count, r_start, r_count];
                gpu_clip.scale_channels[bidx] = [s_start, s_count];
            }

            if !gpu_keyframes.is_empty() {
                let base_offset = animation_pool.append_keyframes(vulkan, &gpu_keyframes)?;
                for ch in &channels {
                    let bidx = ch.bone_index;
                    if gpu_clip.channels[bidx][1] > 0 {
                        gpu_clip.channels[bidx][0] += base_offset;
                    }
                    if gpu_clip.channels[bidx][3] > 0 {
                        gpu_clip.channels[bidx][2] += base_offset;
                    }
                    if gpu_clip.scale_channels[bidx][1] > 0 {
                        gpu_clip.scale_channels[bidx][0] += base_offset;
                    }
                }
            }
            
            let gpu_clip_idx = animation_pool.append_clip(vulkan, &gpu_clip)?;

            self.animation_clips.push(crate::renderer::vulkan::skeleton::AnimationClip {
                name: cname.clone(),
                duration,
                gpu_index: gpu_clip_idx,
                channels,
            });
        }
        
        Some(())
    }

    /// Load and compile a Rhai script.
    pub fn load_script(&mut self, engine: &ScriptEngine, name: &str, path: &str) -> Option<&AST> {
        if !self.scripts.contains_key(name) {
            if let Ok(source) = std::fs::read_to_string(path) {
                match engine.compile(&source) {
                    Ok(ast) => {
                        self.scripts.insert(name.to_string(), ast);
                        self.script_paths.insert(path.to_string(), name.to_string());
                    }
                    Err(e) => {
                        crate::log_info!("Failed to compile script {}: {}", path, e);
                        return None;
                    }
                }
            } else {
                crate::log_info!("Failed to read script file: {}", path);
                return None;
            }
        }
        self.scripts.get(name)
    }

    /// Get a compiled script AST by name.
    pub fn get_script_ast(&self, name: &str) -> Option<&AST> {
        self.scripts.get(name)
    }

    pub fn poll_changes(&mut self, vulkan: &VulkanDevice) -> Vec<AssetEvent> {
        let mut events = Vec::new();
        if let Some(rx) = &self.rx {
            while let Ok(Ok(event)) = rx.try_recv() {
                if let notify::EventKind::Modify(_) = event.kind {
                    for path in event.paths {
                        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                            let path_str = path.to_string_lossy().replace('\\', "/");

                            if ext == "vert" || ext == "frag" {
                                if crate::utils::shader_compiler::compile_shader(&path).is_ok() {
                                    events.push(AssetEvent::ShaderChanged);
                                }
                            } else if ext == "png" || ext == "jpg" || ext == "hdr" {
                                let mut matched_name = None;
                                for (p, name) in &self.texture_paths {
                                    if path_str.ends_with(p) {
                                        matched_name = Some((name.clone(), p.clone()));
                                        break;
                                    }
                                }
                                if let Some((name, p)) = matched_name {
                                    crate::log_info!("Hot-reloading texture: {}", p);
                                    if ext == "hdr" {
                                        if let Some(tex) = Texture::load_hdr(vulkan, &p) {
                                            if let Some(mut old) =
                                                self.textures.insert(name.clone(), tex)
                                            {
                                                unsafe {
                                                    vulkan.device.device_wait_idle().unwrap()
                                                };
                                                old.shutdown(vulkan);
                                            }
                                            events.push(AssetEvent::TextureChanged(name));
                                        }
                                    } else {
                                        if let Some(tex) = Texture::load_from_file(vulkan, &p) {
                                            if let Some(mut old) =
                                                self.textures.insert(name.clone(), tex)
                                            {
                                                unsafe {
                                                    vulkan.device.device_wait_idle().unwrap()
                                                };
                                                old.shutdown(vulkan);
                                            }
                                            events.push(AssetEvent::TextureChanged(name));
                                        }
                                    }
                                }
                            } else if ext == "mesh" || ext == "mat" {
                                let mut matched_name = None;
                                for (p, name) in &self.model_paths {
                                    if path_str.ends_with(p) {
                                        matched_name = Some((name.clone(), p.clone()));
                                        break;
                                    }
                                }
                                if let Some((name, p)) = matched_name {
                                    crate::log_info!("Hot-reloading cooked asset: {}", p);
                                    // The main application loop will catch this and can reload via GeometryPool
                                    events.push(AssetEvent::ModelChanged(name));
                                }
                            } else if ext == "obj" {
                                let mut matched_path = None;
                                for p in self.model_paths.keys() {
                                    if path_str.ends_with(p) {
                                        matched_path = Some(p.clone());
                                        break;
                                    }
                                }
                                if let Some(_p) = matched_path {
                                    // Model hot-reload is disabled — GeometryPool does not
                                    // support hole-punching/reallocation for individual meshes.
                                    // Models must be reloaded via the editor SpawnModel action
                                    // which waits for device idle before appending.
                                }
                            } else if ext == "rhai" {
                                let mut matched_name = None;
                                for (p, name) in &self.script_paths {
                                    if path_str.ends_with(p) {
                                        matched_name = Some((name.clone(), p.clone()));
                                        break;
                                    }
                                }
                                if let Some((name, p)) = matched_name {
                                    crate::log_info!("Hot-reloading script: {}", p);
                                    // Notice: Since we don't have the engine here, we just remove the AST so it will reload next time.
                                    // In a real scenario we might recompile it here if we stored the Engine.
                                    self.scripts.remove(&name);
                                }
                            }
                        }
                    }
                }
            }
        }
        events
    }

    pub fn get_mesh(&self, index: usize) -> Option<&Mesh> {
        self.meshes.get(index)
    }

    pub fn get_mesh_mut(&mut self, index: usize) -> Option<&mut Mesh> {
        self.meshes.get_mut(index)
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        for (_, tex) in self.textures.iter_mut() {
            tex.shutdown(vulkan);
        }
        self.textures.clear();

        for mesh in self.meshes.iter_mut() {
            mesh.shutdown(vulkan);
        }
        self.meshes.clear();
        self.model_map.clear();
    }
}
