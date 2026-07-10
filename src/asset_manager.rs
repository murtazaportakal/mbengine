use crate::renderer::vulkan::skeleton::{AnimationClip, Skeleton};
use crate::renderer::vulkan::{Mesh, Texture, VulkanDevice};
use crate::scripting::engine::ScriptEngine;
use crate::vfs::Vfs;
use notify::{RecursiveMode, Watcher};
use rhai::AST;
use std::collections::HashMap;
use std::path::Path;

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

    // File to Name reverse mappings to know which asset changed
    texture_paths: HashMap<String, String>,
    model_paths: HashMap<String, String>,

    /// Skeletons loaded from GLTF files, keyed by asset name.
    skeletons: HashMap<String, Skeleton>,
    /// Animation clips loaded from GLTF files, keyed by asset name.
    /// Each GLTF can contain multiple clips.
    animation_clips: HashMap<String, Vec<AnimationClip>>,

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
            texture_paths: HashMap::new(),
            model_paths: HashMap::new(),
            skeletons: HashMap::new(),
            animation_clips: HashMap::new(),
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
            if let Some(tex) = Texture::load_from_file(vulkan, path) {
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

    pub fn load_model(
        &mut self,
        vulkan: &VulkanDevice,
        geometry_pool: &mut crate::renderer::vulkan::GeometryPool,
        path: &str,
    ) -> Option<&[usize]> {
        if !self.model_map.contains_key(path) {
            if let Some(loaded_meshes) = Mesh::load_models(path, vulkan, geometry_pool) {
                let mut indices = Vec::with_capacity(loaded_meshes.len());
                for mut mesh in loaded_meshes {
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
                crate::log_info!("Failed to load model: {}", path);
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
            self.animation_clips
                .insert(name.to_string(), gltf_data.clips);
        }

        self.model_map.get(name).map(|v| v.as_slice())
    }

    /// Get a skeleton by name.
    pub fn get_skeleton(&self, name: &str) -> Option<&Skeleton> {
        self.skeletons.get(name)
    }

    /// Get animation clips for a named asset.
    pub fn get_animation_clips(&self, name: &str) -> Option<&[AnimationClip]> {
        self.animation_clips.get(name).map(|v| v.as_slice())
    }

    /// Find an animation clip by asset name and clip name.
    pub fn get_animation_clip(&self, asset_name: &str, clip_name: &str) -> Option<&AnimationClip> {
        self.animation_clips
            .get(asset_name)?
            .iter()
            .find(|c| c.name == clip_name)
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
                            } else if ext == "obj" {
                                let mut matched_path = None;
                                for p in self.model_paths.keys() {
                                    if path_str.ends_with(p) {
                                        matched_path = Some(p.clone());
                                        break;
                                    }
                                }
                                if let Some(_p) = matched_path {
                                    // Currently disable hot-reloading for models until GeometryPool handles reallocation cleanly
                                    /*
                                    crate::log_info!("Hot-reloading model: {}", p);
                                    if let Some(mut loaded_meshes) = Mesh::load_models(&p, vulkan, /* geometry_pool not easily accessible here yet */) {
                                        if let Some(indices) = self.model_map.get(&p) {
                                            unsafe { vulkan.device.device_wait_idle().unwrap() };
                                            for (i, _idx) in indices.iter().enumerate() {
                                                if i < loaded_meshes.len() {
                                                    // We extract the new mesh out of the loaded_meshes array
                                                    // by swapping with a dummy or using Option, but Mesh has no Default.
                                                    // But we can pop from loaded_meshes since we just created it.
                                                    // Since we iterate forward, let's reverse loaded_meshes so we can pop.
                                                }
                                            }

                                            // Simplest way: reverse loaded_meshes and pop
                                            loaded_meshes.reverse();
                                            for idx in indices.iter() {
                                                if let Some(new_mesh) = loaded_meshes.pop() {
                                                    let mut old_mesh = std::mem::replace(
                                                        &mut self.meshes[*idx],
                                                        new_mesh,
                                                    );
                                                    old_mesh.shutdown(vulkan);
                                                }
                                            }
                                            events.push(AssetEvent::ModelChanged(p));
                                        }
                                    }
                                    */
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
