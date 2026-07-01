use crate::renderer::vulkan::{Mesh, Texture, VulkanDevice};
use crate::vfs::Vfs;
use std::collections::HashMap;
use std::path::Path;
use notify::{Watcher, RecursiveMode};

pub enum AssetEvent {
    ShaderChanged,
    TextureChanged(String), // Name of the texture
    ModelChanged(String),   // Name of the model
}

pub struct AssetManager {
    textures: HashMap<String, Texture>,
    meshes: Vec<Mesh>,
    model_map: HashMap<String, Vec<usize>>,
    
    // File to Name reverse mappings to know which asset changed
    texture_paths: HashMap<String, String>, 
    model_paths: HashMap<String, String>,

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
            meshes: Vec::new(),
            model_map: HashMap::new(),
            texture_paths: HashMap::new(),
            model_paths: HashMap::new(),
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

    pub fn load_texture(
        &mut self,
        vulkan: &VulkanDevice,
        name: &str,
        path: &str,
    ) -> Option<&Texture> {
        if !self.textures.contains_key(name) {
            if let Some(tex) = Texture::load_from_file(vulkan, path) {
                self.textures.insert(name.to_string(), tex);
                self.texture_paths.insert(path.to_string(), name.to_string());
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
                self.textures.insert(name.to_string(), tex);
                self.texture_paths.insert(path.to_string(), name.to_string());
            } else {
                crate::log_info!("Failed to load HDR texture: {}", path);
                return None;
            }
        }
        self.textures.get(name)
    }

    /// Provides a fallback checkerboard texture if requested
    pub fn load_checkerboard(&mut self, vulkan: &VulkanDevice, name: &str) -> Option<&Texture> {
        if !self.textures.contains_key(name) {
            if let Some(tex) = Texture::new_checkerboard(vulkan) {
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

    /// Loads an OBJ file (potentially with multiple sub-meshes). Returns a list of indices into the internal mesh array.
    pub fn load_model(&mut self, vulkan: &VulkanDevice, path: &str) -> Option<&[usize]> {
        if !self.model_map.contains_key(path) {
            if let Some(loaded_meshes) = Mesh::load_models(path, vulkan) {
                let mut indices = Vec::with_capacity(loaded_meshes.len());
                for mesh in loaded_meshes {
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
                                            if let Some(mut old) = self.textures.insert(name.clone(), tex) {
                                                unsafe { vulkan.device.device_wait_idle().unwrap() };
                                                old.shutdown(vulkan);
                                            }
                                            events.push(AssetEvent::TextureChanged(name));
                                        }
                                    } else {
                                        if let Some(tex) = Texture::load_from_file(vulkan, &p) {
                                            if let Some(mut old) = self.textures.insert(name.clone(), tex) {
                                                unsafe { vulkan.device.device_wait_idle().unwrap() };
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
                                if let Some(p) = matched_path {
                                    crate::log_info!("Hot-reloading model: {}", p);
                                    if let Some(mut loaded_meshes) = Mesh::load_models(&p, vulkan) {
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
                                            for (_i, idx) in indices.iter().enumerate() {
                                                if let Some(new_mesh) = loaded_meshes.pop() {
                                                    let mut old_mesh = std::mem::replace(&mut self.meshes[*idx], new_mesh);
                                                    old_mesh.shutdown(vulkan);
                                                }
                                            }
                                            events.push(AssetEvent::ModelChanged(p));
                                        }
                                    }
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
