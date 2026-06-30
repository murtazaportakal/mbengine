use crate::renderer::vulkan::{Mesh, Texture, VulkanDevice};
use std::collections::HashMap;

pub struct AssetManager {
    textures: HashMap<String, Texture>,
    meshes: Vec<Mesh>,
    model_map: HashMap<String, Vec<usize>>,
}

impl Default for AssetManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetManager {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            meshes: Vec::new(),
            model_map: HashMap::new(),
        }
    }

    /// Loads a texture from file and caches it. Returns a reference to it.
    pub fn load_texture(
        &mut self,
        vulkan: &VulkanDevice,
        name: &str,
        path: &str,
    ) -> Option<&Texture> {
        if !self.textures.contains_key(name) {
            if let Some(tex) = Texture::load_from_file(vulkan, path) {
                self.textures.insert(name.to_string(), tex);
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
            } else {
                crate::log_info!("Failed to load model: {}", path);
                return None;
            }
        }
        self.model_map.get(path).map(|v| v.as_slice())
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
