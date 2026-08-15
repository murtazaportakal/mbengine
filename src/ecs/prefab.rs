use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PrefabAsset {
    pub name: String,
    pub mesh_path: Option<String>,
    pub material_path: Option<String>,
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: [f32; 3],
    pub rigid_body: bool,
    pub mass: f32,
    pub collider: bool,
}

impl Default for PrefabAsset {
    fn default() -> Self {
        Self {
            name: "New Prefab".to_string(),
            mesh_path: None,
            material_path: None,
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            rigid_body: false,
            mass: 1.0,
            collider: false,
        }
    }
}
