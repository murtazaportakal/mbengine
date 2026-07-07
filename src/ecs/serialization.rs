use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

use crate::ecs::components::TransformComponent;
use crate::ecs::world::World;

#[derive(Serialize, Deserialize, Default)]
pub struct SerializedEntity {
    pub id: u32,
    pub components: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
pub struct Scene {
    pub entities: Vec<SerializedEntity>,
}

pub fn save_scene(world: &World, registry: &crate::ecs::reflection::ComponentRegistry, file_path: &str) {
    let mut scene = Scene {
        entities: Vec::new(),
    };

    // For simplicity, we iterate over all active entity indices using the TransformComponent array
    // as a heuristic for "entities in the scene", or we can just iterate up to MAX_ENTITIES.
    // Ideally we would iterate over `world.entity_manager.dense_entities_slice()`.
    let transforms = world.get_component_array::<TransformComponent>();
    let entities = transforms.dense_entities_slice();

    for &entity_index in entities {
        let components_val = registry.serialize_entity(entity_index, world);
        
        // Only save if it has components (it should)
        if let serde_json::Value::Object(ref map) = components_val {
            if !map.is_empty() {
                let s_ent = SerializedEntity {
                    id: entity_index,
                    components: components_val,
                };
                scene.entities.push(s_ent);
            }
        }
    }

    if let Ok(json) = serde_json::to_string_pretty(&scene) {
        if let Ok(mut file) = File::create(file_path) {
            let _ = file.write_all(json.as_bytes());
            crate::log_info!("Scene saved to {}", file_path);
        } else {
            crate::log_info!("Failed to create file: {}", file_path);
        }
    } else {
        crate::log_info!("Failed to serialize scene to JSON.");
    }
}

pub fn load_scene(world: &mut World, registry: &crate::ecs::reflection::ComponentRegistry, file_path: &str) {
    if let Ok(mut file) = File::open(file_path) {
        let mut json = String::new();
        if file.read_to_string(&mut json).is_ok() {
            if let Ok(scene) = serde_json::from_str::<Scene>(&json) {
                crate::log_info!(
                    "Loaded {} entities from {}",
                    scene.entities.len(),
                    file_path
                );

                for s_ent in scene.entities {
                    let new_id = world.create_entity();
                    registry.deserialize_entity(new_id, world, &s_ent.components);
                }
            } else {
                crate::log_info!("Failed to parse JSON scene.");
            }
        }
    } else {
        crate::log_info!("Failed to open scene file: {}", file_path);
    }
}
