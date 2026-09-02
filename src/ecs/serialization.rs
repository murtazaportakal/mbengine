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

pub fn serialize_scene(
    world: &World,
    registry: &crate::ecs::reflection::ComponentRegistry,
) -> Option<String> {
    let mut scene = Scene {
        entities: Vec::new(),
    };

    let transforms = world.get_component_array::<TransformComponent>();
    let entities = transforms.dense_entities_slice();

    for &entity_index in entities {
        if let Some(entity_id) = world.reconstruct_entity(entity_index) {
            let components_val = registry.serialize_entity(entity_id, world);
            if let serde_json::Value::Object(ref map) = components_val {
                if !map.is_empty() {
                    let s_ent = SerializedEntity {
                        id: entity_index, // Keep ID as index for backwards compatibility in JSON
                        components: components_val,
                    };
                    scene.entities.push(s_ent);
                }
            }
        }
    }

    serde_json::to_string(&scene).ok()
}

pub fn save_scene(
    world: &World,
    registry: &crate::ecs::reflection::ComponentRegistry,
    file_path: &str,
) {
    if let Some(json) = serialize_scene(world, registry) {
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

pub fn deserialize_scene(
    world: &mut World,
    registry: &crate::ecs::reflection::ComponentRegistry,
    json: &str,
) {
    if let Ok(scene) = serde_json::from_str::<Scene>(json) {
        crate::log_info!(
            "Restoring {} entities from state",
            scene.entities.len()
        );

        for s_ent in scene.entities {
            let new_id = world.create_entity();
            registry.deserialize_entity(new_id, world, &s_ent.components);
        }
    } else {
        crate::log_info!("Failed to parse JSON scene.");
    }
}

pub fn load_scene(
    world: &mut World,
    registry: &crate::ecs::reflection::ComponentRegistry,
    file_path: &str,
) {
    if let Ok(mut file) = File::open(file_path) {
        let mut json = String::new();
        if file.read_to_string(&mut json).is_ok() {
            deserialize_scene(world, registry, &json);
        }
    } else {
        crate::log_info!("Failed to open scene file: {}", file_path);
    }
}
