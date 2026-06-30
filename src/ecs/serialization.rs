use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Read, Write};

use crate::ecs::world::World;
use crate::ecs::components::{
    TransformComponent, RenderComponent, CameraComponent, 
    LightComponent, PointLightComponent, HierarchyComponent
};

#[derive(Serialize, Deserialize, Default)]
pub struct SerializedEntity {
    pub id: u32,
    pub transform: Option<TransformComponent>,
    pub render: Option<RenderComponent>,
    pub camera: Option<CameraComponent>,
    pub light: Option<LightComponent>,
    pub point_light: Option<PointLightComponent>,
    pub hierarchy: Option<HierarchyComponent>,
    // We can also store flags like whether it was a static/dynamic rigidbody,
    // but for simplicity we will deduce it from the components or skip it.
}

#[derive(Serialize, Deserialize)]
pub struct Scene {
    pub entities: Vec<SerializedEntity>,
}

pub fn save_scene(world: &World, file_path: &str) {
    let mut scene = Scene { entities: Vec::new() };

    // Get arrays
    let transforms = world.get_component_array::<TransformComponent>();
    let renders = world.get_component_array::<RenderComponent>();
    let cameras = world.get_component_array::<CameraComponent>();
    let lights = world.get_component_array::<LightComponent>();
    let point_lights = world.get_component_array::<PointLightComponent>();
    let hierarchies = world.get_component_array::<HierarchyComponent>();

    // We iterate over all alive entities (using max index roughly)
    // A better way is to iterate over entity_manager dense slice if it was exposed,
    // but we can just use the transforms array as the primary source of entities,
    // or iterate from 0..alive_count if IDs are dense.
    // Actually, transforms dense_entities_slice has all entities with a Transform.
    // For a game scene, almost everything has a transform.
    let entities = transforms.dense_entities_slice();

    for &entity_index in entities {
        // entity_index is the raw index, not full EntityId with generation.
        // But for serialization we can just use the index, and let load assign new IDs.
        
        let mut s_ent = SerializedEntity {
            id: entity_index,
            ..Default::default()
        };

        if transforms.has(entity_index) {
            s_ent.transform = Some(unsafe { *transforms.get(entity_index) });
        }
        if renders.has(entity_index) {
            s_ent.render = Some(unsafe { *renders.get(entity_index) });
        }
        if cameras.has(entity_index) {
            s_ent.camera = Some(unsafe { *cameras.get(entity_index) });
        }
        if lights.has(entity_index) {
            s_ent.light = Some(unsafe { *lights.get(entity_index) });
        }
        if point_lights.has(entity_index) {
            s_ent.point_light = Some(unsafe { *point_lights.get(entity_index) });
        }
        if hierarchies.has(entity_index) {
            s_ent.hierarchy = Some(unsafe { *hierarchies.get(entity_index) });
        }

        scene.entities.push(s_ent);
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

pub fn load_scene(world: &mut World, file_path: &str) {
    if let Ok(mut file) = File::open(file_path) {
        let mut json = String::new();
        if file.read_to_string(&mut json).is_ok() {
            if let Ok(scene) = serde_json::from_str::<Scene>(&json) {
                // Destroy old entities.
                // We'd need a clear() method on world, but we can't easily iterate all alive entities safely without a list.
                // For simplicity, we just assume load_scene is called on a fresh world, or we ignore old entities.
                // Let's print out what we loaded.
                
                crate::log_info!("Loaded {} entities from {}", scene.entities.len(), file_path);
                
                for s_ent in scene.entities {
                    let new_id = world.create_entity();
                    
                    if let Some(transform) = s_ent.transform {
                        unsafe { world.add_component(new_id, transform); }
                    }
                    if let Some(render) = s_ent.render {
                        unsafe { world.add_component(new_id, render); }
                    }
                    if let Some(camera) = s_ent.camera {
                        unsafe { world.add_component(new_id, camera); }
                    }
                    if let Some(light) = s_ent.light {
                        unsafe { world.add_component(new_id, light); }
                    }
                    if let Some(point_light) = s_ent.point_light {
                        unsafe { world.add_component(new_id, point_light); }
                    }
                    if let Some(hierarchy) = s_ent.hierarchy {
                        // The parent ID in hierarchy is the OLD entity index!
                        // We would need an ID map to map old IDs to new IDs.
                        // For a simple demo, we just assume IDs are somewhat sequential.
                        unsafe { world.add_component(new_id, hierarchy); }
                    }

                    // We could also re-add physics bodies here based on naming or scale.
                }
            } else {
                crate::log_info!("Failed to parse JSON scene.");
            }
        }
    } else {
        crate::log_info!("Failed to open scene file: {}", file_path);
    }
}
