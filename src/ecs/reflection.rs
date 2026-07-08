use crate::ecs::{types::get_component_type_id, types::ComponentTypeId, EntityId, World};
use crate::physics::PhysicsSystem;
use std::any::TypeId;

use crate::ui::UiContext;

pub trait ReflectComponent: 'static {
    fn name() -> &'static str;
    fn draw_inspector(&mut self, ui: &mut UiContext, physics: &mut PhysicsSystem) -> bool;
    fn add_to_entity(entity: EntityId, world: &mut World, physics: &mut PhysicsSystem) where Self: Sized;
}

type EditorDrawFn = Box<dyn Fn(EntityId, &mut World, &mut UiContext, &mut PhysicsSystem)>;
type SerializeFn = Box<dyn Fn(EntityId, &World, &mut serde_json::Map<String, serde_json::Value>)>;
type DeserializeFn = Box<dyn Fn(EntityId, &mut World, &serde_json::Value)>;
type AddComponentFn = Box<dyn Fn(EntityId, &mut World, &mut PhysicsSystem)>;

pub struct ComponentRegistry {
    draw_fns: std::collections::HashMap<ComponentTypeId, EditorDrawFn>,
    serialize_fns: std::collections::HashMap<ComponentTypeId, SerializeFn>,
    deserialize_fns: std::collections::HashMap<ComponentTypeId, DeserializeFn>,
    add_fns: std::collections::HashMap<String, AddComponentFn>,
    pub component_names: Vec<String>,
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self {
            draw_fns: std::collections::HashMap::new(),
            serialize_fns: std::collections::HashMap::new(),
            deserialize_fns: std::collections::HashMap::new(),
            add_fns: std::collections::HashMap::new(),
            component_names: Vec::new(),
        }
    }

    pub fn register<T: ReflectComponent>(&mut self) {
        let type_id = get_component_type_id::<T>();

        let expanded = std::cell::Cell::new(true);
        let draw_fn = Box::new(
            move |entity: EntityId,
             world: &mut World,
             ui: &mut UiContext,
             physics: &mut PhysicsSystem| {
                let mut changed = false;
                let mut new_pos = crate::math::vec::Vec3::default();
                let mut new_rot = crate::math::vec::Vec3::default();
                let mut is_expanded = expanded.get();
                {
                    let arrays = world.get_component_array_mut::<T>();
                    if arrays.has(entity) {
                        if ui.collapsing_header(T::name(), &mut is_expanded) {
                            let comp = unsafe { arrays.get_mut(entity) };
                            // Simplified inspector
                            changed = comp.draw_inspector(ui, physics);

                            if changed
                                && TypeId::of::<T>() == TypeId::of::<crate::ecs::TransformComponent>()
                            {
                                let ptr = comp as *const T as *const crate::ecs::TransformComponent;
                                unsafe {
                                    new_pos = (*ptr).position;
                                    new_rot = (*ptr).rotation;
                                }
                            }
                        }
                    }
                }
                expanded.set(is_expanded);

                if changed && TypeId::of::<T>() == TypeId::of::<crate::ecs::components::RenderComponent>() {
                    let mut render_copy = None;
                    {
                        let renders = world.get_component_array::<crate::ecs::components::RenderComponent>();
                        if renders.has(entity) {
                            render_copy = Some(unsafe { *renders.get(entity) });
                        }
                    }
                    if let Some(r_copy) = render_copy {
                        let hierarchies = world.get_component_array::<crate::ecs::components::HierarchyComponent>();
                        let mut target_parent = None;
                        if hierarchies.has(entity) {
                            let hier = unsafe { hierarchies.get(entity) };
                            // If it has a parent, use that parent (so we update all siblings).
                            // If it doesn't have a parent, it might be the parent itself! So use `entity`.
                            target_parent = Some(hier.parent.unwrap_or(entity));
                        }
                        
                        if let Some(parent) = target_parent {
                            let mut related_to_update = Vec::new();
                            let dense_hier = hierarchies.as_slice();
                            let entities = hierarchies.dense_entities_slice();
                            for (i, hier) in dense_hier.iter().enumerate() {
                                if hier.parent == Some(parent) {
                                    related_to_update.push(entities[i]);
                                }
                            }
                            
                            let renders = world.get_component_array_mut::<crate::ecs::components::RenderComponent>();
                            for child in related_to_update {
                                if child != entity && renders.has(child) {
                                    let child_r = unsafe { renders.get_mut(child) };
                                    child_r.visible = r_copy.visible;
                                    child_r.metallic = r_copy.metallic;
                                    child_r.roughness = r_copy.roughness;
                                    child_r.r = r_copy.r;
                                    child_r.g = r_copy.g;
                                    child_r.b = r_copy.b;
                                }
                            }
                        }
                    }
                }

                if changed && TypeId::of::<T>() == TypeId::of::<crate::ecs::TransformComponent>() {
                    let rb_components =
                        world.get_component_array::<crate::ecs::components::RigidBodyComponent>();
                    if rb_components.has(entity) {
                        let rb_comp = unsafe { rb_components.get(entity) };
                        if let Some(rb) = physics.rigid_body_set.get_mut(rb_comp.handle) {
                            rb.set_translation(
                                rapier3d::math::Vector::new(new_pos.x, new_pos.y, new_pos.z),
                                true,
                            );
                            let quat = rapier3d::math::Rotation::from_euler_angles(
                                new_rot.x, new_rot.y, new_rot.z,
                            );
                            rb.set_rotation(quat, true);
                        }
                    }
                }
            },
        );

        self.draw_fns.insert(type_id, draw_fn);
        
        self.add_fns.insert(T::name().to_string(), Box::new(|entity, world, physics| {
            T::add_to_entity(entity, world, physics);
        }));
        self.component_names.push(T::name().to_string());
        self.component_names.sort();
    }

    pub fn add_component(&self, name: &str, entity: EntityId, world: &mut World, physics: &mut PhysicsSystem) {
        if let Some(add_fn) = self.add_fns.get(name) {
            add_fn(entity, world, physics);
        }
    }

    pub fn draw_entity(
        &self,
        entity: EntityId,
        world: &mut World,
        ui: &mut UiContext,
        physics: &mut PhysicsSystem,
    ) {
        for draw_fn in self.draw_fns.values() {
            draw_fn(entity, world, ui, physics);
        }
    }

    pub fn register_serializable<T: ReflectComponent + serde::Serialize + serde::de::DeserializeOwned>(&mut self) {
        let type_id = get_component_type_id::<T>();
        
        let serialize_fn = Box::new(|entity: EntityId, world: &World, map: &mut serde_json::Map<String, serde_json::Value>| {
            let arrays = world.get_component_array::<T>();
            if arrays.has(entity) {
                let comp = unsafe { arrays.get(entity) };
                if let Ok(val) = serde_json::to_value(comp) {
                    map.insert(T::name().to_string(), val);
                }
            }
        });

        let deserialize_fn = Box::new(|entity: EntityId, world: &mut World, val: &serde_json::Value| {
            if let Some(comp_val) = val.get(T::name()) {
                if let Ok(comp) = serde_json::from_value::<T>(comp_val.clone()) {
                    unsafe { world.add_component(entity, comp); }
                }
            }
        });

        self.serialize_fns.insert(type_id, serialize_fn);
        self.deserialize_fns.insert(type_id, deserialize_fn);
    }

    pub fn serialize_entity(&self, entity: EntityId, world: &World) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for serialize_fn in self.serialize_fns.values() {
            serialize_fn(entity, world, &mut map);
        }
        serde_json::Value::Object(map)
    }

    pub fn deserialize_entity(&self, entity: EntityId, world: &mut World, val: &serde_json::Value) {
        for deserialize_fn in self.deserialize_fns.values() {
            deserialize_fn(entity, world, val);
        }
    }
}
