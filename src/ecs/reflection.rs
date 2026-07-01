use crate::ecs::{types::get_component_type_id, types::ComponentTypeId, EntityId, World};
use crate::physics::PhysicsSystem;
use std::any::TypeId;

pub trait ReflectComponent: 'static {
    fn name() -> &'static str;
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool;
}

type EditorDrawFn = Box<dyn Fn(EntityId, &mut World, &mut egui::Ui, &mut PhysicsSystem)>;

pub struct ComponentRegistry {
    draw_fns: std::collections::HashMap<ComponentTypeId, EditorDrawFn>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self {
            draw_fns: std::collections::HashMap::new(),
        }
    }

    pub fn register<T: ReflectComponent>(&mut self) {
        let type_id = get_component_type_id::<T>();

        let draw_fn = Box::new(
            |entity: EntityId,
             world: &mut World,
             ui: &mut egui::Ui,
             physics: &mut PhysicsSystem| {
                let mut changed = false;
                let mut new_pos = crate::math::vec::Vec3::default();
                let mut new_rot = crate::math::vec::Vec3::default();
                {
                    let arrays = world.get_component_array_mut::<T>();
                    if arrays.has(entity) {
                        let comp = unsafe { arrays.get_mut(entity) };
                        ui.collapsing(T::name(), |ui| {
                            changed = comp.draw_inspector(ui);
                        });

                        if changed && TypeId::of::<T>() == TypeId::of::<crate::ecs::TransformComponent>() {
                            let ptr = comp as *const T as *const crate::ecs::TransformComponent;
                            unsafe {
                                new_pos = (*ptr).position;
                                new_rot = (*ptr).rotation;
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
    }

    pub fn draw_entity(
        &self,
        entity: EntityId,
        world: &mut World,
        ui: &mut egui::Ui,
        physics: &mut PhysicsSystem,
    ) {
        for draw_fn in self.draw_fns.values() {
            draw_fn(entity, world, ui, physics);
        }
    }
}
