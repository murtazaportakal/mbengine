use engine::ecs::{World, RenderComponent, HierarchyComponent, TransformComponent, System, ComponentMask};
use engine::ecs::types::{build_mask, get_component_type_id};
use engine::ecs::scheduler::Scheduler;
use engine::physics::PhysicsSystem;
use std::collections::HashSet;

struct SpinSystem;

impl System for SpinSystem {
    fn update(&mut self, dt: f32, world: &World) {
        let render_entities: HashSet<u32> = {
            let renders = world.get_component_array::<RenderComponent>();
            renders.dense_entities_slice().iter().copied().collect()
        };
        let hierarchy_roots: HashSet<u32> = {
            let hier = world.get_component_array::<HierarchyComponent>();
            hier.dense_entities_slice().iter().copied().collect()
        };

        let transforms = unsafe { world.get_component_array_mut_unchecked::<TransformComponent>() };
        let entities = transforms.dense_entities_slice().to_vec();

        for (i, transform) in transforms.as_mut_slice().iter_mut().enumerate() {
            let entity = entities[i];

            if render_entities.contains(&entity) {
                if !hierarchy_roots.contains(&entity) {
                    // Root entity (planet): slow spin
                    transform.rotation.y += 1.0 * dt;
                } else {
                    // Child entities: spin faster
                    transform.rotation.y += 2.5 * dt;
                }
            }
        }
    }

    fn read_components(&self) -> ComponentMask {
        build_mask(&[
            get_component_type_id::<RenderComponent>(),
            get_component_type_id::<HierarchyComponent>(),
        ])
    }

    fn write_components(&self) -> ComponentMask {
        build_mask(&[get_component_type_id::<TransformComponent>()])
    }
}

// Ensure the Scheduler is reused to avoid rebuilding the graph every frame.
static mut SCHEDULER: Option<Scheduler> = None;

#[no_mangle]
pub extern "C" fn game_update(world: &mut World, physics: &mut PhysicsSystem, dt: f32) {
    // 1. Step the Physics Simulation
    physics.update(dt, world);

    // 2. Custom Game Logic using Job System
    let scheduler = unsafe {
        if SCHEDULER.is_none() {
            let mut s = Scheduler::new();
            s.add_system(Box::new(SpinSystem));
            s.build_graph();
            SCHEDULER = Some(s);
        }
        SCHEDULER.as_mut().unwrap()
    };

    scheduler.execute(world, dt);
}
