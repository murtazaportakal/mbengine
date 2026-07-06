use engine::ecs::scheduler::Scheduler;
use engine::ecs::types::{build_mask, get_component_type_id};
use engine::ecs::{
    ComponentMask, HierarchyComponent, RenderComponent, System, TransformComponent, World,
};
use engine::physics::PhysicsSystem;
use std::cell::UnsafeCell;
use std::sync::OnceLock;

struct SpinSystem;

impl System for SpinSystem {
    fn update(&mut self, dt: f32, world: &World) {
        let renders = world.get_component_array::<RenderComponent>();
        let hierarchies = world.get_component_array::<HierarchyComponent>();
        let transforms = unsafe { &mut *world.get_component_array_mut_ptr::<TransformComponent>() };
        let entities = transforms.dense_entities();

        for (i, transform) in transforms.as_mut_slice().iter_mut().enumerate() {
            let entity = unsafe { *entities.add(i) };

            if renders.has(entity) {
                if !hierarchies.has(entity) {
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

// Lock-free scheduler storage. `game_update` is always called from a single thread
// (the engine's main loop), so no synchronization is needed after initialization.
//
// Safety: OnceLock handles the one-time init synchronization. After that,
// the UnsafeCell is only accessed from the main thread.
struct SyncScheduler(UnsafeCell<Scheduler>);
unsafe impl Sync for SyncScheduler {}

static SCHEDULER: OnceLock<SyncScheduler> = OnceLock::new();

#[no_mangle]
pub extern "C" fn game_update(world: &mut World, physics: &mut PhysicsSystem, dt: f32) {
    // 1. Step the Physics Simulation
    physics.update(dt, world);

    // 2. Custom Game Logic using Job System
    let cell = SCHEDULER.get_or_init(|| {
        let mut s = Scheduler::new();
        s.add_system(Box::new(SpinSystem));
        s.build_graph();
        SyncScheduler(UnsafeCell::new(s))
    });

    // Safety: game_update is only called from the engine's main thread.
    let scheduler = unsafe { &mut *cell.0.get() };
    scheduler.execute(world, dt);
}
