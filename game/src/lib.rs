use engine::ecs::{World, RenderComponent, HierarchyComponent, TransformComponent, System};
use engine::physics::PhysicsSystem;
use std::collections::HashSet;

#[no_mangle]
pub extern "C" fn game_update(world: &mut World, physics: &mut PhysicsSystem, dt: f32) {
    println!("game_update called, dt: {}", dt);
    // 1. Step the Physics Simulation
    physics.update(dt, world);

    // 2. Custom Game Logic: Spin entities
    let render_entities: HashSet<u32> = {
        let renders = world.get_component_array::<RenderComponent>();
        renders.dense_entities_slice().iter().copied().collect()
    };
    let hierarchy_roots: HashSet<u32> = {
        let hier = world.get_component_array::<HierarchyComponent>();
        hier.dense_entities_slice().iter().copied().collect()
    };

    let transforms = world.get_component_array_mut::<TransformComponent>();
    let entities = transforms.dense_entities_slice().to_vec();

    for (i, transform) in transforms.as_mut_slice().iter_mut().enumerate() {
        let entity = entities[i];

        if render_entities.contains(&entity) {
            if !hierarchy_roots.contains(&entity) {
                // Root entity (planet): slow spin
                transform.rotation.y += 1.0 * dt;
                println!("Planet {} rotation: {}, dt: {}", entity, transform.rotation.y, dt);
            } else {
                // Child entities: spin faster
                transform.rotation.y += 2.5 * dt;
            }
        }
    }
}
