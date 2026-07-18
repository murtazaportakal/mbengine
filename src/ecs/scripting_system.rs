use crate::asset_manager::AssetManager;
use crate::ecs::{
    components::{ScriptBehaviorComponent, TransformComponent},
    World,
};
use crate::scripting::engine::ScriptEngine;

/// Process all entities with a `ScriptBehaviorComponent` each game frame.
///
/// For each entity the system:
/// 1. Resolves its compiled AST from the asset cache.
/// 2. Calls `init` on first run to obtain the initial state map.
/// 3. Calls `update(dt)` every frame, passing the entity's `TransformComponent`
///    **and** a `RhaiWorld` handle so scripts can spawn/destroy/query entities.
/// 4. Fires `on_trigger_enter` / `on_trigger_exit` for any pending physics events.
pub fn process_scripts(
    world: &mut World,
    asset_manager: &AssetManager,
    script_engine: &ScriptEngine,
    physics: &crate::physics::PhysicsSystem,
    dt: f32,
) {
    let script_components_mut =
        unsafe { &mut *world.get_component_array_mut_ptr::<ScriptBehaviorComponent>() };
    let transforms_mut =
        unsafe { &mut *world.get_component_array_mut_ptr::<TransformComponent>() };

    let entities = script_components_mut.dense_entities();

    // ── Physics trigger events ────────────────────────────────────────────────
    for event in &physics.trigger_events {
        // Fire for entity1 if it has a script
        if script_components_mut.has(event.entity1) {
            let script_comp = unsafe { script_components_mut.get_mut(event.entity1) };
            if let Some(ast) = asset_manager.get_script_ast(script_comp.script_name.as_str()) {
                if event.started {
                    script_engine.call_on_trigger_enter(ast, &mut script_comp.state, event.entity2);
                } else {
                    script_engine.call_on_trigger_exit(ast, &mut script_comp.state, event.entity2);
                }
            }
        }

        // Fire for entity2 if it has a script
        if script_components_mut.has(event.entity2) {
            let script_comp = unsafe { script_components_mut.get_mut(event.entity2) };
            if let Some(ast) = asset_manager.get_script_ast(script_comp.script_name.as_str()) {
                if event.started {
                    script_engine.call_on_trigger_enter(ast, &mut script_comp.state, event.entity1);
                } else {
                    script_engine.call_on_trigger_exit(ast, &mut script_comp.state, event.entity1);
                }
            }
        }
    }

    // ── Per-entity update ─────────────────────────────────────────────────────
    for (i, script_comp) in script_components_mut.as_mut_slice().iter_mut().enumerate() {
        let entity = unsafe { *entities.add(i) };

        if !transforms_mut.has(entity) {
            continue;
        }

        let transform_comp = unsafe { transforms_mut.get_mut(entity) };
        let script_name = script_comp.script_name.as_str();

        if let Some(ast) = asset_manager.get_script_ast(script_name) {
            if !script_comp.initialized {
                script_comp.state = rhai::Dynamic::from(script_engine.call_init(ast));
                script_comp.initialized = true;
            }

            // Call `update(dt)` with full World access so scripts can spawn/destroy/query.
            //
            // SAFETY: `world` is a shared reference so we alias it here as mutable
            // for the Rhai call.  The Scheduler guarantees this system is the only
            // active accessor of ScriptBehaviorComponent and TransformComponent at
            // this point in the frame.  No other system holds a mutable borrow of
            // any component array during this call.
            script_engine.call_update_world(
                ast,
                &mut script_comp.state,
                transform_comp,
                world,
                dt,
            );
        }
    }
}
