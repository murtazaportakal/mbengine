use crate::asset_manager::AssetManager;
use crate::ecs::{
    components::{ScriptBehaviorComponent, TransformComponent},
    World,
};
use crate::scripting::engine::ScriptEngine;

pub fn process_scripts(
    world: &World,
    asset_manager: &AssetManager,
    script_engine: &ScriptEngine,
    physics: &crate::physics::PhysicsSystem,
    dt: f32,
) {
    let script_components_mut =
        unsafe { &mut *world.get_component_array_mut_ptr::<ScriptBehaviorComponent>() };
    let transforms_mut = unsafe { &mut *world.get_component_array_mut_ptr::<TransformComponent>() };

    let entities = script_components_mut.dense_entities();

    // Process physics trigger events
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

            // Call the `update` function in the script
            // The engine will execute it and modify `transform_comp` and `script_comp.state` in-place
            script_engine.call_update(ast, &mut script_comp.state, transform_comp, dt);
        }
    }
}
