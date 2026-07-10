use crate::ecs::components::TransformComponent;
use crate::math::vec::Vec3;
use rhai::{Engine, Scope, AST};

/// Encapsulates the Rhai engine and provides methods for compiling and executing scripts.
pub struct ScriptEngine {
    pub engine: Engine,
}

impl Default for ScriptEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptEngine {
    pub fn new() -> Self {
        let mut engine = Engine::new();

        // 1. Register Math Types
        engine
            .register_type_with_name::<Vec3>("Vec3")
            .register_fn("vec3", Vec3::new)
            .register_get_set("x", |v: &mut Vec3| v.x, |v: &mut Vec3, val| v.x = val)
            .register_get_set("y", |v: &mut Vec3| v.y, |v: &mut Vec3, val| v.y = val)
            .register_get_set("z", |v: &mut Vec3| v.z, |v: &mut Vec3, val| v.z = val)
            .register_fn("+", |a: Vec3, b: Vec3| {
                Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
            })
            .register_fn("-", |a: Vec3, b: Vec3| {
                Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
            })
            .register_fn("*", |a: Vec3, scalar: f32| {
                Vec3::new(a.x * scalar, a.y * scalar, a.z * scalar)
            });

        // 2. Register Transform Component structure for passing back and forth
        engine
            .register_type_with_name::<TransformComponent>("Transform")
            .register_get_set(
                "position",
                |t: &mut TransformComponent| t.position,
                |t: &mut TransformComponent, val: Vec3| t.position = val,
            )
            .register_get_set(
                "rotation",
                |t: &mut TransformComponent| t.rotation,
                |t: &mut TransformComponent, val: Vec3| t.rotation = val,
            )
            .register_get_set(
                "scale",
                |t: &mut TransformComponent| t.scale,
                |t: &mut TransformComponent, val: Vec3| t.scale = val,
            );

        // Limit maximum operations to prevent infinite loops hanging the game engine
        engine.set_max_operations(5000);

        Self { engine }
    }

    /// Compile a script string into an AST
    pub fn compile(&self, script: &str) -> Result<AST, rhai::ParseError> {
        self.engine.compile(script)
    }

    /// Execute the `update` function in the script with the given state and transform.
    /// Execute the `init` function in the script to return the initial state.
    pub fn call_init(&self, ast: &AST) -> rhai::Map {
        let mut scope = Scope::new();
        let result: Result<rhai::Map, Box<rhai::EvalAltResult>> =
            self.engine.call_fn(&mut scope, ast, "init", ());

        match result {
            Ok(map) => map,
            Err(e) => {
                if !matches!(*e, rhai::EvalAltResult::ErrorFunctionNotFound(..)) {
                    crate::log_info!("Script Init Error: {}", e);
                }
                rhai::Map::new()
            }
        }
    }

    /// Execute the `update` function in the script with the given state and transform.
    pub fn call_update(
        &self,
        ast: &AST,
        state: &mut rhai::Dynamic,
        transform: &mut TransformComponent,
        dt: f32,
    ) {
        let mut scope = Scope::new();

        // Take ownership of the state to avoid cloning the map!
        scope.push_dynamic("state", std::mem::take(state));
        scope.push_dynamic("transform", rhai::Dynamic::from(*transform));

        let result: Result<(), Box<rhai::EvalAltResult>> =
            self.engine.call_fn(&mut scope, ast, "update", (dt,));

        // Put the state back into the component
        if let Some(state_ref) = scope.get_mut("state") {
            *state = std::mem::take(state_ref);
        }

        // Sync back the transform
        if let Some(new_t) = scope.get_value::<TransformComponent>("transform") {
            *transform = new_t;
        }

        if let Err(e) = result {
            if !matches!(*e, rhai::EvalAltResult::ErrorFunctionNotFound(..)) {
                crate::log_info!("Script Error: {}", e);
            }
        }
    }
    pub fn call_on_trigger_enter(
        &self,
        ast: &AST,
        state: &mut rhai::Dynamic,
        other_entity: crate::ecs::EntityId,
    ) {
        let mut scope = Scope::new();
        scope.push_dynamic("state", std::mem::take(state));

        let result: Result<(), Box<rhai::EvalAltResult>> =
            self.engine
                .call_fn(&mut scope, ast, "on_trigger_enter", (other_entity as i64,));

        if let Some(state_ref) = scope.get_mut("state") {
            *state = std::mem::take(state_ref);
        }

        if let Err(e) = result {
            if !matches!(*e, rhai::EvalAltResult::ErrorFunctionNotFound(..)) {
                crate::log_info!("Script Error (on_trigger_enter): {}", e);
            }
        }
    }

    pub fn call_on_trigger_exit(
        &self,
        ast: &AST,
        state: &mut rhai::Dynamic,
        other_entity: crate::ecs::EntityId,
    ) {
        let mut scope = Scope::new();
        scope.push_dynamic("state", std::mem::take(state));

        let result: Result<(), Box<rhai::EvalAltResult>> =
            self.engine
                .call_fn(&mut scope, ast, "on_trigger_exit", (other_entity as i64,));

        if let Some(state_ref) = scope.get_mut("state") {
            *state = std::mem::take(state_ref);
        }

        if let Err(e) = result {
            if !matches!(*e, rhai::EvalAltResult::ErrorFunctionNotFound(..)) {
                crate::log_info!("Script Error (on_trigger_exit): {}", e);
            }
        }
    }
}
