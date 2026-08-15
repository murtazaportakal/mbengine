use crate::ecs::components::TransformComponent;
use crate::ecs::World;
use crate::math::vec::Vec3;
use rhai::{Engine, Scope, AST};

// ── RhaiWorld wrapper ────────────────────────────────────────────────────────
//
// A thin, Rhai-registerable wrapper around a raw `*mut World`.  It is only
// valid for the lifetime of a single `call_update` invocation; it must never
// outlive the borrow of the real `World`.  Because Rhai clones Dynamic values,
// the pointer field uses a `usize` cast so Clone can be derived cheaply.
//
// # Safety
// All methods on `RhaiWorld` are internally `unsafe` because they dereference
// the raw pointer.  The caller (`ScriptEngine::call_update_world`) guarantees
// that the pointer remains valid for the entire call.

/// A Rhai-safe handle to the engine's ECS World.
///
/// Exposes entity lifecycle and component queries to game scripts.
#[derive(Clone)]
pub struct RhaiWorld {
    /// Raw pointer stored as usize so Clone can be derived without special-casing.
    world_ptr: usize,
}

impl RhaiWorld {
    /// Create a wrapper from a mutable reference.
    ///
    /// # Safety
    /// The world reference must remain valid for every Rhai function call made
    /// through this wrapper within the enclosing `call_update_world` call.
    pub unsafe fn from_world(world: &mut World) -> Self {
        Self {
            world_ptr: world as *mut World as usize,
        }
    }

    /// Get a mutable reference back to the underlying World.
    ///
    /// # Safety
    /// Must only be called while the pointer is still valid.
    unsafe fn world(&self) -> &mut World {
        &mut *(self.world_ptr as *mut World)
    }

    // ── Entity lifecycle ─────────────────────────────────────────────────────

    /// Create a new entity and return its ID as i64.
    pub fn create_entity(&mut self) -> i64 {
        unsafe { self.world().create_entity() as i64 }
    }

    /// Destroy an entity by its ID.  No-op if the entity is not alive.
    pub fn destroy_entity(&mut self, id: i64) {
        let world = unsafe { self.world() };
        let eid = id as crate::ecs::EntityId;
        if world.is_alive(eid) {
            world.destroy_entity(eid);
        }
    }

    /// Return true if the entity is alive.
    pub fn is_alive(&self, id: i64) -> bool {
        unsafe { self.world().is_alive(id as crate::ecs::EntityId) }
    }

    // ── Entity names ─────────────────────────────────────────────────────────

    pub fn set_entity_name(&mut self, id: i64, name: &str) {
        let world = unsafe { self.world() };
        world.set_entity_name(id as crate::ecs::EntityId, name);
    }

    pub fn get_entity_name(&self, id: i64) -> String {
        let world = unsafe { self.world() };
        world
            .get_entity_name(id as crate::ecs::EntityId)
            .unwrap_or("")
            .to_string()
    }

    pub fn find_entity(&self, name: &str) -> i64 {
        let world = unsafe { self.world() };
        world
            .find_entity_by_name(name)
            .map(|id| id as i64)
            .unwrap_or(-1)
    }

    // ── TransformComponent queries ────────────────────────────────────────────

    /// Return true if the entity has a TransformComponent.
    pub fn has_transform(&self, id: i64) -> bool {
        let world = unsafe { self.world() };
        let eid = id as crate::ecs::EntityId;
        if !world.is_alive(eid) {
            return false;
        }
        world.get_component_array::<TransformComponent>().has(crate::ecs::get_entity_index(eid))
    }

    /// Get the TransformComponent for an entity.
    /// Returns a zero transform if the entity has no transform or is dead.
    pub fn get_transform(&self, id: i64) -> TransformComponent {
        let world = unsafe { self.world() };
        let eid = id as crate::ecs::EntityId;
        if !world.is_alive(eid) {
            return TransformComponent::default();
        }
        let arr = world.get_component_array::<TransformComponent>();
        let idx = crate::ecs::get_entity_index(eid);
        if arr.has(idx) {
            unsafe { *arr.get(idx) }
        } else {
            TransformComponent::default()
        }
    }

    /// Set the TransformComponent for an entity (no-op if entity lacks one).
    pub fn set_transform(&mut self, id: i64, t: TransformComponent) {
        let world = unsafe { self.world() };
        let eid = id as crate::ecs::EntityId;
        if !world.is_alive(eid) {
            return;
        }
        let arr = world.get_component_array_mut::<TransformComponent>();
        let idx = crate::ecs::get_entity_index(eid);
        if arr.has(idx) {
            *unsafe { arr.get_mut(idx) } = t;
        }
    }

    /// Return an array of entity IDs (as i64) that have a TransformComponent.
    pub fn query_with_transform(&self) -> rhai::Array {
        let world = unsafe { self.world() };
        let arr = world.get_component_array::<TransformComponent>();
        arr.dense_entities_slice()
            .iter()
            .map(|&e| rhai::Dynamic::from(e as i64))
            .collect()
    }
}

// ── ScriptEngine ──────────────────────────────────────────────────────────────

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

        // ── 1. Register Math Types ────────────────────────────────────────────
        engine
            .register_type_with_name::<Vec3>("Vec3")
            .register_fn("vec3", Vec3::new)
            .register_get_set("x", |v: &mut Vec3| v.x, |v: &mut Vec3, val| v.x = val)
            .register_get_set("y", |v: &mut Vec3| v.y, |v: &mut Vec3, val| v.y = val)
            .register_get_set("z", |v: &mut Vec3| v.z, |v: &mut Vec3, val| v.z = val)
            .register_fn("+", |a: Vec3, b: Vec3| Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z))
            .register_fn("-", |a: Vec3, b: Vec3| Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z))
            .register_fn("*", |a: Vec3, scalar: f32| {
                Vec3::new(a.x * scalar, a.y * scalar, a.z * scalar)
            });

        // ── 2. Register TransformComponent ────────────────────────────────────
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

        // ── 3. Register RhaiWorld (entity lifecycle + queries) ─────────────────
        //
        // The world is passed *per call* via the scope (not stored permanently
        // in the engine) so scripts always operate on the current frame's state.
        engine
            .register_type_with_name::<RhaiWorld>("World")
            // Entity lifecycle
            .register_fn("create_entity",   RhaiWorld::create_entity)
            .register_fn("destroy_entity",  RhaiWorld::destroy_entity)
            .register_fn("is_alive",        RhaiWorld::is_alive)
            .register_fn("set_entity_name", RhaiWorld::set_entity_name)
            .register_fn("get_entity_name", RhaiWorld::get_entity_name)
            .register_fn("find_entity",     RhaiWorld::find_entity)
            // TransformComponent access
            .register_fn("has_transform",   RhaiWorld::has_transform)
            .register_fn("get_transform",   RhaiWorld::get_transform)
            .register_fn("set_transform",   RhaiWorld::set_transform)
            // Bulk queries
            .register_fn("query_with_transform", RhaiWorld::query_with_transform);

        // Limit maximum operations to prevent infinite loops hanging the game engine.
        engine.set_max_operations(5000);

        Self { engine }
    }

    /// Compile a script string into an AST.
    pub fn compile(&self, script: &str) -> Result<AST, rhai::ParseError> {
        self.engine.compile(script)
    }

    /// Execute the `init` function in the script.  Returns the initial state map.
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

    /// Execute the `update` function.
    ///
    /// Passes `state`, `transform`, and `dt` as before.  The world is NOT
    /// injected here — use [`call_update_world`] when entity queries are needed.
    pub fn call_update(
        &self,
        ast: &AST,
        state: &mut rhai::Dynamic,
        transform: &mut TransformComponent,
        dt: f32,
    ) {
        let mut scope = Scope::new();

        scope.push_dynamic("state", std::mem::take(state));
        scope.push_dynamic("transform", rhai::Dynamic::from(*transform));

        let result: Result<(), Box<rhai::EvalAltResult>> =
            self.engine.call_fn(&mut scope, ast, "update", (dt,));

        if let Some(state_ref) = scope.get_mut("state") {
            *state = std::mem::take(state_ref);
        }

        if let Some(new_t) = scope.get_value::<TransformComponent>("transform") {
            *transform = new_t;
        }

        if let Err(e) = result {
            if !matches!(*e, rhai::EvalAltResult::ErrorFunctionNotFound(..)) {
                crate::log_info!("Script Error: {}", e);
            }
        }
    }

    /// Execute the `update` function, also injecting the World for entity queries.
    ///
    /// Scripts receive a `world: World` variable in scope and can call:
    /// - `world.create_entity()` → i64
    /// - `world.destroy_entity(id)`
    /// - `world.is_alive(id)` → bool
    /// - `world.set_entity_name(id, name)`
    /// - `world.get_entity_name(id)` → String
    /// - `world.find_entity(name)` → i64 (or -1)
    /// - `world.has_transform(id)` → bool
    /// - `world.get_transform(id)` → Transform
    /// - `world.set_transform(id, t)`
    /// - `world.query_with_transform()` → Array of i64
    ///
    /// # Safety
    /// The `world_ref` pointer is valid for the duration of this call only.
    pub fn call_update_world(
        &self,
        ast: &AST,
        state: &mut rhai::Dynamic,
        transform: &mut TransformComponent,
        world_ref: &mut World,
        dt: f32,
    ) {
        let mut scope = Scope::new();

        scope.push_dynamic("state", std::mem::take(state));
        scope.push_dynamic("transform", rhai::Dynamic::from(*transform));

        // SAFETY: RhaiWorld is only valid for this call's scope lifetime.
        let rhai_world = unsafe { RhaiWorld::from_world(world_ref) };
        scope.push("world", rhai_world);

        let result: Result<(), Box<rhai::EvalAltResult>> =
            self.engine.call_fn(&mut scope, ast, "update", (dt,));

        if let Some(state_ref) = scope.get_mut("state") {
            *state = std::mem::take(state_ref);
        }

        if let Some(new_t) = scope.get_value::<TransformComponent>("transform") {
            *transform = new_t;
        }

        if let Err(e) = result {
            if !matches!(*e, rhai::EvalAltResult::ErrorFunctionNotFound(..)) {
                crate::log_info!("Script Error: {}", e);
            }
        }
    }

    /// Fire the `on_trigger_enter` callback.
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

    /// Fire the `on_trigger_exit` callback.
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
