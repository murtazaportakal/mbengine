//! Foundational types and constants for the Entity Component System.
//!
//! Defines:
//!   - EntityId encoding (20-bit index + 12-bit generation in a u32)
//!   - ComponentTypeId and runtime type → ID mapping
//!   - ComponentMask bitset (u64, max 64 component types)
//!   - Capacity constants and sentinel values

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// ── type aliases ────────────────────────────────────────────────────────────

/// 32-bit entity handle: 20-bit index + 12-bit generation.
pub type EntityId = u32;

/// 20-bit entity slot index.
pub type EntityIndex = u32;

/// 12-bit generation counter (stored as u16 for convenience).
pub type EntityGen = u16;

/// Per-entity component presence bitset. One bit per registered component type.
pub type ComponentMask = u64;

/// Numeric identifier for a component type (assigned at registration time).
pub type ComponentTypeId = u8;

// ── capacity constants ──────────────────────────────────────────────────────

/// Maximum simultaneous live entities. Controls sparse-array sizing.
/// 2^20 = 1,048,576.
pub const MAX_ENTITIES: u32 = 1 << 20;

/// Maximum number of distinct component types that can be registered.
/// Must fit in a ComponentMask (u64), so hard cap is 64.
pub const MAX_COMPONENT_TYPES: u32 = 64;

/// Maximum number of systems that can be registered with a World.
pub const MAX_SYSTEMS: u32 = 64;

// ── bit-field parameters ────────────────────────────────────────────────────

pub const ENTITY_INDEX_BITS: u32 = 20;
pub const ENTITY_GENERATION_BITS: u32 = 12;
pub const ENTITY_INDEX_MASK: u32 = (1 << ENTITY_INDEX_BITS) - 1; // 0x000F_FFFF
pub const ENTITY_GENERATION_MASK: u32 = (1 << ENTITY_GENERATION_BITS) - 1; // 0x0000_0FFF

// ── sentinel values ─────────────────────────────────────────────────────────

/// An EntityId that can never be valid. Used as "null handle".
pub const INVALID_ENTITY: EntityId = !0u32;

/// Sentinel for sparse-array slots that have no dense-array entry.
pub const INVALID_INDEX: u32 = !0u32;

// ── EntityId helpers ────────────────────────────────────────────────────────

/// Pack an index and a generation counter into a single EntityId.
#[inline]
pub const fn make_entity_id(index: EntityIndex, generation: EntityGen) -> EntityId {
    ((generation as u32 & ENTITY_GENERATION_MASK) << ENTITY_INDEX_BITS)
        | (index & ENTITY_INDEX_MASK)
}

/// Extract the 20-bit slot index from an EntityId.
#[inline]
pub const fn get_entity_index(id: EntityId) -> EntityIndex {
    id & ENTITY_INDEX_MASK
}

/// Extract the 12-bit generation counter from an EntityId.
#[inline]
pub const fn get_entity_generation(id: EntityId) -> EntityGen {
    ((id >> ENTITY_INDEX_BITS) & ENTITY_GENERATION_MASK) as EntityGen
}

/// Returns true if the ID is not the sentinel value.
/// Does NOT check liveness — use `EntityManager::is_alive()` for that.
#[inline]
pub const fn is_valid_entity(id: EntityId) -> bool {
    id != INVALID_ENTITY
}

// ── component-type ID assignment ────────────────────────────────────────────
//
// Each unique component type T gets a unique ComponentTypeId via
// `get_component_type_id::<T>()`. Core engine component types receive
// deterministic IDs (0–11) via a compile-time table keyed by `TypeId`.
// Additional types are assigned sequentially from 12 onwards via a
// Mutex-protected fallback registry — this only happens during init,
// never on the hot path.

/// Published constants for core component type IDs.
/// Game code can use these directly to avoid function-call overhead.
pub const TRANSFORM_TYPE_ID: ComponentTypeId = 0;
pub const RENDER_TYPE_ID: ComponentTypeId = 1;
pub const CAMERA_TYPE_ID: ComponentTypeId = 2;
pub const LIGHT_TYPE_ID: ComponentTypeId = 3;
pub const POINT_LIGHT_TYPE_ID: ComponentTypeId = 4;
pub const HIERARCHY_TYPE_ID: ComponentTypeId = 5;
pub const RIGID_BODY_TYPE_ID: ComponentTypeId = 6;
pub const COLLIDER_TYPE_ID: ComponentTypeId = 7;
pub const SKELETON_TYPE_ID: ComponentTypeId = 8;
pub const ANIMATOR_TYPE_ID: ComponentTypeId = 9;
pub const AUDIO_LISTENER_TYPE_ID: ComponentTypeId = 10;
pub const AUDIO_EMITTER_TYPE_ID: ComponentTypeId = 11;
pub const NAME_TYPE_ID: ComponentTypeId = 12;

/// First ID available for dynamically registered component types.
const DYNAMIC_ID_START: ComponentTypeId = 13;

/// Table of (TypeId, assigned ComponentTypeId) for core engine components.
/// Populated lazily on first access via `OnceLock`.
fn core_type_table() -> &'static [(TypeId, ComponentTypeId)] {
    use crate::ecs::components::*;

    static TABLE: OnceLock<Vec<(TypeId, ComponentTypeId)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        vec![
            (TypeId::of::<TransformComponent>(), TRANSFORM_TYPE_ID),
            (TypeId::of::<RenderComponent>(), RENDER_TYPE_ID),
            (TypeId::of::<CameraComponent>(), CAMERA_TYPE_ID),
            (TypeId::of::<LightComponent>(), LIGHT_TYPE_ID),
            (TypeId::of::<PointLightComponent>(), POINT_LIGHT_TYPE_ID),
            (TypeId::of::<HierarchyComponent>(), HIERARCHY_TYPE_ID),
            (TypeId::of::<RigidBodyComponent>(), RIGID_BODY_TYPE_ID),
            (TypeId::of::<ColliderComponent>(), COLLIDER_TYPE_ID),
            (TypeId::of::<SkeletonComponent>(), SKELETON_TYPE_ID),
            (TypeId::of::<AnimatorComponent>(), ANIMATOR_TYPE_ID),
            (
                TypeId::of::<AudioListenerComponent>(),
                AUDIO_LISTENER_TYPE_ID,
            ),
            (TypeId::of::<AudioEmitterComponent>(), AUDIO_EMITTER_TYPE_ID),
            (TypeId::of::<NameComponent>(), NAME_TYPE_ID),
        ]
    })
}

struct ComponentTypeRegistry {
    map: HashMap<TypeId, ComponentTypeId>,
    next_id: ComponentTypeId,
}

static REGISTRY: OnceLock<Mutex<ComponentTypeRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<ComponentTypeRegistry> {
    REGISTRY.get_or_init(|| {
        Mutex::new(ComponentTypeRegistry {
            map: HashMap::new(),
            next_id: DYNAMIC_ID_START,
        })
    })
}

/// Returns a unique, stable ComponentTypeId for the given type T.
///
/// Core engine types (Transform, Render, Camera, etc.) receive
/// deterministic IDs via a `TypeId` lookup table — no string matching.
/// Any other type gets a sequentially assigned ID on first call.
///
/// # Panics
/// Panics if more than `MAX_COMPONENT_TYPES` distinct types are registered.
pub fn get_component_type_id<T: 'static>() -> ComponentTypeId {
    let tid = TypeId::of::<T>();

    // Fast path: scan the core type table (small, cache-friendly linear search).
    for &(core_tid, id) in core_type_table() {
        if core_tid == tid {
            return id;
        }
    }

    // Slow path: dynamic fallback for non-core types (init-time only).
    let mut reg = registry().lock().unwrap();

    if let Some(&id) = reg.map.get(&tid) {
        return id;
    }

    let id = reg.next_id;
    assert!(
        (id as u32) < MAX_COMPONENT_TYPES,
        "Exceeded MAX_COMPONENT_TYPES component registrations."
    );
    reg.next_id += 1;
    reg.map.insert(tid, id);
    id
}

/// Build a component mask from a slice of ComponentTypeIds.
#[inline]
pub fn build_mask(type_ids: &[ComponentTypeId]) -> ComponentMask {
    let mut mask: ComponentMask = 0;
    for &id in type_ids {
        mask |= 1u64 << id;
    }
    mask
}

/// Reset the component type registry. Only for testing.
pub fn reset_component_registry() {
    let mut reg = registry().lock().unwrap();
    reg.map.clear();
    reg.next_id = DYNAMIC_ID_START;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_id_packing() {
        let id = make_entity_id(42, 7);
        assert_eq!(get_entity_index(id), 42);
        assert_eq!(get_entity_generation(id), 7);
    }

    #[test]
    fn test_invalid_entity() {
        assert!(!is_valid_entity(INVALID_ENTITY));
        assert!(is_valid_entity(make_entity_id(0, 0)));
    }

    #[test]
    fn test_index_mask_range() {
        let max_index = ENTITY_INDEX_MASK;
        let id = make_entity_id(max_index, 0);
        assert_eq!(get_entity_index(id), max_index);
    }

    #[test]
    fn test_generation_wrap() {
        let max_gen = ENTITY_GENERATION_MASK as EntityGen;
        let id = make_entity_id(0, max_gen);
        assert_eq!(get_entity_generation(id), max_gen);
    }
}
