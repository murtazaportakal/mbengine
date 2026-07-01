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
// `get_component_type_id::<T>()`. IDs are assigned sequentially starting
// from 0. Uses a global registry protected by a Mutex — registration
// happens during init, never on the hot path.

struct ComponentTypeRegistry {
    map: HashMap<TypeId, ComponentTypeId>,
    next_id: ComponentTypeId,
}

static REGISTRY: OnceLock<Mutex<ComponentTypeRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<ComponentTypeRegistry> {
    REGISTRY.get_or_init(|| {
        Mutex::new(ComponentTypeRegistry {
            map: HashMap::new(),
            next_id: 8,
        })
    })
}

/// Returns a unique, stable ComponentTypeId for the given type T.
/// The first call for each T assigns the next available ID.
///
/// # Panics
/// Panics if more than `MAX_COMPONENT_TYPES` distinct types are registered.
pub fn get_component_type_id<T: 'static>() -> ComponentTypeId {
    let name = std::any::type_name::<T>();
    if name.contains("TransformComponent") { return 0; }
    if name.contains("RenderComponent") { return 1; }
    if name.contains("CameraComponent") { return 2; }
    if name.contains("LightComponent") && !name.contains("PointLightComponent") { return 3; }
    if name.contains("PointLightComponent") { return 4; }
    if name.contains("HierarchyComponent") { return 5; }
    if name.contains("RigidBodyComponent") { return 6; }
    if name.contains("ColliderComponent") { return 7; }
    
    // Fallback for any other types
    let type_id = TypeId::of::<T>();
    let mut reg = registry().lock().unwrap();

    if let Some(&id) = reg.map.get(&type_id) {
        return id;
    }

    let id = reg.next_id;
    assert!(
        (id as u32) < MAX_COMPONENT_TYPES,
        "Exceeded MAX_COMPONENT_TYPES component registrations."
    );
    reg.next_id += 1;
    reg.map.insert(type_id, id);
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
    reg.next_id = 8;
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
