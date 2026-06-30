//! Manages entity lifecycle with generational ID recycling.
//!
//! Architecture:
//!   - Maintains a generation counter and component mask per entity slot.
//!   - Destroyed entity indices are pushed into a ring-buffer recycle queue.
//!   - create_entity() pops from the recycle queue (or bumps a fresh counter)
//!     and packs the index + current generation into a 32-bit EntityId.
//!   - is_alive() compares the stored generation against the ID's generation
//!     field to detect stale handles.
//!
//! Memory:
//!   All arrays are allocated from an externally provided ArenaAllocator.
//!   Zero heap allocations.
//!
//! Thread safety:
//!   None. Call from the main thread only.

use super::types::*;
use crate::memory::ArenaAllocator;
use std::ptr;

/// Manages entity creation, destruction, and generational ID recycling.
pub struct EntityManager {
    /// Current generation per slot. Incremented on each destroy.
    generations: *mut EntityGen,

    /// Component presence bitset per slot.
    component_masks: *mut ComponentMask,

    /// Ring-buffer of recycled entity indices.
    recycle_queue: *mut u32,
    recycle_head: u32,
    recycle_tail: u32,
    recycle_count: u32,

    /// Next never-used slot index (monotonically increasing).
    next_fresh_index: u32,

    /// Number of currently alive entities.
    alive_count: u32,
}

impl EntityManager {
    /// Construct the entity manager, allocating internal arrays from
    /// the provided arena.
    ///
    /// # Safety
    /// The arena must outlive this EntityManager.
    pub unsafe fn init(arena: &mut ArenaAllocator) -> Self {
        // Allocate generation counters.
        let generations = arena.allocate_array::<EntityGen>(MAX_ENTITIES as usize);
        assert!(
            !generations.is_null(),
            "EntityManager: arena exhausted allocating generations."
        );
        ptr::write_bytes(generations, 0, MAX_ENTITIES as usize);

        // Allocate component masks.
        let component_masks = arena.allocate_array::<ComponentMask>(MAX_ENTITIES as usize);
        assert!(
            !component_masks.is_null(),
            "EntityManager: arena exhausted allocating component masks."
        );
        ptr::write_bytes(component_masks, 0, MAX_ENTITIES as usize);

        // Allocate recycle queue.
        let recycle_queue = arena.allocate_array::<u32>(MAX_ENTITIES as usize);
        assert!(
            !recycle_queue.is_null(),
            "EntityManager: arena exhausted allocating recycle queue."
        );

        Self {
            generations,
            component_masks,
            recycle_queue,
            recycle_head: 0,
            recycle_tail: 0,
            recycle_count: 0,
            next_fresh_index: 0,
            alive_count: 0,
        }
    }

    // ── entity lifecycle ────────────────────────────────────────────────

    /// Create a new entity.
    ///
    /// If recycled slots are available, one is reused (with an incremented
    /// generation). Otherwise a fresh slot is allocated.
    pub fn create_entity(&mut self) -> EntityId {
        let index: EntityIndex;

        if self.recycle_count > 0 {
            // Reuse a recycled slot.
            unsafe {
                index = *self.recycle_queue.add(self.recycle_head as usize);
            }
            self.recycle_head = (self.recycle_head + 1) % MAX_ENTITIES;
            self.recycle_count -= 1;
        } else {
            // Allocate a fresh slot.
            if self.next_fresh_index >= MAX_ENTITIES {
                debug_assert!(
                    false,
                    "EntityManager::create_entity: all entity slots exhausted."
                );
                return INVALID_ENTITY;
            }
            index = self.next_fresh_index;
            self.next_fresh_index += 1;
        }

        self.alive_count += 1;

        // The generation was already incremented at destroy-time (or is 0 for fresh slots).
        let gen = unsafe { *self.generations.add(index as usize) };
        make_entity_id(index, gen)
    }

    /// Destroy an entity, freeing its slot for recycling.
    pub fn destroy_entity(&mut self, id: EntityId) {
        debug_assert!(
            self.is_alive(id),
            "EntityManager::destroy_entity: entity is not alive."
        );

        let index = get_entity_index(id);

        // Clear component mask.
        self.clear_all_component_bits(index);

        // Increment generation so stale handles become invalid.
        unsafe {
            let gen = self.generations.add(index as usize);
            *gen = ((*gen).wrapping_add(1)) & (ENTITY_GENERATION_MASK as EntityGen);
        }

        // Push the freed index into the recycle queue.
        unsafe {
            *self.recycle_queue.add(self.recycle_tail as usize) = index;
        }
        self.recycle_tail = (self.recycle_tail + 1) % MAX_ENTITIES;
        self.recycle_count += 1;

        self.alive_count -= 1;
    }

    /// Check whether an entity handle is still alive.
    pub fn is_alive(&self, id: EntityId) -> bool {
        if !is_valid_entity(id) {
            return false;
        }

        let index = get_entity_index(id);
        if index >= self.next_fresh_index {
            return false;
        }

        let id_gen = get_entity_generation(id);
        let slot_gen = unsafe { *self.generations.add(index as usize) };

        id_gen == slot_gen
    }

    // ── component mask ──────────────────────────────────────────────────

    /// Get the component presence bitset for the given entity.
    pub fn get_component_mask(&self, id: EntityId) -> ComponentMask {
        debug_assert!(
            self.is_alive(id),
            "EntityManager::get_component_mask: entity is not alive."
        );
        unsafe { *self.component_masks.add(get_entity_index(id) as usize) }
    }

    /// Set a single component bit in the entity's mask.
    pub fn set_component_bit(&mut self, id: EntityId, type_id: ComponentTypeId) {
        debug_assert!(self.is_alive(id));
        debug_assert!((type_id as u32) < MAX_COMPONENT_TYPES);
        unsafe {
            let mask = self.component_masks.add(get_entity_index(id) as usize);
            *mask |= 1u64 << type_id;
        }
    }

    /// Clear a single component bit in the entity's mask.
    pub fn clear_component_bit(&mut self, id: EntityId, type_id: ComponentTypeId) {
        debug_assert!(self.is_alive(id));
        debug_assert!((type_id as u32) < MAX_COMPONENT_TYPES);
        unsafe {
            let mask = self.component_masks.add(get_entity_index(id) as usize);
            *mask &= !(1u64 << type_id);
        }
    }

    /// Clear all component bits for the given entity slot.
    pub fn clear_all_component_bits(&mut self, index: EntityIndex) {
        debug_assert!(index < MAX_ENTITIES);
        unsafe {
            *self.component_masks.add(index as usize) = 0;
        }
    }

    // ── queries ─────────────────────────────────────────────────────────

    /// Number of currently alive entities.
    #[inline]
    pub fn alive_count(&self) -> u32 {
        self.alive_count
    }

    /// Number of entity slots that have ever been used.
    #[inline]
    pub fn high_water_mark(&self) -> u32 {
        self.next_fresh_index
    }

    /// Number of recycled slots currently available for reuse.
    #[inline]
    pub fn recycled_count(&self) -> u32 {
        self.recycle_count
    }
}
