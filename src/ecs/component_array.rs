//! Sparse-set based component storage for a single component type T.
//!
//! Architecture:
//!   This is the performance-critical core of the ECS. It stores components
//!   of a single type in a tightly packed dense array while maintaining O(1)
//!   lookup from entity index to component via a parallel sparse array.
//!
//!   Memory layout:
//!
//!   ```text
//!     sparse[MAX_ENTITIES]       *mut u32   entity-index -> dense-index
//!     dense[denseCapacity]       *mut T     packed component data
//!     denseEntities[denseCap]    *mut u32   dense-index -> entity-index
//!     denseCount                 u32        number of live entries
//!   ```
//!
//!   - Insert:  O(1) — append to dense arrays, write sparse pointer.
//!   - Remove:  O(1) — swap-and-pop from dense arrays, fixup sparse.
//!   - Get:     O(1) — sparse lookup to dense index.
//!   - Has:     O(1) — sparse check against INVALID_INDEX.
//!   - Iterate: O(N) over dense array, maximally cache-friendly.
//!
//! Memory:
//!   All arrays are allocated from an externally provided ArenaAllocator.
//!   Zero heap allocations.
//!
//! Thread safety:
//!   None. Single-writer assumed; synchronise externally if needed.

use super::types::*;
use crate::memory::{ArenaAllocator, DEFAULT_ALIGNMENT};
use std::ptr;

// ── ComponentArrayOps trait ─────────────────────────────────────────────────

/// Type-erased interface for component storage arrays.
/// The World holds heterogeneous ComponentArray<T> instances via this trait.
///
/// Hot-path iteration goes through the concrete ComponentArray<T>
/// (no virtual dispatch). This vtable is only hit during entity destruction.
pub trait ComponentArrayOps {
    /// Called by the World when an entity is destroyed.
    fn entity_destroyed(&mut self, entity_index: u32);

    /// Number of live components currently stored.
    fn count(&self) -> u32;
}

// ── ComponentArray<T> ───────────────────────────────────────────────────────

/// Sparse-set component storage for type T.
pub struct ComponentArray<T> {
    /// [MAX_ENTITIES] entity-index → dense-index
    sparse: *mut u32,
    /// [dense_capacity] packed component data
    dense: *mut T,
    /// [dense_capacity] dense-index → entity-index (reverse map)
    dense_entities: *mut u32,

    /// Number of live components.
    dense_count: u32,
    /// Max components.
    dense_capacity: u32,
}

impl<T> ComponentArray<T> {
    /// Construct a ComponentArray, allocating all internal storage from
    /// the provided arena.
    ///
    /// # Safety
    /// The arena must outlive this ComponentArray. The caller must ensure
    /// no other code is concurrently allocating from the same arena.
    pub unsafe fn init(arena: &mut ArenaAllocator, dense_capacity: u32) -> Self {
        debug_assert!(dense_capacity > 0 && dense_capacity <= MAX_ENTITIES);

        // Allocate sparse array (entity-index → dense-index).
        let sparse = arena.allocate_array::<u32>(MAX_ENTITIES as usize);
        assert!(
            !sparse.is_null(),
            "ComponentArray: arena exhausted allocating sparse array."
        );

        // Initialise all slots to "no component" (0xFFFFFFFF = INVALID_INDEX).
        ptr::write_bytes(sparse, 0xFF, MAX_ENTITIES as usize);

        // Allocate dense component array.
        let component_align = if std::mem::align_of::<T>() > DEFAULT_ALIGNMENT {
            std::mem::align_of::<T>()
        } else {
            DEFAULT_ALIGNMENT
        };
        let dense_raw = arena.allocate(
            std::mem::size_of::<T>() * dense_capacity as usize,
            component_align,
        );
        assert!(
            !dense_raw.is_null(),
            "ComponentArray: arena exhausted allocating dense array."
        );
        let dense = dense_raw as *mut T;

        // Allocate dense-to-entity reverse map.
        let dense_entities = arena.allocate_array::<u32>(dense_capacity as usize);
        assert!(
            !dense_entities.is_null(),
            "ComponentArray: arena exhausted allocating reverse map."
        );

        Self {
            sparse,
            dense,
            dense_entities,
            dense_count: 0,
            dense_capacity,
        }
    }

    // ── insert ──────────────────────────────────────────────────────────

    /// Insert a component for the given entity.
    ///
    /// # Safety
    /// Caller must ensure `entity_index < MAX_ENTITIES`, the entity does not
    /// already have this component, and the dense array is not full.
    pub unsafe fn insert(&mut self, entity_index: u32, component: T) {
        debug_assert!(
            entity_index < MAX_ENTITIES,
            "ComponentArray::insert: entityIndex out of range."
        );
        debug_assert!(
            !self.has(entity_index),
            "ComponentArray::insert: entity already has this component."
        );
        debug_assert!(
            self.dense_count < self.dense_capacity,
            "ComponentArray::insert: dense array full."
        );

        let dense_idx = self.dense_count;

        // Write sparse pointer.
        *self.sparse.add(entity_index as usize) = dense_idx;

        // Write the component into the dense array.
        ptr::write(self.dense.add(dense_idx as usize), component);

        // Write reverse map.
        *self.dense_entities.add(dense_idx as usize) = entity_index;

        self.dense_count += 1;
    }

    // ── remove ──────────────────────────────────────────────────────────

    /// Remove the component for the given entity using swap-and-pop.
    ///
    /// # Safety
    /// Caller must ensure `entity_index < MAX_ENTITIES` and the entity
    /// currently has this component.
    pub unsafe fn remove(&mut self, entity_index: u32) {
        debug_assert!(
            entity_index < MAX_ENTITIES,
            "ComponentArray::remove: entityIndex out of range."
        );
        debug_assert!(
            self.has(entity_index),
            "ComponentArray::remove: entity does not have this component."
        );

        let removed_dense = *self.sparse.add(entity_index as usize);
        let last_dense = self.dense_count - 1;

        // Drop the removed component.
        ptr::drop_in_place(self.dense.add(removed_dense as usize));

        if removed_dense != last_dense {
            // Move the last element into the gap.
            let last_ptr = self.dense.add(last_dense as usize);
            let removed_ptr = self.dense.add(removed_dense as usize);

            // Read the last element (takes ownership), write into the gap.
            let last_val = ptr::read(last_ptr);
            ptr::write(removed_ptr, last_val);

            // Update the reverse map and sparse array for the moved element.
            let moved_entity = *self.dense_entities.add(last_dense as usize);
            *self.dense_entities.add(removed_dense as usize) = moved_entity;
            *self.sparse.add(moved_entity as usize) = removed_dense;
        }

        // Clear the removed entity's sparse slot.
        *self.sparse.add(entity_index as usize) = INVALID_INDEX;

        self.dense_count -= 1;
    }

    // ── access ──────────────────────────────────────────────────────────

    /// Get a reference to the component for the given entity.
    ///
    /// # Safety
    /// Entity must have this component (`has(entity_index)` must be true).
    pub unsafe fn get(&self, entity_index: u32) -> &T {
        debug_assert!(entity_index < MAX_ENTITIES);
        debug_assert!(self.has(entity_index));
        let dense_idx = *self.sparse.add(entity_index as usize);
        &*self.dense.add(dense_idx as usize)
    }

    /// Get a mutable reference to the component for the given entity.
    ///
    /// # Safety
    /// Entity must have this component.
    pub unsafe fn get_mut(&mut self, entity_index: u32) -> &mut T {
        debug_assert!(entity_index < MAX_ENTITIES);
        debug_assert!(self.has(entity_index));
        let dense_idx = *self.sparse.add(entity_index as usize);
        &mut *self.dense.add(dense_idx as usize)
    }

    /// Check whether the given entity has this component.
    pub fn has(&self, entity_index: u32) -> bool {
        debug_assert!(entity_index < MAX_ENTITIES);
        unsafe { *self.sparse.add(entity_index as usize) != INVALID_INDEX }
    }

    // ── dense-array iteration ───────────────────────────────────────────

    /// Returns a slice over the dense component data.
    /// Every element is a live component — no gaps, no indirection.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.dense, self.dense_count as usize) }
    }

    /// Returns a mutable slice over the dense component data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.dense, self.dense_count as usize) }
    }

    /// Raw pointer to the dense component data.
    pub fn data(&self) -> *const T {
        self.dense
    }

    /// Raw pointer to the dense-to-entity reverse map.
    pub fn dense_entities(&self) -> *const u32 {
        self.dense_entities
    }

    /// Returns a slice over the dense entities reverse map.
    pub fn dense_entities_slice(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.dense_entities, self.dense_count as usize) }
    }

    // ── queries ─────────────────────────────────────────────────────────

    /// Maximum number of components this array can hold.
    #[inline]
    pub fn dense_capacity(&self) -> u32 {
        self.dense_capacity
    }

    /// Number of live components.
    #[inline]
    pub fn len(&self) -> u32 {
        self.dense_count
    }

    /// True when no components are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dense_count == 0
    }

    /// True when the dense array is at capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.dense_count >= self.dense_capacity
    }
}

impl<T> ComponentArrayOps for ComponentArray<T> {
    fn entity_destroyed(&mut self, entity_index: u32) {
        if self.has(entity_index) {
            unsafe {
                self.remove(entity_index);
            }
        }
    }

    fn count(&self) -> u32 {
        self.dense_count
    }
}

impl<T> Drop for ComponentArray<T> {
    fn drop(&mut self) {
        // Call destructors on all live components if T needs dropping.
        if std::mem::needs_drop::<T>() {
            for i in 0..self.dense_count {
                unsafe {
                    ptr::drop_in_place(self.dense.add(i as usize));
                }
            }
        }
        // Memory itself is owned by the arena — no deallocation needed.
    }
}
