//! Top-level ECS container — owns entities, components, and systems.
//!
//! Architecture:
//!   The World is the single entry point for all ECS operations. It owns:
//!     - An EntityManager for entity lifecycle
//!     - Up to MAX_COMPONENT_TYPES ComponentArray<T> instances
//!     - Up to MAX_SYSTEMS System trait objects
//!
//!   All internal storage is allocated from an externally provided
//!   ArenaAllocator (typically PersistentArena from MemorySubsystem).
//!   Zero heap allocations beyond the system trait objects.
//!
//! Thread safety:
//!   None. All calls from the main thread.

use super::component_array::{ComponentArray, ComponentArrayOps};
use super::entity_manager::EntityManager;
use super::system::System;
use super::types::*;
use crate::memory::ArenaAllocator;
use std::ptr;

/// Top-level ECS container.
pub struct World {
    arena: *mut ArenaAllocator,

    entity_manager: *mut EntityManager,

    /// Flat array of type-erased component arrays, indexed by ComponentTypeId.
    /// Each entry is a raw pointer to a ComponentArray<T> cast to *mut dyn ComponentArrayOps.
    /// Using a pair of (data_ptr, vtable_ptr) stored as raw parts.
    component_arrays: [Option<ComponentArrayEntry>; MAX_COMPONENT_TYPES as usize],

    /// Number of component types registered (tracks highest ID + 1).
    registered_component_count: u32,

    /// Boxed system trait objects.
    systems: Vec<Box<dyn System>>,
}

/// Stores a type-erased component array pointer with its vtable for trait dispatch.
struct ComponentArrayEntry {
    /// Raw pointer to the ComponentArray<T> allocation.
    ptr: *mut u8,
    /// Fat pointer to the trait object for type-erased calls.
    ops: *mut dyn ComponentArrayOps,
}

impl World {
    /// Construct a World, allocating internal storage from the provided arena.
    ///
    /// # Safety
    /// The arena must outlive this World. All entity data, component arrays,
    /// and the entity manager are carved from this arena.
    pub unsafe fn new(arena: &mut ArenaAllocator) -> Self {
        // Construct the EntityManager in the arena.
        let em_ptr = arena.allocate(
            std::mem::size_of::<EntityManager>(),
            std::mem::align_of::<EntityManager>(),
        );
        assert!(
            !em_ptr.is_null(),
            "World: arena exhausted allocating EntityManager."
        );

        let entity_manager = em_ptr as *mut EntityManager;
        ptr::write(entity_manager, EntityManager::init(arena));

        const NONE: Option<ComponentArrayEntry> = None;

        Self {
            arena: arena as *mut ArenaAllocator,
            entity_manager,
            component_arrays: [NONE; MAX_COMPONENT_TYPES as usize],
            registered_component_count: 0,
            systems: Vec::new(),
        }
    }

    // ── component registration ──────────────────────────────────────────

    /// Register a component type T with the World.
    ///
    /// Must be called before any entity uses this component type.
    ///
    /// # Safety
    /// T must be a valid component type. The arena must have sufficient space.
    pub unsafe fn register_component<T: 'static>(&mut self, dense_capacity: u32) {
        let type_id = get_component_type_id::<T>();

        debug_assert!(
            (type_id as u32) < MAX_COMPONENT_TYPES,
            "World::register_component: too many component types."
        );
        debug_assert!(
            self.component_arrays[type_id as usize].is_none(),
            "World::register_component: type already registered."
        );

        // Allocate space for ComponentArray<T> in the arena.
        let arena = &mut *self.arena;
        let mem = arena.allocate(
            std::mem::size_of::<ComponentArray<T>>(),
            std::mem::align_of::<ComponentArray<T>>(),
        );
        assert!(
            !mem.is_null(),
            "World::register_component: arena exhausted."
        );

        let array_ptr = mem as *mut ComponentArray<T>;
        ptr::write(array_ptr, ComponentArray::<T>::init(arena, dense_capacity));

        // Store both the raw pointer and the trait object pointer.
        let ops: *mut dyn ComponentArrayOps = array_ptr;

        self.component_arrays[type_id as usize] = Some(ComponentArrayEntry { ptr: mem, ops });

        if type_id as u32 >= self.registered_component_count {
            self.registered_component_count = type_id as u32 + 1;
        }
    }

    // ── system registration ─────────────────────────────────────────────

    /// Register a system with the World.
    ///
    /// Returns a mutable reference so the caller can configure the system.
    pub fn register_system(&mut self, system: Box<dyn System>) -> &mut dyn System {
        debug_assert!(
            (self.systems.len() as u32) < MAX_SYSTEMS,
            "World::register_system: too many systems."
        );

        self.systems.push(system);
        self.systems.last_mut().unwrap().as_mut()
    }

    // ── entity lifecycle ────────────────────────────────────────────────

    /// Create a new entity.
    pub fn create_entity(&mut self) -> EntityId {
        unsafe { (*self.entity_manager).create_entity() }
    }

    /// Destroy an entity and remove all its components.
    pub fn destroy_entity(&mut self, id: EntityId) {
        debug_assert!(
            self.is_alive(id),
            "World::destroy_entity: entity is not alive."
        );

        let index = get_entity_index(id);

        // Notify all registered component arrays to remove this entity's data.
        for i in 0..self.registered_component_count as usize {
            if let Some(ref mut entry) = self.component_arrays[i] {
                unsafe {
                    (*entry.ops).entity_destroyed(index);
                }
            }
        }

        // Destroy the entity (increments generation, clears mask, recycles slot).
        unsafe {
            (*self.entity_manager).destroy_entity(id);
        }
    }

    /// Check whether an entity is still alive.
    pub fn is_alive(&self, id: EntityId) -> bool {
        unsafe { (*self.entity_manager).is_alive(id) }
    }

    // ── component operations ────────────────────────────────────────────

    /// Add a component to an entity.
    ///
    /// # Safety
    /// The component type T must be registered. The entity must be alive.
    pub unsafe fn add_component<T: 'static>(&mut self, id: EntityId, component: T) {
        debug_assert!(
            self.is_alive(id),
            "World::add_component: entity is not alive."
        );

        let type_id = get_component_type_id::<T>();
        let index = get_entity_index(id);

        self.get_component_array_mut::<T>().insert(index, component);
        (*self.entity_manager).set_component_bit(id, type_id);
    }

    /// Remove a component from an entity.
    ///
    /// # Safety
    /// The component type T must be registered. The entity must be alive
    /// and have this component.
    pub unsafe fn remove_component<T: 'static>(&mut self, id: EntityId) {
        debug_assert!(
            self.is_alive(id),
            "World::remove_component: entity is not alive."
        );

        let type_id = get_component_type_id::<T>();
        let index = get_entity_index(id);

        self.get_component_array_mut::<T>().remove(index);
        (*self.entity_manager).clear_component_bit(id, type_id);
    }

    /// Get a reference to an entity's component.
    ///
    /// # Safety
    /// The entity must be alive and have the component.
    pub unsafe fn get_component<T: 'static>(&self, id: EntityId) -> &T {
        debug_assert!(
            self.is_alive(id),
            "World::get_component: entity is not alive."
        );
        self.get_component_array::<T>().get(get_entity_index(id))
    }

    /// Get a mutable reference to an entity's component.
    ///
    /// # Safety
    /// The entity must be alive and have the component.
    pub unsafe fn get_component_mut<T: 'static>(&mut self, id: EntityId) -> &mut T {
        debug_assert!(
            self.is_alive(id),
            "World::get_component_mut: entity is not alive."
        );
        self.get_component_array_mut::<T>()
            .get_mut(get_entity_index(id))
    }

    /// Check whether an entity has a specific component.
    pub fn has_component<T: 'static>(&self, id: EntityId) -> bool {
        if !self.is_alive(id) {
            return false;
        }
        self.get_component_array::<T>().has(get_entity_index(id))
    }

    /// Get direct access to a ComponentArray<T> for cache-optimal iteration.
    pub fn get_component_array<T: 'static>(&self) -> &ComponentArray<T> {
        let type_id = get_component_type_id::<T>();
        debug_assert!(
            (type_id as u32) < MAX_COMPONENT_TYPES,
            "World::get_component_array: type ID out of range."
        );
        let entry = self.component_arrays[type_id as usize]
            .as_ref()
            .expect("World::get_component_array: component type not registered.");
        unsafe { &*(entry.ptr as *const ComponentArray<T>) }
    }

    /// Get mutable access to a ComponentArray<T>.
    pub fn get_component_array_mut<T: 'static>(&mut self) -> &mut ComponentArray<T> {
        let type_id = get_component_type_id::<T>();
        debug_assert!((type_id as u32) < MAX_COMPONENT_TYPES);
        let entry = self.component_arrays[type_id as usize]
            .as_mut()
            .expect("World::get_component_array_mut: component type not registered.");
        unsafe { &mut *(entry.ptr as *mut ComponentArray<T>) }
    }

    // ── system execution ────────────────────────────────────────────────

    /// Run all registered systems in registration order.
    ///
    /// # Safety
    /// Systems receive a raw pointer to the World to work around Rust's
    /// borrowing rules (system needs &mut self AND &mut world). This is
    /// safe because systems don't modify the systems array.
    pub fn update_systems(&mut self, dt: f32) {
        // We need to temporarily take ownership of systems to avoid
        // borrowing self mutably twice.
        let mut systems = std::mem::take(&mut self.systems);
        for system in systems.iter_mut() {
            system.update(dt, self);
        }
        self.systems = systems;
    }

    // ── queries ─────────────────────────────────────────────────────────

    /// Number of currently alive entities.
    pub fn alive_entity_count(&self) -> u32 {
        unsafe { (*self.entity_manager).alive_count() }
    }

    /// Get the component mask for an alive entity.
    pub fn get_component_mask(&self, id: EntityId) -> ComponentMask {
        unsafe { (*self.entity_manager).get_component_mask(id) }
    }
}

impl Drop for World {
    fn drop(&mut self) {
        // Destroy component arrays (calls destructors of live components via Drop).
        for i in 0..self.registered_component_count as usize {
            if let Some(entry) = self.component_arrays[i].take() {
                unsafe {
                    // Drop the trait object to invoke ComponentArray<T>::drop()
                    ptr::drop_in_place(entry.ops);
                }
            }
        }

        // Destroy entity manager.
        if !self.entity_manager.is_null() {
            // EntityManager has no Drop impl needed (arrays owned by arena),
            // but we drop it for completeness.
            unsafe {
                ptr::drop_in_place(self.entity_manager);
            }
            self.entity_manager = ptr::null_mut();
        }

        // Memory is owned by the arena — no deallocation needed.
    }
}
