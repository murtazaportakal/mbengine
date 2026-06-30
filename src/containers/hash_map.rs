//! Open-addressing HashMap with Robin Hood hashing.
//!
//! Backed by an `ArenaAllocator`. Robin Hood hashing reduces the variance
//! of probe lengths, leading to better cache coherency and faster lookups
//! compared to standard open-addressing, without the pointer-chasing overhead
//! of linked-list separate chaining.

use crate::memory::ArenaAllocator;
use std::hash::{BuildHasher, Hash};
use std::mem::MaybeUninit;
use std::ptr;

/// A hash map entry.
struct Entry<K, V> {
    /// 0 means empty. Since hashes are typically not exactly 0, we use 0 to indicate
    /// an empty slot. If a hash actually is 0, we store it as 1.
    hash: u64,
    /// Distance from the initial bucket to this bucket.
    probe_dist: usize,
    key: MaybeUninit<K>,
    value: MaybeUninit<V>,
}

/// A cache-friendly open-addressing hash map using Robin Hood hashing.
pub struct HashMap<K, V, S = std::collections::hash_map::RandomState> {
    entries: *mut Entry<K, V>,
    capacity: usize,
    len: usize,
    build_hasher: S,
}

impl<K, V> HashMap<K, V, std::collections::hash_map::RandomState>
where
    K: Eq + Hash,
{
    /// Create a new, empty HashMap with a pre-allocated capacity.
    ///
    /// # Safety
    /// The returned `HashMap` must not outlive the `ArenaAllocator`.
    pub fn with_capacity(arena: &mut ArenaAllocator, capacity: usize) -> Self {
        Self::with_capacity_and_hasher(arena, capacity, std::collections::hash_map::RandomState::new())
    }
}

impl<K, V, S> HashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Create a new HashMap with a specific capacity and hasher.
    pub fn with_capacity_and_hasher(arena: &mut ArenaAllocator, capacity: usize, build_hasher: S) -> Self {
        let entries = if capacity > 0 {
            let p = arena.allocate_array::<Entry<K, V>>(capacity);
            assert!(!p.is_null(), "HashMap: arena exhausted during allocation");
            
            unsafe {
                for i in 0..capacity {
                    ptr::write(p.add(i), Entry {
                        hash: 0,
                        probe_dist: 0,
                        key: MaybeUninit::uninit(),
                        value: MaybeUninit::uninit(),
                    });
                }
            }
            p
        } else {
            ptr::null_mut()
        };

        Self {
            entries,
            capacity,
            len: 0,
            build_hasher,
        }
    }

    /// Number of elements in the map.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the map contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Capacity of the map.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn hash_key(&self, key: &K) -> u64 {
        let hash = self.build_hasher.hash_one(key);
        if hash == 0 { 1 } else { hash }
    }

    /// Insert a key-value pair into the map.
    ///
    /// If the map is at ~90% capacity, it will reallocate and rehash using the
    /// provided `ArenaAllocator`. Returns the old value if the key was already present.
    pub fn insert(&mut self, arena: &mut ArenaAllocator, key: K, value: V) -> Option<V> {
        // Grow if load factor > ~90%
        if self.len >= self.capacity * 9 / 10 {
            self.grow(arena);
        }

        let hash = self.hash_key(&key);
        let mut current_entry = Entry {
            hash,
            probe_dist: 0,
            key: MaybeUninit::new(key),
            value: MaybeUninit::new(value),
        };

        let mut idx = (hash as usize) % self.capacity;

        loop {
            let slot = unsafe { &mut *self.entries.add(idx) };

            // Empty slot found
            if slot.hash == 0 {
                unsafe {
                    ptr::write(slot, current_entry);
                }
                self.len += 1;
                return None;
            }

            // Key already exists, update value
            if slot.hash == current_entry.hash {
                let slot_key = unsafe { slot.key.assume_init_ref() };
                let current_key = unsafe { current_entry.key.assume_init_ref() };
                if slot_key == current_key {
                    let old_val = unsafe {
                        let old_val_ptr = slot.value.as_mut_ptr();
                        ptr::replace(old_val_ptr, current_entry.value.assume_init())
                    };
                    return Some(old_val);
                }
            }

            // Robin Hood: if the current entry has probed further than the entry in the slot,
            // swap them to minimize maximum probe length.
            if current_entry.probe_dist > slot.probe_dist {
                unsafe {
                    let temp = ptr::read(slot);
                    ptr::write(slot, current_entry);
                    current_entry = temp;
                }
            }

            current_entry.probe_dist += 1;
            idx = (idx + 1) % self.capacity;
        }
    }

    /// Retrieve a reference to the value associated with the given key.
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.capacity == 0 { return None; }

        let hash = self.hash_key(key);
        let mut idx = (hash as usize) % self.capacity;
        let mut probe_dist = 0;

        loop {
            let slot = unsafe { &*self.entries.add(idx) };

            if slot.hash == 0 || probe_dist > slot.probe_dist {
                return None; // Hit an empty slot or a slot that probed less than we have.
            }

            if slot.hash == hash {
                let slot_key = unsafe { slot.key.assume_init_ref() };
                if slot_key == key {
                    return Some(unsafe { slot.value.assume_init_ref() });
                }
            }

            probe_dist += 1;
            idx = (idx + 1) % self.capacity;
        }
    }

    /// Remove a key-value pair from the map and return the value.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if self.capacity == 0 { return None; }

        let hash = self.hash_key(key);
        let mut idx = (hash as usize) % self.capacity;
        let mut probe_dist = 0;

        loop {
            let slot = unsafe { &mut *self.entries.add(idx) };

            if slot.hash == 0 || probe_dist > slot.probe_dist {
                return None;
            }

            if slot.hash == hash {
                let slot_key = unsafe { slot.key.assume_init_ref() };
                if slot_key == key {
                    // Found it. Extract value and key.
                    let val = unsafe { ptr::read(slot.value.as_ptr()) };
                    unsafe { ptr::drop_in_place(slot.key.as_mut_ptr()) };
                    slot.hash = 0;
                    slot.probe_dist = 0;
                    self.len -= 1;

                    // Shift subsequent elements backwards to keep the probe chain intact.
                    let mut curr_idx = idx;
                    loop {
                        let next_idx = (curr_idx + 1) % self.capacity;
                        let next_slot = unsafe { &mut *self.entries.add(next_idx) };

                        if next_slot.hash == 0 || next_slot.probe_dist == 0 {
                            break;
                        }

                        next_slot.probe_dist -= 1;
                        unsafe {
                            ptr::copy_nonoverlapping(next_slot, self.entries.add(curr_idx), 1);
                        }
                        next_slot.hash = 0;
                        next_slot.probe_dist = 0;
                        curr_idx = next_idx;
                    }

                    return Some(val);
                }
            }

            probe_dist += 1;
            idx = (idx + 1) % self.capacity;
        }
    }

    // (clear() moved to unbounded impl below)

    #[cold]
    fn grow(&mut self, arena: &mut ArenaAllocator) {
        let new_capacity = if self.capacity == 0 { 8 } else { self.capacity * 2 };
        
        let new_entries = arena.allocate_array::<Entry<K, V>>(new_capacity);
        assert!(!new_entries.is_null(), "HashMap: arena exhausted during grow");

        unsafe {
            for i in 0..new_capacity {
                ptr::write(new_entries.add(i), Entry {
                    hash: 0,
                    probe_dist: 0,
                    key: MaybeUninit::uninit(),
                    value: MaybeUninit::uninit(),
                });
            }
        }

        let old_entries = self.entries;
        let old_capacity = self.capacity;

        self.entries = new_entries;
        self.capacity = new_capacity;
        self.len = 0; // Will be incremented by insert_internal

        if !old_entries.is_null() && old_capacity > 0 {
            for i in 0..old_capacity {
                let old_slot = unsafe { &mut *old_entries.add(i) };
                if old_slot.hash != 0 {
                    // Extract key and value to move them
                    let key = unsafe { ptr::read(old_slot.key.as_ptr()) };
                    let value = unsafe { ptr::read(old_slot.value.as_ptr()) };
                    // Re-insert into the new backing array
                    self.insert(arena, key, value);
                }
            }
        }
    }
    }

impl<K, V, S> HashMap<K, V, S> {
    /// Clear the map, dropping all elements.
    pub fn clear(&mut self) {
        if self.capacity == 0 { return; }

        for i in 0..self.capacity {
            let slot = unsafe { &mut *self.entries.add(i) };
            if slot.hash != 0 {
                unsafe {
                    ptr::drop_in_place(slot.key.as_mut_ptr());
                    ptr::drop_in_place(slot.value.as_mut_ptr());
                }
                slot.hash = 0;
                slot.probe_dist = 0;
            }
        }
        self.len = 0;
    }
}

impl<K, V, S> Drop for HashMap<K, V, S> {
    fn drop(&mut self) {
        self.clear();
    }
}
