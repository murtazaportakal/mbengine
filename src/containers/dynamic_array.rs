//! Arena-backed dynamic array.
//!
//! Replaces `Vec` for growable arrays, but allocates entirely from a provided
//! `ArenaAllocator`. Because arenas don't support `realloc` easily (unless
//! the allocation is at the top of the arena), growing this array requires
//! allocating a new, larger block and copying the old elements over.

use crate::memory::ArenaAllocator;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::ptr;

/// A growable array backed by an `ArenaAllocator`.
pub struct DynamicArray<T> {
    ptr: *mut T,
    capacity: usize,
    len: usize,
}

impl<T> DynamicArray<T> {
    /// Create a new, empty DynamicArray with a pre-allocated capacity.
    ///
    /// # Safety
    /// The returned `DynamicArray` must not outlive the `ArenaAllocator` it was created from.
    pub fn with_capacity(arena: &mut ArenaAllocator, capacity: usize) -> Self {
        let ptr = if capacity > 0 {
            let p = arena.allocate_array::<T>(capacity);
            assert!(
                !p.is_null(),
                "DynamicArray: arena exhausted during allocation"
            );
            p
        } else {
            ptr::null_mut()
        };

        Self {
            ptr,
            capacity,
            len: 0,
        }
    }

    /// Number of elements currently in the array.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the array contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum number of elements the array can currently hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Append an element to the back of the array.
    ///
    /// If the capacity is exceeded, this will allocate a new block from the arena
    /// that is twice as large, and copy the existing elements over.
    pub fn push(&mut self, arena: &mut ArenaAllocator, value: T) {
        if self.len >= self.capacity {
            self.grow(arena);
        }

        unsafe {
            ptr::write(self.ptr.add(self.len), value);
        }
        self.len += 1;
    }

    /// Remove the last element from the array and return it, or `None` if it is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe { Some(ptr::read(self.ptr.add(self.len))) }
        }
    }

    /// Removes all elements from the array.
    pub fn clear(&mut self) {
        // We must drop the elements, but we don't deallocate the memory from the arena.
        if std::mem::needs_drop::<T>() {
            for i in 0..self.len {
                unsafe {
                    ptr::drop_in_place(self.ptr.add(i));
                }
            }
        }
        self.len = 0;
    }

    /// View the initialized elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    /// View the initialized elements as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    #[cold]
    fn grow(&mut self, arena: &mut ArenaAllocator) {
        let new_capacity = if self.capacity == 0 {
            4
        } else {
            self.capacity * 2
        };

        let new_ptr = arena.allocate_array::<T>(new_capacity);
        assert!(
            !new_ptr.is_null(),
            "DynamicArray: arena exhausted during grow"
        );

        if !self.ptr.is_null() && self.len > 0 {
            unsafe {
                // Copy old elements to the new allocation.
                ptr::copy_nonoverlapping(self.ptr, new_ptr, self.len);
                // Note: we DO NOT drop the old elements because they've been bitwise moved to new_ptr.
                // We also DO NOT free the old memory block because it's managed by an ArenaAllocator
                // which only resets in bulk.
            }
        }

        self.ptr = new_ptr;
        self.capacity = new_capacity;
    }
}

impl<T> Drop for DynamicArray<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<T> Deref for DynamicArray<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for DynamicArray<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// Convert a `Vec<T>` into a `DynamicArray<T>` using an `ArenaAllocator`.
impl<T> DynamicArray<T> {
    pub fn from_vec(arena: &mut ArenaAllocator, vec: Vec<T>) -> Self {
        let mut arr = Self::with_capacity(arena, vec.len());
        let vec = ManuallyDrop::new(vec);
        unsafe {
            ptr::copy_nonoverlapping(vec.as_ptr(), arr.ptr, vec.len());
        }
        arr.len = vec.len();
        arr
    }
}
