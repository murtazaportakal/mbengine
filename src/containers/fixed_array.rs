//! Stack-allocated fixed-capacity array.
//!
//! Replaces `Vec` for small arrays where the maximum capacity is known at
//! compile time, avoiding heap allocations. Memory is inline and contiguous.

use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

/// A fixed-capacity array stored inline on the stack (or within its parent struct).
pub struct FixedArray<T, const N: usize> {
    data: [MaybeUninit<T>; N],
    len: usize,
}

impl<T, const N: usize> FixedArray<T, N> {
    /// Create a new, empty FixedArray.
    #[inline]
    pub fn new() -> Self {
        // Create an uninitialized array of MaybeUninit
        let data = unsafe { MaybeUninit::uninit().assume_init() };
        Self { data, len: 0 }
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

    /// Maximum number of elements the array can hold.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns true if the array is at maximum capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == N
    }

    /// Append an element to the back of the array.
    ///
    /// # Panics
    /// Panics if the array is already at full capacity.
    #[inline]
    pub fn push(&mut self, value: T) {
        assert!(self.len < N, "FixedArray::push: capacity exceeded");
        self.data[self.len] = MaybeUninit::new(value);
        self.len += 1;
    }

    /// Remove the last element from the array and return it, or `None` if it is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            // Safety: We just decremented len, so the item at self.len is initialized.
            // We read the value and leave the MaybeUninit effectively uninitialized again.
            let val = unsafe { self.data[self.len].assume_init_read() };
            Some(val)
        }
    }

    /// Removes all elements from the array.
    #[inline]
    pub fn clear(&mut self) {
        // Drop elements safely
        for i in 0..self.len {
            unsafe {
                self.data[i].assume_init_drop();
            }
        }
        self.len = 0;
    }

    /// View the initialized elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // Safety: The first `len` elements are initialized.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.len) }
    }

    /// View the initialized elements as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // Safety: The first `len` elements are initialized.
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len) }
    }
}

impl<T, const N: usize> Default for FixedArray<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for FixedArray<T, N> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<T, const N: usize> Deref for FixedArray<T, N> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, const N: usize> DerefMut for FixedArray<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}
