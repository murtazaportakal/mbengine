//! Lock-free Single-Producer Single-Consumer (SPSC) ring buffer.
//!
//! Suitable for command queues, audio buffers, or event streams between
//! two threads (e.g., Main thread and Render thread).

use crate::memory::ArenaAllocator;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A lock-free SPSC queue backed by an `ArenaAllocator`.
/// Capacity must be a power of two for efficient bitwise wrapping.
pub struct RingBuffer<T> {
    buffer: *mut MaybeUninit<T>,
    capacity_mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

// Safety: A RingBuffer is safe to share between threads if T is Send.
unsafe impl<T: Send> Send for RingBuffer<T> {}
unsafe impl<T: Send> Sync for RingBuffer<T> {}

impl<T> RingBuffer<T> {
    /// Create a new `RingBuffer` with a capacity of a power of two.
    ///
    /// # Safety
    /// The returned `RingBuffer` must not outlive the `ArenaAllocator`.
    /// `capacity` must be a power of two.
    pub fn new(arena: &mut ArenaAllocator, capacity: usize) -> Self {
        assert!(
            capacity.is_power_of_two(),
            "RingBuffer capacity must be a power of two"
        );

        let buffer = arena.allocate_array::<MaybeUninit<T>>(capacity);
        assert!(!buffer.is_null(), "RingBuffer: arena exhausted");

        Self {
            buffer,
            capacity_mask: capacity - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Attempts to push an element into the buffer.
    /// Returns `Ok(())` if successful, or `Err(value)` if the buffer is full.
    pub fn push(&self, value: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        // If the buffer is full (head wrapped around to tail)
        if head.wrapping_sub(tail) > self.capacity_mask {
            return Err(value);
        }

        unsafe {
            // Write to the buffer
            self.buffer.add(head & self.capacity_mask).write(MaybeUninit::new(value));
        }

        // Increment head
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Attempts to pop an element from the buffer.
    /// Returns `Some(value)` if successful, or `None` if the buffer is empty.
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        // If the buffer is empty
        if head == tail {
            return None;
        }

        let value = unsafe {
            // Read from the buffer
            self.buffer.add(tail & self.capacity_mask).read().assume_init()
        };

        // Increment tail
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    /// Returns the number of elements currently in the buffer.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }

    /// Returns `true` if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the maximum capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity_mask + 1
    }
}

impl<T> Drop for RingBuffer<T> {
    fn drop(&mut self) {
        // Drop any remaining elements in the buffer.
        while let Some(value) = self.pop() {
            drop(value);
        }
    }
}
