//! Linear (Arena/Bump) memory allocator.
//!
//! Architecture:
//!   - Takes an externally owned, pre-allocated memory block.
//!   - O(1) allocation via pointer bump.
//!   - O(1) full reset (rewind to base).
//!   - O(1) partial reset via save-point markers.
//!   - Zero heap allocations internally — suitable for hot-loop use.
//!   - All returned pointers are aligned to the requested boundary.
//!
//! Intended usage:
//!   Per-frame scratch memory, temporary command buffers, staging geometry,
//!   string formatting, and any transient allocation pattern where individual
//!   frees are unnecessary.
//!
//! Thread safety:
//!   None. One ArenaAllocator per thread, or synchronise externally.

use super::memory_utils::{align_forward, is_power_of_two, DEFAULT_ALIGNMENT};
use std::ptr;

// ── ArenaMarker ─────────────────────────────────────────────────────────────

/// Lightweight save-point for partial rewinds.
/// Obtain one via `save_point()`, restore via `restore_to_save_point()`.
#[derive(Clone, Copy, Debug)]
pub struct ArenaMarker {
    /// Byte offset from arena base at capture time.
    pub offset: usize,
}

// ── ArenaAllocator ──────────────────────────────────────────────────────────

/// Linear bump allocator over an externally owned memory block.
pub struct ArenaAllocator {
    /// Start of the externally owned block.
    base: *mut u8,
    /// Total block size in bytes.
    total_bytes: usize,
    /// Current bump position (bytes from base).
    current_offset: usize,
}

impl ArenaAllocator {
    // ── lifecycle ───────────────────────────────────────────────────────

    /// Construct an arena over an externally owned memory block.
    ///
    /// # Safety
    /// - `base_memory` must point to a valid, writable block of at least
    ///   `total_bytes` bytes.
    /// - The block must remain valid for the lifetime of this allocator.
    /// - The caller is responsible for ensuring no aliasing violations.
    pub unsafe fn new(base_memory: *mut u8, total_bytes: usize) -> Self {
        debug_assert!(
            !base_memory.is_null(),
            "ArenaAllocator: base memory must not be null."
        );
        debug_assert!(total_bytes > 0, "ArenaAllocator: total bytes must be > 0.");

        Self {
            base: base_memory,
            total_bytes,
            current_offset: 0,
        }
    }

    // ── allocation ──────────────────────────────────────────────────────

    /// Bump-allocate `size_bytes` with the given alignment.
    ///
    /// Returns a pointer to the allocated block, or null if the arena is
    /// exhausted.
    ///
    /// # Safety
    /// The returned pointer is valid for writes of `size_bytes` bytes.
    /// The caller must ensure proper type alignment and lifetime.
    pub fn allocate(&mut self, size_bytes: usize, alignment: usize) -> *mut u8 {
        debug_assert!(
            is_power_of_two(alignment),
            "ArenaAllocator::allocate: alignment must be a power of two."
        );

        // Current absolute address = base + offset.
        let current_addr = self.base as usize + self.current_offset;

        // Align forward to the requested boundary.
        let aligned_addr = align_forward(current_addr, alignment);

        // Padding introduced by alignment.
        let padding = aligned_addr - current_addr;

        // Check for overflow: offset + padding + size must fit within the block.
        let new_offset = self.current_offset + padding + size_bytes;
        if new_offset > self.total_bytes {
            // Arena exhausted — return null pointer.
            return ptr::null_mut();
        }

        // Bump the offset.
        self.current_offset = new_offset;

        // Return the aligned pointer.
        aligned_addr as *mut u8
    }

    /// Allocate space for `count` elements of type T with proper alignment.
    /// Does NOT call constructors (returns raw pointer).
    ///
    /// # Safety
    /// Caller must initialise the memory before reading and ensure T's
    /// alignment requirements are met.
    pub fn allocate_array<T>(&mut self, count: usize) -> *mut T {
        let align = if std::mem::align_of::<T>() > DEFAULT_ALIGNMENT {
            std::mem::align_of::<T>()
        } else {
            DEFAULT_ALIGNMENT
        };
        let ptr = self.allocate(std::mem::size_of::<T>() * count, align);
        ptr as *mut T
    }

    /// Allocate and initialise a single value of type T.
    ///
    /// # Safety
    /// Caller must ensure the arena outlives any references to the returned
    /// pointer.
    pub fn alloc_new<T>(&mut self, value: T) -> *mut T {
        let align = if std::mem::align_of::<T>() > DEFAULT_ALIGNMENT {
            std::mem::align_of::<T>()
        } else {
            DEFAULT_ALIGNMENT
        };
        let ptr = self.allocate(std::mem::size_of::<T>(), align);
        if ptr.is_null() {
            return ptr::null_mut();
        }
        let typed = ptr as *mut T;
        unsafe {
            ptr::write(typed, value);
        }
        typed
    }

    // ── reset / rewind ──────────────────────────────────────────────────

    /// O(1) full reset — rewinds the bump pointer to the base.
    /// All prior allocations are logically invalidated.
    ///
    /// If `zero_memory` is true, the used region is zeroed before reset.
    pub fn reset(&mut self, zero_memory: bool) {
        if zero_memory && self.current_offset > 0 {
            unsafe {
                ptr::write_bytes(self.base, 0, self.current_offset);
            }
        }
        self.current_offset = 0;
    }

    /// Capture a save-point at the current bump position.
    pub fn save_point(&self) -> ArenaMarker {
        ArenaMarker {
            offset: self.current_offset,
        }
    }

    /// Rewind the allocator to a previously captured save-point.
    ///
    /// If `zero_memory` is true, the released region is zeroed.
    pub fn restore_to_save_point(&mut self, marker: ArenaMarker, zero_memory: bool) {
        debug_assert!(
            marker.offset <= self.current_offset,
            "ArenaAllocator::restore_to_save_point: marker is ahead of current offset."
        );

        if zero_memory && marker.offset < self.current_offset {
            unsafe {
                let start = self.base.add(marker.offset);
                let bytes_to_zero = self.current_offset - marker.offset;
                ptr::write_bytes(start, 0, bytes_to_zero);
            }
        }

        self.current_offset = marker.offset;
    }

    // ── queries ─────────────────────────────────────────────────────────

    /// Total capacity of the backing block, in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.total_bytes
    }

    /// Number of bytes currently in use (including alignment padding).
    #[inline]
    pub fn used(&self) -> usize {
        self.current_offset
    }

    /// Remaining bytes available before exhaustion.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.total_bytes - self.current_offset
    }

    /// Raw pointer to the base of the backing block.
    #[inline]
    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// Returns true when no allocations are outstanding (offset == 0).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.current_offset == 0
    }

    /// Returns true when no more bytes can be allocated (even unaligned).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.current_offset >= self.total_bytes
    }

    /// Check whether a pointer falls within this arena's backing block.
    pub fn owns(&self, ptr: *const u8) -> bool {
        let addr = ptr as usize;
        let base = self.base as usize;
        addr >= base && addr < (base + self.total_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{alloc, dealloc, Layout};

    /// Helper: allocate a page-aligned block for testing.
    fn alloc_test_block(size: usize) -> (*mut u8, Layout) {
        let layout = Layout::from_size_align(size, 64).unwrap();
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null());
        (ptr, layout)
    }

    #[test]
    fn test_basic_allocation() {
        let (block, layout) = alloc_test_block(1024);
        let mut arena = unsafe { ArenaAllocator::new(block, 1024) };

        assert!(arena.is_empty());
        assert_eq!(arena.capacity(), 1024);

        let p1 = arena.allocate(64, 16);
        assert!(!p1.is_null());
        assert!(arena.owns(p1));
        assert!(!arena.is_empty());

        let p2 = arena.allocate(64, 16);
        assert!(!p2.is_null());
        assert_ne!(p1, p2);

        unsafe { dealloc(block, layout) };
    }

    #[test]
    fn test_exhaustion() {
        let (block, layout) = alloc_test_block(128);
        let mut arena = unsafe { ArenaAllocator::new(block, 128) };

        let p = arena.allocate(256, 16);
        assert!(p.is_null());

        unsafe { dealloc(block, layout) };
    }

    #[test]
    fn test_save_point_restore() {
        let (block, layout) = alloc_test_block(1024);
        let mut arena = unsafe { ArenaAllocator::new(block, 1024) };

        let _ = arena.allocate(64, 16);
        let marker = arena.save_point();
        let used_at_marker = arena.used();

        let _ = arena.allocate(128, 16);
        assert!(arena.used() > used_at_marker);

        arena.restore_to_save_point(marker, false);
        assert_eq!(arena.used(), used_at_marker);

        unsafe { dealloc(block, layout) };
    }

    #[test]
    fn test_reset() {
        let (block, layout) = alloc_test_block(1024);
        let mut arena = unsafe { ArenaAllocator::new(block, 1024) };

        let _ = arena.allocate(256, 16);
        assert!(!arena.is_empty());

        arena.reset(false);
        assert!(arena.is_empty());
        assert_eq!(arena.used(), 0);

        unsafe { dealloc(block, layout) };
    }

    #[test]
    fn test_alignment() {
        let (block, layout) = alloc_test_block(4096);
        let mut arena = unsafe { ArenaAllocator::new(block, 4096) };

        // Allocate 1 byte to misalign, then allocate with 64-byte alignment.
        let _ = arena.allocate(1, 1);
        let p = arena.allocate(64, 64);
        assert!(!p.is_null());
        assert_eq!((p as usize) % 64, 0);

        unsafe { dealloc(block, layout) };
    }
}
