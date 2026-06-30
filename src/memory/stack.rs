//! LIFO (stack-based) memory allocator with per-allocation headers.
//!
//! Architecture:
//!   - Takes an externally owned, pre-allocated memory block.
//!   - O(1) allocation via pointer bump (like ArenaAllocator), but each
//!     allocation stores a small header so it can be freed in LIFO order.
//!   - O(1) individual free — but ONLY the most-recently-allocated block.
//!   - O(1) full reset (rewind to base).
//!   - O(1) partial reset via save-point markers.
//!   - Zero heap allocations internally.
//!
//! Memory layout per allocation:
//!
//!   ┌──────────────────────┬────────────────────────────────┐
//!   │  AllocationHeader    │  User payload (aligned)        │
//!   │  (padding + prevOff) │                                │
//!   └──────────────────────┴────────────────────────────────┘
//!   ^                      ^
//!   headerStart            alignedPayload (returned to caller)
//!
//! Intended usage:
//!   Scoped temporary allocations that unwind in reverse order:
//!   render command building, recursive algorithms, nested scratch buffers.
//!
//! Thread safety:
//!   None. One StackAllocator per thread, or synchronise externally.

use super::memory_utils::{align_forward, is_power_of_two};
use std::ptr;

// ── StackMarker ─────────────────────────────────────────────────────────────

/// Lightweight save-point for partial rewinds.
#[derive(Clone, Copy, Debug)]
pub struct StackMarker {
    /// Byte offset from stack base at capture time.
    pub offset: usize,
}

// ── AllocationHeader ────────────────────────────────────────────────────────

/// Per-allocation header stored immediately before each aligned user payload.
///
/// `prev_offset`: the value of `current_offset` *before* this allocation,
///                so `free()` can rewind to it.
/// `adjustment`:  total bytes from the raw bump position to the start
///                of the user payload (= padding + sizeof(header)).
#[repr(C)]
struct AllocationHeader {
    prev_offset: usize,
    adjustment: usize,
}

// ── StackAllocator ──────────────────────────────────────────────────────────

/// LIFO stack allocator with per-allocation headers.
pub struct StackAllocator {
    base: *mut u8,
    total_bytes: usize,
    current_offset: usize,

    /// In debug builds, track the most-recent payload address so `free()`
    /// can assert correct LIFO ordering.
    #[cfg(debug_assertions)]
    last_allocation: *mut u8,
}

impl StackAllocator {
    // ── lifecycle ───────────────────────────────────────────────────────

    /// Construct a stack allocator over an externally owned memory block.
    ///
    /// # Safety
    /// - `base_memory` must point to a valid, writable block of at least
    ///   `total_bytes` bytes.
    /// - The block must remain valid for the lifetime of this allocator.
    pub unsafe fn new(base_memory: *mut u8, total_bytes: usize) -> Self {
        debug_assert!(!base_memory.is_null(), "StackAllocator: base memory must not be null.");
        debug_assert!(total_bytes > 0, "StackAllocator: total bytes must be > 0.");

        Self {
            base: base_memory,
            total_bytes,
            current_offset: 0,
            #[cfg(debug_assertions)]
            last_allocation: ptr::null_mut(),
        }
    }

    // ── allocation ──────────────────────────────────────────────────────

    /// Allocate `size_bytes` from the top of the stack.
    ///
    /// The returned pointer is aligned to `alignment`.
    /// Returns null if the stack is exhausted.
    pub fn allocate(&mut self, size_bytes: usize, alignment: usize) -> *mut u8 {
        debug_assert!(
            is_power_of_two(alignment),
            "StackAllocator::allocate: alignment must be a power of two."
        );

        let base_addr = self.base as usize;

        // Raw address at the current bump position.
        let raw_addr = base_addr + self.current_offset;

        // We need to fit: [padding] [AllocationHeader] [aligned payload]
        let after_header = raw_addr + std::mem::size_of::<AllocationHeader>();
        let aligned_payload = align_forward(after_header, alignment);

        // Total bytes consumed from the raw position to the end of the payload.
        let adjustment = aligned_payload - raw_addr;
        let total_needed = adjustment + size_bytes;

        // Bounds check.
        if self.current_offset + total_needed > self.total_bytes {
            return ptr::null_mut(); // Stack exhausted.
        }

        // Store the previous offset so free() can rewind.
        let prev_offset = self.current_offset;

        // Bump the offset.
        self.current_offset += total_needed;

        // Write the header immediately before the aligned payload.
        unsafe {
            let header_ptr =
                (aligned_payload - std::mem::size_of::<AllocationHeader>()) as *mut AllocationHeader;
            ptr::write(
                header_ptr,
                AllocationHeader {
                    prev_offset,
                    adjustment,
                },
            );
        }

        let payload = aligned_payload as *mut u8;

        #[cfg(debug_assertions)]
        {
            self.last_allocation = payload;
        }

        payload
    }

    // ── deallocation ────────────────────────────────────────────────────

    /// Free the most recent allocation (LIFO).
    ///
    /// # Safety
    /// `ptr` must be the pointer returned by the last `allocate()` call
    /// that has not yet been freed. Passing null is a safe no-op.
    pub unsafe fn free(&mut self, ptr: *mut u8) {
        if ptr.is_null() {
            return;
        }

        debug_assert!(
            self.owns(ptr),
            "StackAllocator::free: pointer does not belong to this allocator."
        );

        #[cfg(debug_assertions)]
        {
            if !self.last_allocation.is_null() {
                debug_assert!(
                    ptr == self.last_allocation,
                    "StackAllocator::free: out-of-order free detected. \
                     You must free in strict LIFO order."
                );
            }
        }

        // Locate the header just before the payload.
        let payload_addr = ptr as usize;
        let header = (payload_addr - std::mem::size_of::<AllocationHeader>())
            as *const AllocationHeader;

        // Rewind the bump pointer to where it was before this allocation.
        self.current_offset = (*header).prev_offset;

        #[cfg(debug_assertions)]
        {
            // See C++ comment: we can't trivially recover the previous
            // allocation's start address, so set to null.
            self.last_allocation = ptr::null_mut();
        }
    }

    // ── reset / rewind ──────────────────────────────────────────────────

    /// O(1) full reset — rewinds the stack to the base.
    pub fn reset(&mut self, zero_memory: bool) {
        if zero_memory && self.current_offset > 0 {
            unsafe {
                ptr::write_bytes(self.base, 0, self.current_offset);
            }
        }
        self.current_offset = 0;

        #[cfg(debug_assertions)]
        {
            self.last_allocation = ptr::null_mut();
        }
    }

    /// Capture a save-point at the current stack position.
    pub fn save_point(&self) -> StackMarker {
        StackMarker {
            offset: self.current_offset,
        }
    }

    /// Rewind the stack to a previously captured save-point.
    pub fn restore_to_save_point(&mut self, marker: StackMarker, zero_memory: bool) {
        debug_assert!(
            marker.offset <= self.current_offset,
            "StackAllocator::restore_to_save_point: marker is ahead of current offset."
        );

        if zero_memory && marker.offset < self.current_offset {
            unsafe {
                let start = self.base.add(marker.offset);
                let bytes_to_zero = self.current_offset - marker.offset;
                ptr::write_bytes(start, 0, bytes_to_zero);
            }
        }

        self.current_offset = marker.offset;

        #[cfg(debug_assertions)]
        {
            self.last_allocation = ptr::null_mut();
        }
    }

    // ── queries ─────────────────────────────────────────────────────────

    /// Total capacity of the backing block, in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.total_bytes
    }

    /// Number of bytes currently in use.
    #[inline]
    pub fn used(&self) -> usize {
        self.current_offset
    }

    /// Remaining bytes available.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.total_bytes - self.current_offset
    }

    /// Raw pointer to the base of the backing block.
    #[inline]
    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// Returns true when no allocations are outstanding.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.current_offset == 0
    }

    /// Returns true when the stack is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.current_offset >= self.total_bytes
    }

    /// Check whether a pointer falls within this allocator's backing block.
    pub fn owns(&self, ptr: *const u8) -> bool {
        let addr = ptr as usize;
        let base = self.base as usize;
        addr >= base && addr < (base + self.total_bytes)
    }
}

// ── RAII scope guard ────────────────────────────────────────────────────────

/// Scoped save-point guard for StackAllocator.
///
/// Usage:
/// ```ignore
/// {
///     let _scope = StackScope::new(&mut stack, false);
///     let temp = stack.allocate(1024, 16);
///     // ... use temp ...
/// } // automatically restores to the save-point
/// ```
pub struct StackScope<'a> {
    allocator: &'a mut StackAllocator,
    marker: StackMarker,
    zero_on_restore: bool,
}

impl<'a> StackScope<'a> {
    /// Create a new scope guard, capturing the current stack position.
    pub fn new(allocator: &'a mut StackAllocator, zero_on_restore: bool) -> Self {
        let marker = allocator.save_point();
        Self {
            allocator,
            marker,
            zero_on_restore,
        }
    }
}

impl<'a> Drop for StackScope<'a> {
    fn drop(&mut self) {
        self.allocator
            .restore_to_save_point(self.marker, self.zero_on_restore);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{alloc, dealloc, Layout};

    fn alloc_test_block(size: usize) -> (*mut u8, Layout) {
        let layout = Layout::from_size_align(size, 64).unwrap();
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null());
        (ptr, layout)
    }

    #[test]
    fn test_stack_basic() {
        let (block, layout) = alloc_test_block(4096);
        let mut stack = unsafe { StackAllocator::new(block, 4096) };

        assert!(stack.is_empty());

        let p1 = stack.allocate(64, 16);
        assert!(!p1.is_null());
        assert!(stack.owns(p1));

        let p2 = stack.allocate(128, 16);
        assert!(!p2.is_null());

        // Free in LIFO order.
        unsafe { stack.free(p2) };
        unsafe { stack.free(p1) };

        assert!(stack.is_empty());

        unsafe { dealloc(block, layout) };
    }

    #[test]
    fn test_stack_save_restore() {
        let (block, layout) = alloc_test_block(4096);
        let mut stack = unsafe { StackAllocator::new(block, 4096) };

        let _ = stack.allocate(64, 16);
        let marker = stack.save_point();

        let _ = stack.allocate(256, 16);
        let _ = stack.allocate(256, 16);

        stack.restore_to_save_point(marker, false);
        // Only the first allocation should remain.

        unsafe { dealloc(block, layout) };
    }

    #[test]
    fn test_stack_alignment() {
        let (block, layout) = alloc_test_block(4096);
        let mut stack = unsafe { StackAllocator::new(block, 4096) };

        let _ = stack.allocate(1, 1);
        let p = stack.allocate(64, 64);
        assert!(!p.is_null());
        assert_eq!((p as usize) % 64, 0);

        unsafe { dealloc(block, layout) };
    }
}
