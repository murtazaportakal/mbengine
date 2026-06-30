//! Fixed-size block pool allocator with embedded free-list.
//!
//! Architecture:
//!   - Takes an externally owned, pre-allocated memory block.
//!   - Divides it into N equally-sized chunks at initialisation.
//!   - O(1) allocation: pop the head of a singly-linked free-list.
//!   - O(1) deallocation: push the returned block onto the free-list head.
//!   - Zero fragmentation — every block is the same size.
//!   - Zero heap allocations internally.
//!   - The free-list node pointer is embedded *inside* the free block itself
//!     (intrusive), so there is zero per-block metadata overhead.
//!
//! Intended usage:
//!   ECS component arrays, particle pools, command objects, fixed-size
//!   message queues — any pattern where objects are the same size and
//!   individual alloc/free is required.
//!
//! Thread safety:
//!   None. One PoolAllocator per thread, or synchronise externally.

use super::memory_utils::{align_forward, is_power_of_two};
use std::ptr;

// ── FreeNode ────────────────────────────────────────────────────────────────

/// Intrusive free-list node. When a block is free, its first bytes store a
/// pointer to the next free block. When allocated, this is overwritten by
/// the user's data — zero overhead.
#[repr(C)]
struct FreeNode {
    next: *mut FreeNode,
}

// ── PoolAllocator ───────────────────────────────────────────────────────────

/// Fixed-size block pool allocator with intrusive embedded free-list.
pub struct PoolAllocator {
    base: *mut u8,
    total_bytes: usize,
    block_size: usize,
    block_alignment: usize,
    block_count: usize,
    free_count: usize,
    free_list_head: *mut FreeNode,
}

impl PoolAllocator {
    // ── lifecycle ───────────────────────────────────────────────────────

    /// Construct a pool over an externally owned memory block.
    ///
    /// # Safety
    /// - `base_memory` must point to a valid, writable block of at least
    ///   `total_bytes` bytes.
    /// - The block must remain valid for the lifetime of this allocator.
    /// - `block_size` must be >= size_of::<*mut u8>() (pointer size).
    /// - `block_alignment` must be a power of two.
    pub unsafe fn new(
        base_memory: *mut u8,
        total_bytes: usize,
        block_size: usize,
        block_alignment: usize,
    ) -> Self {
        debug_assert!(
            !base_memory.is_null(),
            "PoolAllocator: base memory must not be null."
        );
        debug_assert!(total_bytes > 0, "PoolAllocator: total bytes must be > 0.");
        debug_assert!(
            block_size >= std::mem::size_of::<FreeNode>(),
            "PoolAllocator: block size must be >= size_of pointer."
        );
        debug_assert!(
            is_power_of_two(block_alignment),
            "PoolAllocator: alignment must be a power of two."
        );

        // Round block_size up to a multiple of block_alignment so that every
        // block start is naturally aligned when laid out sequentially.
        let mask = block_alignment - 1;
        let aligned_block_size = (block_size + mask) & !mask;

        // Compute the aligned start of the first block.
        let raw_start = base_memory as usize;
        let aligned_start = align_forward(raw_start, block_alignment);
        let front_padding = aligned_start - raw_start;

        // How many blocks fit after the initial alignment padding?
        let count = if front_padding >= total_bytes {
            0
        } else {
            (total_bytes - front_padding) / aligned_block_size
        };

        let mut pool = Self {
            base: base_memory,
            total_bytes,
            block_size: aligned_block_size,
            block_alignment,
            block_count: count,
            free_count: count,
            free_list_head: ptr::null_mut(),
        };

        pool.build_free_list();
        pool
    }

    // ── allocation ──────────────────────────────────────────────────────

    /// O(1) allocate a single block from the pool.
    ///
    /// Returns a pointer to a block of `block_size()` usable bytes,
    /// or null if the pool is exhausted.
    pub fn allocate(&mut self) -> *mut u8 {
        if self.free_list_head.is_null() {
            return ptr::null_mut();
        }

        // Pop the head of the free-list.
        let node = self.free_list_head;
        unsafe {
            self.free_list_head = (*node).next;
        }
        self.free_count -= 1;

        node as *mut u8
    }

    // ── deallocation ────────────────────────────────────────────────────

    /// O(1) return a block to the pool.
    ///
    /// # Safety
    /// `ptr` must be a pointer previously returned by `allocate()` on this
    /// pool instance. Passing null is a safe no-op. Double-free is undefined
    /// behaviour in release builds.
    pub unsafe fn free(&mut self, ptr: *mut u8) {
        if ptr.is_null() {
            return;
        }

        debug_assert!(
            self.owns(ptr),
            "PoolAllocator::free: pointer does not belong to this pool."
        );

        // Push onto the head of the free-list.
        let node = ptr as *mut FreeNode;
        (*node).next = self.free_list_head;
        self.free_list_head = node;
        self.free_count += 1;
    }

    // ── reset ───────────────────────────────────────────────────────────

    /// O(N) full reset — rebuilds the entire free-list from scratch.
    /// All outstanding allocations are logically invalidated.
    pub fn reset(&mut self, zero_memory: bool) {
        if zero_memory {
            unsafe {
                ptr::write_bytes(self.base, 0, self.total_bytes);
            }
        }

        self.free_count = self.block_count;
        self.build_free_list();
    }

    // ── queries ─────────────────────────────────────────────────────────

    /// The usable size of each block (after alignment padding).
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// The alignment guarantee of each block.
    #[inline]
    pub fn block_alignment(&self) -> usize {
        self.block_alignment
    }

    /// Maximum number of blocks that fit in the backing memory.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.block_count
    }

    /// Number of blocks currently allocated (in use).
    #[inline]
    pub fn allocated_count(&self) -> usize {
        self.block_count - self.free_count
    }

    /// Number of blocks currently free.
    #[inline]
    pub fn free_count(&self) -> usize {
        self.free_count
    }

    /// Total capacity of the backing block, in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.total_bytes
    }

    /// True when no blocks are allocated.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.free_count == self.block_count
    }

    /// True when all blocks are allocated.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.free_count == 0
    }

    /// Raw pointer to the base of the backing block.
    #[inline]
    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// Check whether a pointer falls within this pool's backing block
    /// AND is properly aligned to a block boundary.
    pub fn owns(&self, ptr: *const u8) -> bool {
        let addr = ptr as usize;
        let base = self.base as usize;

        let aligned_base = align_forward(base, self.block_alignment);

        if addr < aligned_base {
            return false;
        }

        let offset = addr - aligned_base;

        // Must be within the allocated region.
        if offset >= self.block_count * self.block_size {
            return false;
        }

        // Must be exactly at a block boundary.
        offset.is_multiple_of(self.block_size)
    }

    // ── internal ────────────────────────────────────────────────────────

    /// Build (or rebuild) the free-list across all blocks.
    fn build_free_list(&mut self) {
        if self.block_count == 0 {
            self.free_list_head = ptr::null_mut();
            return;
        }

        let raw_base = self.base as usize;
        let aligned_base = align_forward(raw_base, self.block_alignment);

        // Thread blocks in forward order so that sequential allocations
        // yield sequential (cache-friendly) memory addresses.
        self.free_list_head = aligned_base as *mut FreeNode;

        unsafe {
            let mut current = self.free_list_head;
            for i in 1..self.block_count {
                let next = (aligned_base + i * self.block_size) as *mut FreeNode;
                (*current).next = next;
                current = next;
            }
            (*current).next = ptr::null_mut(); // Terminate the list.
        }
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
    fn test_basic_pool() {
        let (block, layout) = alloc_test_block(4096);
        let mut pool = unsafe { PoolAllocator::new(block, 4096, 64, 16) };

        assert!(pool.is_empty());
        assert!(pool.block_count() > 0);

        let p1 = pool.allocate();
        assert!(!p1.is_null());
        assert!(pool.owns(p1));
        assert_eq!(pool.allocated_count(), 1);

        let p2 = pool.allocate();
        assert!(!p2.is_null());
        assert_ne!(p1, p2);

        unsafe { pool.free(p1) };
        assert_eq!(pool.allocated_count(), 1);

        unsafe { pool.free(p2) };
        assert!(pool.is_empty());

        unsafe { dealloc(block, layout) };
    }

    #[test]
    fn test_pool_exhaustion_and_reset() {
        let (block, layout) = alloc_test_block(256);
        let mut pool = unsafe { PoolAllocator::new(block, 256, 64, 16) };

        let count = pool.block_count();
        for _ in 0..count {
            let p = pool.allocate();
            assert!(!p.is_null());
        }
        assert!(pool.is_full());

        let p_exhaust = pool.allocate();
        assert!(p_exhaust.is_null());

        pool.reset(false);
        assert!(pool.is_empty());

        unsafe { dealloc(block, layout) };
    }
}
