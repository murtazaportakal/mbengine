//! Top-level memory management system for the engine.
//!
//! Architecture:
//!   - At startup, reserves a single large contiguous block from the OS
//!     (via platform-specific virtual memory: VirtualAlloc / mmap).
//!   - Carves that block into named regions, each backing one allocator.
//!   - Provides a centralised Init/Shutdown lifecycle.
//!   - Provides factory access to pre-configured allocator instances.
//!   - Zero dynamic allocations after init() completes.
//!
//! Region layout (configurable via MemoryConfig):
//!
//!   ┌─────────────┬────────────────┬──────────────┬──────────────┬─────────┐
//!   │ Frame Arena  │ Persistent     │ ECS Pool     │ Stack Temp   │ Reserve │
//!   │ (scratch)    │ Arena          │              │              │         │
//!   └─────────────┴────────────────┴──────────────┴──────────────┴─────────┘
//!
//! Thread safety:
//!   init/shutdown are NOT thread-safe — call from the main thread only.
//!   Individual allocators are per-thread or externally synchronised.

use super::arena::ArenaAllocator;
use super::memory_utils::*;
use super::pool::PoolAllocator;
use super::stack::StackAllocator;

// ── platform imports ────────────────────────────────────────────────────────

#[cfg(windows)]
mod platform {
    #[link(name = "kernel32")]
    extern "system" {
        pub fn VirtualAlloc(
            lp_address: *mut u8,
            dw_size: usize,
            fl_allocation_type: u32,
            fl_protect: u32,
        ) -> *mut u8;

        pub fn VirtualFree(lp_address: *mut u8, dw_size: usize, dw_free_type: u32) -> i32;

        pub fn GetLastError() -> u32;
    }

    pub const MEM_RESERVE: u32 = 0x00002000;
    pub const MEM_COMMIT: u32 = 0x00001000;
    pub const MEM_RELEASE: u32 = 0x00008000;
    pub const PAGE_READWRITE: u32 = 0x04;
}

#[cfg(not(windows))]
mod platform {
    extern "C" {
        pub fn mmap(
            addr: *mut u8,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut u8;

        pub fn munmap(addr: *mut u8, len: usize) -> i32;
    }

    pub const PROT_READ: i32 = 0x1;
    pub const PROT_WRITE: i32 = 0x2;
    pub const MAP_PRIVATE: i32 = 0x02;
    pub const MAP_ANONYMOUS: i32 = 0x20;
    pub const MAP_FAILED: *mut u8 = !0usize as *mut u8;
}

// ── MemoryConfig ────────────────────────────────────────────────────────────

/// Configuration for the memory subsystem.
/// All sizes are in bytes. Use the `MB` / `GB` constants for readability.
///
/// Defaults target a mid-range desktop (≈256 MB total reservation).
pub struct MemoryConfig {
    /// Per-frame scratch arena — reset every frame.
    pub frame_arena_size: usize,
    /// Persistent arena — lives across frames, reset on level transitions.
    pub persistent_arena_size: usize,
    /// ECS component pool — fixed-size blocks.
    pub ecs_pool_size: usize,
    /// Default block size for the ECS pool allocator.
    pub ecs_pool_block_size: usize,
    /// Alignment for ECS pool blocks.
    pub ecs_pool_block_align: usize,
    /// Stack allocator for ordered temporary allocations.
    pub stack_size: usize,
    /// Unassigned reserve — available for future subsystems.
    pub reserve_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            frame_arena_size: 64 * MB,
            persistent_arena_size: 64 * MB,
            ecs_pool_size: 64 * MB,
            ecs_pool_block_size: 64,
            ecs_pool_block_align: 16,
            stack_size: 32 * MB,
            reserve_size: 32 * MB,
        }
    }
}

impl MemoryConfig {
    /// Compute the total reservation size.
    pub fn total_size(&self) -> usize {
        self.frame_arena_size
            + self.persistent_arena_size
            + self.ecs_pool_size
            + self.stack_size
            + self.reserve_size
    }
}

// ── MemorySubsystem ─────────────────────────────────────────────────────────

/// Top-level memory management system. Reserves a single OS allocation and
/// carves it into typed allocator regions.
pub struct MemorySubsystem {
    config: MemoryConfig,

    /// OS-reserved block.
    memory_block: *mut u8,
    memory_block_size: usize,

    /// Allocators stored as heap Boxes. In C++ these were placement-new'd
    /// into the reserve region; in Rust we use Box for safe ownership since
    /// the allocator structs are tiny (< 64 bytes each).
    frame_arena: Option<Box<ArenaAllocator>>,
    persistent_arena: Option<Box<ArenaAllocator>>,
    ecs_pool: Option<Box<PoolAllocator>>,
    temp_stack: Option<Box<StackAllocator>>,

    reserve_base: *mut u8,
}

// Safety: MemorySubsystem is single-threaded by design (same as C++ version).
// The raw pointers inside point to OS-reserved memory owned by this struct.
unsafe impl Send for MemorySubsystem {}

impl MemorySubsystem {
    /// Create an uninitialised memory subsystem. Call `init()` before use.
    pub fn new() -> Self {
        Self {
            config: MemoryConfig::default(),
            memory_block: std::ptr::null_mut(),
            memory_block_size: 0,
            frame_arena: None,
            persistent_arena: None,
            ecs_pool: None,
            temp_stack: None,
            reserve_base: std::ptr::null_mut(),
        }
    }

    /// Reserve virtual memory from the OS and carve it into regions.
    ///
    /// Returns `true` on success, `false` if the OS reservation failed.
    ///
    /// Must not be called more than once without an intervening `shutdown()`.
    pub fn init(&mut self, config: MemoryConfig) -> bool {
        debug_assert!(
            self.memory_block.is_null(),
            "MemorySubsystem::init: already initialised. Call shutdown() first."
        );

        self.memory_block_size = config.total_size();

        // 1. Reserve virtual memory from the OS.
        self.memory_block = Self::platform_reserve(self.memory_block_size);
        if self.memory_block.is_null() {
            eprintln!(
                "[MemorySubsystem] FATAL: Failed to reserve {} bytes from the OS.",
                self.memory_block_size
            );
            return false;
        }

        // 2. Carve the block into contiguous regions.
        let mut cursor = self.memory_block;

        unsafe {
            // Frame Arena
            let frame_region = cursor;
            cursor = cursor.add(config.frame_arena_size);

            // Persistent Arena
            let persist_region = cursor;
            cursor = cursor.add(config.persistent_arena_size);

            // ECS Pool
            let ecs_region = cursor;
            cursor = cursor.add(config.ecs_pool_size);

            // Temp Stack
            let stack_region = cursor;
            cursor = cursor.add(config.stack_size);

            // Reserve (unassigned)
            self.reserve_base = cursor;

            // 3. Construct the allocators.
            self.frame_arena = Some(Box::new(ArenaAllocator::new(
                frame_region,
                config.frame_arena_size,
            )));

            self.persistent_arena = Some(Box::new(ArenaAllocator::new(
                persist_region,
                config.persistent_arena_size,
            )));

            self.ecs_pool = Some(Box::new(PoolAllocator::new(
                ecs_region,
                config.ecs_pool_size,
                config.ecs_pool_block_size,
                config.ecs_pool_block_align,
            )));

            self.temp_stack = Some(Box::new(StackAllocator::new(
                stack_region,
                config.stack_size,
            )));
        }

        println!(
            "[MemorySubsystem] Initialised: {} MB total \
             (frame={} MB, persist={} MB, ecs={} MB, stack={} MB, reserve={} MB)",
            self.memory_block_size / MB,
            config.frame_arena_size / MB,
            config.persistent_arena_size / MB,
            config.ecs_pool_size / MB,
            config.stack_size / MB,
            config.reserve_size / MB,
        );

        self.config = config;
        true
    }

    /// Convenience: init with default config.
    pub fn init_default(&mut self) -> bool {
        self.init(MemoryConfig::default())
    }

    /// Release all OS-reserved memory.
    /// All allocator references become invalid after this call.
    pub fn shutdown(&mut self) {
        if self.memory_block.is_null() {
            return;
        }

        // Drop allocators in reverse order.
        self.temp_stack = None;
        self.ecs_pool = None;
        self.persistent_arena = None;
        self.frame_arena = None;

        // Release the OS reservation.
        Self::platform_release(self.memory_block, self.memory_block_size);

        self.memory_block = std::ptr::null_mut();
        self.memory_block_size = 0;
        self.reserve_base = std::ptr::null_mut();

        println!("[MemorySubsystem] Shutdown complete.");
    }

    /// Returns true after a successful `init()` and before `shutdown()`.
    #[inline]
    pub fn is_initialised(&self) -> bool {
        !self.memory_block.is_null()
    }

    // ── allocator access ────────────────────────────────────────────────

    /// Per-frame scratch arena. Reset this at the end of every frame.
    pub fn frame_arena(&mut self) -> &mut ArenaAllocator {
        self.frame_arena
            .as_mut()
            .expect("MemorySubsystem::frame_arena: not initialised.")
    }

    /// Persistent arena. Lives across frames; reset on level transitions.
    pub fn persistent_arena(&mut self) -> &mut ArenaAllocator {
        self.persistent_arena
            .as_mut()
            .expect("MemorySubsystem::persistent_arena: not initialised.")
    }

    /// ECS component pool.
    pub fn ecs_pool(&mut self) -> &mut PoolAllocator {
        self.ecs_pool
            .as_mut()
            .expect("MemorySubsystem::ecs_pool: not initialised.")
    }

    /// General-purpose stack allocator.
    pub fn temp_stack(&mut self) -> &mut StackAllocator {
        self.temp_stack
            .as_mut()
            .expect("MemorySubsystem::temp_stack: not initialised.")
    }

    // ── queries ─────────────────────────────────────────────────────────

    /// The raw OS-reserved block base.
    #[inline]
    pub fn memory_block_base(&self) -> *mut u8 {
        self.memory_block
    }

    /// Total bytes reserved from the OS.
    #[inline]
    pub fn memory_block_size(&self) -> usize {
        self.memory_block_size
    }

    /// The active configuration.
    #[inline]
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Pointer to the start of the unassigned reserve region.
    #[inline]
    pub fn reserve_base(&self) -> *mut u8 {
        self.reserve_base
    }

    // ── platform virtual memory ─────────────────────────────────────────

    fn platform_reserve(size: usize) -> *mut u8 {
        #[cfg(windows)]
        {
            unsafe {
                let block = platform::VirtualAlloc(
                    std::ptr::null_mut(),
                    size,
                    platform::MEM_RESERVE | platform::MEM_COMMIT,
                    platform::PAGE_READWRITE,
                );
                if block.is_null() {
                    eprintln!(
                        "[MemorySubsystem] VirtualAlloc failed (requested {} bytes, error {}).",
                        size,
                        platform::GetLastError()
                    );
                }
                block
            }
        }

        #[cfg(not(windows))]
        {
            unsafe {
                let block = platform::mmap(
                    std::ptr::null_mut(),
                    size,
                    platform::PROT_READ | platform::PROT_WRITE,
                    platform::MAP_PRIVATE | platform::MAP_ANONYMOUS,
                    -1,
                    0,
                );
                if block == platform::MAP_FAILED {
                    eprintln!("[MemorySubsystem] mmap failed (requested {} bytes).", size);
                    return std::ptr::null_mut();
                }
                block
            }
        }
    }

    fn platform_release(block: *mut u8, _size: usize) {
        if block.is_null() {
            return;
        }

        #[cfg(windows)]
        {
            unsafe {
                platform::VirtualFree(block, 0, platform::MEM_RELEASE);
            }
        }

        #[cfg(not(windows))]
        {
            unsafe {
                platform::munmap(block, _size);
            }
        }
    }
}

impl Default for MemorySubsystem {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MemorySubsystem {
    fn drop(&mut self) {
        if !self.memory_block.is_null() {
            self.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_shutdown() {
        let mut mem = MemorySubsystem::new();
        let ok = mem.init_default();
        assert!(ok, "MemorySubsystem::init() failed");
        assert!(mem.is_initialised());

        // Verify allocators are accessible.
        assert!(mem.frame_arena().capacity() == 64 * MB);
        assert!(mem.persistent_arena().capacity() == 64 * MB);
        assert!(mem.temp_stack().capacity() == 32 * MB);

        mem.shutdown();
        assert!(!mem.is_initialised());
    }

    #[test]
    fn test_allocators_work_after_init() {
        let mut mem = MemorySubsystem::new();
        assert!(mem.init_default());

        // Allocate from frame arena.
        let p = mem.frame_arena().allocate(1024, 16);
        assert!(!p.is_null());

        // Allocate from persistent arena.
        let p2 = mem.persistent_arena().allocate(2048, 16);
        assert!(!p2.is_null());

        // Allocate from temp stack.
        let p3 = mem.temp_stack().allocate(512, 16);
        assert!(!p3.is_null());

        mem.shutdown();
    }
}
