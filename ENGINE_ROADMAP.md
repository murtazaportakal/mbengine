# Custom Game Engine — Architecture & Roadmap

> **Last updated:** 2026-06-30 — Migrated from C++20 to Rust.

---

## Core Architectural Constraints

These rules are **non-negotiable** across all future sessions:

| Constraint | Detail |
|---|---|
| **Language** | Rust (2021 edition, stable toolchain) |
| **Paradigm** | Strict Data-Oriented Design (DoD). No deep inheritance trees. Flat, tightly packed arrays. Entity Component System (ECS). |
| **Memory** | **Zero** heap allocations in the main game loop. All allocations go through custom allocators (Arena, Pool, Stack). |
| **Standard Library** | Minimal `std` usage in performance-critical paths. Custom cache-friendly containers. |
| **Graphics** | Vulkan API via `ash` crate (decoupled — built later). |
| **Cache coherency** | Prioritised in every data structure. |
| **Code style** | One file at a time. No partial code or `// TODO` stubs unless explicitly outlining a module. Production-ready on first write. |
| **Unsafe** | Encapsulated behind safe public APIs. All `unsafe` blocks documented with `# Safety` comments. |

---

## Completed — Memory Management Layer

### File Inventory

```
src/memory/
├── mod.rs                  Module declarations + re-exports
├── memory_utils.rs         Alignment utilities + size constants
├── arena.rs                Linear bump allocator (ArenaAllocator)
├── pool.rs                 Fixed-size block free-list (PoolAllocator)
├── stack.rs                LIFO allocator + StackScope RAII guard
└── subsystem.rs            OS reservation + region management (MemorySubsystem)
```

### Per-File Summary

#### memory_utils.rs
- `is_power_of_two()`, `align_forward()`, `align_size()`, `is_aligned()` — `const fn` helpers.
- Size constants: `KB`, `MB`, `GB`, `PAGE_SIZE`, `DEFAULT_ALIGNMENT` (16).

#### arena.rs
- Takes an externally owned memory block.
- O(1) bump allocation with configurable alignment (default 16-byte).
- O(1) full `reset()` with optional zeroing.
- `ArenaMarker` save-point system for partial rewinds.
- Typed helpers: `allocate_array::<T>(count)`, `alloc_new::<T>(value)`.
- `owns(ptr)` for debug bounds checking.
- Module: `crate::memory`.

#### pool.rs
- Fixed-size block allocator with **intrusive embedded free-list** (zero per-block metadata overhead).
- O(1) `allocate()` (pop free-list head), O(1) `free()` (push to head).
- Constructor rounds `block_size` up to `block_alignment`, computes block count after front-padding.
- `owns(ptr)` validates both range AND block-boundary alignment.
- O(N) `reset()` rebuilds free-list.

#### stack.rs
- LIFO allocator with per-allocation `AllocationHeader` (stores `prev_offset` + `adjustment`).
- O(1) `allocate()`: reserves header space, aligns payload, bumps pointer.
- O(1) `free()`: reads header, rewinds to `prev_offset`. Debug builds assert LIFO order via `#[cfg(debug_assertions)]`.
- `StackMarker` save-point and `restore_to_save_point()` for bulk rewind.
- **`StackScope`** RAII guard via `Drop` trait — automatically restores save-point on scope exit.

#### subsystem.rs
- `MemoryConfig` struct with defaults: 64 MB frame arena, 64 MB persistent arena, 64 MB ECS pool, 32 MB temp stack, 32 MB reserve = **256 MB total**.
- `init()`: single OS syscall (`VirtualAlloc` on Windows, `mmap` on POSIX via `extern` FFI), carves block into regions.
- `shutdown()`: drops allocators, OS release.
- Typed accessors: `frame_arena()`, `persistent_arena()`, `ecs_pool()`, `temp_stack()`, `reserve_base()`.
- **Zero heap allocations** for OS memory — allocator metadata uses `Box` (tiny, < 64 bytes each).

---

## Completed — ECS Core

### File Inventory

```
src/ecs/
├── mod.rs                  Module declarations + re-exports
├── types.rs                EntityId encoding, ComponentTypeId, masks, constants
├── component_array.rs      Sparse-set generic — performance-critical core
├── entity_manager.rs       Entity lifecycle
├── system.rs               System trait
└── world.rs                Top-level ECS container
```

### Per-File Summary

#### types.rs
- `EntityId` = `u32` with 20-bit index + 12-bit generation packing.
- `ComponentMask` = `u64` bitset (one bit per component type).
- `ComponentTypeId` = `u8`, assigned via `get_component_type_id::<T>()` using `TypeId`-keyed global registry.
- Constants: `MAX_ENTITIES` (1M), `MAX_COMPONENT_TYPES` (64), `MAX_SYSTEMS` (64).
- Helpers: `make_entity_id()`, `get_entity_index()`, `get_entity_generation()`, `is_valid_entity()`.
- Module: `crate::ecs`.

#### component_array.rs
- `ComponentArrayOps` trait (type-erased interface): `entity_destroyed()`, `count()`.
- `ComponentArray<T>` generic struct with **sparse-set** data structure.
- O(1) `insert()`, O(1) `remove()` (swap-and-pop), O(1) `get()`, O(1) `has()`.
- Dense-array `as_slice()` / `as_mut_slice()` for cache-optimal iteration — zero gaps, zero indirection.
- All memory from `ArenaAllocator`. `Drop` impl calls destructors on cleanup.

#### entity_manager.rs
- Per-slot `generations` and `component_masks` arrays.
- Ring-buffer `recycle_queue` for freed entity indices.
- `create_entity()`: pop recycled or bump fresh counter, pack into EntityId.
- `destroy_entity()`: increment generation, clear mask, push to recycle queue.
- `is_alive()`: compare stored generation vs. ID generation field.
- All arrays from `ArenaAllocator`.

#### system.rs
- `System` trait: `fn update(&mut self, dt: f32, world: &mut World)`.
- `required_components()` / `set_required_components()` for entity filtering.
- `build_mask()` helper function for type-safe mask construction.

#### world.rs
- Owns `EntityManager`, up to 64 `ComponentArrayOps` trait objects, up to 64 `Box<dyn System>`.
- `register_component::<T>(dense_capacity)`: constructs `ComponentArray<T>` in arena.
- `register_system()`: accepts `Box<dyn System>`.
- `create_entity()`, `destroy_entity()`, `is_alive()`.
- `add_component::<T>()`, `remove_component::<T>()`, `get_component::<T>()`, `has_component::<T>()`.
- `get_component_array::<T>()` for direct dense-array access in systems.
- `update_systems(dt)` runs all systems in registration order.
- `Drop` impl properly destroys component arrays, systems, and entity manager.
- **All component/entity memory from `PersistentArena`. Zero heap allocations for game data.**

---

## Build System

```
Cargo.toml                  Rust 2021 edition, zero dependencies
src/                        Engine library source
tests/test_ecs.rs           ECS smoke test (all assertions passing)
```

### Build Commands

```bash
cargo build                 # Debug build
cargo build --release       # Optimized release build
cargo test                  # Run all tests
cargo clippy                # Lint check
```

---

## Test Suite

`tests/test_ecs.rs` — Comprehensive integration test covering:
1. MemorySubsystem init/shutdown
2. World creation + component registration
3. Entity create / destroy / generation recycling
4. Component add / get / remove (swap-and-pop correctness)
5. Dense-array iteration
6. System execution (MovementSystem)
7. Stale handle detection

All assertions passing. **20 unit tests + 1 integration test = 21 total.**

---

## Completed — Custom Containers & Platform

### Phase 3: Custom Containers
- **`FixedArray<T, N>`**: Stack-allocated fixed-capacity array.
- **`DynamicArray<T>`**: Arena-backed growable array.
- **`RingBuffer<T>`**: Lock-free SPSC ring buffer.
- **`HashMap<K, V>`**: Robin Hood hashed map.
- **`FixedString<N>`**: Stack-allocated string.

### Phase 4: Platform & Logging
- **`win32.rs`**: Pure zero-dependency FFI bindings.
- **`window.rs`**: Win32 window creation and message pumping (`HWND` access).
- **`timer.rs`**: High-resolution nanosecond timer via `QueryPerformanceCounter`.
- **`logger.rs`**: Lock-free global logger backed by `RingBuffer<FixedString>`.

---

## Backlog — Prioritised Next Steps


### Phase 5: Math Library
| Priority | File(s) | Description |
|---|---|---|
| **P1** | `src/math/vec.rs` | SIMD-friendly Vec2/Vec3/Vec4 types (16-byte aligned). |
| **P1** | `src/math/mat4.rs` | Column-major 4×4 matrix. |
| **P1** | `src/math/quat.rs` | Quaternion for rotations. |
| **P2** | `src/math/transform.rs` | Position + Rotation + Scale, SoA-friendly. |
| **P2** | `src/math/aabb.rs` | Axis-aligned bounding box for broad-phase. |

### Phase 6: Renderer Foundation
| Priority | File(s) | Description |
|---|---|---|
| **P2** | `src/renderer/device.rs` | Abstract render backend interface (Vulkan impl via `ash` crate). |
| **P2** | `src/renderer/vulkan/device.rs` | Instance, physical/logical device, queues. |
| **P2** | `src/renderer/vulkan/swapchain.rs` | Swapchain creation and present. |
| **P3** | `src/renderer/vulkan/pipeline.rs` | Shader modules, pipeline layout, graphics pipeline. |

### Phase 7: Game Loop & Application
| Priority | File(s) | Description |
|---|---|---|
| **P2** | `src/app/application.rs` | Main loop: Init → while(running) { Input → Update → Render → FrameArena.reset() } → Shutdown. |
| **P2** | `src/app/input.rs` | Keyboard/mouse state, event queue. |

---

## Session Handoff Notes

- **All files compile cleanly** under Rust stable 1.96.0 with `cargo clippy` passing clean.
- **Build system:** `Cargo.toml` at project root. `cargo build` / `cargo test`.
- **Toolchain:** `stable-x86_64-pc-windows-gnu` (MinGW linker).
- **ECS uses `PersistentArena`** (not `ECSPool`). The sparse-set design needs variable-size arrays, not fixed-block pools. The `ECSPool` remains available for fixed-size runtime objects.
- **Destruction order matters:** `World` must be dropped before `MemorySubsystem::shutdown()` is called, since World's `Drop` impl accesses arena memory. Use `drop(world)` or ensure correct declaration order.
- **`MemoryConfig` defaults are tunable.** Current: 256 MB total.
- **Entity capacity:** 1M max entities (20-bit index). Each ComponentArray sparse array costs 4 MB. With all 64 types registered, that's 256 MB.
- **Component type limit:** 64 (fits in `u64` mask). Sufficient for most indie games.
- **Test suite:** `tests/test_ecs.rs`, `tests/test_containers.rs`, `tests/test_platform.rs`. Comprehensive testing for ECS, memory, containers, and platform. All passing flawlessly.
- **C++ source files are completely removed** (Phase 2 constraint completed).

---

*Start the next session with: "Continue from ENGINE_ROADMAP.md — build Phase 5: Math Library"*
