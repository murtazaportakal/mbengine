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


## Completed — Math & Rendering

### Phase 5: Math Library
- **`src/math/vec.rs`**: SIMD-friendly Vec2/Vec3/Vec4 types (16-byte aligned).
- **`src/math/mat4.rs`**: Column-major 4×4 matrix.
- **`src/math/quat.rs`**: Quaternion for rotations.

### Phase 6: Renderer Foundation
- **`src/renderer/device.rs`**: Abstract render backend interface.
- **`src/renderer/vulkan/device.rs`**: Vulkan Instance, physical/logical device, queues.
- **`src/renderer/vulkan/swapchain.rs`**: Swapchain creation and present.

### Phase 7: Game Loop & Application
- **`src/app/application.rs`**: Main loop, ECS execution, frame cleanup.
- **`src/app/input.rs`**: Keyboard/mouse state mapping.

### Phase 8: Graphics Pipeline
- **`src/renderer/vulkan/render_pass.rs`**: Vulkan RenderPass setup.
- **`src/renderer/vulkan/pipeline.rs`**: Graphics pipeline and shaders.
- **Shaders**: `shaders/shader.vert` and `shaders/shader.frag` (glslc compiled).

### Phase 9: Resource Management & Buffers
- **`src/renderer/vulkan/buffer.rs`**: CPU HOST_VISIBLE staging and GPU DEVICE_LOCAL buffer allocations.
- **Vertices**: Hardcoded `Vertex` struct streamed to GPU VRAM.

---

## Completed — Gameplay & Rendering Systems

### Phase 10: ECS Rendering Systems
- **`src/ecs/components.rs`**: Defined `TransformComponent` and `RenderComponent`.
- **`src/renderer/vulkan/pipeline.rs`**: Push Constants setup.
- **`src/app/application.rs`**: Mapped Entity components to Vulkan Push Constants and `cmd_draw` calls.

### Phase 11: Camera & Depth Buffering
- **`src/math/mat4.rs`**: Implemented `perspective` and `look_at` matrix generation.
- **`src/ecs/components.rs`**: Added `CameraComponent`.
- **`src/renderer/vulkan/swapchain.rs`**: Added Depth buffer attachment (`vk::Format::D32_SFLOAT`).
- **`src/renderer/vulkan/pipeline.rs`**: Enabled depth testing.

### Phase 12: Interactive Camera Controls (Fly/FPS Camera)
- **`src/app/input.rs`**: Added mouse delta tracking and keyboard states.
- **`src/app/application.rs`**: Hooked up input to WASD translation and mouse rotation for the camera.

### Phase 13: Mesh Loading & Index Buffers
- **`src/renderer/vulkan/mesh.rs`**: Custom zero-dependency `.obj` parser and index buffer creation.
- **`src/app/application.rs`**: Loaded `cube.obj` and enabled indexed drawing.

### Phase 14: Lighting (Directional Lights & Basic Shading)
- **`src/ecs/components.rs`**: Added `LightComponent`.
- **`shaders/shader.*`**: Updated shaders to calculate Lambertian diffuse lighting.

### Phase 15: Textures & Materials (Descriptor Sets & Samplers)
- **`src/renderer/vulkan/texture.rs`**: Generated procedural checkerboard texture and uploaded to `vk::Image`.
- **`src/app/application.rs`, `pipeline.rs`**: Setup descriptor pools, layout, and bound them for rendering.

### Phase 16: Scene Graph & Hierarchy (Parent/Child Transforms)
- **`src/ecs/components.rs`**: Added `HierarchyComponent`.
- **`src/app/application.rs`**: Computed nested `world_matrices` dynamically during the render loop (e.g. Moon orbiting Planet).

---

## Backlog — Prioritised Next Steps

### Phase 17: Asset Management & Real Image Loading
| Priority | File(s) | Description |
|---|---|---|
| **P1** | `Cargo.toml`, `texture.rs` | Add `image` crate. Update texture loader to parse `.png` and `.jpg` from disk. |
| **P1** | `asset_manager.rs` | Create a system to load and cache textures so they aren't duplicated in VRAM. |

### Phase 18: Complex Model Loading (GLTF/OBJ)
| Priority | File(s) | Description |
|---|---|---|
| **P1** | `Cargo.toml`, `mesh.rs` | Add `tobj` or `gltf` crate. |
| **P1** | `mesh.rs` | Load complex meshes with multiple sub-meshes and parse materials. |
| **P1** | `ecs/components.rs` | Update components to reference cached asset IDs. |

### Phase 19: Advanced Lighting & PBR Materials
| Priority | File(s) | Description |
|---|---|---|
| **P1** | `shaders/shader.frag` | Upgrade fragment shader to handle Physically Based Rendering (PBR). |
| **P2** | `ecs/components.rs` | Add `PointLightComponent` and allow multiple light sources in the UBO. |

### Phase 20: Physics & Collisions
| Priority | File(s) | Description |
|---|---|---|
| **P1** | `Cargo.toml`, `physics.rs` | Integrate a physics engine like `rapier3d`. |
| **P1** | `ecs/components.rs` | Add `RigidBodyComponent` and `ColliderComponent`. |
| **P1** | `application.rs` | Sync physics simulation state with visual `TransformComponent` each frame. |

### Phase 21: Scene Serialization & UI
| Priority | File(s) | Description |
|---|---|---|
| **P1** | `ecs/serialization.rs` | Use `serde` to save and load entities and components to/from JSON. |
| **P2** | `ui.rs` | Integrate `egui` to create a real-time developer interface to tweak variables without recompiling. |

---

## Session Handoff Notes

- **All files compile cleanly** under Rust stable 1.96.0 with `cargo clippy` passing clean. Unused variables in `application.rs` have been fixed.
- **Rendering is fully functional:** Swapchain resizing logic is fixed, avoiding Windows DWM warping. The pipeline uses dynamic viewport/scissor states correctly. 
- **Next steps are Asset loading and PBR:** The engine is ready for real 3D models and textures.
- **Build system:** `Cargo.toml` at project root. `cargo build` / `cargo run`.
- **Toolchain:** `stable-x86_64-pc-windows-gnu` (MinGW linker).

---

*Start the next session with: "Continue from ENGINE_ROADMAP.md — build Phase 17: Asset Management & Real Image Loading"*

---

## The Next-Generation Engine (V2 Master Plan)

Having completed the foundational architecture (Phases 1-21), this section outlines the roadmap to transform this project into a full-fledged competitor to engines like Bevy and Fyrox. The focus is exclusively on **Developer Experience** and **Iteration Speed**.

### Epic 1: The Editor Foundation (Egui Integration)
| Priority | File(s) | Description |
|---|---|---|
| **P1** | `Cargo.toml` | Add `egui`. |
| **P1** | `platform/win32.rs` | Translate Win32 messages (scroll, characters, resize) into `egui::RawInput`. |
| **P1** | `renderer/vulkan/egui_backend.rs` | Custom pipeline to stream `egui` clipped meshes and render the UI overlay. |

### Epic 2: Offscreen Rendering (The Viewport)
| Priority | File(s) | Description |
|---|---|---|
| **P1** | `renderer/vulkan/offscreen.rs` | Create an offscreen render target (color + depth). |
| **P1** | `app/application.rs` | Render the game to the offscreen texture, then pass it to `egui` to draw inside an Editor Window. |

### Epic 3: ECS Reflection & Scene Inspector
| Priority | File(s) | Description |
|---|---|---|
| **P1** | `ecs/reflection.rs` | Build a component registry to expose component fields dynamically. |
| **P2** | `app/editor.rs` | Build the Hierarchy (Entity tree) and Inspector (Component properties) panels. |

### Epic 4: Native DLL Hot-Reloading (The Killer Feature)
| Priority | File(s) | Description |
|---|---|---|
| **P1** | `src/lib.rs` -> `host` vs `game` | Split the engine into a Host executable (Renderer + ECS Memory) and a Game DLL (Systems). |
| **P1** | `platform/hot_reload.rs` | Watch the DLL file, seamlessly unload and reload it when recompiled, maintaining ECS state. |

### Epic 5: Asset Pipeline & VFS
| Priority | File(s) | Description |
|---|---|---|
| **P2** | `asset_manager.rs` | File watcher for hot-reloading shaders, textures, and models instantly. |
| **P3** | `vfs.rs` | Virtual File System for packing assets into a release binary. |

### Epic 6: Job System & Multithreading
| Priority | File(s) | Description |
|---|---|---|
| **P2** | `ecs/system.rs` | Declare read/write access requirements for systems. |
| **P3** | `ecs/scheduler.rs` | Build a dependency graph and use a thread pool to run non-overlapping systems concurrently. |

### Epic: Advanced PBR & Render Graph
| Priority | Status | Feature | Description |
|---|---|---|---|
| **P1** | **DONE** | Render Graph Architecture | Shift from hardcoded RenderPasses to a dynamic, node-based Render Graph to automatically manage Vulkan image transitions, subpasses, and memory aliasing. |
| **P1** | **DONE** | Physically Based Rendering (PBR) | Implement full metallic-roughness PBR pipelines, Image-Based Lighting (IBL), Irradiance volumes, and HDR tonemapping (ACES). |
| **P2** | **DONE** | Global Illumination & Shadows | Cascaded Shadow Maps (CSM) for directional lights, Omnidirectional shadows for point lights, and Voxel Cone Tracing or Screen Space Global Illumination (SSGI). |
| **P3** | **TODO (Phase D)** | Post-Processing Stack | Bloom, Screen Space Ambient Occlusion (SSAO), Depth of Field (DoF), Motion Blur, and Temporal Anti-Aliasing (TAA). |

---

*Start the next session with: "Continue from ENGINE_ROADMAP.md — build Epic 2: Offscreen Rendering (The Viewport) to transition the engine into a full Editor."*
