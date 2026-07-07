# Custom Game Engine — Architecture & Roadmap

> **Last updated:** 2026-07-06 — P0/P1 Zero-Heap Violations Resolved

---

## Core Architectural Constraints

These rules are **non-negotiable** across all future sessions and govern the engine's design:

| Constraint | Detail |
|---|---|
| **Language** | Rust (2021 edition, stable toolchain) |
| **Paradigm** | Strict Data-Oriented Design (DoD). No deep inheritance trees. Flat, tightly packed arrays. Entity Component System (ECS). |
| **Memory** | **Zero** heap allocations in the main game loop. All allocations go through custom allocators (Arena, Pool, Stack). |
| **Standard Library** | Minimal `std` usage in performance-critical paths. Custom cache-friendly containers. |
| **Graphics** | Vulkan API via `ash` crate (decoupled rendering pipeline). |
| **Cache coherency** | Prioritized in every data structure for maximum CPU throughput. |
| **Code style** | One file at a time. No partial code or `// TODO` stubs unless explicitly outlining a module. Production-ready on first write. |
| **Unsafe** | Encapsulated behind safe public APIs. All `unsafe` blocks documented with `# Safety` comments. |

---

## Engine Features & Architecture (V1 & V2 Completed)

The engine has reached full maturity for its baseline requirements. All V1 and V2 epics are 100% complete and actively functional in the repository.

### Recent Polish (July 3, 2026)
- **ECS Parallelization**: Upgraded component iteration to use `rayon`'s `par_iter_chunks` for massively improved data-oriented traversal.
- **Editor UX**: Revamped Egui theme (light gray, sharp corners, distinct dropdown shadows).
- **Vulkan Fixes**: Corrected Swapchain format selection to strictly prioritize `B8G8R8A8_UNORM` to prevent red/blue channel swaps, and fixed font texture descriptor tracking.
- **Asynchronous IO**: Offloaded blocking native file dialogs (`rfd`) to background threads to prevent OS 'ghosting' overlays and thread deadlocks.

### Zero-Heap Audit & Philosophy Fixes (July 4, 2026)
- **Component Type IDs**: Replaced fragile `type_name().contains()` string matching with `TypeId`-based lookup table. Core types (0–11) resolved via compile-time table, immune to module path changes and substring collisions. Published constants (`TRANSFORM_TYPE_ID`, etc.) for direct use.
- **TransformComponent Rotation**: Implemented full `T × Rz × Ry × Rx × S` Euler rotation matrix. Previously only `T × S` was computed, completely ignoring the rotation field.
- **Game DLL Lock-Free**: Removed unnecessary `Mutex<Scheduler>` wrapper in `game.dll`. Replaced with `UnsafeCell` + `OnceLock` — `OnceLock` handles one-time init, `UnsafeCell` provides zero-overhead access since `game_update` is always single-threaded.
- **Hot-Path HashMap Elimination**: Replaced `std::collections::HashMap` with pre-allocated `Vec`-based storage for `world_matrices` (flat `Vec<(u32, Mat4)>` with 256 reserved capacity) and `compute_descriptor_sets` (`Vec<Option<DescriptorSet>>` indexed by mesh index). Zero heap allocations during the game loop.

### 1. Memory Management Subsystem (`src/memory/`)
- **Zero OS Heap on Hot Path**: The engine pre-allocates a 256 MB block from the OS (via `VirtualAlloc`/`mmap`) and slices it into distinct regions (Frame, Persistent, ECS, Stack).
- **Custom Allocators**: 
  - `ArenaAllocator`: O(1) linear bump allocation with save-point rewinds.
  - `PoolAllocator`: Fixed-size blocks with intrusive zero-overhead free-lists.
  - `StackAllocator`: LIFO allocator with automatic RAII `StackScope` cleanup.

### 2. Entity Component System (ECS) & Job System (`src/ecs/`)
- **Core Architecture**: Dense `ComponentArray<T>` sparse-set design ensuring cache-contiguous iteration loops.
- **Multithreading**: `Scheduler` dependency graph dynamically partitions `System`s based on read/write component masks. Uses `std::thread::scope` for fork-join execution, guaranteeing lock-free data parallelism safely across stages.
- **Lock-free Access**: Replaced interior mutability (`Mutex`/`RwLock`) with `get_component_array_mut_unchecked` verified entirely by the Scheduler.

### 3. Advanced Renderer (`src/renderer/`)
- **Render Graph Architecture**: Dynamic, node-based Render Graph that automatically tracks resource states and resolves Vulkan memory barriers/image transitions.
- **Physically Based Rendering (PBR)**: Full Metallic-Roughness PBR pipeline with Image-Based Lighting (IBL).
- **Global Illumination & Shadows**: Cascaded Shadow Maps (CSM) for directional lights, omnidirectional point light shadows.
- **Post-Processing**: ACES Tonemapping, Bloom downsample/upsample chains, and custom full-screen triangle generation.

### 4. Hot-Reloading & Editor Tooling (`src/app/`, `src/platform/`)
- **Native DLL Hot-Reloading**: Engine split into `engine.exe` (host/memory) and `game.dll` (systems). The host transparently unloads/reloads the DLL upon recompilation while persisting ECS data, enabling zero-downtime iteration.
- **Custom IMGUI Integration**: Replaced `egui` with a fully custom, in-house, zero-heap Immediate Mode GUI framework optimized for our data-oriented engine.
- **Scene Inspector**: ECS reflection (`ecs/reflection.rs`) automatically parses entities and exposes component properties for real-time editor tweaking.
- **Offscreen Rendering (Viewport)**: The 3D scene renders to an offscreen target, embedded directly into scalable custom UI windows.

### 5. Asset Pipeline & Virtual File System (`src/asset_manager.rs`, `src/vfs.rs`)
- **File Watcher**: `notify`-based asynchronous file watching automatically hot-reloads GLTF models, PNG/JPG textures, and SPIR-V shaders instantly.
- **Asset Caching**: De-duplicated asset loads to minimize VRAM usage.
- **VFS (Virtual File System)**: Abstraction layer allowing assets to be loaded from disk during development and packed into bundled archives for release builds.

### 6. Math, Physics & Core Utils
- **Math**: Custom `nalgebra`-inspired, SIMD-friendly linear algebra library (`vec`, `mat4`, `quat`, `transform`).
- **Physics**: Seamless integration with `rapier3d` (rigid bodies, colliders) synced dynamically with the visual ECS transforms.
- **Containers**: Cache-friendly generic collections (`FixedArray`, `DynamicArray`, `RingBuffer`, `HashMap`, `FixedString`).

---

## Zero-Heap Compliance — Status

### Resolved (July 6, 2026)
- **P0 — Scheduler scratch buffer** (`src/ecs/scheduler.rs`): Replaced per-frame `Vec::with_capacity()` in `execute()` with a pre-allocated `scratch_systems` buffer, sized once in `build_graph()`. `clear()` + `push()` reuses capacity every frame — zero heap allocations.
- **P0 — Resource tracker flat map** (`src/renderer/vulkan/render_graph.rs`, `src/app/application.rs`): Replaced `std::collections::HashMap<ResourceHandle, ResourceState>` with an inline `ResourceTracker` struct using a `[(ResourceHandle, ResourceState); 16]` fixed array with linear scan. Zero heap allocations.
- **P1 — Audio sinks flat map** (`src/audio.rs`): Replaced `HashMap<u32, SpatialSink>` with `Vec<Option<SpatialSink>>` indexed by entity index. Pre-allocated to 128 slots at construction. Growth only occurs at entity creation time, never during the frame loop.
- **P2 — Skeleton computed matrices** (`src/ecs/components.rs`): Replaced `Vec<Mat4>` with `FixedArray<Mat4, MAX_BONES>` to eliminate heap allocations inside `SkeletonComponent`.

### Remaining Violations

*None! The engine is now 100% Zero-Heap compliant on the hot path.*

---

## The Next-Generation Engine (V3 Master Plan)

With the rendering, hot-reloading, and multithreaded ECS foundations complete, the focus now shifts entirely to expanding the engine's capability as a full-suite game development platform.

### Epic 1: Spatial Audio Subsystem (Completed)
| Priority | Feature | Description |
|---|---|---|
| **P1** | Core Mixer & Output | Integrate a low-latency audio backend (`cpal` or `rodio`) respecting the zero-heap constraints. *(Done)* |
| **P1** | Spatial 3D Audio | Add `AudioEmitter` and `AudioListener` components. Implement HRTF/3D panning and distance attenuation based on ECS Transforms. *(Done)* |
| **P2** | Audio Streaming | Stream large `.ogg` or `.wav` music tracks from the VFS to avoid high RAM consumption. *(Done)* |

### Epic 2: Skeletal Animation & Blend Trees (Completed)
| Priority | Feature | Description |
|---|---|---|
| **P1** | GLTF Skinning | Expand the GLTF loader to parse inverse bind matrices and bone weights. *(Done)* |
| **P1** | Compute Shader Skinning | Move skeletal vertex deformation to a Vulkan compute shader for massive performance scaling. *(Done)* |
| **P2** | Animation Graphs | Introduce an `AnimatorComponent` supporting 1D/2D blend trees, state machines, and cross-fading between animation clips. *(Done)* |

### Epic 3: Gameplay Scripting (Completed)
| Priority | Feature | Description |
|---|---|---|
| **P1** | VM Integration | Embed a lightweight scripting language (e.g., `rhai` or `mlua`) to allow rapid behavior iteration without recompiling Rust DLLs. *(Done)* |
| **P1** | API Bindings | Expose the ECS (entity creation, component modification, queries) and Math library to the scripting context securely. *(Done)* |
| **P3** | Visual Node Graph | Build a visual node-based scripting tool inside the custom editor that transpiles to the embedded VM language. *(Done)* |

### Epic 4: Advanced Physics & Queries (Completed)
| Priority | Feature | Description |
|---|---|---|
| **P1** | Raycasting & Spatial Queries | Expose a clean API for line-of-sight checks, mouse picking (click-to-select), and sweep tests via `rapier3d`. *(Done)* |
| **P2** | Triggers & Sensor Volumes | Implement sensor colliders that fire ECS events (e.g., `OnTriggerEnter`) without physical resolution. *(Done)* |
| **P3** | Soft Bodies / Cloth | Expand physics support to handle deformable bodies, integrating with the compute shader mesh pipeline. *(Done)* |

### Epic 5: Project Export & Build Pipeline (Completed)
| Priority | Feature | Description |
|---|---|---|
| **P1** | VFS Archiver | Create a CLI tool to bundle all textures, shaders, and models into a single compressed binary package (e.g., `.pak`). *(Done)* |
| **P1** | Standalone Executable | Provide a build step that strips the custom editor and hot-reloading components, linking `game.dll` statically into a highly optimized standalone `.exe`. *(Done)* |
| **P2** | Build Profiles | Support multi-target output (e.g., Windows, Linux) via cross-compilation configurations. *(Done)* |

### Epic 6: Custom IMGUI Framework (Completed)
| Priority | Feature | Description |
|---|---|---|
| **P1** | Core Primitive Rendering | Implement basic immediate-mode UI shapes (quads, borders) and text using `FixedString` zero-heap constraints. *(Done)* |
| **P1** | Layout Engine | Cursor-based automatic UI layout with support for `begin_panel`, vertical, and horizontal flows. *(Done)* |
| **P1** | Component Inspector | Fully functional Right Inspector Panel utilizing `drag_float` and `checkbox` primitives to directly edit `ReflectComponent` structures. *(Done)* |

---

## Build System & Tests

### Build Commands
```bash
cargo build                 # Debug build (generates engine.exe & game.dll)
cargo build --release       # Optimized release build
cargo test                  # Run the entire test suite (ECS, Memory, Integration)
cargo run                   # Launch the Engine Editor
```

### Test Suite
Run tests via `cargo test`. Contains 40 tests covering:
- Memory alignment, stack limits, arena save/restores, and OS region mapping.
- ECS Entity ID lifecycle, generational bounds, component mapping.
- Mutability safety across the Multithreaded Job System execution graph.
- Application boot-up, initialization, and hot-reload DLL teardown loops. 
*(Note: A minor known Vulkan validation leak occurs during teardown in `test_app` solely due to OS `libloading` unload order, not affecting actual runtime).*
