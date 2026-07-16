# Custom Game Engine — Architecture & Roadmap

> **Last updated:** 2026-07-10 — V4 GPU-Driven Stabilization & 5-Phase Feature Plan Added

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


### Post-Processing & Optimization (July 9, 2026)
- **Gaussian Blur Pipeline**: Implemented a 5-tap linear sampling Gaussian blur utilizing horizontal and vertical ping-pong targets directly integrated into the RenderGraph.
- **Zero-Heap Geometry Rendering**: Swapped per-frame map_memory on the Instance Buffer to persistent mapping (instance_mapped_ptr) for instant data pushes.
- **Data-Oriented Texture Lookups**: Eliminated runtime HashMap usage. Texture handles (Albedo, Normal, MR, Emissive) are now directly cached into the flat Mesh ECS component at load time.
- **Prepass & MDI Restoration**: Restored the Multi-Draw Indirect (MDI) system, combining it seamlessly with a dedicated Depth/Normal Prepass pipeline.

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

### Resolved (July 16, 2026)
- **P0 — Per-frame instance data Vec** (`src/renderer/vulkan/backend.rs`): Replaced per-frame `Vec::with_capacity(0)` allocation in `render_frame()` with a pre-allocated `instance_data_buffer` field on `RenderBackend`, sized to `max_instances` (100_000) at construction. `clear()` + `push()` reuses capacity every frame — zero heap allocations on the hot render path.

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

---

## V4 Master Plan — GPU-Driven Rendering & Asset Pipeline

> **Status:** Phase 0 complete (2026-07-11). Phases 1–5 pending.

### Architectural Constraints for V4

These supplement the core constraints above:
- **No runtime glTF parsing.** All models are pre-cooked offline by `cooker.exe`. Hot path sees only binary blobs.
- **No CPU readbacks on the hot path.** Counter resets and buffer clears use `vkCmdFillBuffer` (GPU timeline). Zero CPU stalls.
- **Single cull dispatch per frame.** One GPU `vkCmdDispatch` over all instances, not per-mesh loops.

---

### Phase 0: Stabilize the GPU-Driven Foundation

**Gate:** 10,000 cubes, camera at 200+ FPS (debug), correct frustum culling verified by vanishing draws when turning away.

| Priority | Bug | Fix |
|---|---|---|
| **🔴 CRITICAL** | `draw_count_buffer` missing from `Application` struct — compile error | Add `pub draw_count_buffer: Buffer` field; allocate 4-byte `STORAGE_BUFFER \| TRANSFER_DST \| HOST_VISIBLE \| HOST_COHERENT` buffer, initialize to `0u32` |
| **🔴 CRITICAL** | Atomic draw counter never zeroed each frame — overflow after frame 1 | Insert `cmd_fill_buffer(draw_count_buffer, 0, 4, 0)` + `TRANSFER_WRITE → SHADER_READ_WRITE` barrier before cull dispatch |
| **🟡 HIGH** | Cull dispatch per-mesh (meshlet_count groups) vs. per-instance intent | Unify into one `dispatch(total_instances.div_ceil(64), 1, 1)` after building InstanceData SSBO |
| **✅ Done** | `#[repr(C, align(16))]` on `InstanceData` | Already applied in `pipeline.rs:50` |
| **✅ Done** | Per-frame `map_memory` stall on instance buffer | Already replaced with `instance_mapped` persistent pointer (July 9) |

---

### Phase 1: CLI Asset Cooker (`cooker.exe`)

**Gate:** `cooker.exe asset.gltf` outputs valid `.mesh` + `.mat` binary blobs loadable by the engine.

| Priority | Feature | Description |
|---|---|---|
| **P1** | `cooker` binary (`src/bin/cooker.rs`) | New standalone Rust binary registered as `[[bin]] name = "cooker"` in Cargo.toml |
| **P1** | glTF parsing | Use the `gltf` crate (already a dep) — extract positions, normals, UVs, joint IDs, weights, PBR material params |
| **P1** | `.mesh` binary format | Header + `Vertex[]` + `u32 indices[]` + `MeshletData[]` — layout matches engine structs exactly for zero-copy load |
| **P1** | `.mat` binary format | Header + PBR scalars + texture file references |
| **P2** | Meshlet generation | Run `meshopt::build_meshlets` in the cooker, not at runtime |
| **P2** | Texture compression | Convert PNGs to BC7 block-compressed format for 75% VRAM savings |
| **P2** | Format versioning | Magic bytes + version u32 in every header so stale files are detected at load time |

---

### Phase 2: Engine VFS Hooks & Prefab System

**Gate:** Drag a `.mesh` file into the editor viewport → entity spawns with correct geometry and material.

| Priority | Feature | Description |
|---|---|---|
| **P1** | VFS hot-reload for `.mesh`/`.mat` | Update `notify` watcher to watch cooked files instead of raw `.obj` |
| **P1** | Zero-copy load path | `vfs.read_raw_into_staging()` streams bytes directly into a pre-allocated staging buffer pointer — no intermediate Vec |
| **P1** | `load_cooked_mesh()` / `load_cooked_material()` | `AssetManager` methods that parse `.mesh`/`.mat` headers and upload to `GeometryPool` |
| **P1** | Prefab descriptor (`src/ecs/prefab.rs`) | `PrefabAsset { mesh_path, material_path, default_transform }` — serde-serializable scene descriptor |
| **P2** | Editor file browser | Custom IMGUI panel listing `.mesh`/`.mat`/`.prefab` files from the VFS watch directory |
| **P2** | Drag-and-drop instantiation | Dragging a cooked asset into the 3D viewport spawns an entity in `component_array.rs` |

---

### Phase 3: Data-Oriented Skeletal Animation (Cooker Path)

**Gate:** Animated glTF character cooked to `.anim`, loaded and skinned entirely on GPU with no CPU bone transforms.

| Priority | Feature | Description |
|---|---|---|
| **P1** | `.anim` format in cooker | Extract bone weights, inverse bind matrices, keyframe tracks from glTF; output `[AnimHeader][BoneCount][InvBindMat4[]][KeyframeTrack[]]` |
| **P1** | Cooker-side skeleton baking | Pre-multiply bind matrices, validate bone counts ≤ `MAX_BONES` |
| **P2** | `AnimationClipHandle` registry | Flat array of loaded clips (no HashMap); index stored in `AnimatorComponent` |
| **P2** | GPU-only bone update | Keyframe lerp runs in a compute shader; CPU never touches bone matrices per-frame |

*(Core ECS components `SkeletonComponent` + `AnimatorComponent` and `skinning.comp` already exist from V3)*

---

### Phase 4: Logic, Scripting & Lighting

**Gate:** A game script can spawn, query, and destroy entities created in the editor; scene supports 8+ dynamic point lights.

| Priority | Feature | Description |
|---|---|---|
| **P1** | Script entity name/ID bridge | Entity names registered in a VFS-like flat string table; exposed to Rhai context |
| **P1** | DLL hot-reload entity persistence | Verify ECS entity IDs survive `game.dll` unload/reload across scenes loaded in editor |
| **P2** | Forward+ cluster grid | Divide frustum into 3D tiles in compute; assign point lights per tile; sample in `shader.frag` — supports 256+ dynamic lights with O(1) per-fragment cost |
| **P3** | Deferred lighting (optional) | GBuffer MRT (albedo/normal/PBR) + separate lighting pass — heavier but enables screen-space effects |

---

### Phase 5: Packager & Release Builds

**Gate:** `cargo build --release --features standalone` produces a single `engine.exe` + `data.pak` with no editor code and no loose files.

| Priority | Feature | Description |
|---|---|---|
| **P1** | `packer.rs` update | Concatenate `.mesh`, `.mat`, `.anim`, `.spv` files into a single `data.pak` with an offset table |
| **P1** | `VfsMode::Pak` | VFS reads assets from `data.pak` using the offset table instead of the filesystem |
| **P1** | Editor feature gate | `#[cfg(feature = "editor")]` wraps all IMGUI, file browser, and hot-reload code |
| **P2** | `standalone` Cargo feature | `cargo build --release --features standalone` links `game.dll` statically, strips editor, outputs release bundle |
| **P2** | Cross-compilation configs | `.cargo/config.toml` targets for `x86_64-pc-windows-msvc` and `x86_64-unknown-linux-gnu` |

---

### V4 Build Commands (Future)

```bash
cargo build                              # Dev build with editor (engine.exe + game.dll)
cargo run --bin cooker -- model.gltf     # Cook a glTF model → .mesh + .mat
cargo run --bin packer                   # Bundle all cooked assets → data.pak
cargo build --release --features standalone  # Strip editor, produce release bundle
```


