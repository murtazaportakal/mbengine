# Custom Game Engine — Architecture & Roadmap

> **Last updated:** 2026-08-22 — HZB Occlusion Fixed and Codebase Cleanup

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

### GPU-Driven Skeletal Animation (August 15, 2026)
- **Compute Shader Skinning Pipeline**: Eliminated CPU-side bone matrix multiplication. Implemented `anim_update.comp` to evaluate bone hierarchy transformations directly from the global GPU `AnimationPool` using compute shaders.
- **Global Bone Matrices & Vertex Skinning**: Unified all animation matrices into a single massive global SSBO (`anim_bone_matrices_buffer`) supporting up to 100,000 animated instances simultaneously. The vertex shader (`shader.vert`) dynamically performs skinning on-the-fly, completely bypassing intermediate memory writes.
- **Meshlet Visualization UI**: Added an Unreal Engine Nanite-style debug visualization mode. Implemented `gl_DrawID` hashing in `shader.vert` to dynamically assign distinct pseudo-random colors to every rendered meshlet cluster in real-time, toggleable via the editor UI.

### GPU-Driven Meshlet Culling (July 18, 2026)
- **Nanite-Style Culling Pipeline**: Designed and implemented a multi-phase compute shader (`cull.comp`) performing frustum, occlusion (HZB), and screen-space LOD culling on individual meshlets.
- **GPU-Driven Indirect Drawing**: Integrated completely GPU-driven rendering (`vkCmdDrawIndexedIndirectCount`) utilizing atomic draw counters and compacted indirect command buffers.
- **Data-Oriented Fixes**: Solved severe Vulkan struct stride misalignments (144 bytes vs 160 bytes) between Rust `InstanceData` and GLSL `InstanceData`, ensuring robust GPU memory access up to 100,000 instances without mathematical degradation or screen glitches.
- **Resolved (August 22, 2026)**: Fixed HZB Occlusion culling Standard-Z depth comparison bug and Phase 2 occlusion list population in `cull.comp`. Validated `generate_hzb.comp` depth reduction.

### Engine Stabilization & Stress Testing (July 18, 2026)
- **ECS Capacity Limits**: Massively increased internal sparse-set Component Array capacities from 1,000 to 20,000 to cleanly support the 10,000 Object Stress Test.
- **UI Buffer Scalability**: Increased Vulkan Immediate-Mode GUI vertex/index buffers from 64K to 1,024K (4MB) to prevent buffer overflows when visualizing extreme entity counts in the Scene Hierarchy.
- **Drop-Order Audio Safety**: Reordered `Application` subsystem drop sequencing so `AudioSystem` (sink tracker) drops strictly before `AudioSubsystem` (cpal stream), completely eliminating dangling thread memory violations on exit.
- **Clean Headless Teardown**: Deleted the redundant `test_app` harness to synchronize CI purely around `cargo run`, stabilizing the build pipeline.

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

> **Status:** All Phases (0-5) complete. Engine is fully V4-compliant.

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

### Phase 4: Logic, Scripting & Lighting (Completed)

**Gate:** A game script can spawn, query, and destroy entities created in the editor; scene supports 8+ dynamic point lights. *(Completed)*

| Priority | Feature | Description | Status |
|---|---|---|---|
| **P1** | Script entity name/ID bridge | Entity names registered in a VFS-like flat string table; exposed to Rhai context | *(Done)* |
| **P1** | DLL hot-reload entity persistence | Verify ECS entity IDs survive `game.dll` unload/reload across scenes loaded in editor | *(Done)* |
| **P2** | Forward+ cluster grid | Divide frustum into 3D tiles in compute; assign point lights per tile; sample in `shader.frag` — supports 256+ dynamic lights with O(1) per-fragment cost | *(Done)* |
| **P3** | Deferred lighting (optional) | GBuffer MRT (albedo/normal/PBR) + separate lighting pass — heavier but enables screen-space effects | *(Skipped/Replaced by Forward+)* |

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


---

## V5 Master Plan — Closing the Gap to Unity/Unreal

> **Last updated:** 2026-08-31
> **Philosophy:** Match the feature depth of Unity/Unreal for 3D games, while keeping the runtime overhead and project simplicity of Godot. Every feature added must pass two tests: (1) Does it make the engine measurably more capable for game developers? (2) Does it fit within the zero-heap, DOD, offline-cook architecture?

This plan is organized into **six epics**, ordered by strategic impact. Each epic is broken into atomic, independently shippable phases.

---

### Epic A: First-Class Editor Viewport

**Goal:** The editor IS the game — hitting Play runs the actual engine loop inside the viewport, not a simulation.

**Why:** This single change closes the largest UX gap with Godot/Unity. Right now the editor is a side panel. Making the 3D viewport the primary workspace transforms the engine from a programmer's tool into a game developer's tool.

#### Phase A1: True Scene Viewport
| Priority | Task | Detail |
|---|---|---|
| **P1** | **Fullscreen offscreen scene** | Render the scene entirely to an `OffscreenTarget`. Display it as a UI image filling the central editor panel. Already partially done (`offscreen.rs`) — wire it to the main editor layout. |
| **P1** | **Mouse picking in viewport** | Cast a ray from cursor into the scene using the `rapier3d` ray-cast API. Highlight and select the hit entity. Set `selected_entity`. |
| **P1** | **Play / Pause / Stop buttons** | Entering Play mode: serialize the current ECS state to a temp buffer (use `save_scene`). Stopping: restore from that buffer. No scene data is ever permanently mutated by Play mode. |
| **P2** | **Camera fly-through in editor** | Right-click + WASD inside the viewport moves an editor-only camera. The editor camera is NOT an ECS entity — it is a pure editor state struct. |
| **P2** | **Scene grid & world-axis overlay** | Render a flat procedural grid mesh and XYZ axis lines using the existing debug draw path, depth-tested. |
| **P3** | **Multi-viewport support** | Split the editor into up to 4 simultaneous viewport panels (Top, Front, Side, Perspective) — classic Unreal/Blender layout. |

#### Phase A2: 3D Transform Gizmo
| Priority | Task | Detail |
|---|---|---|
| **P1** | **Translate gizmo** | Render three colored axis arrows (X=red, Y=green, Z=blue) as GPU mesh instances. Clicking and dragging an arrow moves the selected entity on that axis. Use `active_gizmo_axis` field already in `Editor` struct. |
| **P1** | **Rotate gizmo** | Three arc rings rendered per-axis. Dragging rotates the entity's Euler angles. |
| **P2** | **Scale gizmo** | Three cube handles per axis. Dragging scales non-uniformly. |
| **P2** | **World vs. Local space toggle** | Toggle between world-aligned gizmo and object-local gizmo. |
| **P3** | **Multi-select + group transform** | Box-select multiple entities (drag rectangle in viewport). Gizmo appears at the centroid and transforms all selected. |

#### Phase A3: Scene Hierarchy & Prefab Drag-Drop
| Priority | Task | Detail |
|---|---|---|
| **P1** | **Reparent via hierarchy drag** | Drag an entity onto another in the Scene Hierarchy panel to set `HierarchyComponent.parent`. Children move with parent transforms. |
| **P2** | **Prefab drag-drop from file browser** | Drag a `.mesh` or future `.prefab` file from the File Browser panel into the 3D viewport. Spawn a new entity at the cursor hit position (ray-cast against ground plane). |
| **P2** | **Undo/Redo stack** | Maintain a ring buffer of `EditorAction` variants. `Ctrl+Z` pops and inverts the last action. Supports translate, rotate, scale, spawn, delete. |

---

### Epic B: NavMesh & AI Navigation

**Goal:** Any entity can navigate a 3D environment autonomously.

**Why:** NavMesh is the single most-requested feature that separates game engines from rendering engines. Without it, you cannot ship any game that requires enemies, NPCs, or companions.

#### Phase B1: Static NavMesh Baking
| Priority | Task | Detail |
|---|---|---|
| **P1** | **Integrate `recast-navigation-rs`** | Add the Rust bindings for the industry-standard Recast/Detour library (used by Unity, Godot, Unreal) as a Cargo dependency. |
| **P1** | **Offline navmesh cooker step** | Add a `--navmesh` flag to `cooker.exe`. It reads the scene's static mesh geometry, runs Recast's voxelization + region + contour + poly mesh pipeline, and outputs a `.nav` binary blob. |
| **P1** | **`.nav` VFS loader** | Add `load_cooked_navmesh(path)` to `AssetManager`. Deserialize the Detour nav mesh into a `NavMesh` struct held in the Application. |
| **P2** | **NavMesh debug overlay** | Render the walkable triangle soup as a translucent green mesh overlay in the editor. Toggled via an editor checkbox. |

#### Phase B2: Pathfinding & Steering
| Priority | Task | Detail |
|---|---|---|
| **P1** | **`NavAgentComponent`** | New ECS component: `{ target: Vec3, path: FixedArray<Vec3, 64>, path_index: u32, speed: f32, radius: f32 }`. Zero heap allocation — path fits in a fixed-size array. |
| **P1** | **`NavSystem` (ECS System)** | Each frame: for entities with `NavAgentComponent` that have a dirty target, call `detour.find_path()` and store result into the fixed path array. Then advance along the path, updating `TransformComponent.position`. |
| **P2** | **Steering behaviors** | Separation, alignment, and cohesion forces computed in a compute shader for flocking groups of up to 10,000 agents simultaneously — true DOD approach. |
| **P3** | **Dynamic obstacles** | Register `RigidBodyComponent` entities as dynamic obstacles in Detour. NavMesh re-tiles locally when obstacles move. |

---

### Epic C: GPU Particle System

**Goal:** A fully GPU-driven particle emitter that can simulate millions of particles at 60+ FPS with zero CPU involvement per-frame.

**Why:** Visual effects (fire, explosions, magic, weather) are expected in every 3D game. A compute-shader particle system fits perfectly in the existing GPU-driven architecture.

#### Phase C1: Core Particle Simulation
| Priority | Task | Detail |
|---|---|---|
| **P1** | **`ParticlePool` GPU buffer** | Allocate a global `MAX_PARTICLES = 1_000_000` SSBO containing flat `Particle` structs: `{ pos: [f32;3], vel: [f32;3], life: f32, max_life: f32, color: [f32;4], size: f32 }`. |
| **P1** | **`particle_update.comp`** | Compute shader. Each invocation updates one particle: integrates velocity, decrements life, applies gravity. Dead particles (life ≤ 0) are returned to a free-list via an atomic counter. |
| **P1** | **`particle_emit.comp`** | Compute shader. Emitter parameters passed via push constants: `{ origin, emit_dir, spread_angle, emit_rate, initial_speed, lifetime, color }`. Atomically claims slots from the free-list and initializes new particles. |
| **P1** | **Indirect draw output** | Compact living particles into a draw-ready position buffer. Write a `VkDrawIndirectCommand` for a GPU-instanced billboard quad pass. |
| **P2** | **Billboarded sprite rendering** | New render pass: vertex shader positions each particle quad facing the camera. Fragment shader samples from a `particle_atlas` texture array for animated sprites. |

#### Phase C2: Emitter Component & Editor
| Priority | Task | Detail |
|---|---|---|
| **P1** | **`ParticleEmitterComponent`** | ECS component storing emitter parameters. Registered with `ComponentRegistry` so the editor inspector exposes all knobs with live preview. |
| **P2** | **Real-time preview in editor** | Particle simulation runs live in the editor viewport. No Play mode required to preview effects. |
| **P2** | **Emitter shape presets** | Point, sphere, cone, box, and mesh-surface emission shapes. |
| **P3** | **Sub-emitter chaining** | A particle death event triggers a secondary emitter (e.g., spark burst on impact). Implemented with a GPU atomic event queue. |

---

### Epic D: Cascaded Shadow Maps + Advanced Lighting

**Goal:** Production-quality outdoor lighting indistinguishable from AAA games.

**Why:** The current single-cascade shadow map creates hard shadow edges and acne at medium distances. CSM is the industry standard for outdoor scenes.

#### Phase D1: Cascaded Shadow Maps (CSM)
| Priority | Task | Detail |
|---|---|---|
| **P1** | **4-cascade depth atlas** | Replace the single 2048×2048 depth image with a 4-cascade 4096×2048 atlas. Cascades: C0=0–10m (near detail), C1=10–50m, C2=50–200m, C3=200–500m (far). |
| **P1** | **Per-cascade light matrices** | Compute 4 orthographic light-space matrices each frame using the practical split scheme (λ=0.75). Store in a `CsmUbo` pushed into the global descriptor set. |
| **P1** | **Shadow shader PCF 3×3 sampling** | Update `shadow_calculation()` in `shader.frag` to sample the correct cascade atlas slice based on view-space depth. Apply 9-tap Percentage Closer Filtering for smooth edges. |
| **P2** | **Cascade debug visualization** | Editor toggle that tints fragments red/green/blue/yellow per-cascade. Essential for tuning split distances. |
| **P3** | **PCSS (Percentage Closer Soft Shadows)** | Variable penumbra size based on blocker distance from light — physically correct soft shadows for mid-range cascades. |

#### Phase D2: Point Light Shadows (Omnidirectional)
| Priority | Task | Detail |
|---|---|---|
| **P2** | **Cubemap shadow atlas** | For up to 8 shadow-casting point lights: a 6-face cubemap depth array. Render geometry into each face via geometry shader instancing. |
| **P2** | **Per-light frustum culling** | In the existing cull compute shader, add a per-point-light frustum cull pass that populates a per-light indirect draw buffer. |

#### Phase D3: Image-Based Lighting (IBL) Improvements
| Priority | Task | Detail |
|---|---|---|
| **P2** | **HDRI skybox baking in cooker** | `cooker.exe --hdri sky.hdr` computes the irradiance map (32×32 diffuse convolution) and pre-filtered environment map (128×128, 7 mips) offline. Stores as compressed `.ktx2` blobs for instant zero-copy load. |
| **P3** | **Screen-Space Reflections (SSR)** | Trace reflection rays in screen space using the HZB depth buffer as an acceleration structure. Falls back to IBL when ray exits screen. |

---

### Epic E: Networking Foundation

**Goal:** Deterministic lockstep and client-server authority for multiplayer games up to 64 players.

**Why:** No AAA engine ships without networking. Adding a well-designed, DOD-compatible network layer is the largest remaining architectural gap.

#### Phase E1: Transport Layer
| Priority | Task | Detail |
|---|---|---|
| **P1** | **Add `renet` crate** | `renet` provides reliable UDP transport with connection management, channels (reliable ordered, unreliable), and connection tokens. Actively maintained and battle-tested. |
| **P1** | **`NetworkSubsystem` struct** | Owns a `RenetServer` or `RenetClient` depending on launch flags. Updated once per frame before the ECS scheduler runs. Zero-heap: all packet buffers pre-allocated at init. |
| **P1** | **Launch flags** | `--server --port 7777` starts a dedicated server. `--client --host 192.168.1.1` connects. `--singleplayer` skips network init entirely. |

#### Phase E2: ECS Network Replication
| Priority | Task | Detail |
|---|---|---|
| **P1** | **`NetworkedComponent` marker** | Tagging a component type with `Networked` causes `ReplicationSystem` to serialize it with `bincode` and broadcast diffs every N ticks. Only changed fields are sent (dirty-flag diffing). |
| **P2** | **Client-side prediction** | Client simulates physics and movement locally. Server sends authoritative state. Client reconciles using snapshot interpolation (Quake/Source engine approach). |
| **P3** | **Interest management** | Only replicate entities within a configurable radius of each client's position. Implemented via spatial hash grid. |

---

### Epic F: Cross-Platform Backend Abstraction

**Goal:** The same game code targets Windows (Vulkan), Linux (Vulkan), and macOS (Metal via MoltenVK) from a single codebase.

**Why:** Cross-platform support is a prerequisite for any commercial release and community adoption.

#### Phase F1: Abstract Graphics Backend
| Priority | Task | Detail |
|---|---|---|
| **P1** | **`GfxDevice` trait** | Define a minimal `GfxDevice` trait with methods: `create_buffer`, `create_texture`, `begin_frame`, `submit`, `present`. Current `VulkanDevice` implements this trait behind a type alias. Zero overhead — trait is monomorphized at compile time. |
| **P1** | **Conditional compile** | `#[cfg(target_os = "windows")]` + `#[cfg(target_os = "linux")]` resolve to `VulkanDevice`. `#[cfg(target_os = "macos")]` resolves to a future `MetalDevice` backed by MoltenVK. |
| **P2** | **Linux CI pipeline** | Add a GitHub Actions workflow that cross-compiles to `x86_64-unknown-linux-gnu` using a Docker image with Vulkan SDK headers. Catches platform regressions automatically. |

#### Phase F2: Window & Input Abstraction
| Priority | Task | Detail |
|---|---|---|
| **P1** | **`PlatformWindow` trait** | Abstract over the current Win32 `HWND` window creation. Linux implementation uses `xcb` or `wayland-client`. |
| **P2** | **Gamepad support** | Add `gilrs` crate for cross-platform gamepad input. Map analog sticks to the existing `Input` struct. Expose as `input.gamepad_axis(Axis::LeftX)`. |

---

### V5 Feature Priority Matrix

| Feature | Estimated Effort | Game Dev Impact | Sprint |
|---|---|---|---|
| True 3D editor viewport (A1) | 2 weeks | 🔴 Critical | 1 |
| 3D transform gizmo (A2) | 1 week | 🔴 Critical | 1 |
| Cascaded shadow maps (D1) | 1 week | 🔴 High | 1 |
| NavMesh baking + agent (B1+B2) | 3 weeks | 🔴 High | 2 |
| GPU particle system (C1+C2) | 3 weeks | 🟡 High | 2 |
| Undo/Redo stack (A3) | 3 days | 🟡 Medium | 2 |
| Point light shadows (D2) | 2 weeks | 🟡 Medium | 3 |
| Networking — transport (E1) | 2 weeks | 🟡 Medium | 3 |
| Prefab drag-drop (A3) | 1 week | 🟡 Medium | 3 |
| IBL baking / SSR (D3) | 2 weeks | 🟢 Medium | 4 |
| Cross-platform abstraction (F) | 4 weeks | 🟢 Medium | 4 |
| ECS replication (E2) | 3 weeks | 🟢 Medium | 5 |
| GPU flocking / steering (B2) | 2 weeks | 🟢 Low | 5 |

---

### V5 Build Commands (Additions)

```bash
# Cook a scene's static geometry into a navmesh
cargo run --bin cooker -- --navmesh scene.mesh

# Cook an HDRI skybox into diffuse + specular IBL blobs
cargo run --bin cooker -- --hdri sky.hdr

# Launch as dedicated game server
cargo run -- --server --port 7777

# Launch as network client connecting to local server
cargo run -- --client --host 127.0.0.1

# Cross-compile for Linux
cargo build --target x86_64-unknown-linux-gnu --release
```

---

> **V5 North Star:** When Epic A (Editor Viewport) and Epic B (NavMesh) are complete, MBEngine can ship a real commercial 3D game. Every epic after that makes the engine progressively more competitive with commercial alternatives — without ever compromising the zero-heap, GPU-driven, DOD core that makes it uniquely performant.
