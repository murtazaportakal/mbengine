# Custom Game Engine

## Overview
A high-performance, data-oriented custom 3D game engine written from scratch in Rust, utilizing the Vulkan graphics API via `ash`. The engine is strictly designed to avoid heap allocations on the hot path, leveraging an Entity Component System (ECS) architecture, a custom Job System, and advanced GPU-driven rendering techniques.

## Features & Architecture

### Core Architecture
- **Data-Oriented ECS**: Fast, cache-friendly iteration loops with flat Component Arrays.
- **Multithreading & Job System**: Dynamic lock-free dependency scheduling utilizing `std::thread::scope` and `rayon`.
- **Zero-Heap Philosophy**: The hot-path frame loop relies on custom allocators (Arena, Pool, Stack) and persistent memory mapping to eliminate runtime heap fragmentation and garbage collection stutters.

### Advanced Rendering (Vulkan)
- **GPU-Driven Pipeline**: Nanite-style compute shader culling (frustum, HZB occlusion, LOD selection) coupled with Multi-Draw Indirect (`vkCmdDrawIndexedIndirectCount`).
- **Physically Based Rendering (PBR)**: Metallic-roughness material workflow with Image-Based Lighting.
- **Lighting & Shadows**: Cascaded Shadow Maps (CSM) for directional lighting and omnidirectional point light shadows.
- **Render Graph**: Automated resource tracking, barriers, and render pass synchronization using a node-based graph.
- **Post-Processing**: ACES tonemapping, Bloom downsample/upsample chains, and 5-tap Gaussian Blur.

### Asset Pipeline & Tooling
- **Offline Asset Cooker (`cooker.exe`)**: Parses `.gltf`/`.glb` models and builds engine-ready zero-copy binary blobs (`.mesh`, `.mat`) with precomputed meshlets.
- **Virtual File System (VFS)**: Asynchronous file watching (`notify`) and hot-reloading for models, textures, and shaders.
- **Custom IMGUI Framework**: In-house immediate-mode UI optimized for zero-allocation rendering, complete with a Component Inspector and File Browser.
- **Native DLL Hot-Reloading**: The core game logic resides in `game.dll`, hot-swappable at runtime without losing ECS state, ensuring zero-downtime iteration.

### Subsystems
- **Physics**: Integrated with `rapier3d` (rigid bodies, colliders, raycasting).
- **Skeletal Animation**: GPU-accelerated compute shader vertex skinning.
- **Audio**: Spatial 3D audio powered by `cpal` and `rodio`.
- **Scripting**: Embedded virtual machine (`rhai`/`mlua`) for rapid gameplay logic prototyping.

## Build Instructions
```bash
cargo build                              # Dev build with editor (engine.exe + game.dll)
cargo run --bin cooker -- model.gltf     # Cook a glTF model → .mesh + .mat
cargo build --release --features standalone  # Strip editor, produce standalone release bundle
cargo test                               # Run the full test suite (ECS, Memory, Integration)
```

## V4 Master Plan (Completed)
The V4 Master Plan focused on scaling the engine to a production-ready state. All phases are now **100% complete**:

1. **Phase 0 & 1: GPU-Driven Foundation & Cooker**
   - Implemented an offline `cooker.exe` to bake `.gltf` assets into highly optimized `.mesh` and `.mat` zero-copy binary blobs.
   - Fully GPU-driven rendering pipeline utilizing Nanite-style compute shader culling.

2. **Phase 2 & 3: Data-Oriented Skeletal Animation & Prefabs**
   - VFS hot-reloading for cooked formats.
   - Pre-baking skeletal `.anim` formats directly in the offline cooker.
   - GPU-only bone matrix lerping and keyframe blending using compute shaders, avoiding CPU traversal entirely.

3. **Phase 4: Logic, Scripting & Lighting**
   - Expanded script-to-ECS integration with entity name registration.
   - **Forward+ Cluster Grid**: Dividing the frustum into 3D tiles to support 256+ dynamic lights with O(1) per-fragment lighting evaluation cost.

4. **Phase 5: Packager & Release Builds**
   - VFS Archiver (`packer.rs`) effectively bundles all cooked assets (`.mesh`, `.mat`, `.anim`, `.spv`) into a single contiguous `data.pak`.
   - Hardened `standalone` Cargo feature locking down editor code and outputting a highly optimized, fully self-contained game client.
   - Cross-compilation configurations included for Windows and Linux release targets.

---

> **Engine Status:** All V1, V2, V3, and V4 systems are **100% complete**. The engine is now a fully self-contained, production-ready, zero-heap runtime capable of producing packed release builds.
