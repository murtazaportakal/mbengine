//! # GPU Binary Format Definitions — Single Source of Truth
//!
//! Every struct that crosses the **cooker ↔ runtime** boundary lives here.
//! The cooker (`src/bin/cooker.rs`) imports these types to write `.mesh`
//! blobs.  The engine runtime imports these same types to read them.
//! Nobody copy-pastes them.
//!
//! ## Rules
//!
//! 1. All structs are `#[repr(C)]` with `Pod + Zeroable` for safe
//!    byte-level reinterpretation via `bytemuck`.
//! 2. Layouts are frozen once `MESH_VERSION` is bumped — the cooker
//!    and runtime must always agree on the byte layout.
//! 3. Compile-time assertions at the bottom of this file verify sizes
//!    so any accidental field addition breaks the build immediately.

use bytemuck::{Pod, Zeroable};

// ── Vertex ───────────────────────────────────────────────────────────────────

/// GPU vertex layout shared by the cooker, the runtime mesh loader,
/// the geometry pool, and all vertex shaders.
///
/// 64 bytes, tightly packed at natural alignment:
/// ```text
/// offset  0: pos            [f32; 3]  (12 bytes)
/// offset 12: normal         [f32; 3]  (12 bytes)
/// offset 24: uv             [f32; 2]  ( 8 bytes)
/// offset 32: joint_ids      [u32; 4]  (16 bytes)
/// offset 48: joint_weights  [f32; 4]  (16 bytes)
/// total: 64 bytes
/// ```
///
/// This is consumed via traditional vertex attributes (`layout(location = N)`)
/// in the current pipeline.  When the engine migrates to BDA, the struct
/// will gain explicit std430 padding and all consumers will update in lockstep
/// because there is only one definition.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub _pad0: u32,
    pub normal: [f32; 3],
    pub _pad1: u32,
    pub uv: [f32; 2],
    pub _pad2: [u32; 2],
    pub joint_ids: [u32; 4],
    pub joint_weights: [f32; 4],
}

// ── Mesh Header ──────────────────────────────────────────────────────────────

/// Magic bytes identifying a `.mesh` file.
pub const MESH_MAGIC: [u8; 4] = *b"MESH";

/// Bump this whenever `MeshHeader`, `Vertex`, or `MeshletData` layout changes.
/// The runtime rejects files whose version does not match.
pub const MESH_VERSION: u32 = 1;

/// Flag: mesh contains skinning data (joint_ids / joint_weights are populated).
pub const MESH_FLAG_SKINNED: u32 = 1 << 0;

/// File header for the `.mesh` binary format.
///
/// The cooker writes this, the runtime reads it.  Immediately followed on
/// disk by `Vertex[vertex_count]`, then `u32[index_count]`, then
/// `MeshletData[meshlet_count]`.
///
/// ```text
/// offset  0: magic          [u8; 4]   ( 4 bytes)
/// offset  4: version         u32      ( 4 bytes)
/// offset  8: flags           u32      ( 4 bytes)
/// offset 12: vertex_count    u32      ( 4 bytes)
/// offset 16: index_count     u32      ( 4 bytes)
/// offset 20: meshlet_count   u32      ( 4 bytes)
/// offset 24: aabb_min       [f32; 3]  (12 bytes)
/// offset 36: aabb_max       [f32; 3]  (12 bytes)
/// offset 48: _pad            u32      ( 4 bytes)
/// total: 52 bytes
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeshHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub flags: u32,
    pub vertex_count: u32,
    pub index_count: u32,
    pub meshlet_count: u32,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub _pad: u32,
}

// ── Meshlet Data ─────────────────────────────────────────────────────────────

/// Per-meshlet bounding/culling descriptor.
///
/// 32 bytes, used by the compute culling shader (`cull.comp`) to perform
/// frustum + cone-based back-face culling per meshlet.
///
/// ```text
/// offset  0: center         [f32; 3]  (12 bytes)
/// offset 12: radius          f32      ( 4 bytes)
/// offset 16: cone_axis      [f32; 3]  (12 bytes)
/// offset 28: cone_cutoff     f32      ( 4 bytes)
/// offset 32: index_offset    u32      ( 4 bytes)
/// offset 36: triangle_count  u32      ( 4 bytes)
/// offset 40: _pad           [u32; 2]  ( 8 bytes)
/// total: 48 bytes
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeshletData {
    pub center: [f32; 3],
    pub radius: f32,
    pub cone_axis: [f32; 3],
    pub cone_cutoff: f32,
    pub index_offset: u32,
    pub triangle_count: u32,
    pub _pad: [u32; 2],
}

// ── Compile-time layout assertions ───────────────────────────────────────────
//
// If anyone adds a field and forgets to bump the version, these fire at
// compile time — not at runtime when a user loads a stale file.

const _: () = {
    assert!(core::mem::size_of::<Vertex>() == 80);
    assert!(core::mem::size_of::<MeshHeader>() == 52);
    assert!(core::mem::size_of::<MeshletData>() == 48);
};
