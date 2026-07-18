//! # cooker — Offline Asset Cooker for the Custom Game Engine
//!
//! Converts glTF 2.0 / GLB files into engine-native binary blobs:
//!   - `<name>.mesh`  — `MeshHeader + Vertex[] + u32[] + MeshletData[]`
//!   - `<name>_<n>.mat` — `MatHeader + MatData + texture path strings`
//!
//! ## Usage
//! ```text
//! cargo run --bin cooker -- model.gltf [--out-dir cooked/]
//! ```
//!
//! ## Design constraints
//! - Zero engine library dependencies — no Vulkan, no ECS, no allocators.
//! - Structs are `#[repr(C)]` and byte-for-byte identical to the engine's
//!   `Vertex`, `MeshletData`, and supporting types so the engine can
//!   `mmap` / stream the blobs directly into GPU staging memory (zero-copy).
//! - Format versioning: every file starts with a 4-byte magic + `u32` version
//!   so the engine loader can reject stale/mismatched files at runtime.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem;
use std::path::{Path, PathBuf};

use bytemuck::{Pod, Zeroable};

// ── Constants ────────────────────────────────────────────────────────────────

/// Maximum meshlet vertex count (matches engine/mesh.rs).
const MAX_MESHLET_VERTICES: usize = 64;
/// Maximum meshlet triangle count (matches engine/mesh.rs).
const MAX_MESHLET_TRIANGLES: usize = 124;
/// meshopt cone-weight for back-face culling aggressiveness.
const MESHLET_CONE_WEIGHT: f32 = 0.5;

// ── Vertex layout (must be byte-identical to pipeline.rs::Vertex) ────────────

/// `#[repr(C)]` vertex layout identical to the engine's `pipeline::Vertex`.
///
/// Offsets:
///   - pos:          0  (12 bytes)
///   - normal:      12  (12 bytes)
///   - uv:          24  (8  bytes)
///   - joint_ids:   32  (16 bytes)
///   - joint_weights:48 (16 bytes)
///   Total: 64 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    joint_ids: [u32; 4],
    joint_weights: [f32; 4],
}

// ── MeshletData (must be byte-identical to mesh.rs::MeshletData) ─────────────

/// `#[repr(C)]` meshlet descriptor identical to the engine's `mesh::MeshletData`.
///
/// 32 bytes total, 16-byte aligned (padding makes it a power-of-two size).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MeshletData {
    center: [f32; 3],
    radius: f32,
    cone_axis: [f32; 3],
    cone_cutoff: f32,
    index_offset: u32,
    triangle_count: u32,
    _pad: [u32; 2],
}

// ── .mesh binary format ───────────────────────────────────────────────────────

/// Magic bytes written at the start of every `.mesh` file.
const MESH_MAGIC: &[u8; 4] = b"MESH";
/// Increment when `MeshHeader` or any layout changes.
const MESH_VERSION: u32 = 1;

/// Flag bit: mesh contains skinning data (joint_ids / joint_weights).
const MESH_FLAG_SKINNED: u32 = 1 << 0;

/// Header preceding the vertex, index, and meshlet data arrays.
///
/// The engine reads this struct first, then seeks to `sizeof(MeshHeader)` to
/// begin streaming vertex data.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MeshHeader {
    magic: [u8; 4],      // b"MESH"
    version: u32,        // MESH_VERSION
    flags: u32,          // MESH_FLAG_* bitmask
    vertex_count: u32,
    index_count: u32,
    meshlet_count: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    _padding: u32,       // Pad to 48 bytes for 16-byte alignment
}

// ── .mat binary format ────────────────────────────────────────────────────────

/// Magic bytes written at the start of every `.mat` file.
const MAT_MAGIC: &[u8; 4] = b"MATL";
/// Increment when `MatHeader` or `MatData` layout changes.
const MAT_VERSION: u32 = 1;

/// Fixed header for `.mat` files.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MatHeader {
    magic: [u8; 4], // b"MATL"
    version: u32,   // MAT_VERSION
}

/// PBR material scalars and flags for texture presence.
///
/// Immediately follows `MatHeader`. Texture path strings follow this struct
/// as a null-terminated UTF-8 sequence; the engine looks them up relative to
/// the cooked asset directory.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MatData {
    base_color_factor: [f32; 4],
    metallic_factor: f32,
    roughness_factor: f32,
    /// Byte offset from file start to the albedo path string (0 = no texture).
    albedo_path_offset: u32,
    /// Byte offset from file start to the normal map path string (0 = none).
    normal_path_offset: u32,
    /// Byte offset from file start to the metal-roughness path string (0 = none).
    mr_path_offset: u32,
    /// Byte offset from file start to the emissive texture path string (0 = none).
    emissive_path_offset: u32,
    _pad: [u32; 2],
}

// ── Internal intermediate representation ─────────────────────────────────────

/// Raw mesh data extracted from a single glTF primitive.
struct RawPrimitive {
    vertices: Vec<Vertex>,
    /// Meshlet-reindexed global indices (triangle list).
    indices: Vec<u32>,
    meshlets: Vec<MeshletData>,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    /// True if any vertex has non-zero joint weights.
    is_skinned: bool,
    material_index: Option<usize>,
}

/// Material data extracted from a glTF material entry.
struct RawMaterial {
    base_color_factor: [f32; 4],
    metallic_factor: f32,
    roughness_factor: f32,
    albedo_tex: Option<String>,
    normal_tex: Option<String>,
    mr_tex: Option<String>,
    emissive_tex: Option<String>,
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let (input_path, out_dir) = parse_args(&args).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        eprintln!();
        eprintln!("Usage: cooker <input.gltf|input.glb> [--out-dir <directory>]");
        std::process::exit(1);
    });

    println!("[cooker] Input:   {}", input_path.display());
    println!("[cooker] Out dir: {}", out_dir.display());

    // Parse the glTF file
    let (primitives, materials) = load_gltf(&input_path).unwrap_or_else(|e| {
        eprintln!("[cooker] Failed to load glTF: {}", e);
        std::process::exit(1);
    });

    if primitives.is_empty() {
        eprintln!("[cooker] No mesh primitives found in '{}'.", input_path.display());
        std::process::exit(1);
    }

    // Derive the base name for output files (e.g. "warrior" from "warrior.gltf")
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| {
        eprintln!("[cooker] Cannot create output directory '{}': {}", out_dir.display(), e);
        std::process::exit(1);
    });

    // Write one .mesh per primitive
    for (i, prim) in primitives.iter().enumerate() {
        let mesh_name = if primitives.len() == 1 {
            format!("{}.mesh", stem)
        } else {
            format!("{}_{}.mesh", stem, i)
        };
        let mesh_path = out_dir.join(&mesh_name);

        write_mesh(prim, &mesh_path).unwrap_or_else(|e| {
            eprintln!("[cooker] Failed to write '{}': {}", mesh_path.display(), e);
            std::process::exit(1);
        });

        println!(
            "[cooker] ✓ {} — {} verts, {} indices, {} meshlets",
            mesh_name,
            prim.vertices.len(),
            prim.indices.len(),
            prim.meshlets.len()
        );

        // Write matching .mat if this primitive has a material
        if let Some(mat_idx) = prim.material_index {
            if let Some(mat) = materials.get(mat_idx) {
                let mat_name = if primitives.len() == 1 {
                    format!("{}.mat", stem)
                } else {
                    format!("{}_{}.mat", stem, i)
                };
                let mat_path = out_dir.join(&mat_name);

                write_mat(mat, &mat_path).unwrap_or_else(|e| {
                    eprintln!("[cooker] Failed to write '{}': {}", mat_path.display(), e);
                    std::process::exit(1);
                });

                println!("[cooker] ✓ {} — PBR material (metallic={:.2}, roughness={:.2})",
                    mat_name, mat.metallic_factor, mat.roughness_factor);
            }
        }
    }

    println!("[cooker] Done.");
}

// ── Argument parsing ──────────────────────────────────────────────────────────

fn parse_args(args: &[String]) -> Result<(PathBuf, PathBuf), String> {
    if args.len() < 2 {
        return Err("Missing required argument: <input.gltf>".to_string());
    }

    let input = PathBuf::from(&args[1]);
    if !input.exists() {
        return Err(format!("File not found: '{}'", input.display()));
    }

    let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext != "gltf" && ext != "glb" {
        return Err(format!(
            "Unsupported file type '{}'. Expected .gltf or .glb.",
            ext
        ));
    }

    // --out-dir flag
    let mut out_dir: Option<PathBuf> = None;
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--out-dir" {
            i += 1;
            if i >= args.len() {
                return Err("--out-dir requires a path argument".to_string());
            }
            out_dir = Some(PathBuf::from(&args[i]));
        }
        i += 1;
    }

    // Default out-dir: same directory as the input file
    let out_dir = out_dir.unwrap_or_else(|| {
        input.parent().unwrap_or(Path::new(".")).to_path_buf()
    });

    Ok((input, out_dir))
}

// ── glTF loading ──────────────────────────────────────────────────────────────

/// Parse a glTF / GLB file into raw intermediate primitives and materials.
fn load_gltf(path: &Path) -> Result<(Vec<RawPrimitive>, Vec<RawMaterial>), String> {
    let (document, buffers, _images) =
        gltf::import(path).map_err(|e| format!("gltf::import failed: {}", e))?;

    // ── Materials ────────────────────────────────────────────────────────────
    let mut materials: Vec<RawMaterial> = Vec::new();
    for mat in document.materials() {
        let pbr = mat.pbr_metallic_roughness();

        // Helper: convert a glTF texture reference into a stable file path string.
        // We record the source image index so the engine can find the right file.
        // For embedded textures we record "embedded:<index>" as a sentinel that
        // Phase 2 (VFS hooks) will resolve when loading the cooked asset.
        let tex_path = |info: Option<gltf::texture::Texture<'_>>| -> Option<String> {
            info.map(|t| {
                let src = t.source();
                match src.source() {
                    gltf::image::Source::Uri { uri, .. } => {
                        // Resolve relative to the glTF directory
                        let gltf_dir = path.parent().unwrap_or(Path::new("."));
                        gltf_dir
                            .join(uri)
                            .to_string_lossy()
                            .replace('\\', "/")
                    }
                    gltf::image::Source::View { .. } => {
                        // Embedded image — record index as sentinel for Phase 2
                        format!("embedded:{}", src.index())
                    }
                }
            })
        };

        materials.push(RawMaterial {
            base_color_factor: pbr.base_color_factor(),
            metallic_factor: pbr.metallic_factor(),
            roughness_factor: pbr.roughness_factor(),
            albedo_tex: tex_path(pbr.base_color_texture().map(|t| t.texture())),
            normal_tex: tex_path(mat.normal_texture().map(|t| t.texture())),
            mr_tex: tex_path(pbr.metallic_roughness_texture().map(|t| t.texture())),
            emissive_tex: tex_path(mat.emissive_texture().map(|t| t.texture())),
        });
    }

    // ── Primitives ───────────────────────────────────────────────────────────
    let mut primitives: Vec<RawPrimitive> = Vec::new();
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

            // Positions (required)
            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(iter) => iter.collect(),
                None => {
                    eprintln!(
                        "[cooker] Warning: primitive in mesh '{}' has no POSITION data — skipping.",
                        mesh.name().unwrap_or("<unnamed>")
                    );
                    continue;
                }
            };
            let n = positions.len();

            // Normals
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|it| it.collect())
                .unwrap_or_else(|| compute_flat_normals(&positions));

            // UV0
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|it| it.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0f32; 2]; n]);

            // Skinning: JOINTS_0 + WEIGHTS_0
            let joints: Vec<[u32; 4]> = reader
                .read_joints(0)
                .map(|it| {
                    it.into_u16()
                        .map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32])
                        .collect()
                })
                .unwrap_or_else(|| vec![[0u32; 4]; n]);

            let weights: Vec<[f32; 4]> = reader
                .read_weights(0)
                .map(|it| it.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0f32; 4]; n]);

            // Indices (fall back to sequential if absent)
            let base_indices: Vec<u32> = reader
                .read_indices()
                .map(|it| it.into_u32().collect())
                .unwrap_or_else(|| (0..n as u32).collect());

            if base_indices.is_empty() || positions.is_empty() {
                continue;
            }

            // Detect skinning (any non-zero weight)
            let is_skinned = weights.iter().any(|w| w.iter().any(|&v| v > 0.0));

            // Build the flat Vertex array
            let vertices: Vec<Vertex> = (0..n)
                .map(|i| Vertex {
                    pos: positions[i],
                    normal: normals[i],
                    uv: uvs[i],
                    joint_ids: joints[i],
                    joint_weights: weights[i],
                })
                .collect();

            // AABB
            let (aabb_min, aabb_max) = compute_aabb(&vertices);

            // Meshlet generation via meshopt
            let (indices, meshlets) =
                build_meshlets_for_primitive(&vertices, &base_indices);

            let material_index = primitive.material().index();

            primitives.push(RawPrimitive {
                vertices,
                indices,
                meshlets,
                aabb_min,
                aabb_max,
                is_skinned,
                material_index,
            });
        }
    }

    Ok((primitives, materials))
}

// ── Meshlet generation ────────────────────────────────────────────────────────

/// Run `meshopt::build_meshlets` and flatten the output into engine-ready
/// `(global_indices: Vec<u32>, meshlets: Vec<MeshletData>)`.
///
/// The returned `global_indices` are the fully expanded triangle list in meshlet
/// order — identical to how `mesh.rs::Mesh::load_models` builds them today.
fn build_meshlets_for_primitive(
    vertices: &[Vertex],
    base_indices: &[u32],
) -> (Vec<u32>, Vec<MeshletData>) {
    // Build the meshopt VertexDataAdapter pointing at the position field (offset 0).
    let vertices_u8: &[u8] = bytemuck::cast_slice(vertices);
    let vertex_data = meshopt::VertexDataAdapter::new(
        vertices_u8,
        mem::size_of::<Vertex>(),
        0, // position offset in struct
    )
    .expect("VertexDataAdapter construction failed");

    let meshlets_raw = meshopt::build_meshlets(
        base_indices,
        &vertex_data,
        MAX_MESHLET_VERTICES,
        MAX_MESHLET_TRIANGLES,
        MESHLET_CONE_WEIGHT,
    );

    let mut global_indices: Vec<u32> = Vec::new();
    let mut meshlet_data: Vec<MeshletData> = Vec::new();

    for i in 0..meshlets_raw.meshlets.len() {
        let raw_m = &meshlets_raw.meshlets[i];
        let index_offset = global_indices.len() as u32;

        // Expand local (meshlet-relative) indices into global vertex indices.
        for tri_idx in 0..(raw_m.triangle_count * 3) {
            let local = meshlets_raw.triangles[(raw_m.triangle_offset + tri_idx) as usize];
            let global =
                meshlets_raw.vertices[(raw_m.vertex_offset + local as u32) as usize];
            global_indices.push(global);
        }

        let bounds = meshopt::compute_meshlet_bounds(meshlets_raw.get(i), &vertex_data);

        meshlet_data.push(MeshletData {
            center: bounds.center,
            radius: bounds.radius,
            cone_axis: bounds.cone_axis,
            cone_cutoff: bounds.cone_cutoff,
            index_offset,
            triangle_count: raw_m.triangle_count,
            _pad: [0; 2],
        });
    }

    (global_indices, meshlet_data)
}

// ── AABB & normal helpers ─────────────────────────────────────────────────────

fn compute_aabb(vertices: &[Vertex]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for v in vertices {
        for i in 0..3 {
            if v.pos[i] < min[i] { min[i] = v.pos[i]; }
            if v.pos[i] > max[i] { max[i] = v.pos[i]; }
        }
    }
    (min, max)
}

/// Generate flat (face) normals when the glTF primitive omits NORMAL data.
fn compute_flat_normals(positions: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let n = positions.len();
    let mut normals = vec![[0.0f32; 3]; n];

    // Walk triangles in groups of 3; assign the face normal to all 3 verts.
    let mut tri_start = 0;
    while tri_start + 2 < n {
        let v0 = positions[tri_start];
        let v1 = positions[tri_start + 1];
        let v2 = positions[tri_start + 2];
        let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
        let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
        let cross = [
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0],
        ];
        let len = (cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]).sqrt();
        let face_normal = if len > 1e-6 {
            [cross[0]/len, cross[1]/len, cross[2]/len]
        } else {
            [0.0, 1.0, 0.0]
        };
        normals[tri_start]     = face_normal;
        normals[tri_start + 1] = face_normal;
        normals[tri_start + 2] = face_normal;
        tri_start += 3;
    }
    normals
}

// ── .mesh writer ──────────────────────────────────────────────────────────────

/// Serialise a `RawPrimitive` into a `.mesh` binary blob.
///
/// Layout on disk:
/// ```text
/// MeshHeader          (48 bytes)
/// Vertex[vertex_count] (64 bytes each)
/// u32[index_count]    (4 bytes each)
/// MeshletData[meshlet_count] (32 bytes each)
/// ```
fn write_mesh(prim: &RawPrimitive, path: &Path) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    let flags = if prim.is_skinned { MESH_FLAG_SKINNED } else { 0 };

    let header = MeshHeader {
        magic: *MESH_MAGIC,
        version: MESH_VERSION,
        flags,
        vertex_count: prim.vertices.len() as u32,
        index_count: prim.indices.len() as u32,
        meshlet_count: prim.meshlets.len() as u32,
        aabb_min: prim.aabb_min,
        aabb_max: prim.aabb_max,
        _padding: 0,
    };

    // Write header
    w.write_all(bytemuck::bytes_of(&header))?;

    // Write vertex array
    w.write_all(bytemuck::cast_slice(&prim.vertices))?;

    // Write index array
    w.write_all(bytemuck::cast_slice(&prim.indices))?;

    // Write meshlet array
    w.write_all(bytemuck::cast_slice(&prim.meshlets))?;

    w.flush()?;
    Ok(())
}

// ── .mat writer ───────────────────────────────────────────────────────────────

/// Serialise a `RawMaterial` into a `.mat` binary blob.
///
/// Layout on disk:
/// ```text
/// MatHeader  (8 bytes)
/// MatData    (48 bytes)
/// [string section: variable-length null-terminated UTF-8 paths]
/// ```
/// `MatData.{albedo,normal,mr,emissive}_path_offset` stores the byte offset
/// **from the start of the file** to the corresponding null-terminated string.
/// An offset of 0 means the texture is absent.
fn write_mat(mat: &RawMaterial, path: &Path) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // Compute the base offset: where string data begins
    let fixed_size = (mem::size_of::<MatHeader>() + mem::size_of::<MatData>()) as u32;

    // Build the string section and compute offsets
    let mut string_section: Vec<u8> = Vec::new();

    let mut alloc_string = |s: &Option<String>| -> u32 {
        match s {
            None => 0,
            Some(text) => {
                let offset = fixed_size + string_section.len() as u32;
                string_section.extend_from_slice(text.as_bytes());
                string_section.push(0); // null terminator
                offset
            }
        }
    };

    let albedo_offset   = alloc_string(&mat.albedo_tex);
    let normal_offset   = alloc_string(&mat.normal_tex);
    let mr_offset       = alloc_string(&mat.mr_tex);
    let emissive_offset = alloc_string(&mat.emissive_tex);

    let header = MatHeader {
        magic: *MAT_MAGIC,
        version: MAT_VERSION,
    };

    let data = MatData {
        base_color_factor: mat.base_color_factor,
        metallic_factor: mat.metallic_factor,
        roughness_factor: mat.roughness_factor,
        albedo_path_offset: albedo_offset,
        normal_path_offset: normal_offset,
        mr_path_offset: mr_offset,
        emissive_path_offset: emissive_offset,
        _pad: [0; 2],
    };

    w.write_all(bytemuck::bytes_of(&header))?;
    w.write_all(bytemuck::bytes_of(&data))?;
    w.write_all(&string_section)?;

    w.flush()?;
    Ok(())
}
