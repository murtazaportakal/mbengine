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

// ── Shared binary formats ────────────────────────────────────────────────────
//
// Imported from the engine library — single source of truth.
// No more "must be byte-identical to..." copy-paste contracts.
use engine::renderer::vulkan::gpu_format::{
    MeshHeader, MeshletData, Vertex, MESH_FLAG_SKINNED, MESH_MAGIC, MESH_VERSION,
};

// ── Constants ────────────────────────────────────────────────────────────────

/// Maximum meshlet vertex count (matches engine/mesh.rs).
const MAX_MESHLET_VERTICES: usize = 64;
/// Maximum meshlet triangle count (matches engine/mesh.rs).
const MAX_MESHLET_TRIANGLES: usize = 124;
/// meshopt cone-weight for back-face culling aggressiveness.
const MESHLET_CONE_WEIGHT: f32 = 0.5;

// ── .mat binary format ────────────────────────────────────────────────────────

/// Magic bytes written at the start of every `.mat` file.
const MAT_MAGIC: &[u8; 4] = b"MATL";
/// Increment when `MatHeader` or `MatData` layout changes.
const MAT_VERSION: u32 = 1;

// ── .tex binary format ────────────────────────────────────────────────────────

/// Magic bytes written at the start of every `.tex` file.
const TEX_MAGIC: &[u8; 4] = b"TEXL";
const TEX_VERSION: u32 = 1;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(dead_code)]
pub enum TexFormat {
    Rgba8Unorm = 0,
    Bc7UnormBlock = 1,
    Bc7SrgbBlock = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TexHeader {
    magic: [u8; 4],
    version: u32,
    width: u32,
    height: u32,
    format: u32,
    mip_count: u32,
    data_size: u32,
    _pad: u32,
}

// ── .anim binary format ───────────────────────────────────────────────────────

const ANIM_MAGIC: &[u8; 4] = b"ANIM";
const ANIM_VERSION: u32 = 1;
pub const MAX_BONES: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct AnimHeader {
    magic: [u8; 4],
    version: u32,
    bone_count: u32,
    clip_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BoneHeader {
    inverse_bind_matrix: [f32; 16],
    parent_index: i32,
    name_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ClipHeader {
    name_len: u32,
    duration: f32,
    channel_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ChannelHeader {
    bone_index: u32,
    translation_count: u32,
    rotation_count: u32,
    scale_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vec3Key {
    time: f32,
    value: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct QuatKey {
    time: f32,
    value: [f32; 4],
}

struct RawBone {
    name: String,
    inverse_bind_matrix: [f32; 16],
    parent_index: i32,
}

struct RawChannel {
    bone_index: u32,
    translations: Vec<Vec3Key>,
    rotations: Vec<QuatKey>,
    scales: Vec<Vec3Key>,
}

struct RawClip {
    name: String,
    duration: f32,
    channels: Vec<RawChannel>,
}

struct RawSkeleton {
    bones: Vec<RawBone>,
    clips: Vec<RawClip>,
}

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

// ── Core Architecture ────────────────────────────────────────────────────────

pub struct RawScene {
    primitives: Vec<RawPrimitive>,
    materials: Vec<RawMaterial>,
    skeleton: Option<RawSkeleton>,
}

trait AssetImporter {
    fn import(&self, path: &Path) -> Result<RawScene, String>;
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let (input_path, out_dir) = parse_args(&args).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        eprintln!();
        eprintln!("Usage: cooker <input.file> [--out-dir <directory>]");
        std::process::exit(1);
    });

    println!("[cooker] Input:   {}", input_path.display());
    println!("[cooker] Out dir: {}", out_dir.display());

    let ext = input_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

    match ext.as_str() {
        "gltf" | "glb" => {
            let importer = GltfImporter;
            let scene = importer.import(&input_path).unwrap_or_else(|e| {
                eprintln!("[cooker] Failed to load glTF: {}", e);
                std::process::exit(1);
            });
            cook_scene(scene, &input_path, &out_dir);
        }
        "obj" => {
            let importer = ObjImporter;
            let scene = importer.import(&input_path).unwrap_or_else(|e| {
                eprintln!("[cooker] Failed to load OBJ: {}", e);
                std::process::exit(1);
            });
            cook_scene(scene, &input_path, &out_dir);
        }
        "png" | "jpg" | "tga" | "bmp" => {
            let stem = input_path.file_stem().unwrap().to_string_lossy();
            let out_tex = out_dir.join(format!("{}.tex", stem));
            std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| {
                eprintln!("[cooker] Cannot create output directory '{}': {}", out_dir.display(), e);
                std::process::exit(1);
            });
            cook_texture(&input_path, &out_tex, true).unwrap_or_else(|e| {
                eprintln!("[cooker] Failed to cook texture: {}", e);
                std::process::exit(1);
            });
            println!("[cooker] Done.");
        }
        _ => {
            eprintln!("[cooker] Unsupported file type '{}'.", ext);
            std::process::exit(1);
        }
    }
}

fn cook_scene(mut scene: RawScene, input_path: &Path, out_dir: &Path) {
    if scene.primitives.is_empty() {
        eprintln!("[cooker] No mesh primitives found in '{}'.", input_path.display());
        std::process::exit(1);
    }

    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| {
        eprintln!("[cooker] Cannot create output directory '{}': {}", out_dir.display(), e);
        std::process::exit(1);
    });

    let input_parent = input_path.parent().unwrap_or(Path::new(""));

    // Pre-cook all textures in materials
    for mat in &mut scene.materials {
        let process_tex = |tex_opt: &mut Option<String>, is_srgb: bool| {
            if let Some(tex) = tex_opt {
                let in_tex_path = input_parent.join(&tex);
                
                // Change extension to .tex
                let out_tex = std::path::Path::new(tex).with_extension("tex").to_string_lossy().into_owned();
                let out_tex_path = out_dir.join(&out_tex);
                
                if let Some(parent) = out_tex_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                
                // Skip if already cooked
                if !out_tex_path.exists() {
                    if let Err(e) = cook_texture(&in_tex_path, &out_tex_path, is_srgb) {
                        eprintln!("[cooker] WARNING: Failed to cook texture {}: {}", in_tex_path.display(), e);
                    }
                }
                
                *tex_opt = Some(out_tex);
            }
        };
        
        process_tex(&mut mat.albedo_tex, true);
        process_tex(&mut mat.normal_tex, false);
        process_tex(&mut mat.mr_tex, false);
        process_tex(&mut mat.emissive_tex, true);
    }

    // Write one .mesh per primitive
    for (i, prim) in scene.primitives.iter().enumerate() {
        let mesh_name = if scene.primitives.len() == 1 {
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
            if let Some(mat) = scene.materials.get(mat_idx) {
                let mat_name = if scene.primitives.len() == 1 {
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

    if let Some(skel) = scene.skeleton {
        let anim_name = format!("{}.anim", stem);
        let anim_path = out_dir.join(&anim_name);
        write_anim(&skel, &anim_path).unwrap_or_else(|e| {
            eprintln!("[cooker] Failed to write '{}': {}", anim_path.display(), e);
            std::process::exit(1);
        });
        println!(
            "[cooker] ✓ {} — {} bones, {} clips",
            anim_name,
            skel.bones.len(),
            skel.clips.len()
        );
    }

    println!("[cooker] Done.");
}

// ── Argument parsing ──────────────────────────────────────────────────────────

fn parse_args(args: &[String]) -> Result<(PathBuf, PathBuf), String> {
    if args.len() < 2 {
        return Err("Missing required argument: <input.file>".to_string());
    }

    let input = PathBuf::from(&args[1]);
    if !input.exists() {
        return Err(format!("File not found: '{}'", input.display()));
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

// ── Importers ──────────────────────────────────────────────────────────────

struct GltfImporter;

impl AssetImporter for GltfImporter {
    fn import(&self, path: &Path) -> Result<RawScene, String> {
        let (primitives, materials, skeleton) = load_gltf_internal(path)?;
        Ok(RawScene {
            primitives,
            materials,
            skeleton,
        })
    }
}

struct ObjImporter;

impl AssetImporter for ObjImporter {
    fn import(&self, path: &Path) -> Result<RawScene, String> {
        let (models, materials) = tobj::load_obj(
            path,
            &tobj::LoadOptions {
                single_index: true,
                triangulate: true,
                ignore_points: true,
                ignore_lines: true,
            },
        ).map_err(|e| format!("Failed to load OBJ: {}", e))?;

        let mut out_materials = Vec::new();
        if let Ok(mats) = materials {
            for m in mats {
                out_materials.push(RawMaterial {
                    base_color_factor: m.diffuse.map_or([1.0, 1.0, 1.0, 1.0], |d| [d[0], d[1], d[2], 1.0]),
                    metallic_factor: 0.0,
                    roughness_factor: 0.8,
                    albedo_tex: m.diffuse_texture.filter(|s| !s.is_empty()),
                    normal_tex: m.ambient_texture.filter(|s| !s.is_empty()), // Often misused in OBJs for normal map
                    mr_tex: None,
                    emissive_tex: None,
                });
            }
        }

        let mut primitives = Vec::new();
        for m in models {
            let mesh = m.mesh;
            
            if mesh.positions.is_empty() || mesh.indices.is_empty() {
                continue;
            }
            
            let mut vertices = Vec::new();
            let num_vertices = mesh.positions.len() / 3;

            let has_normals = !mesh.normals.is_empty();
            let has_uvs = !mesh.texcoords.is_empty();

            let normals = if has_normals {
                mesh.normals.clone()
            } else {
                let mut flat = vec![0.0; num_vertices * 3];
                for idx in mesh.indices.chunks_exact(3) {
                    let i0 = idx[0] as usize * 3;
                    let i1 = idx[1] as usize * 3;
                    let i2 = idx[2] as usize * 3;

                    let v0 = nalgebra::Vector3::new(mesh.positions[i0], mesh.positions[i0+1], mesh.positions[i0+2]);
                    let v1 = nalgebra::Vector3::new(mesh.positions[i1], mesh.positions[i1+1], mesh.positions[i1+2]);
                    let v2 = nalgebra::Vector3::new(mesh.positions[i2], mesh.positions[i2+1], mesh.positions[i2+2]);

                    let normal = (v1 - v0).cross(&(v2 - v0));
                    for j in 0..3 {
                        flat[i0+j] += normal[j];
                        flat[i1+j] += normal[j];
                        flat[i2+j] += normal[j];
                    }
                }
                for i in (0..flat.len()).step_by(3) {
                    let n = nalgebra::Vector3::new(flat[i], flat[i+1], flat[i+2]).normalize();
                    flat[i] = n[0]; flat[i+1] = n[1]; flat[i+2] = n[2];
                }
                flat
            };

            for i in 0..num_vertices {
                vertices.push(Vertex {
                    pos: [mesh.positions[i * 3], mesh.positions[i * 3 + 1], mesh.positions[i * 3 + 2]],
                    normal: [normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]],
                    uv: if has_uvs { [mesh.texcoords[i * 2], mesh.texcoords[i * 2 + 1]] } else { [0.0, 0.0] },
                    joint_ids: [0, 0, 0, 0],
                    joint_weights: [0.0, 0.0, 0.0, 0.0],
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: [0, 0],
                });
            }

            let (aabb_min, aabb_max) = compute_aabb(&vertices);
            let (indices, meshlets) = build_meshlets_for_primitive(&vertices, &mesh.indices);

            primitives.push(RawPrimitive {
                vertices,
                indices,
                meshlets,
                aabb_min,
                aabb_max,
                is_skinned: false,
                material_index: mesh.material_id,
            });
        }

        Ok(RawScene {
            primitives,
            materials: out_materials,
            skeleton: None,
        })
    }
}

/// Parse a glTF / GLB file into raw intermediate primitives and materials.
fn load_gltf_internal(path: &std::path::Path) -> Result<(Vec<RawPrimitive>, Vec<RawMaterial>, Option<RawSkeleton>), String> {
    let (document, buffers, _images) =
        gltf::import(path).map_err(|e| format!("gltf::import failed: {}", e))?;

    let mut mesh_transforms = std::collections::HashMap::new();
    fn traverse_node(node: &gltf::Node, parent_transform: nalgebra::Matrix4<f32>, map: &mut std::collections::HashMap<usize, nalgebra::Matrix4<f32>>) {
        let m = node.transform().matrix();
        let local = nalgebra::Matrix4::new(
            m[0][0], m[1][0], m[2][0], m[3][0],
            m[0][1], m[1][1], m[2][1], m[3][1],
            m[0][2], m[1][2], m[2][2], m[3][2],
            m[0][3], m[1][3], m[2][3], m[3][3],
        );
        let world = parent_transform * local;
        map.insert(node.index(), world);
        for child in node.children() {
            traverse_node(&child, world, map);
        }
    }
    for scene in document.scenes() {
        for node in scene.nodes() {
            traverse_node(&node, nalgebra::Matrix4::identity(), &mut mesh_transforms);
        }
    }

    let mut materials: Vec<RawMaterial> = Vec::new();
    for mat in document.materials() {
        let pbr = mat.pbr_metallic_roughness();
        let tex_path = |info: Option<gltf::texture::Texture<'_>>| -> Option<String> {
            info.map(|t| {
                let src = t.source();
                match src.source() {
                    gltf::image::Source::Uri { uri, .. } => {
                        let gltf_dir = path.parent().unwrap_or(std::path::Path::new("."));
                        gltf_dir.join(uri).to_string_lossy().replace('\\', "/")
                    }
                    gltf::image::Source::View { .. } => format!("embedded:{}", src.index()),
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

    let has_real_skin = document.skins().next().is_some();
    let has_animations = document.animations().count() > 0;
    let use_synthetic_skeleton = !has_real_skin && has_animations;

    let mut node_to_synthetic_bone = std::collections::HashMap::new();
    let mut synthetic_joints = Vec::new();
    if use_synthetic_skeleton {
        let mut bone_idx = 0;
        for scene in document.scenes() {
            fn map_nodes(node: &gltf::Node, map: &mut std::collections::HashMap<usize, u32>, idx: &mut u32, joints: &mut Vec<usize>) {
                map.insert(node.index(), *idx);
                joints.push(node.index());
                *idx += 1;
                for child in node.children() {
                    map_nodes(&child, map, idx, joints);
                }
            }
            for node in scene.nodes() {
                map_nodes(&node, &mut node_to_synthetic_bone, &mut bone_idx, &mut synthetic_joints);
            }
        }
    }

    let mut primitives: Vec<RawPrimitive> = Vec::new();
    for scene in document.scenes() {
        fn extract_from_node(
            node: &gltf::Node,
            document: &gltf::Document,
            buffers: &[gltf::buffer::Data],
            mesh_transforms: &std::collections::HashMap<usize, nalgebra::Matrix4<f32>>,
            use_synthetic_skeleton: bool,
            node_to_synthetic_bone: &std::collections::HashMap<usize, u32>,
            primitives: &mut Vec<RawPrimitive>
        ) {
            if let Some(mesh) = node.mesh() {
                for primitive in mesh.primitives() {
                    let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));
                    let positions: Vec<[f32; 3]> = match reader.read_positions() {
                        Some(iter) => iter.collect(),
                        None => continue,
                    };
                    let n = positions.len();

                    let normals: Vec<[f32; 3]> = reader.read_normals().map(|it| it.collect()).unwrap_or_else(|| compute_flat_normals(&positions));
                    let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0).map(|it| it.into_f32().collect()).unwrap_or_else(|| vec![[0.0f32; 2]; n]);
                    
                    let mut joints: Vec<[u32; 4]> = reader.read_joints(0).map(|it| {
                        it.into_u16().map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32]).collect()
                    }).unwrap_or_else(|| vec![[0u32; 4]; n]);
                    
                    let mut weights: Vec<[f32; 4]> = reader.read_weights(0).map(|it| it.into_f32().collect()).unwrap_or_else(|| vec![[0.0f32; 4]; n]);
                    
                    let mut is_skinned = reader.read_weights(0).is_some();
                    
                    if use_synthetic_skeleton {
                        if let Some(&bone_idx) = node_to_synthetic_bone.get(&node.index()) {
                            joints = vec![[bone_idx, 0, 0, 0]; n];
                            weights = vec![[1.0, 0.0, 0.0, 0.0]; n];
                            is_skinned = true;
                        }
                    }

                    let base_indices: Vec<u32> = reader.read_indices().map(|it| it.into_u32().collect()).unwrap_or_else(|| (0..n as u32).collect());
                    if base_indices.is_empty() || positions.is_empty() { continue; }

                    let transform = mesh_transforms.get(&node.index()).copied().unwrap_or(nalgebra::Matrix4::identity());
                    let normal_transform = transform.try_inverse().unwrap_or_else(|| nalgebra::Matrix4::identity()).transpose();

                    let mut vertices = Vec::new();
                    for i in 0..n {
                        let pos_v = transform * nalgebra::Vector4::new(positions[i][0], positions[i][1], positions[i][2], 1.0);
                        let norm_v = normal_transform * nalgebra::Vector4::new(normals[i][0], normals[i][1], normals[i][2], 0.0);
                        let mut norm_vec = nalgebra::Vector3::new(norm_v.x, norm_v.y, norm_v.z);
                        if norm_vec.norm_squared() > 0.0 { norm_vec.normalize_mut(); }
                        vertices.push(Vertex {
                            pos: [pos_v.x, pos_v.y, pos_v.z],
                            _pad0: 0,
                            normal: [norm_vec.x, norm_vec.y, norm_vec.z],
                            _pad1: 0,
                            uv: uvs[i],
                            _pad2: [0; 2],
                            joint_ids: joints[i],
                            joint_weights: weights[i],
                        });
                    }

                    let (aabb_min, aabb_max) = compute_aabb(&vertices);
                    let (indices, meshlets) = build_meshlets_for_primitive(&vertices, &base_indices);
                    let material_index = primitive.material().index();

                    primitives.push(RawPrimitive { vertices, indices, meshlets, aabb_min, aabb_max, is_skinned, material_index });
                }
            }
            for child in node.children() {
                extract_from_node(&child, document, buffers, mesh_transforms, use_synthetic_skeleton, node_to_synthetic_bone, primitives);
            }
        }
        for node in scene.nodes() {
            extract_from_node(&node, &document, &buffers, &mesh_transforms, use_synthetic_skeleton, &node_to_synthetic_bone, &mut primitives);
        }
    }

    let skeleton = extract_skeleton_and_animations(&document, &buffers, &mesh_transforms, use_synthetic_skeleton, &node_to_synthetic_bone, &synthetic_joints);
    Ok((primitives, materials, skeleton))
}

fn extract_skeleton_and_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    mesh_transforms: &std::collections::HashMap<usize, nalgebra::Matrix4<f32>>,
    use_synthetic_skeleton: bool,
    node_to_synthetic_bone: &std::collections::HashMap<usize, u32>,
    synthetic_joints: &[usize],
) -> Option<RawSkeleton> {
    let mut joints: Vec<gltf::Node<'_>> = Vec::new();
    let mut node_to_bone = std::collections::HashMap::new();

    if use_synthetic_skeleton {
        for &node_idx in synthetic_joints {
            for scene in document.scenes() {
                fn find_node<'a>(n: &gltf::Node<'a>, idx: usize) -> Option<gltf::Node<'a>> {
                    if n.index() == idx { return Some(n.clone()); }
                    for child in n.children() {
                        if let Some(res) = find_node(&child, idx) { return Some(res); }
                    }
                    None
                }
                for node in scene.nodes() {
                    if let Some(res) = find_node(&node, node_idx) {
                        joints.push(res);
                        break;
                    }
                }
            }
        }
        node_to_bone = node_to_synthetic_bone.clone();
    } else {
        if let Some(skin) = document.skins().next() {
            joints = skin.joints().collect();
            for (i, joint) in joints.iter().enumerate() {
                node_to_bone.insert(joint.index(), i as u32);
            }
        } else {
            return None; // No skin and no synthetic skeleton
        }
    }

    if joints.len() > MAX_BONES {
        eprintln!("[cooker] Warning: Skeleton has {} bones (max is {}).", joints.len(), MAX_BONES);
        return None;
    }

    let mut parent_map = std::collections::HashMap::new();
    fn visit(node: &gltf::Node<'_>, map: &mut std::collections::HashMap<usize, usize>) {
        for child in node.children() {
            map.insert(child.index(), node.index());
            visit(&child, map);
        }
    }
    for scene in document.scenes() {
        for node in scene.nodes() {
            visit(&node, &mut parent_map);
        }
    }

    let find_parent_bone = |node: &gltf::Node<'_>| -> i32 {
        let mut curr = node.index();
        loop {
            match parent_map.get(&curr) {
                Some(&parent_node) => {
                    if let Some(&bone_idx) = node_to_bone.get(&parent_node) {
                        return bone_idx as i32;
                    }
                    curr = parent_node;
                }
                None => return -1,
            }
        }
    };

    let inverse_bind_matrices: Vec<[f32; 16]> = if use_synthetic_skeleton {
        joints.iter().map(|j| {
            let t = mesh_transforms.get(&j.index()).copied().unwrap_or(nalgebra::Matrix4::identity());
            let inv = t.try_inverse().unwrap_or(nalgebra::Matrix4::identity());
            [
                inv[0], inv[4], inv[8], inv[12],
                inv[1], inv[5], inv[9], inv[13],
                inv[2], inv[6], inv[10], inv[14],
                inv[3], inv[7], inv[11], inv[15],
            ]
        }).collect()
    } else {
        let skin = document.skins().next().unwrap();
        match skin.reader(|buf| Some(&buffers[buf.index()])).read_inverse_bind_matrices() {
            Some(iter) => iter.map(|m| {
                [
                    m[0][0], m[0][1], m[0][2], m[0][3],
                    m[1][0], m[1][1], m[1][2], m[1][3],
                    m[2][0], m[2][1], m[2][2], m[2][3],
                    m[3][0], m[3][1], m[3][2], m[3][3],
                ]
            }).collect(),
            None => {
                let id = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
                vec![id; joints.len()]
            }
        }
    };

    let mut bones = Vec::new();
    for (i, joint) in joints.iter().enumerate() {
        bones.push(RawBone {
            name: joint.name().unwrap_or("unnamed").to_string(),
            inverse_bind_matrix: inverse_bind_matrices[i],
            parent_index: find_parent_bone(joint),
        });
    }

    let mut clips = Vec::new();
    for animation in document.animations() {
        let mut duration = 0.0f32;
        let mut channels = Vec::new();
        let mut channel_map = std::collections::HashMap::new();

        for channel in animation.channels() {
            let target_node = channel.target().node().index();
            let bone_idx = match node_to_bone.get(&target_node) {
                Some(&idx) => idx,
                None => continue,
            };

            let reader = channel.reader(|buf| Some(&buffers[buf.index()]));
            let timestamps: Vec<f32> = match reader.read_inputs() {
                Some(iter) => iter.collect(),
                None => continue,
            };

            if let Some(&max_t) = timestamps.last() {
                if max_t > duration { duration = max_t; }
            }

            let entry = channel_map.entry(bone_idx).or_insert_with(|| RawChannel {
                bone_index: bone_idx,
                translations: Vec::new(),
                rotations: Vec::new(),
                scales: Vec::new(),
            });

            match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::Translations(iter)) => {
                    for (t, val) in timestamps.iter().zip(iter) {
                        entry.translations.push(Vec3Key { time: *t, value: val });
                    }
                }
                Some(gltf::animation::util::ReadOutputs::Rotations(iter)) => {
                    for (t, val) in timestamps.iter().zip(iter.into_f32()) {
                        entry.rotations.push(QuatKey { time: *t, value: val });
                    }
                }
                Some(gltf::animation::util::ReadOutputs::Scales(iter)) => {
                    for (t, val) in timestamps.iter().zip(iter) {
                        entry.scales.push(Vec3Key { time: *t, value: val });
                    }
                }
                _ => {}
            }
        }

        for (_, ch) in channel_map {
            channels.push(ch);
        }

        clips.push(RawClip {
            name: animation.name().unwrap_or("unnamed_clip").to_string(),
            duration,
            channels,
        });
    }

    Some(RawSkeleton { bones, clips })
}
fn write_anim(skel: &RawSkeleton, path: &Path) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    let header = AnimHeader {
        magic: *ANIM_MAGIC,
        version: ANIM_VERSION,
        bone_count: skel.bones.len() as u32,
        clip_count: skel.clips.len() as u32,
    };
    w.write_all(bytemuck::bytes_of(&header))?;

    // Helper to write string with 4-byte padding
    let write_str = |s: &str, w: &mut BufWriter<File>| -> std::io::Result<()> {
        let bytes = s.as_bytes();
        w.write_all(bytes)?;
        let pad = (4 - (bytes.len() % 4)) % 4;
        for _ in 0..pad { w.write_all(&[0])?; }
        Ok(())
    };

    // Bones
    for bone in &skel.bones {
        let bh = BoneHeader {
            inverse_bind_matrix: bone.inverse_bind_matrix,
            parent_index: bone.parent_index,
            name_len: bone.name.len() as u32,
        };
        w.write_all(bytemuck::bytes_of(&bh))?;
        write_str(&bone.name, &mut w)?;
    }

    // Clips
    for clip in &skel.clips {
        let ch = ClipHeader {
            name_len: clip.name.len() as u32,
            duration: clip.duration,
            channel_count: clip.channels.len() as u32,
        };
        w.write_all(bytemuck::bytes_of(&ch))?;
        write_str(&clip.name, &mut w)?;

        for ch in &clip.channels {
            let channel_head = ChannelHeader {
                bone_index: ch.bone_index,
                translation_count: ch.translations.len() as u32,
                rotation_count: ch.rotations.len() as u32,
                scale_count: ch.scales.len() as u32,
            };
            w.write_all(bytemuck::bytes_of(&channel_head))?;
            w.write_all(bytemuck::cast_slice(&ch.translations))?;
            w.write_all(bytemuck::cast_slice(&ch.rotations))?;
            w.write_all(bytemuck::cast_slice(&ch.scales))?;
        }
    }

    w.flush()?;
    Ok(())
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
        magic: MESH_MAGIC,
        version: MESH_VERSION,
        flags,
        vertex_count: prim.vertices.len() as u32,
        index_count: prim.indices.len() as u32,
        meshlet_count: prim.meshlets.len() as u32,
        aabb_min: prim.aabb_min,
        aabb_max: prim.aabb_max,
        _pad: 0,
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

fn cook_texture(in_path: &Path, out_path: &Path, _is_srgb: bool) -> std::io::Result<()> {
    println!("[Cooker] Compressing texture: {}", in_path.display());
    
    // Load image
    let img = image::open(in_path)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    
    let rgba8 = img.into_rgba8();
    let (width, height) = rgba8.dimensions();
    
    // Write .tex file
    let file = File::create(out_path)?;
    let mut w = BufWriter::new(file);
    
    let data_size = rgba8.as_raw().len() as u32;
    let header = TexHeader {
        magic: *TEX_MAGIC,
        version: TEX_VERSION,
        width,
        height,
        format: TexFormat::Rgba8Unorm as u32,
        mip_count: 1,
        data_size,
        _pad: 0,
    };
    
    w.write_all(bytemuck::bytes_of(&header))?;
    w.write_all(rgba8.as_raw())?;
    w.flush()?;
    
    Ok(())
}
