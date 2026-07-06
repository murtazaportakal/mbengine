//! GLTF/GLB mesh, skeleton, and animation loader.
//!
//! Parses glTF 2.0 files via the `gltf` crate and produces:
//!   - `Vec<(Vec<Vertex>, Vec<u32>)>` — mesh primitives (vertices + indices)
//!   - `Option<Skeleton>` — bone hierarchy with inverse bind matrices
//!   - `Vec<AnimationClip>` — sampled animation clips
//!
//! Supports JOINTS_0 and WEIGHTS_0 attributes for skinned meshes.

use crate::math::mat4::Mat4;
use crate::math::vec::{Vec3, Vec4};
use crate::renderer::vulkan::pipeline::Vertex;
use crate::renderer::vulkan::skeleton::{AnimationClip, Bone, BoneChannel, Skeleton};

/// Result of loading a GLTF file.
pub struct GltfData {
    /// One entry per mesh primitive: (vertices, indices).
    pub primitives: Vec<(Vec<Vertex>, Vec<u32>)>,
    /// Skeleton extracted from the first skin, if any.
    pub skeleton: Option<Skeleton>,
    /// All animation clips in the file.
    pub clips: Vec<AnimationClip>,
}

/// Load a GLTF or GLB file from disk.
///
/// Returns `None` if the file cannot be parsed or contains no mesh data.
pub fn load_gltf(path: &str) -> Option<GltfData> {
    let (document, buffers, _images) = gltf::import(path).ok()?;

    // ── Extract Skeleton ────────────────────────────────────────────────
    let (skeleton, joint_node_to_bone_index) = extract_skeleton(&document, &buffers);

    // ── Extract Meshes ──────────────────────────────────────────────────
    let primitives = extract_meshes(&document, &buffers);

    // ── Extract Animations ──────────────────────────────────────────────
    let clips = extract_animations(&document, &buffers, &joint_node_to_bone_index);

    if primitives.is_empty() {
        return None;
    }

    Some(GltfData {
        primitives,
        skeleton,
        clips,
    })
}

/// Extract all mesh primitives from the GLTF document.
fn extract_meshes(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Vec<(Vec<Vertex>, Vec<u32>)> {
    let mut primitives = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            // Positions (required)
            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(iter) => iter.collect(),
                None => continue,
            };

            // Normals
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

            // UVs
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            // Joint IDs (JOINTS_0) — up to 4 bone influences per vertex
            let joints: Vec<[u32; 4]> = reader
                .read_joints(0)
                .map(|iter| {
                    iter.into_u16()
                        .map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32])
                        .collect()
                })
                .unwrap_or_else(|| vec![[0u32; 4]; positions.len()]);

            // Joint Weights (WEIGHTS_0)
            let weights: Vec<[f32; 4]> = reader
                .read_weights(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0f32; 4]; positions.len()]);

            // Indices
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|iter| iter.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());

            // Assemble vertices
            let vertices: Vec<Vertex> = (0..positions.len())
                .map(|i| Vertex {
                    pos: positions[i],
                    normal: normals[i],
                    uv: uvs[i],
                    joint_ids: joints[i],
                    joint_weights: weights[i],
                })
                .collect();

            primitives.push((vertices, indices));
        }
    }

    primitives
}

/// Extract the skeleton (first skin) from the GLTF document.
///
/// Returns the skeleton and a mapping from GLTF node index to bone index.
fn extract_skeleton(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> (Option<Skeleton>, std::collections::HashMap<usize, usize>) {
    let mut joint_node_to_bone_index = std::collections::HashMap::new();

    let skin = match document.skins().next() {
        Some(s) => s,
        None => return (None, joint_node_to_bone_index),
    };

    let joints: Vec<gltf::Node<'_>> = skin.joints().collect();

    // Read inverse bind matrices
    let inverse_bind_matrices: Vec<Mat4> = match skin
        .reader(|buffer| Some(&buffers[buffer.index()]))
        .read_inverse_bind_matrices()
    {
        Some(iter) => iter.map(|m| gltf_mat4_to_mat4(&m)).collect(),
        None => vec![Mat4::identity(); joints.len()],
    };

    // Build a map from GLTF node index → bone index for parent lookups
    for (bone_idx, joint) in joints.iter().enumerate() {
        joint_node_to_bone_index.insert(joint.index(), bone_idx);
    }

    // Build bones
    let bones: Vec<Bone> = joints
        .iter()
        .enumerate()
        .map(|(bone_idx, joint)| {
            // Find parent: walk up the GLTF node tree until we find a joint
            let parent = find_parent_bone(joint, document, &joint_node_to_bone_index);

            Bone {
                index: bone_idx,
                name: joint.name().unwrap_or("unnamed_bone").to_string(),
                inverse_bind_matrix: inverse_bind_matrices[bone_idx],
                parent,
            }
        })
        .collect();

    let skeleton = Skeleton::new(bones);
    (Some(skeleton), joint_node_to_bone_index)
}

/// Find the parent bone index for a given joint node.
/// Walks up the GLTF node graph looking for a node that is also a joint.
fn find_parent_bone(
    joint: &gltf::Node<'_>,
    document: &gltf::Document,
    joint_map: &std::collections::HashMap<usize, usize>,
) -> Option<usize> {
    // GLTF doesn't expose parent directly, so we build a parent map from scenes.
    let parent_map = build_parent_map(document);
    let mut current = joint.index();

    loop {
        match parent_map.get(&current) {
            Some(&parent_node_idx) => {
                if let Some(&bone_idx) = joint_map.get(&parent_node_idx) {
                    return Some(bone_idx);
                }
                current = parent_node_idx;
            }
            None => return None,
        }
    }
}

/// Build a child→parent node index map from the GLTF scene graph.
fn build_parent_map(document: &gltf::Document) -> std::collections::HashMap<usize, usize> {
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

    parent_map
}

/// Extract all animation clips from the GLTF document.
fn extract_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    joint_map: &std::collections::HashMap<usize, usize>,
) -> Vec<AnimationClip> {
    let mut clips = Vec::new();

    for animation in document.animations() {
        let mut channels = Vec::new();
        let mut duration: f32 = 0.0;

        // Group channels by target node
        let mut channel_map: std::collections::HashMap<usize, BoneChannel> =
            std::collections::HashMap::new();

        for channel in animation.channels() {
            let target_node = channel.target().node().index();

            // Only process channels that target a bone
            let bone_index = match joint_map.get(&target_node) {
                Some(&idx) => idx,
                None => continue,
            };

            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
            let timestamps: Vec<f32> = match reader.read_inputs() {
                Some(iter) => iter.collect(),
                None => continue,
            };

            // Track max time for duration
            if let Some(&max_t) = timestamps.last() {
                if max_t > duration {
                    duration = max_t;
                }
            }

            let entry = channel_map
                .entry(bone_index)
                .or_insert_with(|| BoneChannel {
                    bone_index,
                    translation_keys: Vec::new(),
                    rotation_keys: Vec::new(),
                    scale_keys: Vec::new(),
                });

            match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::Translations(iter)) => {
                    for (t, val) in timestamps.iter().zip(iter) {
                        entry
                            .translation_keys
                            .push((*t, Vec3::new(val[0], val[1], val[2])));
                    }
                }
                Some(gltf::animation::util::ReadOutputs::Rotations(iter)) => {
                    for (t, val) in timestamps.iter().zip(iter.into_f32()) {
                        // GLTF stores quaternions as [x, y, z, w]
                        entry.rotation_keys.push((*t, val));
                    }
                }
                Some(gltf::animation::util::ReadOutputs::Scales(iter)) => {
                    for (t, val) in timestamps.iter().zip(iter) {
                        entry
                            .scale_keys
                            .push((*t, Vec3::new(val[0], val[1], val[2])));
                    }
                }
                _ => {}
            }
        }

        for (_, bone_channel) in channel_map {
            channels.push(bone_channel);
        }

        let clip_name = animation.name().unwrap_or("unnamed_animation").to_string();

        clips.push(AnimationClip {
            name: clip_name,
            duration,
            channels,
        });
    }

    clips
}

/// Convert a GLTF row-major [[f32; 4]; 4] matrix to our column-major Mat4.
fn gltf_mat4_to_mat4(m: &[[f32; 4]; 4]) -> Mat4 {
    // GLTF stores matrices in column-major order as flat arrays,
    // and the gltf crate returns them as [[f32;4];4] in column-major layout.
    Mat4::new(
        Vec4::new(m[0][0], m[0][1], m[0][2], m[0][3]),
        Vec4::new(m[1][0], m[1][1], m[1][2], m[1][3]),
        Vec4::new(m[2][0], m[2][1], m[2][2], m[2][3]),
        Vec4::new(m[3][0], m[3][1], m[3][2], m[3][3]),
    )
}
