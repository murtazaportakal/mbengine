//! Skeletal animation data structures.
//!
//! Contains:
//!   - `Bone`: Single bone with inverse bind matrix and parent linkage
//!   - `Skeleton`: Flat array of bones forming a skeletal hierarchy
//!   - `AnimationClip`: Sampled keyframes per bone over time
//!   - `AnimationSampler`: Interpolation utilities for sampling clips

use crate::math::mat4::Mat4;
use crate::math::vec::Vec3;

/// Maximum number of bones per skeleton.
/// Matches the SSBO layout in the skinning compute shader.
pub const MAX_BONES: usize = 128;

// ── Bone ────────────────────────────────────────────────────────────────────

/// A single bone in a skeleton hierarchy.
#[derive(Clone, Debug)]
pub struct Bone {
    /// Index of this bone in the skeleton's flat array.
    pub index: usize,
    /// Human-readable name (e.g., "LeftArm", "Spine").
    pub name: String,
    /// Transforms from mesh space to bone-local space (bind pose).
    pub inverse_bind_matrix: Mat4,
    /// Index of the parent bone, or `None` for root bones.
    pub parent: Option<usize>,
}

// ── Skeleton ────────────────────────────────────────────────────────────────

/// A complete skeleton: a flat array of bones with hierarchy encoded via parent indices.
#[derive(Clone, Debug)]
pub struct Skeleton {
    /// Flat array of bones. Index == `Bone::index`.
    pub bones: Vec<Bone>,
    /// Map bone name → bone index for quick lookup.
    pub name_to_index: std::collections::HashMap<String, usize>,
}

impl Skeleton {
    /// Create a new skeleton from a list of bones.
    pub fn new(bones: Vec<Bone>) -> Self {
        let name_to_index = bones.iter().map(|b| (b.name.clone(), b.index)).collect();
        Self {
            bones,
            name_to_index,
        }
    }

    /// Number of bones in this skeleton.
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Look up a bone index by name.
    #[inline]
    pub fn find_bone(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    /// Compute the final bone matrices (model-space) given a set of local-space
    /// transforms (one per bone). Returns the array suitable for GPU upload:
    ///   `final[i] = global_transform[i] * inverse_bind_matrix[i]`
    ///
    /// `local_transforms` must have exactly `bone_count()` entries.
    pub fn compute_bone_matrices(
        &self,
        local_transforms: &[Mat4],
        out_matrices: &mut crate::containers::FixedArray<Mat4, MAX_BONES>,
    ) {
        debug_assert_eq!(local_transforms.len(), self.bones.len());

        let count = self.bones.len();
        let mut global_transforms = [Mat4::identity(); MAX_BONES];

        // Forward pass: bones are stored in topological order (parents before children).
        for i in 0..count {
            let local = &local_transforms[i];
            global_transforms[i] = match self.bones[i].parent {
                Some(parent_idx) => global_transforms[parent_idx] * *local,
                None => *local,
            };
        }

        // Final: global * inverse_bind
        out_matrices.clear();
        for (global, bone) in global_transforms.iter().take(count).zip(self.bones.iter()) {
            out_matrices.push(*global * bone.inverse_bind_matrix);
        }
    }
}

// ── Keyframe ────────────────────────────────────────────────────────────────

/// A single keyframe for one bone channel.
#[derive(Clone, Debug)]
pub struct Keyframe {
    pub time: f32,
    pub translation: Vec3,
    pub rotation: [f32; 4], // Quaternion (x, y, z, w)
    pub scale: Vec3,
}

impl Default for Keyframe {
    fn default() -> Self {
        Self {
            time: 0.0,
            translation: Vec3::new(0.0, 0.0, 0.0),
            rotation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
            scale: Vec3::new(1.0, 1.0, 1.0),
        }
    }
}

// ── BoneChannel ─────────────────────────────────────────────────────────────

/// All keyframes for a single bone within an animation clip.
#[derive(Clone, Debug)]
pub struct BoneChannel {
    pub bone_index: usize,
    pub translation_keys: Vec<(f32, Vec3)>,
    pub rotation_keys: Vec<(f32, [f32; 4])>,
    pub scale_keys: Vec<(f32, Vec3)>,
}

// ── TransformTRS & SkeletonPose ─────────────────────────────────────────────

/// Represents a raw decomposed transform (Translation, Rotation, Scale).
#[derive(Clone, Copy, Debug)]
pub struct TransformTRS {
    pub translation: Vec3,
    pub rotation: [f32; 4], // Quaternion (x, y, z, w)
    pub scale: Vec3,
}

impl Default for TransformTRS {
    fn default() -> Self {
        Self {
            translation: Vec3::new(0.0, 0.0, 0.0),
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: Vec3::new(1.0, 1.0, 1.0),
        }
    }
}

impl TransformTRS {
    /// Blend two TRS transforms mathematically.
    pub fn blend(a: &Self, b: &Self, weight: f32) -> Self {
        Self {
            translation: Vec3::new(
                a.translation.x + (b.translation.x - a.translation.x) * weight,
                a.translation.y + (b.translation.y - a.translation.y) * weight,
                a.translation.z + (b.translation.z - a.translation.z) * weight,
            ),
            rotation: AnimationClip::slerp(&a.rotation, &b.rotation, weight),
            scale: Vec3::new(
                a.scale.x + (b.scale.x - a.scale.x) * weight,
                a.scale.y + (b.scale.y - a.scale.y) * weight,
                a.scale.z + (b.scale.z - a.scale.z) * weight,
            ),
        }
    }

    /// Convert to a Mat4.
    pub fn to_matrix(&self) -> Mat4 {
        AnimationClip::compose_trs(&self.translation, &self.rotation, &self.scale)
    }
}

/// A pose of the skeleton containing a TRS for each bone.
#[derive(Clone, Debug, Default)]
pub struct SkeletonPose {
    pub bones: crate::containers::FixedArray<TransformTRS, MAX_BONES>,
}

impl SkeletonPose {
    pub fn new(bone_count: usize) -> Self {
        let mut bones = crate::containers::FixedArray::new();
        for _ in 0..bone_count {
            bones.push(TransformTRS::default());
        }
        Self { bones }
    }

    pub fn blend(a: &Self, b: &Self, weight: f32, out_pose: &mut Self) {
        out_pose.bones.clear();
        for i in 0..a.bones.len() {
            if i < b.bones.len() {
                out_pose
                    .bones
                    .push(TransformTRS::blend(&a.bones[i], &b.bones[i], weight));
            } else {
                out_pose.bones.push(a.bones[i]); // Fallback
            }
        }
    }

    pub fn to_matrices(&self, out_transforms: &mut crate::containers::FixedArray<Mat4, MAX_BONES>) {
        out_transforms.clear();
        for trs in self.bones.as_slice() {
            out_transforms.push(trs.to_matrix());
        }
    }
}

// ── AnimationClip ───────────────────────────────────────────────────────────

/// A named animation clip containing keyframes for multiple bones.
#[derive(Clone, Debug)]
pub struct AnimationClip {
    pub name: String,
    /// Total duration in seconds.
    pub duration: f32,
    /// Per-bone animation channels.
    pub channels: Vec<BoneChannel>,
}

impl AnimationClip {
    /// Sample the clip at a given time `t` (in seconds), producing a local-space
    /// TRS pose for each bone. `bone_count` is the total number of bones
    /// in the skeleton; channels without data at this bone default to identity.
    pub fn sample_pose(&self, t: f32, bone_count: usize, out_pose: &mut SkeletonPose) {
        out_pose.bones.clear();
        for _ in 0..bone_count {
            out_pose.bones.push(TransformTRS::default());
        }

        for channel in &self.channels {
            let translation = Self::sample_vec3(&channel.translation_keys, t);
            let rotation = Self::sample_quat(&channel.rotation_keys, t);
            let scale = Self::sample_vec3(&channel.scale_keys, t);

            out_pose.bones[channel.bone_index] = TransformTRS {
                translation,
                rotation,
                scale,
            };
        }
    }

    /// Linearly interpolate a Vec3 channel at time `t`.
    fn sample_vec3(keys: &[(f32, Vec3)], t: f32) -> Vec3 {
        if keys.is_empty() {
            return Vec3::new(0.0, 0.0, 0.0);
        }
        if keys.len() == 1 || t <= keys[0].0 {
            return keys[0].1;
        }
        if t >= keys.last().unwrap().0 {
            return keys.last().unwrap().1;
        }

        // Find the two keyframes bracketing `t`.
        let mut i = 0;
        while i < keys.len() - 1 && keys[i + 1].0 < t {
            i += 1;
        }

        let (t0, v0) = &keys[i];
        let (t1, v1) = &keys[i + 1];
        let factor = (t - t0) / (t1 - t0);

        Vec3::new(
            v0.x + (v1.x - v0.x) * factor,
            v0.y + (v1.y - v0.y) * factor,
            v0.z + (v1.z - v0.z) * factor,
        )
    }

    /// Spherically interpolate a quaternion channel at time `t`.
    fn sample_quat(keys: &[(f32, [f32; 4])], t: f32) -> [f32; 4] {
        if keys.is_empty() {
            return [0.0, 0.0, 0.0, 1.0];
        }
        if keys.len() == 1 || t <= keys[0].0 {
            return keys[0].1;
        }
        if t >= keys.last().unwrap().0 {
            return keys.last().unwrap().1;
        }

        let mut i = 0;
        while i < keys.len() - 1 && keys[i + 1].0 < t {
            i += 1;
        }

        let (t0, q0) = &keys[i];
        let (t1, q1) = &keys[i + 1];
        let factor = (t - t0) / (t1 - t0);

        Self::slerp(q0, q1, factor)
    }

    /// Quaternion spherical linear interpolation.
    pub fn slerp(a: &[f32; 4], b: &[f32; 4], t: f32) -> [f32; 4] {
        let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];

        // If dot < 0, negate one to take the short path.
        let mut b_adj = *b;
        if dot < 0.0 {
            dot = -dot;
            b_adj = [-b[0], -b[1], -b[2], -b[3]];
        }

        // If very close, do linear interpolation to avoid division by zero.
        if dot > 0.9995 {
            let result = [
                a[0] + (b_adj[0] - a[0]) * t,
                a[1] + (b_adj[1] - a[1]) * t,
                a[2] + (b_adj[2] - a[2]) * t,
                a[3] + (b_adj[3] - a[3]) * t,
            ];
            return Self::normalize_quat(&result);
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let wa = ((1.0 - t) * theta).sin() / sin_theta;
        let wb = (t * theta).sin() / sin_theta;

        [
            a[0] * wa + b_adj[0] * wb,
            a[1] * wa + b_adj[1] * wb,
            a[2] * wa + b_adj[2] * wb,
            a[3] * wa + b_adj[3] * wb,
        ]
    }

    fn normalize_quat(q: &[f32; 4]) -> [f32; 4] {
        let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        if len < 1e-10 {
            return [0.0, 0.0, 0.0, 1.0];
        }
        let inv = 1.0 / len;
        [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
    }

    /// Compose a TRS (Translation * Rotation * Scale) matrix from components.
    pub fn compose_trs(translation: &Vec3, rotation: &[f32; 4], scale: &Vec3) -> Mat4 {
        // Quaternion to rotation matrix
        let (x, y, z, w) = (rotation[0], rotation[1], rotation[2], rotation[3]);

        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        // Column-major mat4: T * R * S
        let mut m = Mat4::identity();
        m.cols[0].x = (1.0 - (yy + zz)) * scale.x;
        m.cols[0].y = (xy + wz) * scale.x;
        m.cols[0].z = (xz - wy) * scale.x;

        m.cols[1].x = (xy - wz) * scale.y;
        m.cols[1].y = (1.0 - (xx + zz)) * scale.y;
        m.cols[1].z = (yz + wx) * scale.y;

        m.cols[2].x = (xz + wy) * scale.z;
        m.cols[2].y = (yz - wx) * scale.z;
        m.cols[2].z = (1.0 - (xx + yy)) * scale.z;

        m.cols[3].x = translation.x;
        m.cols[3].y = translation.y;
        m.cols[3].z = translation.z;

        m
    }
}
