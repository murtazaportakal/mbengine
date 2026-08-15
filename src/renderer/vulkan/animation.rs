
use bytemuck::{Pod, Zeroable};
use crate::renderer::vulkan::skeleton::MAX_BONES;

/// A GPU-friendly bone definition.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, Pod, Zeroable)]
pub struct GpuBone {
    pub inverse_bind_matrix: [f32; 16],
    pub parent_index: i32, // -1 if no parent
    pub _pad: [u32; 3],
}

/// A GPU-friendly skeleton definition (array of bones).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuSkeleton {
    pub bones: [GpuBone; MAX_BONES],
    pub bone_count: u32,
    pub _pad: [u32; 3],
}

impl Default for GpuSkeleton {
    fn default() -> Self {
        Self {
            bones: [GpuBone::default(); MAX_BONES],
            bone_count: 0,
            _pad: [0; 3],
        }
    }
}

/// A single translation, rotation, or scale keyframe.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, Pod, Zeroable)]
pub struct GpuKeyframe {
    pub time: f32,
    pub value: [f32; 4], // For translation/scale, w is unused. For rotation, it's a quaternion.
}

/// GPU Clip definition. We use flat indexing into a global keyframe buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuClip {
    pub duration: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,

    /// Channel indices/counts for up to MAX_BONES.
    /// x: offset into global keyframe buffer (Translation)
    /// y: count of translation keyframes
    /// z: offset into global keyframe buffer (Rotation)
    /// w: count of rotation keyframes
    pub channels: [[u32; 4]; MAX_BONES], 
    
    /// Scale channels
    /// x: offset into global keyframe buffer
    /// y: count of scale keyframes
    pub scale_channels: [[u32; 2]; MAX_BONES],
}

impl Default for GpuClip {
    fn default() -> Self {
        Self {
            duration: 0.0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            channels: [[0; 4]; MAX_BONES],
            scale_channels: [[0; 2]; MAX_BONES],
        }
    }
}

/// The state of an entity's animation for the compute shader to evaluate.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, Pod, Zeroable)]
pub struct InstanceAnimData {
    pub skeleton_index: u32,
    pub state_type: u32, // 0 = Clip, 1 = Blend1D, 2 = Blend2D

    // Primary State (or target state if crossfading)
    pub clip_a: u32,
    pub clip_b: u32,
    pub clip_c: u32,
    pub clip_d: u32,
    pub weights_ab: [f32; 2], // (weight_a, weight_b)
    pub weights_cd: [f32; 2], // (weight_c, weight_d)
    pub current_time: f32,

    // Previous state (if crossfading)
    pub prev_state_type: u32,
    pub prev_clip_a: u32,
    pub prev_clip_b: u32,
    pub prev_clip_c: u32,
    pub prev_clip_d: u32,
    pub prev_weights_ab: [f32; 2],
    pub prev_weights_cd: [f32; 2],
    pub prev_time: f32,

    pub crossfade_weight: f32, // 0.0 = primary only, >0.0 = blend between prev and primary
    
    pub _pad: [u32; 2],
}
