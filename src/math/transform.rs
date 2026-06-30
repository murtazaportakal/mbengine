//! Spatial transform for entities.

use crate::math::{Mat4, Quat, Vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform {
    #[inline]
    pub const fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    /// Computes the local model matrix for this transform.
    pub fn to_mat4(&self) -> Mat4 {
        let t = Mat4::translation(self.position);
        let r = self.rotation.to_mat4();
        let s = Mat4::scale(self.scale);

        // TRS order: Translation * Rotation * Scale
        t * r * s
    }
}
