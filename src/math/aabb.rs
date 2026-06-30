//! Axis-Aligned Bounding Box for spatial queries.

use crate::math::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    #[inline]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Extends this AABB to include the given point.
    pub fn extend(&mut self, point: Vec3) {
        self.min.x = self.min.x.min(point.x);
        self.min.y = self.min.y.min(point.y);
        self.min.z = self.min.z.min(point.z);

        self.max.x = self.max.x.max(point.x);
        self.max.y = self.max.y.max(point.y);
        self.max.z = self.max.z.max(point.z);
    }

    /// Merges another AABB into this one.
    pub fn merge(&mut self, other: &AABB) {
        self.extend(other.min);
        self.extend(other.max);
    }

    /// Returns true if this AABB intersects with another.
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Returns true if this AABB completely contains the other.
    pub fn contains(&self, other: &AABB) -> bool {
        self.min.x <= other.min.x
            && self.max.x >= other.max.x
            && self.min.y <= other.min.y
            && self.max.y >= other.max.y
            && self.min.z <= other.min.z
            && self.max.z >= other.max.z
    }

    /// Center point of the AABB.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half-extents of the AABB.
    pub fn extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }
}
