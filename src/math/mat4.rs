//! Column-major 4x4 matrix.

use crate::math::{Vec3, Vec4};
use std::ops::{Mul, MulAssign};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[repr(C, align(16))]
pub struct Mat4 {
    /// Columns of the matrix.
    pub cols: [Vec4; 4],
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        cols: [
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        ],
    };

    pub const ZERO: Self = Self {
        cols: [Vec4::ZERO; 4],
    };

    #[inline]
    pub const fn new(c0: Vec4, c1: Vec4, c2: Vec4, c3: Vec4) -> Self {
        Self { cols: [c0, c1, c2, c3] }
    }

    #[inline(always)]
    pub fn identity() -> Self {
        Self {
            cols: [
                Vec4::new(1.0, 0.0, 0.0, 0.0),
                Vec4::new(0.0, 1.0, 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            ],
        }
    }

    /// Creates a translation matrix.
    #[inline]
    pub fn translation(v: Vec3) -> Self {
        Self {
            cols: [
                Vec4::new(1.0, 0.0, 0.0, 0.0),
                Vec4::new(0.0, 1.0, 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0, 0.0),
                Vec4::new(v.x, v.y, v.z, 1.0),
            ],
        }
    }

    /// Creates a scale matrix.
    #[inline]
    pub fn scale(v: Vec3) -> Self {
        Self {
            cols: [
                Vec4::new(v.x, 0.0, 0.0, 0.0),
                Vec4::new(0.0, v.y, 0.0, 0.0),
                Vec4::new(0.0, 0.0, v.z, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            ],
        }
    }

    /// Creates a look-at view matrix.
    pub fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        Self {
            cols: [
                Vec4::new(s.x, u.x, -f.x, 0.0),
                Vec4::new(s.y, u.y, -f.y, 0.0),
                Vec4::new(s.z, u.z, -f.z, 0.0),
                Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
            ],
        }
    }

    /// Creates a perspective projection matrix (Vulkan clip space conventions).
    /// y points down, depth is [0, 1].
    pub fn perspective(fov_y_radians: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        let f = 1.0 / (fov_y_radians / 2.0).tan();
        
        let mut result = Self::ZERO;
        result.cols[0].x = f / aspect_ratio;
        result.cols[1].y = -f; // Invert Y for Vulkan
        result.cols[2].z = z_far / (z_near - z_far);
        result.cols[2].w = -1.0;
        result.cols[3].z = -(z_far * z_near) / (z_far - z_near);
        
        result
    }

    /// Computes the inverse of this matrix using nalgebra.
    pub fn try_inverse(&self) -> Option<Self> {
        let mut na_mat = nalgebra::Matrix4::zeros();
        for c in 0..4 {
            na_mat[(0, c)] = self.cols[c].x;
            na_mat[(1, c)] = self.cols[c].y;
            na_mat[(2, c)] = self.cols[c].z;
            na_mat[(3, c)] = self.cols[c].w;
        }

        if let Some(inv) = na_mat.try_inverse() {
            let mut result = Self::ZERO;
            for c in 0..4 {
                result.cols[c].x = inv[(0, c)];
                result.cols[c].y = inv[(1, c)];
                result.cols[c].z = inv[(2, c)];
                result.cols[c].w = inv[(3, c)];
            }
            Some(result)
        } else {
            None
        }
    }
}

impl Mul<Mat4> for Mat4 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Mat4) -> Self::Output {
        let mut result = Mat4::ZERO;

        for c in 0..4 {
            for r in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    // self.cols[k] gives the k-th column, so .x/.y/.z/.w gives the r-th row element.
                    let self_elem = match r {
                        0 => self.cols[k].x,
                        1 => self.cols[k].y,
                        2 => self.cols[k].z,
                        3 => self.cols[k].w,
                        _ => unreachable!(),
                    };
                    
                    let rhs_elem = match k {
                        0 => rhs.cols[c].x,
                        1 => rhs.cols[c].y,
                        2 => rhs.cols[c].z,
                        3 => rhs.cols[c].w,
                        _ => unreachable!(),
                    };

                    sum += self_elem * rhs_elem;
                }

                match r {
                    0 => result.cols[c].x = sum,
                    1 => result.cols[c].y = sum,
                    2 => result.cols[c].z = sum,
                    3 => result.cols[c].w = sum,
                    _ => unreachable!(),
                }
            }
        }

        result
    }
}

impl MulAssign<Mat4> for Mat4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Mat4) {
        *self = *self * rhs;
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;

    #[inline]
    fn mul(self, rhs: Vec4) -> Self::Output {
        Vec4 {
            x: self.cols[0].x * rhs.x + self.cols[1].x * rhs.y + self.cols[2].x * rhs.z + self.cols[3].x * rhs.w,
            y: self.cols[0].y * rhs.x + self.cols[1].y * rhs.y + self.cols[2].y * rhs.z + self.cols[3].y * rhs.w,
            z: self.cols[0].z * rhs.x + self.cols[1].z * rhs.y + self.cols[2].z * rhs.z + self.cols[3].z * rhs.w,
            w: self.cols[0].w * rhs.x + self.cols[1].w * rhs.y + self.cols[2].w * rhs.z + self.cols[3].w * rhs.w,
        }
    }
}
