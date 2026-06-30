//! Vector types optimized for game development.

use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ── Vec2 ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self::new(0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 1.0);
    pub const UP: Self = Self::new(0.0, 1.0);
    pub const RIGHT: Self = Self::new(1.0, 0.0);

    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 0.0 {
            self / len
        } else {
            Self::ZERO
        }
    }
}

// ── Vec3 ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 1.0, 1.0);
    pub const UP: Self = Self::new(0.0, 1.0, 0.0);
    pub const RIGHT: Self = Self::new(1.0, 0.0, 0.0);
    pub const FORWARD: Self = Self::new(0.0, 0.0, 1.0);

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 0.0 {
            self / len
        } else {
            Self::ZERO
        }
    }
}

// ── Vec4 ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[repr(C, align(16))]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 1.0, 1.0, 1.0);

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 0.0 {
            self / len
        } else {
            Self::ZERO
        }
    }
}

impl From<Vec3> for Vec4 {
    #[inline]
    fn from(v: Vec3) -> Self {
        Self::new(v.x, v.y, v.z, 0.0)
    }
}

// ── Operator Overloads ──────────────────────────────────────────────────────

// Expand for Vec2
impl Add for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            x: self.x + o.x,
            y: self.y + o.y,
        }
    }
}
impl AddAssign for Vec2 {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl Sub for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            x: self.x - o.x,
            y: self.y - o.y,
        }
    }
}
impl SubAssign for Vec2 {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}
impl Mul for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self {
            x: self.x * o.x,
            y: self.y * o.y,
        }
    }
}
impl MulAssign for Vec2 {
    #[inline]
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}
impl Div for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        Self {
            x: self.x / o.x,
            y: self.y / o.y,
        }
    }
}
impl DivAssign for Vec2 {
    #[inline]
    fn div_assign(&mut self, o: Self) {
        *self = *self / o;
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }
}
impl MulAssign<f32> for Vec2 {
    #[inline]
    fn mul_assign(&mut self, s: f32) {
        *self = *self * s;
    }
}
impl Div<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, s: f32) -> Self {
        Self {
            x: self.x / s,
            y: self.y / s,
        }
    }
}
impl DivAssign<f32> for Vec2 {
    #[inline]
    fn div_assign(&mut self, s: f32) {
        *self = *self / s;
    }
}
impl Neg for Vec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}
impl Mul<Vec2> for f32 {
    type Output = Vec2;
    #[inline]
    fn mul(self, v: Vec2) -> Vec2 {
        v * self
    }
}

// Expand for Vec3
impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
        }
    }
}
impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            x: self.x - o.x,
            y: self.y - o.y,
            z: self.z - o.z,
        }
    }
}
impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}
impl Mul for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self {
            x: self.x * o.x,
            y: self.y * o.y,
            z: self.z * o.z,
        }
    }
}
impl MulAssign for Vec3 {
    #[inline]
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}
impl Div for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        Self {
            x: self.x / o.x,
            y: self.y / o.y,
            z: self.z / o.z,
        }
    }
}
impl DivAssign for Vec3 {
    #[inline]
    fn div_assign(&mut self, o: Self) {
        *self = *self / o;
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}
impl MulAssign<f32> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, s: f32) {
        *self = *self * s;
    }
}
impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, s: f32) -> Self {
        Self {
            x: self.x / s,
            y: self.y / s,
            z: self.z / s,
        }
    }
}
impl DivAssign<f32> for Vec3 {
    #[inline]
    fn div_assign(&mut self, s: f32) {
        *self = *self / s;
    }
}
impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

// Expand for Vec4
impl Add for Vec4 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
            w: self.w + o.w,
        }
    }
}
impl AddAssign for Vec4 {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl Sub for Vec4 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            x: self.x - o.x,
            y: self.y - o.y,
            z: self.z - o.z,
            w: self.w - o.w,
        }
    }
}
impl SubAssign for Vec4 {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}
impl Mul for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self {
            x: self.x * o.x,
            y: self.y * o.y,
            z: self.z * o.z,
            w: self.w * o.w,
        }
    }
}
impl MulAssign for Vec4 {
    #[inline]
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}
impl Div for Vec4 {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        Self {
            x: self.x / o.x,
            y: self.y / o.y,
            z: self.z / o.z,
            w: self.w / o.w,
        }
    }
}
impl DivAssign for Vec4 {
    #[inline]
    fn div_assign(&mut self, o: Self) {
        *self = *self / o;
    }
}

impl Mul<f32> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
            w: self.w * s,
        }
    }
}
impl MulAssign<f32> for Vec4 {
    #[inline]
    fn mul_assign(&mut self, s: f32) {
        *self = *self * s;
    }
}
impl Div<f32> for Vec4 {
    type Output = Self;
    #[inline]
    fn div(self, s: f32) -> Self {
        Self {
            x: self.x / s,
            y: self.y / s,
            z: self.z / s,
            w: self.w / s,
        }
    }
}
impl DivAssign<f32> for Vec4 {
    #[inline]
    fn div_assign(&mut self, s: f32) {
        *self = *self / s;
    }
}
impl Neg for Vec4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}
impl Mul<Vec4> for f32 {
    type Output = Vec4;
    #[inline]
    fn mul(self, v: Vec4) -> Vec4 {
        v * self
    }
}
