//! Integration tests for Math Library.

use engine::math::{Vec2, Vec3, Vec4, Mat4, Quat, Transform, AABB};

#[test]
fn test_vec_alignment() {
    // Ensure Vec4 is 16-byte aligned for SIMD Auto-vectorization
    assert_eq!(std::mem::align_of::<Vec4>(), 16);
    assert_eq!(std::mem::size_of::<Vec4>(), 16);
    
    // Mat4 should also be 16-byte aligned, containing 4 Vec4s
    assert_eq!(std::mem::align_of::<Mat4>(), 16);
    assert_eq!(std::mem::size_of::<Mat4>(), 64);
}

#[test]
fn test_vec_math() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);
    
    let c = a + b;
    assert_eq!(c, Vec3::new(5.0, 7.0, 9.0));
    
    let d = a * 2.0;
    assert_eq!(d, Vec3::new(2.0, 4.0, 6.0));
    
    let dot = a.dot(b);
    assert_eq!(dot, 1.0*4.0 + 2.0*5.0 + 3.0*6.0);
    
    let cross = Vec3::RIGHT.cross(Vec3::UP);
    assert_eq!(cross, Vec3::FORWARD);
}

#[test]
fn test_matrix_multiplication() {
    let t = Mat4::translation(Vec3::new(10.0, 20.0, 30.0));
    let s = Mat4::scale(Vec3::new(2.0, 2.0, 2.0));
    
    let m = t * s;
    
    // Transform a point
    let p = Vec4::new(1.0, 1.0, 1.0, 1.0);
    let p_transformed = m * p;
    
    // Expected: First scale (1*2=2), then translate (+10 = 12)
    assert_eq!(p_transformed.x, 12.0);
    assert_eq!(p_transformed.y, 22.0);
    assert_eq!(p_transformed.z, 32.0);
    assert_eq!(p_transformed.w, 1.0);
}

#[test]
fn test_quaternion() {
    let q = Quat::from_axis_angle(Vec3::UP, std::f32::consts::PI / 2.0);
    let m = q.to_mat4();
    
    // Rotating (1, 0, 0) by 90 degrees around Y should yield (0, 0, -1) in left-handed or right-handed depending on formulation.
    // Our math uses standard right-handed. X rotates to -Z.
    let p = Vec4::new(1.0, 0.0, 0.0, 1.0);
    let p_rot = m * p;
    
    // Floating point math might be slightly off zero
    assert!(p_rot.x.abs() < 0.0001);
    assert!(p_rot.y.abs() < 0.0001);
    assert!((p_rot.z - (-1.0)).abs() < 0.0001);
}

#[test]
fn test_transform() {
    let mut t = Transform::new();
    t.position = Vec3::new(0.0, 5.0, 0.0);
    t.scale = Vec3::new(2.0, 2.0, 2.0);
    
    let m = t.to_mat4();
    let p = Vec4::new(0.0, 1.0, 0.0, 1.0);
    let p_out = m * p;
    
    assert_eq!(p_out.x, 0.0);
    assert_eq!(p_out.y, 7.0); // 1.0 * 2.0 (scale) + 5.0 (trans)
    assert_eq!(p_out.z, 0.0);
}

#[test]
fn test_aabb() {
    let mut a = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 10.0, 10.0));
    let b = AABB::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(15.0, 15.0, 15.0));
    let c = AABB::new(Vec3::new(20.0, 20.0, 20.0), Vec3::new(30.0, 30.0, 30.0));
    
    assert!(a.intersects(&b));
    assert!(!a.intersects(&c));
    
    a.merge(&c);
    assert!(a.contains(&b));
    assert!(a.contains(&c));
    assert_eq!(a.max, Vec3::new(30.0, 30.0, 30.0));
}
