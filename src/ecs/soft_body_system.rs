use crate::ecs::World;
use crate::physics::PhysicsSystem;
use crate::renderer::vulkan::compute_cloth::{ComputeClothPipeline, SphereCollider};

use crate::renderer::vulkan::VulkanDevice;

pub struct SoftBodySystem;

impl SoftBodySystem {
    pub fn update(
        _world: &mut World,
        physics: &PhysicsSystem,
        cloth_pipeline: &mut ComputeClothPipeline,
        vulkan: &VulkanDevice,
    ) {
        let mut colliders = Vec::new();

        // Extract all sphere colliders from physics system
        for (_, collider) in physics.collider_set.iter() {
            if let Some(sphere) = collider.shape().as_ball() {
                let translation = collider.position().translation.vector;
                colliders.push(SphereCollider {
                    pos: [translation.x, translation.y, translation.z],
                    radius: sphere.radius,
                });
            }
        }

        // Also add custom scene colliders if we want (e.g. falling cube could be approximated as a sphere for cloth).
        // Since we only handle spheres for now, we'll just approximate boxes as spheres based on their bounds.
        for (_, collider) in physics.collider_set.iter() {
            if let Some(cuboid) = collider.shape().as_cuboid() {
                let translation = collider.position().translation.vector;
                // Approximate with a sphere that encloses it
                let radius = cuboid.half_extents.max();
                colliders.push(SphereCollider {
                    pos: [translation.x, translation.y, translation.z],
                    radius,
                });
            }
        }

        // Limit to 64 colliders as per our buffer size
        if !colliders.is_empty() {
            cloth_pipeline.colliders_buffer.upload(vulkan, &colliders);
        } else {
            // Upload an empty array to clear previous frame's colliders
            cloth_pipeline.colliders_buffer.upload(
                vulkan,
                &[SphereCollider {
                    pos: [0.0; 3],
                    radius: 0.0,
                }],
            );
        }
    }
}
