use crate::ecs::components::{RigidBodyComponent, TransformComponent};
use crate::ecs::system::System;
use crate::ecs::types::ComponentMask;
use crate::ecs::world::World;
use crate::math::vec::Vec3;

use rapier3d::prelude::*;

pub struct PhysicsSystem {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub gravity: Vector<Real>,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: BroadPhase,
    pub narrow_phase: NarrowPhase,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    pub ccd_solver: CCDSolver,
    pub query_pipeline: QueryPipeline,
    required_components: ComponentMask,
}

impl Default for PhysicsSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsSystem {
    pub fn new() -> Self {
        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            gravity: vector![0.0, -9.81, 0.0],
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            required_components: 0,
        }
    }
}

impl System for PhysicsSystem {
    fn update(&mut self, dt: f32, world: &mut World) {
        // Step the simulation
        self.integration_parameters.dt = dt.max(0.001); // Prevent 0 dt

        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );

        // Synchronize state from Rapier to ECS Transforms
        let mut updates = Vec::new();
        {
            let rb_components = world.get_component_array::<RigidBodyComponent>();
            let entities = rb_components.dense_entities_slice();
            for (i, rb_comp) in rb_components.as_slice().iter().enumerate() {
                updates.push((entities[i], rb_comp.handle));
            }
        }

        let transforms = world.get_component_array_mut::<TransformComponent>();
        for (entity, handle) in updates {
            if transforms.has(entity) {
                if let Some(rb) = self.rigid_body_set.get(handle) {
                    let translation = rb.translation();
                    let transform = unsafe { transforms.get_mut(entity) };
                    transform.position = Vec3::new(translation.x, translation.y, translation.z);

                    // Convert rotation (Quaternion -> Euler)
                    let rot = rb.rotation().euler_angles();
                    transform.rotation = Vec3::new(rot.0, rot.1, rot.2);
                }
            }
        }
    }

    fn required_components(&self) -> ComponentMask {
        self.required_components
    }

    fn set_required_components(&mut self, mask: ComponentMask) {
        self.required_components = mask;
    }
}
