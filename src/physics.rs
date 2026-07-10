use crate::ecs::components::{RigidBodyComponent, TransformComponent};
use crate::ecs::system::System;
use crate::ecs::types::ComponentMask;
use crate::ecs::world::World;
use crate::math::vec::Vec3;

use rapier3d::prelude::*;

pub struct RayCastResult {
    pub entity: crate::ecs::EntityId,
    pub hit_point: crate::math::vec::Vec3,
    pub normal: crate::math::vec::Vec3,
    pub toi: f32,
}

pub struct SweepTestResult {
    pub entity: crate::ecs::EntityId,
    pub toi: f32,
    pub witness1: crate::math::vec::Vec3,
    pub witness2: crate::math::vec::Vec3,
    pub normal1: crate::math::vec::Vec3,
    pub normal2: crate::math::vec::Vec3,
}

#[derive(Clone, Copy)]
pub struct TriggerEvent {
    pub entity1: crate::ecs::EntityId,
    pub entity2: crate::ecs::EntityId,
    pub started: bool,
}

struct RawTriggerEvent {
    pub handle1: rapier3d::geometry::ColliderHandle,
    pub handle2: rapier3d::geometry::ColliderHandle,
    pub started: bool,
}

pub struct PhysicsEventCollector {
    raw_events: std::sync::Mutex<Vec<RawTriggerEvent>>,
}

impl Default for PhysicsEventCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsEventCollector {
    pub fn new() -> Self {
        Self {
            raw_events: std::sync::Mutex::new(Vec::with_capacity(128)),
        }
    }
}

impl EventHandler for PhysicsEventCollector {
    fn handle_collision_event(
        &self,
        _rigid_body_set: &RigidBodySet,
        _collider_set: &ColliderSet,
        event: CollisionEvent,
        _contact_pair: Option<&ContactPair>,
    ) {
        if let Ok(mut events) = self.raw_events.lock() {
            match event {
                CollisionEvent::Started(handle1, handle2, _) => {
                    events.push(RawTriggerEvent {
                        handle1,
                        handle2,
                        started: true,
                    });
                }
                CollisionEvent::Stopped(handle1, handle2, _) => {
                    events.push(RawTriggerEvent {
                        handle1,
                        handle2,
                        started: false,
                    });
                }
            }
        }
    }

    fn handle_contact_force_event(
        &self,
        _dt: f32,
        _bodies: &RigidBodySet,
        _colliders: &ColliderSet,
        _contact_pair: &ContactPair,
        _total_force_magnitude: f32,
    ) {
    }
}

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
    pub event_collector: PhysicsEventCollector,
    pub trigger_events: Vec<TriggerEvent>,
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
            event_collector: PhysicsEventCollector::new(),
            trigger_events: Vec::with_capacity(128),
        }
    }

    pub fn cast_ray(
        &self,
        origin: crate::math::vec::Vec3,
        dir: crate::math::vec::Vec3,
        max_toi: f32,
        solid: bool,
    ) -> Option<RayCastResult> {
        let ray = rapier3d::geometry::Ray::new(
            rapier3d::math::Point::new(origin.x, origin.y, origin.z),
            rapier3d::math::Vector::new(dir.x, dir.y, dir.z),
        );
        let query_filter = rapier3d::pipeline::QueryFilter::default();

        if let Some((handle, intersection)) = self.query_pipeline.cast_ray_and_get_normal(
            &self.rigid_body_set,
            &self.collider_set,
            &ray,
            max_toi,
            solid,
            query_filter,
        ) {
            let collider = self.collider_set.get(handle)?;
            let entity = collider.user_data as crate::ecs::EntityId;
            let hit_point = ray.point_at(intersection.toi);

            Some(RayCastResult {
                entity,
                hit_point: crate::math::vec::Vec3::new(hit_point.x, hit_point.y, hit_point.z),
                normal: crate::math::vec::Vec3::new(
                    intersection.normal.x,
                    intersection.normal.y,
                    intersection.normal.z,
                ),
                toi: intersection.toi,
            })
        } else {
            None
        }
    }

    pub fn cast_shape(
        &self,
        origin: crate::math::vec::Vec3,
        dir: crate::math::vec::Vec3,
        shape: &dyn rapier3d::geometry::Shape,
        max_toi: f32,
    ) -> Option<SweepTestResult> {
        let shape_pos = rapier3d::math::Isometry::new(
            rapier3d::math::Vector::new(origin.x, origin.y, origin.z),
            rapier3d::math::Vector::zeros(),
        );
        let shape_vel = rapier3d::math::Vector::new(dir.x, dir.y, dir.z);
        let query_filter = rapier3d::pipeline::QueryFilter::default();

        if let Some((handle, hit)) = self.query_pipeline.cast_shape(
            &self.rigid_body_set,
            &self.collider_set,
            &shape_pos,
            &shape_vel,
            shape,
            max_toi,
            true, // stop_at_penetration
            query_filter,
        ) {
            let collider = self.collider_set.get(handle)?;
            let entity = collider.user_data as crate::ecs::EntityId;

            Some(SweepTestResult {
                entity,
                toi: hit.toi,
                witness1: crate::math::vec::Vec3::new(
                    hit.witness1.x,
                    hit.witness1.y,
                    hit.witness1.z,
                ),
                witness2: crate::math::vec::Vec3::new(
                    hit.witness2.x,
                    hit.witness2.y,
                    hit.witness2.z,
                ),
                normal1: crate::math::vec::Vec3::new(hit.normal1.x, hit.normal1.y, hit.normal1.z),
                normal2: crate::math::vec::Vec3::new(hit.normal2.x, hit.normal2.y, hit.normal2.z),
            })
        } else {
            None
        }
    }
}

impl System for PhysicsSystem {
    fn update(&mut self, dt: f32, world: &World) {
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
            &self.event_collector,
        );

        // Map RawTriggerEvents to TriggerEvents
        self.trigger_events.clear();
        if let Ok(mut raw_events) = self.event_collector.raw_events.lock() {
            for raw in raw_events.drain(..) {
                if let (Some(c1), Some(c2)) = (
                    self.collider_set.get(raw.handle1),
                    self.collider_set.get(raw.handle2),
                ) {
                    self.trigger_events.push(TriggerEvent {
                        entity1: c1.user_data as crate::ecs::EntityId,
                        entity2: c2.user_data as crate::ecs::EntityId,
                        started: raw.started,
                    });
                }
            }
        }

        // Synchronize state from Rapier to ECS Transforms.
        let rb_components = world.get_component_array::<RigidBodyComponent>();
        let entities = rb_components.dense_entities_slice();
        let transforms = unsafe { &mut *world.get_component_array_mut_ptr::<TransformComponent>() };

        for (i, rb_comp) in rb_components.as_slice().iter().enumerate() {
            let entity = entities[i];
            let handle = rb_comp.handle;
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

    fn read_components(&self) -> ComponentMask {
        crate::ecs::types::build_mask(&[crate::ecs::types::get_component_type_id::<
            RigidBodyComponent,
        >()])
    }

    fn write_components(&self) -> ComponentMask {
        crate::ecs::types::build_mask(&[crate::ecs::types::get_component_type_id::<
            TransformComponent,
        >()])
    }
}
