use crate::ecs::reflection::ReflectComponent;
use crate::math::mat4::Mat4;
use crate::math::vec::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TransformComponent {
    pub position: Vec3,
    pub rotation: Vec3, // Euler angles for now
    pub scale: Vec3,
    pub matrix: Mat4,
}

impl Default for TransformComponent {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Vec3::new(0.0, 0.0, 0.0),
            scale: Vec3::new(1.0, 1.0, 1.0),
            matrix: Mat4::identity(),
        }
    }
}

impl TransformComponent {
    pub fn update_matrix(&mut self) {
        // Build a translation matrix
        let mut t = Mat4::identity();
        t.cols[3].x = self.position.x;
        t.cols[3].y = self.position.y;
        t.cols[3].z = self.position.z;

        // Build scale matrix
        let mut s = Mat4::identity();
        s.cols[0].x = self.scale.x;
        s.cols[1].y = self.scale.y;
        s.cols[2].z = self.scale.z;

        // Skip rotation for now to keep it simple, or implement basic XYZ rotation.
        // For our test, T * S is enough.
        self.matrix = t * s;
    }
}

impl ReflectComponent for TransformComponent {
    fn name() -> &'static str {
        "Transform"
    }
    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        _physics: &mut crate::physics::PhysicsSystem,
    ) {
        unsafe {
            world.add_component(entity, Self::default());
        }
    }
    fn draw_inspector(
        &mut self,
        ui: &mut crate::ui::UiContext,
        _physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        let mut changed = false;
        changed |= ui.drag_float("Position X", &mut self.position.x);
        changed |= ui.drag_float("Position Y", &mut self.position.y);
        changed |= ui.drag_float("Position Z", &mut self.position.z);

        changed |= ui.drag_float("Rotation X", &mut self.rotation.x);
        changed |= ui.drag_float("Rotation Y", &mut self.rotation.y);
        changed |= ui.drag_float("Rotation Z", &mut self.rotation.z);

        changed |= ui.drag_float("Scale X", &mut self.scale.x);
        changed |= ui.drag_float("Scale Y", &mut self.scale.y);
        changed |= ui.drag_float("Scale Z", &mut self.scale.z);
        changed
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RenderComponent {
    // In the future this will hold mesh_id and material_id.
    // For now it acts as a tag to indicate this entity should be drawn.
    pub visible: bool,
    pub mesh_index: usize,
    pub metallic: f32,
    pub roughness: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Default for RenderComponent {
    fn default() -> Self {
        Self {
            visible: true,
            mesh_index: 0,
            metallic: 0.0,
            roughness: 0.5,
            r: 1.0,
            g: 1.0,
            b: 1.0,
        }
    }
}

impl ReflectComponent for RenderComponent {
    fn name() -> &'static str {
        "Render"
    }
    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        _physics: &mut crate::physics::PhysicsSystem,
    ) {
        unsafe {
            world.add_component(entity, Self::default());
        }
    }
    fn draw_inspector(
        &mut self,
        ui: &mut crate::ui::UiContext,
        _physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        let mut changed = false;
        changed |= ui.checkbox("Visible", &mut self.visible);
        changed |= ui.drag_float("Metallic", &mut self.metallic);
        changed |= ui.drag_float("Roughness", &mut self.roughness);
        changed |= ui.drag_float("Color R", &mut self.r);
        changed |= ui.drag_float("Color G", &mut self.g);
        changed |= ui.drag_float("Color B", &mut self.b);
        changed
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CameraComponent {
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub view: Mat4,
    pub proj: Mat4,
}

impl Default for CameraComponent {
    fn default() -> Self {
        Self {
            fov: 45.0,
            near: 0.1,
            far: 100.0,
            view: Mat4::identity(),
            proj: Mat4::perspective(std::f32::consts::FRAC_PI_4, 800.0 / 600.0, 0.1, 100.0),
        }
    }
}

impl ReflectComponent for CameraComponent {
    fn name() -> &'static str {
        "Camera"
    }
    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        _physics: &mut crate::physics::PhysicsSystem,
    ) {
        unsafe {
            world.add_component(entity, Self::default());
        }
    }
    fn draw_inspector(
        &mut self,
        ui: &mut crate::ui::UiContext,
        _physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        let mut changed = false;
        changed |= ui.drag_float("FOV", &mut self.fov);
        changed |= ui.drag_float("Near", &mut self.near);
        changed |= ui.drag_float("Far", &mut self.far);
        changed
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LightComponent {
    pub direction: Vec3,
    pub color: Vec3,
}

impl Default for LightComponent {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.0, -1.0, 0.0),
            color: Vec3::new(1.0, 1.0, 1.0),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PointLightComponent {
    pub color: Vec3,
    pub intensity: f32,
}

impl Default for PointLightComponent {
    fn default() -> Self {
        Self {
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 1.0,
        }
    }
}

impl ReflectComponent for PointLightComponent {
    fn name() -> &'static str {
        "Point Light"
    }
    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        _physics: &mut crate::physics::PhysicsSystem,
    ) {
        unsafe {
            world.add_component(entity, Self::default());
        }
    }
    fn draw_inspector(
        &mut self,
        ui: &mut crate::ui::UiContext,
        _physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        let mut changed = false;
        changed |= ui.drag_float("Color R", &mut self.color.x);
        changed |= ui.drag_float("Color G", &mut self.color.y);
        changed |= ui.drag_float("Color B", &mut self.color.z);
        changed |= ui.drag_float("Intensity", &mut self.intensity);
        changed
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct HierarchyComponent {
    pub parent: Option<u32>,
    pub local_matrix: crate::math::mat4::Mat4,
}

impl ReflectComponent for HierarchyComponent {
    fn name() -> &'static str {
        "Hierarchy"
    }
    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        _physics: &mut crate::physics::PhysicsSystem,
    ) {
        unsafe {
            world.add_component(entity, Self::default());
        }
    }
    fn draw_inspector(
        &mut self,
        ui: &mut crate::ui::UiContext,
        _physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        if let Some(parent) = self.parent {
            use core::fmt::Write;
            let mut label = crate::containers::FixedString::<128>::new();
            let _ = write!(label, "Parent: {}", parent);
            ui.label(&label);
        } else {
            ui.label("Parent: None");
        }
        false
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RigidBodyComponent {
    pub handle: rapier3d::dynamics::RigidBodyHandle,
}

impl ReflectComponent for RigidBodyComponent {
    fn name() -> &'static str {
        "Rigid Body"
    }
    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        physics: &mut crate::physics::PhysicsSystem,
    ) {
        let rb = rapier3d::prelude::RigidBodyBuilder::dynamic().build();
        let handle = physics.rigid_body_set.insert(rb);
        unsafe {
            world.add_component(entity, Self { handle });
        }
    }
    fn draw_inspector(
        &mut self,
        ui: &mut crate::ui::UiContext,
        physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        let mut changed = false;
        if let Some(rb) = physics.rigid_body_set.get_mut(self.handle) {
            let mut lin_damp = rb.linear_damping();
            if ui.drag_float("Linear Damping", &mut lin_damp) {
                rb.set_linear_damping(lin_damp);
                changed = true;
            }
            let mut ang_damp = rb.angular_damping();
            if ui.drag_float("Angular Damping", &mut ang_damp) {
                rb.set_angular_damping(ang_damp);
                changed = true;
            }
        }
        changed
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ColliderComponent {
    pub handle: rapier3d::geometry::ColliderHandle,
}

impl ReflectComponent for ColliderComponent {
    fn name() -> &'static str {
        "Collider"
    }
    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        physics: &mut crate::physics::PhysicsSystem,
    ) {
        let collider = rapier3d::prelude::ColliderBuilder::cuboid(0.5, 0.5, 0.5).build();
        let arrays = world.get_component_array::<RigidBodyComponent>();
        let handle = if arrays.has(entity) {
            let rb_comp = unsafe { arrays.get(entity) };
            physics.collider_set.insert_with_parent(
                collider,
                rb_comp.handle,
                &mut physics.rigid_body_set,
            )
        } else {
            physics.collider_set.insert(collider)
        };
        unsafe {
            world.add_component(entity, Self { handle });
        }
    }
    fn draw_inspector(
        &mut self,
        ui: &mut crate::ui::UiContext,
        physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        let mut changed = false;
        if let Some(col) = physics.collider_set.get_mut(self.handle) {
            let mut friction = col.friction();
            if ui.drag_float("Friction", &mut friction) {
                col.set_friction(friction);
                changed = true;
            }
            let mut restitution = col.restitution();
            if ui.drag_float("Restitution", &mut restitution) {
                col.set_restitution(restitution);
                changed = true;
            }
        }
        changed
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AudioListenerComponent;

impl Default for AudioListenerComponent {
    fn default() -> Self {
        Self
    }
}

impl ReflectComponent for AudioListenerComponent {
    fn name() -> &'static str {
        "Audio Listener"
    }
    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        _physics: &mut crate::physics::PhysicsSystem,
    ) {
        unsafe {
            world.add_component(entity, Self::default());
        }
    }
    fn draw_inspector(
        &mut self,
        _ui: &mut crate::ui::UiContext,
        _physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        false
    }
}

#[derive(Clone, Debug)]
pub struct AudioEmitterComponent {
    pub asset_path: crate::containers::FixedString<128>,
    pub volume: f32,
    pub max_distance: f32,
    pub is_playing: bool,
    pub loop_audio: bool,
}

impl Default for AudioEmitterComponent {
    fn default() -> Self {
        let mut path = crate::containers::FixedString::new();
        path.push_str("sounds/engine.ogg");
        Self {
            asset_path: path,
            volume: 1.0,
            max_distance: 100.0,
            is_playing: true,
            loop_audio: true,
        }
    }
}

impl ReflectComponent for AudioEmitterComponent {
    fn name() -> &'static str {
        "Audio Emitter"
    }
    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        _physics: &mut crate::physics::PhysicsSystem,
    ) {
        unsafe {
            world.add_component(entity, Self::default());
        }
    }
    fn draw_inspector(
        &mut self,
        _ui: &mut crate::ui::UiContext,
        _physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        false
    }
}

/// Links an entity to a named skeleton for GPU skinning.
///
/// The `skeleton_name` references a skeleton loaded by the AssetManager.
/// The `skinning_instance_index` is the index into the application's skinning
/// instance storage, set during scene setup.
#[derive(Clone, Debug, Default)]
pub struct SkeletonComponent {
    pub skeleton_name: crate::containers::FixedString<64>,
    /// Index into the application's `skinning_instances` array.
    /// Set during GLTF loading. `None` if not yet initialized.
    pub skinning_instance_index: Option<usize>,
    /// The final bone matrices computed by the AnimationSystem.
    pub computed_matrices: crate::containers::FixedArray<
        crate::math::mat4::Mat4,
        { crate::renderer::vulkan::skeleton::MAX_BONES },
    >,
}

impl ReflectComponent for SkeletonComponent {
    fn name() -> &'static str {
        "Skeleton"
    }
    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        _physics: &mut crate::physics::PhysicsSystem,
    ) {
        unsafe {
            world.add_component(entity, Self::default());
        }
    }
    fn draw_inspector(
        &mut self,
        _ui: &mut crate::ui::UiContext,
        _physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        false
    }
}

#[derive(Clone, Debug)]
pub enum AnimationState {
    Clip {
        clip_name: crate::containers::FixedString<64>,
    },
    Blend1D {
        clip_a: crate::containers::FixedString<64>,
        clip_b: crate::containers::FixedString<64>,
        weight: f32, // 0.0 = clip_a, 1.0 = clip_b
    },
}

impl Default for AnimationState {
    fn default() -> Self {
        AnimationState::Clip {
            clip_name: crate::containers::FixedString::new(),
        }
    }
}

/// Drives skeletal animation playback for entities with a SkeletonComponent.
#[derive(Clone, Debug)]
pub struct AnimatorComponent {
    /// Current state (Single Clip or Blend1D).
    pub state: AnimationState,
    /// Current playback time in seconds for the primary state.
    pub current_time: f32,

    /// Target state if currently in a transition.
    pub target_state: Option<AnimationState>,
    /// Playback time for the target state.
    pub transition_time: f32,

    /// Current time in the crossfade.
    pub crossfade_current: f32,
    /// Total duration of the crossfade.
    pub crossfade_duration: f32,

    /// Playback speed multiplier (1.0 = normal).
    pub speed: f32,
    /// Whether the animation is currently playing.
    pub is_playing: bool,
    /// Whether the animation loops.
    pub is_looping: bool,
}

impl Default for AnimatorComponent {
    fn default() -> Self {
        Self {
            state: AnimationState::default(),
            current_time: 0.0,
            target_state: None,
            transition_time: 0.0,
            crossfade_current: 0.0,
            crossfade_duration: 0.0,
            speed: 1.0,
            is_playing: true,
            is_looping: true,
        }
    }
}

impl AnimatorComponent {
    /// Start a crossfade to a new state over a given duration (in seconds).
    pub fn crossfade_to(&mut self, new_state: AnimationState, duration: f32) {
        if duration <= 0.0 {
            self.state = new_state;
            self.current_time = 0.0;
            self.target_state = None;
        } else {
            self.target_state = Some(new_state);
            self.transition_time = 0.0;
            self.crossfade_current = 0.0;
            self.crossfade_duration = duration;
        }
    }
}

impl ReflectComponent for AnimatorComponent {
    fn name() -> &'static str {
        "Animator"
    }
    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        _physics: &mut crate::physics::PhysicsSystem,
    ) {
        unsafe {
            world.add_component(entity, Self::default());
        }
    }
    fn draw_inspector(
        &mut self,
        _ui: &mut crate::ui::UiContext,
        _physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        false
    }
}

/// A component that executes a Rhai script to modify the entity's state.
#[derive(Clone, Debug)]
pub struct ScriptBehaviorComponent {
    pub script_name: crate::containers::FixedString<64>,
    pub state: rhai::Dynamic,
    pub initialized: bool,
}

impl Default for ScriptBehaviorComponent {
    fn default() -> Self {
        Self {
            script_name: crate::containers::FixedString::new(),
            state: rhai::Dynamic::from(rhai::Map::new()),
            initialized: false,
        }
    }
}

impl ReflectComponent for ScriptBehaviorComponent {
    fn name() -> &'static str {
        "Script Behavior"
    }
    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        _physics: &mut crate::physics::PhysicsSystem,
    ) {
        unsafe {
            world.add_component(entity, Self::default());
        }
    }
    fn draw_inspector(
        &mut self,
        _ui: &mut crate::ui::UiContext,
        _physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        false
    }
}

/// Defines a deformable cloth or soft body.
#[derive(Clone, Debug)]
pub struct SoftBodyComponent {
    /// Grid dimensions (width, height) in vertices
    pub grid_size: (u32, u32),
    pub stiffness: f32,
    pub mass: f32,
    pub damping: f32,
    /// Index into the application's cloth instances array
    pub cloth_instance_index: Option<usize>,
}

impl Default for SoftBodyComponent {
    fn default() -> Self {
        Self {
            grid_size: (10, 10),
            stiffness: 1000.0,
            mass: 1.0,
            damping: 0.98,
            cloth_instance_index: None,
        }
    }
}

impl ReflectComponent for SoftBodyComponent {
    fn name() -> &'static str {
        "Soft Body"
    }
    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        _physics: &mut crate::physics::PhysicsSystem,
    ) {
        unsafe {
            world.add_component(entity, Self::default());
        }
    }
    fn draw_inspector(
        &mut self,
        _ui: &mut crate::ui::UiContext,
        _physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        false
    }
}
