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
    /// Builds the full world matrix: T × Rz × Ry × Rx × S
    ///
    /// Rotation is applied in extrinsic XYZ order: first around X (pitch),
    /// then around Y (yaw), then around Z (roll).
    pub fn update_matrix(&mut self) {
        // Scale matrix
        let mut sc = Mat4::identity();
        sc.cols[0].x = self.scale.x;
        sc.cols[1].y = self.scale.y;
        sc.cols[2].z = self.scale.z;

        // Rotation around X axis (pitch)
        let rx = self.rotation.x;
        let mut rot_x = Mat4::identity();
        rot_x.cols[1].y = rx.cos();
        rot_x.cols[1].z = rx.sin();
        rot_x.cols[2].y = -rx.sin();
        rot_x.cols[2].z = rx.cos();

        // Rotation around Y axis (yaw)
        let ry = self.rotation.y;
        let mut rot_y = Mat4::identity();
        rot_y.cols[0].x = ry.cos();
        rot_y.cols[0].z = -ry.sin();
        rot_y.cols[2].x = ry.sin();
        rot_y.cols[2].z = ry.cos();

        // Rotation around Z axis (roll)
        let rz = self.rotation.z;
        let mut rot_z = Mat4::identity();
        rot_z.cols[0].x = rz.cos();
        rot_z.cols[0].y = rz.sin();
        rot_z.cols[1].x = -rz.sin();
        rot_z.cols[1].y = rz.cos();

        // Translation matrix
        let mut t = Mat4::identity();
        t.cols[3].x = self.position.x;
        t.cols[3].y = self.position.y;
        t.cols[3].z = self.position.z;

        // T × Rz × Ry × Rx × S
        self.matrix = t * rot_z * rot_y * rot_x * sc;
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
            world.add_component(entity, Self);
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
    /// Single clip playback.
    Clip {
        clip_handle: u32,
    },
    /// 1D linear blend between two clips using `weight` ∈ [0, 1].
    Blend1D {
        clip_a: u32,
        clip_b: u32,
        weight: f32, // 0.0 = clip_a, 1.0 = clip_b
    },
    /// 2D bilinear blend between four clips arranged on a unit-square grid.
    ///
    /// ```text
    ///   param_y=1  clip_tl ──── clip_tr
    ///              │              │
    ///   param_y=0  clip_bl ──── clip_br
    ///              param_x=0   param_x=1
    /// ```
    Blend2D {
        clip_bl: u32, // bottom-left  (0, 0)
        clip_br: u32, // bottom-right (1, 0)
        clip_tl: u32, // top-left     (0, 1)
        clip_tr: u32, // top-right    (1, 1)
        param_x: f32, // horizontal blend param [0, 1]
        param_y: f32, // vertical   blend param [0, 1]
    },
}

impl Default for AnimationState {
    fn default() -> Self {
        AnimationState::Clip {
            clip_handle: 0,
        }
    }
}

/// A single named transition between two state-machine states.
#[derive(Clone, Debug)]
pub struct StateMachineTransition {
    /// Source state name.  Empty string matches any state.
    pub from: crate::containers::FixedString<32>,
    /// Destination state name.
    pub to: crate::containers::FixedString<32>,
    /// The float parameter whose value drives this transition.
    pub condition_param: crate::containers::FixedString<32>,
    /// Transition fires when `param >= threshold` (for ≥ semantics) or
    /// when `param < threshold` (for < semantics, indicated by negative threshold).
    pub threshold: f32,
    /// Cross-fade duration in seconds (0 = instant switch).
    pub blend_duration: f32,
}

/// Named float parameters accessible from both the state machine and Rhai scripts.
#[derive(Clone, Debug, Default)]
pub struct StateMachineParams {
    pub entries: crate::containers::FixedArray<(crate::containers::FixedString<32>, f32), 16>,
}

impl StateMachineParams {
    pub fn set(&mut self, name: &str, value: f32) {
        for (n, v) in self.entries.as_mut_slice() {
            if n.as_str() == name { *v = value; return; }
        }
        // Not found — insert if capacity allows
        let mut key = crate::containers::FixedString::<32>::new();
        key.push_str(name);
        self.entries.push((key, value));
    }

    pub fn get(&self, name: &str) -> f32 {
        for (n, v) in self.entries.as_slice() {
            if n.as_str() == name { return *v; }
        }
        0.0
    }
}

/// A state machine layered on top of `AnimatorComponent`.
///
/// Holds up to 16 named animation states and 32 transitions.  Each frame
/// `evaluate_transitions` is called by the animation system; when a condition
/// fires it triggers a `crossfade_to` on the owning `AnimatorComponent`.
#[derive(Clone, Debug, Default)]
pub struct AnimatorStateMachine {
    /// Named states: (name → AnimationState).
    pub states: crate::containers::FixedArray<
        (crate::containers::FixedString<32>, AnimationState), 16,
    >,
    /// All possible transitions.
    pub transitions: crate::containers::FixedArray<StateMachineTransition, 32>,
    /// Shared float parameters accessible by transitions and scripts.
    pub params: StateMachineParams,
    /// Name of the currently active state.
    pub current_state_name: crate::containers::FixedString<32>,
}

impl AnimatorStateMachine {
    /// Register a named animation state.
    pub fn add_state(&mut self, name: &str, state: AnimationState) {
        let mut key = crate::containers::FixedString::<32>::new();
        key.push_str(name);
        self.states.push((key, state));
    }

    /// Register a transition.
    pub fn add_transition(
        &mut self,
        from: &str,
        to: &str,
        param: &str,
        threshold: f32,
        blend_duration: f32,
    ) {
        let mut t = StateMachineTransition {
            from: crate::containers::FixedString::new(),
            to: crate::containers::FixedString::new(),
            condition_param: crate::containers::FixedString::new(),
            threshold,
            blend_duration,
        };
        t.from.push_str(from);
        t.to.push_str(to);
        t.condition_param.push_str(param);
        self.transitions.push(t);
    }

    /// Set a float parameter by name.
    pub fn set_param(&mut self, name: &str, value: f32) {
        self.params.set(name, value);
    }

    /// Get a float parameter by name.
    pub fn get_param(&self, name: &str) -> f32 {
        self.params.get(name)
    }

    /// Evaluate all transitions from the current state.
    /// Returns `Some((target_state, blend_duration))` if a transition fires.
    pub fn evaluate_transitions(&self) -> Option<(&AnimationState, f32)> {
        let current = self.current_state_name.as_str();
        for t in self.transitions.as_slice() {
            let from_matches = t.from.as_str().is_empty() || t.from.as_str() == current;
            if !from_matches { continue; }

            let param_val = self.params.get(t.condition_param.as_str());
            let fires = if t.threshold >= 0.0 {
                param_val >= t.threshold
            } else {
                param_val < -t.threshold
            };
            if !fires { continue; }
            // Don't self-transition
            if t.to.as_str() == current { continue; }

            // Find the target state
            for (name, state) in self.states.as_slice() {
                if name.as_str() == t.to.as_str() {
                    return Some((state, t.blend_duration));
                }
            }
        }
        None
    }

    /// Update the current state name after a crossfade commits.
    pub fn commit_transition(&mut self, new_state_name: &str) {
        self.current_state_name = crate::containers::FixedString::new();
        self.current_state_name.push_str(new_state_name);
    }
}

/// Drives skeletal animation playback for entities with a `SkeletonComponent`.
#[derive(Clone, Debug)]
pub struct AnimatorComponent {
    /// Current state (Single Clip, Blend1D, or Blend2D).
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

    /// Optional state machine layered on top of direct state control.
    ///
    /// When `Some`, `evaluate_transitions` is called each frame and may
    /// override the active state via `crossfade_to`.
    pub state_machine: Option<AnimatorStateMachine>,
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
            state_machine: None,
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


// �� ClothComponent �����������������������������������������������������������

/// Marks an entity as a GPU-simulated cloth / soft body.
///
/// The `RenderBackend` uses this component to maintain a `ClothGpuInstance`
/// for the entity.  The ECS component stores simulation parameters; the GPU
/// state (velocity buffer, descriptor sets) lives in `RenderBackend::cloth_instances`.
#[derive(Clone, Debug)]
pub struct ClothComponent {
    pub grid_width: u32,
    pub grid_height: u32,
    pub stiffness: f32,
    pub damping: f32,
    pub solver_iterations: u32,
    /// Flat indices of pinned grid particles (row * grid_width + col).
    pub pinned_vertices: crate::containers::FixedArray<u32, 16>,
    /// Sphere colliders in local space: [cx, cy, cz, radius].
    pub sphere_colliders: crate::containers::FixedArray<[f32; 4], 8>,
    /// Index into `RenderBackend::cloth_instances` once the GPU instance is created.
    pub gpu_instance_index: Option<usize>,
}

impl Default for ClothComponent {
    fn default() -> Self {
        Self {
            grid_width: 16,
            grid_height: 16,
            stiffness: 0.8,
            damping: 0.02,
            solver_iterations: 8,
            pinned_vertices: crate::containers::FixedArray::new(),
            sphere_colliders: crate::containers::FixedArray::new(),
            gpu_instance_index: None,
        }
    }
}

impl ReflectComponent for ClothComponent {
    fn name() -> &'static str { "Cloth" }

    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        _physics: &mut crate::physics::PhysicsSystem,
    ) {
        unsafe { world.add_component(entity, Self::default()); }
    }

    fn draw_inspector(
        &mut self,
        ui: &mut crate::ui::UiContext,
        _physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        let mut changed = false;
        let mut gw = self.grid_width as f32;
        if ui.drag_float("Grid Width", &mut gw) {
            self.grid_width = (gw as u32).clamp(2, 64);
            changed = true;
        }
        let mut gh = self.grid_height as f32;
        if ui.drag_float("Grid Height", &mut gh) {
            self.grid_height = (gh as u32).clamp(2, 64);
            changed = true;
        }
        if ui.drag_float("Stiffness", &mut self.stiffness) { changed = true; }
        if ui.drag_float("Damping", &mut self.damping) { changed = true; }
        let mut iters = self.solver_iterations as f32;
        if ui.drag_float("Solver Iterations", &mut iters) {
            self.solver_iterations = (iters as u32).clamp(1, 32);
            changed = true;
        }
        changed
    }
}

// ── NameComponent ────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct NameComponent {
    pub name: crate::containers::FixedString<64>,
}

impl ReflectComponent for NameComponent {
    fn name() -> &'static str { "Name" }

    fn add_to_entity(
        entity: crate::ecs::EntityId,
        world: &mut crate::ecs::World,
        _physics: &mut crate::physics::PhysicsSystem,
    ) {
        unsafe { world.add_component(entity, Self::default()); }
    }

    fn draw_inspector(
        &mut self,
        _ui: &mut crate::ui::UiContext,
        _physics: &mut crate::physics::PhysicsSystem,
    ) -> bool {
        // Name edits via Inspector could be done here later
        false
    }
}
