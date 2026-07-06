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
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::Grid::new("transform_grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Position");
                ui.horizontal(|ui| {
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.position.x)
                                .speed(0.1)
                                .prefix("X: "),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.position.y)
                                .speed(0.1)
                                .prefix("Y: "),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.position.z)
                                .speed(0.1)
                                .prefix("Z: "),
                        )
                        .changed();
                });
                ui.end_row();

                ui.label("Rotation");
                ui.horizontal(|ui| {
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.rotation.x)
                                .speed(0.05)
                                .prefix("X: "),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.rotation.y)
                                .speed(0.05)
                                .prefix("Y: "),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.rotation.z)
                                .speed(0.05)
                                .prefix("Z: "),
                        )
                        .changed();
                });
                ui.end_row();

                ui.label("Scale");
                ui.horizontal(|ui| {
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.scale.x)
                                .speed(0.1)
                                .prefix("X: "),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.scale.y)
                                .speed(0.1)
                                .prefix("Y: "),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.scale.z)
                                .speed(0.1)
                                .prefix("Z: "),
                        )
                        .changed();
                });
                ui.end_row();
            });

        if changed {
            self.update_matrix();
        }
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
}

impl Default for RenderComponent {
    fn default() -> Self {
        Self {
            visible: true,
            mesh_index: 0,
            metallic: 0.0,
            roughness: 0.5,
        }
    }
}

impl ReflectComponent for RenderComponent {
    fn name() -> &'static str {
        "Render"
    }
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::Grid::new("render_grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Visible");
                changed |= ui.checkbox(&mut self.visible, "").changed();
                ui.end_row();

                ui.label("Metallic");
                changed |= ui
                    .add(egui::Slider::new(&mut self.metallic, 0.0..=1.0))
                    .changed();
                ui.end_row();

                ui.label("Roughness");
                changed |= ui
                    .add(egui::Slider::new(&mut self.roughness, 0.0..=1.0))
                    .changed();
                ui.end_row();
            });
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
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::Grid::new("camera_grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("FOV");
                changed |= ui
                    .add(egui::Slider::new(&mut self.fov, 10.0..=120.0))
                    .changed();
                ui.end_row();

                ui.label("Near Plane");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.near).speed(0.1))
                    .changed();
                ui.end_row();

                ui.label("Far Plane");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.far).speed(1.0))
                    .changed();
                ui.end_row();
            });
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
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::Grid::new("pointlight_grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Color");
                let mut color_arr = [self.color.x, self.color.y, self.color.z];
                if ui.color_edit_button_rgb(&mut color_arr).changed() {
                    self.color.x = color_arr[0];
                    self.color.y = color_arr[1];
                    self.color.z = color_arr[2];
                    changed = true;
                }
                ui.end_row();

                ui.label("Intensity");
                changed |= ui
                    .add(egui::Slider::new(&mut self.intensity, 0.0..=100.0))
                    .changed();
                ui.end_row();
            });
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
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        if let Some(parent) = self.parent {
            ui.label(format!("Parent Entity: {}", parent));
        } else {
            ui.label("Parent Entity: None");
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
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        ui.label(format!("Handle Index: {}", self.handle.into_raw_parts().0));
        false
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
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        ui.label(format!("Handle Index: {}", self.handle.into_raw_parts().0));
        false
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
    fn draw_inspector(&mut self, _ui: &mut egui::Ui) -> bool {
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
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui.checkbox(&mut self.is_playing, "Playing").changed();
        changed |= ui.checkbox(&mut self.loop_audio, "Loop").changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.volume, 0.0..=2.0).text("Volume"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.max_distance, 1.0..=500.0).text("Max Distance"))
            .changed();
        changed
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
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        ui.label(format!("Skeleton: {}", self.skeleton_name.as_str()));
        if let Some(idx) = self.skinning_instance_index {
            ui.label(format!("Skinning Instance: {}", idx));
        } else {
            ui.label("Skinning Instance: Not bound");
        }
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
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        
        match &self.state {
            AnimationState::Clip { clip_name } => {
                ui.label(format!("State: Clip ({})", clip_name.as_str()));
            }
            AnimationState::Blend1D { clip_a, clip_b, weight } => {
                ui.label(format!("State: Blend1D ({} <-> {})", clip_a.as_str(), clip_b.as_str()));
                ui.label(format!("Weight: {:.2}", weight));
            }
        }
        
        if self.target_state.is_some() {
            ui.label(format!("Transitioning... ({:.2}/{:.2}s)", self.crossfade_current, self.crossfade_duration));
        }

        changed |= ui.checkbox(&mut self.is_playing, "Playing").changed();
        changed |= ui.checkbox(&mut self.is_looping, "Looping").changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.speed, 0.0..=5.0).text("Speed"))
            .changed();
        ui.label(format!("Time: {:.2}s", self.current_time));
        changed
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
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        
        let mut script_name = self.script_name.as_str().to_string();
        ui.horizontal(|ui| {
            ui.label("Script Name:");
            if ui.text_edit_singleline(&mut script_name).changed() {
                self.script_name.clear();
                self.script_name.push_str(&script_name);
                self.initialized = false;
                changed = true;
            }
        });

        ui.label(format!("Initialized: {}", self.initialized));
        
        if ui.button("Reload").clicked() {
            self.initialized = false;
            changed = true;
        }

        changed
    }
}
