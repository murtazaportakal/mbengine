use crate::math::mat4::Mat4;
use crate::math::vec::Vec3;
use serde::{Deserialize, Serialize};
use crate::ecs::reflection::ReflectComponent;

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
    fn name() -> &'static str { "Transform" }
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Pos X");
            changed |= ui.add(egui::DragValue::new(&mut self.position.x).speed(0.1)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Pos Y");
            changed |= ui.add(egui::DragValue::new(&mut self.position.y).speed(0.1)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Pos Z");
            changed |= ui.add(egui::DragValue::new(&mut self.position.z).speed(0.1)).changed();
        });

        ui.horizontal(|ui| {
            ui.label("Rot X");
            changed |= ui.add(egui::DragValue::new(&mut self.rotation.x).speed(0.05)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Rot Y");
            changed |= ui.add(egui::DragValue::new(&mut self.rotation.y).speed(0.05)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Rot Z");
            changed |= ui.add(egui::DragValue::new(&mut self.rotation.z).speed(0.05)).changed();
        });

        ui.horizontal(|ui| {
            ui.label("Scale  ");
            changed |= ui.add(egui::DragValue::new(&mut self.scale.x).speed(0.1)).changed();
            changed |= ui.add(egui::DragValue::new(&mut self.scale.y).speed(0.1)).changed();
            changed |= ui.add(egui::DragValue::new(&mut self.scale.z).speed(0.1)).changed();
        });
        
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
    fn name() -> &'static str { "Render" }
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui.checkbox(&mut self.visible, "Visible").changed();
        changed |= ui.add(egui::Slider::new(&mut self.metallic, 0.0..=1.0).text("Metallic")).changed();
        changed |= ui.add(egui::Slider::new(&mut self.roughness, 0.0..=1.0).text("Roughness")).changed();
        changed
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CameraComponent {
    pub view: Mat4,
    pub proj: Mat4,
}

impl Default for CameraComponent {
    fn default() -> Self {
        Self {
            view: Mat4::identity(),
            proj: Mat4::perspective(std::f32::consts::FRAC_PI_4, 800.0 / 600.0, 0.1, 100.0),
        }
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
    fn name() -> &'static str { "Point Light" }
    fn draw_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui.add(egui::Slider::new(&mut self.intensity, 0.0..=100.0).text("Intensity")).changed();
        
        // Color picker
        let mut rgb = [self.color.x, self.color.y, self.color.z];
        ui.horizontal(|ui| {
            ui.label("Color");
            if ui.color_edit_button_rgb(&mut rgb).changed() {
                self.color.x = rgb[0];
                self.color.y = rgb[1];
                self.color.z = rgb[2];
                changed = true;
            }
        });
        
        changed
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct HierarchyComponent {
    pub parent: Option<u32>,
    pub local_matrix: crate::math::mat4::Mat4,
}

#[derive(Clone, Copy, Debug)]
pub struct RigidBodyComponent {
    pub handle: rapier3d::dynamics::RigidBodyHandle,
}

#[derive(Clone, Copy, Debug)]
pub struct ColliderComponent {
    pub handle: rapier3d::geometry::ColliderHandle,
}
