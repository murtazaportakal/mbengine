use crate::ecs::{EntityId, World};
use crate::ecs::{RenderComponent, TransformComponent};
use crate::physics::PhysicsSystem;
use crate::ecs::reflection::ComponentRegistry;

pub struct Editor {
    pub registry: ComponentRegistry,
}

impl Editor {
    pub fn new() -> Self {
        let mut registry = ComponentRegistry::new();
        registry.register::<crate::ecs::TransformComponent>();
        registry.register::<crate::ecs::RenderComponent>();
        registry.register::<crate::ecs::PointLightComponent>();
        registry.register::<crate::ecs::components::CameraComponent>();
        registry.register::<crate::ecs::components::HierarchyComponent>();
        registry.register::<crate::ecs::components::RigidBodyComponent>();
        registry.register::<crate::ecs::components::ColliderComponent>();
        registry.register::<crate::ecs::components::AudioEmitterComponent>();
        registry.register::<crate::ecs::components::AudioListenerComponent>();
        
        Self { registry }
    }

    pub fn draw(
        &mut self,
        ctx: &egui::Context,
        world: &mut World,
        physics: &mut PhysicsSystem,
        selected_entity: &mut Option<EntityId>,
        bloom_threshold: &mut f32,
        fps: f32,
    ) {
        egui::SidePanel::left("hierarchy_panel")
            .resizable(true)
            .min_width(200.0)
            .show(ctx, |ui| {
                ui.heading("Hierarchy");
                ui.label(format!("FPS: {:.1}", fps));
                ui.separator();
                if ui.button("Add Entity").clicked() {
                    let new_entity = world.create_entity();
                    let x = 0.0;
                    let y = 0.0;
                    let z = 0.0;
                    unsafe {
                        world.add_component(
                            new_entity,
                            TransformComponent {
                                position: crate::math::vec::Vec3::new(x, y, z),
                                rotation: crate::math::vec::Vec3::new(0.0, 0.0, 0.0),
                                scale: crate::math::vec::Vec3::new(1.0, 1.0, 1.0),
                                matrix: crate::math::mat4::Mat4::identity(),
                            },
                        );
                        world.add_component(
                            new_entity,
                            RenderComponent {
                                mesh_index: 0,
                                visible: true,
                                metallic: 0.1,
                                roughness: 0.8,
                            },
                        );
                    }
                }
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    let entities = world
                        .get_component_array::<TransformComponent>()
                        .dense_entities_slice()
                        .to_vec(); // clone so we don't hold the borrow
                        
                    let hierarchies = world.get_component_array::<crate::ecs::components::HierarchyComponent>();

                    let mut children_map: std::collections::HashMap<EntityId, Vec<EntityId>> =
                        std::collections::HashMap::new();
                    let mut roots = Vec::new();

                    for &entity in &entities {
                        if hierarchies.has(entity) {
                            let parent_opt = unsafe { hierarchies.get(entity) }.parent;
                            if let Some(parent) = parent_opt {
                                children_map.entry(parent).or_default().push(entity);
                            } else {
                                roots.push(entity);
                            }
                        } else {
                            roots.push(entity);
                        }
                    }

                    for &root in &roots {
                        Self::draw_entity_tree(ui, root, &children_map, selected_entity);
                    }
                });
            });

        egui::SidePanel::right("inspector_panel")
            .resizable(true)
            .min_width(250.0)
            .show(ctx, |ui| {
                ui.heading("Inspector");
                ui.separator();
                if let Some(entity_id) = *selected_entity {
                    ui.label(format!("Entity ID: {}", entity_id));
                    ui.separator();

                    ui.heading("Post Processing");
                    ui.add(egui::Slider::new(bloom_threshold, 0.0..=10.0).text("Bloom Threshold"));
                    ui.separator();

                    self.registry.draw_entity(entity_id, world, ui, physics);
                } else {
                    ui.label("No Entity Selected.");
                }
            });
    }

    fn draw_entity_tree(
        ui: &mut egui::Ui,
        entity: EntityId,
        children_map: &std::collections::HashMap<EntityId, Vec<EntityId>>,
        selected_entity: &mut Option<EntityId>,
    ) {
        let label = format!("Entity {}", entity);
        let is_selected = *selected_entity == Some(entity);

        if let Some(children) = children_map.get(&entity) {
            let id = ui.id().with(entity);
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true)
                .show_header(ui, |ui| {
                    if ui.selectable_label(is_selected, &label).clicked() {
                        *selected_entity = Some(entity);
                    }
                })
                .body(|ui| {
                    for &child in children {
                        Self::draw_entity_tree(ui, child, children_map, selected_entity);
                    }
                });
        } else {
            if ui.selectable_label(is_selected, &label).clicked() {
                *selected_entity = Some(entity);
            }
        }
    }
}
