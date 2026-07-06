use crate::ecs::reflection::ComponentRegistry;
use crate::ecs::TransformComponent;
use crate::ecs::{EntityId, World};
use crate::physics::PhysicsSystem;
use crate::app::docking::{DockingManager, PanelType};
use crate::app::slate_theme::EngineTheme;

pub enum EditorAction {
    Play,
    Pause,
    SpawnModel(String),
}

pub struct Editor {
    pub registry: ComponentRegistry,
    pub file_dialog_receiver: Option<std::sync::mpsc::Receiver<String>>,
    pub node_editor: crate::app::node_editor::NodeGraphEditor,
    pub docking: DockingManager,
    pub theme: EngineTheme,
}

impl Default for Editor {
    fn default() -> Self {
        Self::new()
    }
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
        registry.register::<crate::ecs::components::SkeletonComponent>();
        registry.register::<crate::ecs::components::ScriptBehaviorComponent>();
        registry.register::<crate::ecs::components::AnimatorComponent>();

        Self {
            registry,
            file_dialog_receiver: None,
            node_editor: crate::app::node_editor::NodeGraphEditor::new(),
            docking: DockingManager::new(),
            theme: EngineTheme::slate(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        ctx: &egui::Context,
        world: &mut World,
        physics: &mut PhysicsSystem,
        selected_entity: &mut Option<EntityId>,
        bloom_threshold: &mut f32,
        fps: f32,
        is_playing: bool,
        offscreen_texture_id: egui::TextureId,
    ) -> (Vec<EditorAction>, Option<(u32, u32)>, Option<(f32, f32)>, bool) {
        let mut actions = Vec::new();
        let mut new_viewport_size = None;
        let mut raycast_request = None;
        let mut viewport_hovered = false;

        // Check if a file dialog completed
        if let Some(rx) = &self.file_dialog_receiver {
            if let Ok(path) = rx.try_recv() {
                actions.push(EditorAction::SpawnModel(path));
                self.file_dialog_receiver = None; // Reset after receiving
            }
        }

        // --- Top Menu Bar ---
        let top_frame = crate::app::slate_theme::EditorFrame::panel();
        egui::TopBottomPanel::top("top_menu_bar")
            .frame(top_frame)
            .show(ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        if ui.button("New Scene").clicked() { /* ... */ }
                        if ui.button("Open Scene...").clicked() { /* ... */ }
                        ui.separator();
                        if ui.button("Save Scene").clicked() { /* ... */ }
                        ui.separator();
                        if ui.button("Exit").clicked() {
                            std::process::exit(0);
                        }
                    });
                    ui.menu_button("Edit", |ui| {
                        if ui.button("Undo").clicked() { /* ... */ }
                        if ui.button("Redo").clicked() { /* ... */ }
                    });
                    ui.menu_button("View", |ui| {
                        if ui.button("Toggle Fullscreen").clicked() { /* ... */ }
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(format!("FPS: {:.1}", fps));
                        ui.separator();
                        if is_playing {
                            if ui.button("⏸ Pause").clicked() {
                                actions.push(EditorAction::Pause);
                            }
                        } else {
                            if ui.button("▶ Play").clicked() {
                                actions.push(EditorAction::Play);
                            }
                        }
                    });
                });
            });

        // --- Central Panel (Docking Area) ---
        egui::CentralPanel::default()
            .frame(crate::app::slate_theme::EditorFrame::central_viewport())
            .show(ctx, |ui| {
                
                // Borrow extraction for closure
                let docking = &mut self.docking;
                let theme = &self.theme;
                let node_editor = &mut self.node_editor;
                let registry = &mut self.registry;
                let mut receiver = self.file_dialog_receiver.take();
                let mut triggered_file_dialog = false;

                docking.draw(ui, theme, |panel_type, ui, _rect| {
                    match panel_type {
                        PanelType::Console => {
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                ui.label("[System] Engine Booted Successfully.");
                                ui.label("[Memory] Initialized 256MB arena block.");
                                ui.label("[HotReload] game.dll attached.");
                                ui.label(format!("Current FPS: {:.1}", fps));
                            });
                        }
                        PanelType::NodeGraph => {
                            ui.horizontal(|ui| {
                                if ui.button("Compile to Rhai").clicked() {
                                    let code = node_editor.compile_to_rhai();
                                    let path = std::path::Path::new("assets/scripts/visual_graph.rhai");
                                    if let Some(parent) = path.parent() {
                                        let _ = std::fs::create_dir_all(parent);
                                    }
                                    let mut msg = crate::containers::FixedString::<128>::new();
                                    use std::fmt::Write;
                                    if let Err(e) = std::fs::write(path, code) {
                                        let _ = write!(&mut msg, "Error saving script: {}", e);
                                    } else {
                                        let _ = write!(&mut msg, "Success! Saved to assets/scripts/visual_graph.rhai");
                                    }
                                    node_editor.compile_message = Some((msg, std::time::Instant::now()));
                                }
                                
                                if let Some((msg, time)) = &node_editor.compile_message {
                                    if time.elapsed().as_secs_f32() < 3.0 {
                                        ui.label(egui::RichText::new(msg.as_str()).color(theme.accent));
                                    } else {
                                        node_editor.compile_message = None;
                                    }
                                }
                            });
                            ui.separator();
                            node_editor.draw(ui);
                        }
                        PanelType::Hierarchy => {
                            if ui.button("Add Entity").clicked() && receiver.is_none() {
                                triggered_file_dialog = true;
                            }
                            ui.separator();
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                let entities = world
                                    .get_component_array::<TransformComponent>()
                                    .dense_entities_slice()
                                    .to_vec();

                                let hierarchies =
                                    world.get_component_array::<crate::ecs::components::HierarchyComponent>();

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
                                    Self::draw_entity_tree(ui, root, &children_map, selected_entity, world);
                                }
                            });
                        }
                        PanelType::Inspector => {
                            if let Some(entity_id) = *selected_entity {
                                ui.label(format!("Entity ID: {}", entity_id));
                                ui.separator();

                                ui.heading("Post Processing");
                                ui.add(egui::Slider::new(bloom_threshold, 0.0..=10.0).text("Bloom Threshold"));
                                ui.separator();

                                registry.draw_entity(entity_id, world, ui, physics);
                            } else {
                                ui.label("No Entity Selected.");
                            }
                        }
                        PanelType::Viewport => {
                            let size = ui.available_size();
                            new_viewport_size = Some((size.x.max(1.0) as u32, size.y.max(1.0) as u32));
                            let image = egui::Image::new(egui::load::SizedTexture::new(
                                offscreen_texture_id,
                                size,
                            ))
                            .sense(egui::Sense::click() | egui::Sense::drag());

                            let response = ui.add(image);
                            viewport_hovered = response.hovered() || response.dragged();

                            if response.clicked() {
                                *selected_entity = None;
                                if let Some(pos) = response.interact_pointer_pos() {
                                    let local_pos = pos - response.rect.min;
                                    let ndc_x = (local_pos.x / response.rect.width()) * 2.0 - 1.0;
                                    let ndc_y = (local_pos.y / response.rect.height()) * 2.0 - 1.0;
                                    raycast_request = Some((ndc_x, ndc_y));
                                }
                            }
                        }
                        PanelType::None => {}
                    }
                });

                if triggered_file_dialog {
                    let (tx, rx) = std::sync::mpsc::channel();
                    receiver = Some(rx);
                    std::thread::spawn(move || {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("3D Models", &["obj", "gltf", "glb"])
                            .pick_file()
                        {
                            let _ = tx.send(path.to_string_lossy().into_owned());
                        }
                    });
                }
                
                self.file_dialog_receiver = receiver;
            });

        (actions, new_viewport_size, raycast_request, viewport_hovered)
    }

    fn draw_entity_tree(
        ui: &mut egui::Ui,
        entity: EntityId,
        children_map: &std::collections::HashMap<EntityId, Vec<EntityId>>,
        selected_entity: &mut Option<EntityId>,
        world: &World,
    ) {
        let mut icon = "📦";
        if world.get_component_array::<crate::ecs::components::CameraComponent>().has(entity) {
            icon = "🎥";
        } else if world.get_component_array::<crate::ecs::PointLightComponent>().has(entity) {
            icon = "💡";
        } else if world.get_component_array::<crate::ecs::RenderComponent>().has(entity) {
            icon = "🧊";
        } else if world.get_component_array::<crate::ecs::components::AudioEmitterComponent>().has(entity) {
            icon = "🔊";
        }

        let label = format!("{} Entity {}", icon, entity);
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
                        Self::draw_entity_tree(ui, child, children_map, selected_entity, world);
                    }
                });
        } else {
            if ui.selectable_label(is_selected, &label).clicked() {
                *selected_entity = Some(entity);
            }
        }
    }
}
