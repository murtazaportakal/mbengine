use crate::ecs::{EntityId, World};
use crate::ecs::TransformComponent;
use crate::physics::PhysicsSystem;
use crate::ecs::reflection::ComponentRegistry;
pub fn configure_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::light();
    
    // Light gray backgrounds
    let bg_color = egui::Color32::from_rgb(245, 245, 245);
    let panel_color = egui::Color32::from_rgb(255, 255, 255); // Solid white for distinct windows/menus
    let text_color = egui::Color32::from_rgb(20, 20, 20);
    let accent_color = egui::Color32::from_rgb(0, 112, 224); 
    let accent_hovered = egui::Color32::from_rgb(30, 130, 240);
    
    visuals.widgets.noninteractive.bg_fill = panel_color;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 200, 200));
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, text_color);
    
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(225, 225, 225);
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 200, 200));
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, text_color);
    
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(210, 210, 210);
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, accent_hovered);
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, text_color);
    
    visuals.widgets.active.bg_fill = accent_color;
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, accent_color);
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);

    visuals.selection.bg_fill = accent_color;
    visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 80, 180));
    
    visuals.window_fill = panel_color;
    visuals.panel_fill = bg_color;
    visuals.faint_bg_color = bg_color;
    visuals.extreme_bg_color = panel_color;
    
    visuals.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(180, 180, 180));
    
    visuals.window_shadow = egui::epaint::Shadow {
        offset: egui::vec2(0.0, 8.0),
        blur: 16.0,
        spread: 0.0,
        color: egui::Color32::from_black_alpha(60),
    };
    visuals.popup_shadow = egui::epaint::Shadow {
        offset: egui::vec2(0.0, 4.0),
        blur: 8.0,
        spread: 0.0,
        color: egui::Color32::from_black_alpha(60),
    };
    
    let rounding = egui::Rounding::same(2.0);
    visuals.widgets.noninteractive.rounding = rounding;
    visuals.widgets.inactive.rounding = rounding;
    visuals.widgets.hovered.rounding = rounding;
    visuals.widgets.active.rounding = rounding;
    visuals.window_rounding = rounding;
    visuals.menu_rounding = rounding;
    
    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(6.0, 4.0);
    style.spacing.window_margin = egui::Margin::same(4.0);
    ctx.set_style(style);
}

pub enum EditorAction {
    Play,
    Pause,
    SpawnModel(String),
}

#[derive(PartialEq)]
pub enum BottomTab {
    Console,
    AssetBrowser,
    Profiler,
}

pub struct Editor {
    pub registry: ComponentRegistry,
    pub bottom_tab: BottomTab,
    pub file_dialog_receiver: Option<std::sync::mpsc::Receiver<String>>,
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
        registry.register::<crate::ecs::components::AnimatorComponent>();
        
        Self { 
            registry, 
            bottom_tab: BottomTab::Console,
            file_dialog_receiver: None,
        }
    }

    pub fn draw(
        &mut self,
        ctx: &egui::Context,
        world: &mut World,
        physics: &mut PhysicsSystem,
        selected_entity: &mut Option<EntityId>,
        bloom_threshold: &mut f32,
        fps: f32,
        is_playing: bool,
    ) -> Vec<EditorAction> {
        let mut actions = Vec::new();

        // Check if a file dialog completed
        if let Some(rx) = &self.file_dialog_receiver {
            if let Ok(path) = rx.try_recv() {
                actions.push(EditorAction::SpawnModel(path));
                self.file_dialog_receiver = None; // Reset after receiving
            }
        }

        // --- Top Menu Bar ---
        let top_frame = egui::Frame::default()
            .fill(ctx.style().visuals.panel_fill)
            .inner_margin(4.0);
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
                    if ui.button("Exit").clicked() { std::process::exit(0); }
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
                        if ui.button("⏸ Pause").clicked() { actions.push(EditorAction::Pause); }
                    } else {
                        if ui.button("▶ Play").clicked() { actions.push(EditorAction::Play); }
                    }
                });
            });
        });

        // --- Bottom Panel ---
        let bottom_frame = egui::Frame::default()
            .fill(ctx.style().visuals.window_fill)
            .stroke(ctx.style().visuals.window_stroke())
            .inner_margin(4.0);
        egui::TopBottomPanel::bottom("bottom_panel")
            .frame(bottom_frame)
            .resizable(true)
            .min_height(150.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.selectable_label(self.bottom_tab == BottomTab::Console, "Console").clicked() { self.bottom_tab = BottomTab::Console; }
                    if ui.selectable_label(self.bottom_tab == BottomTab::AssetBrowser, "Asset Browser").clicked() { self.bottom_tab = BottomTab::AssetBrowser; }
                    if ui.selectable_label(self.bottom_tab == BottomTab::Profiler, "Profiler").clicked() { self.bottom_tab = BottomTab::Profiler; }
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    match self.bottom_tab {
                        BottomTab::Console => {
                            ui.label("[System] Engine Booted Successfully.");
                            ui.label("[Memory] Initialized 256MB arena block.");
                            ui.label("[HotReload] game.dll attached.");
                        }
                        BottomTab::AssetBrowser => {
                            ui.label("Asset Browser is not yet implemented.");
                        }
                        BottomTab::Profiler => {
                            ui.label(format!("Current FPS: {:.1}", fps));
                            ui.label(format!("Frame Time: {:.2} ms", 1000.0 / fps));
                            ui.label("More profiling metrics coming soon...");
                        }
                    }
                });
            });

        // --- Left Panel (Hierarchy) ---
        egui::SidePanel::left("hierarchy_panel")
            .resizable(true)
            .min_width(200.0)
            .show(ctx, |ui| {
                ui.heading("Hierarchy");
                ui.label(format!("FPS: {:.1}", fps));
                ui.separator();
                if ui.button("Add Entity").clicked() {
                    if self.file_dialog_receiver.is_none() {
                        let (tx, rx) = std::sync::mpsc::channel();
                        self.file_dialog_receiver = Some(rx);
                        std::thread::spawn(move || {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("3D Models", &["obj", "gltf", "glb"])
                                .pick_file()
                            {
                                let _ = tx.send(path.to_string_lossy().into_owned());
                            }
                        });
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
                        Self::draw_entity_tree(ui, root, &children_map, selected_entity, world);
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
        
        actions
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
