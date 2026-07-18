use crate::ecs::reflection::ComponentRegistry;
use crate::ecs::{EntityId, World};
use crate::physics::PhysicsSystem;
use crate::ui::UiContext;
use crate::scripting::visual_graph::VisualGraph;

#[derive(PartialEq, Clone, Copy)]
pub enum EditorTab {
    Scene3D,
    ScriptGraph,
    Profiler,
}

pub enum EditorAction {
    Play,
    Pause,
    SpawnModel(String),
    SpawnEntity,
    DeleteEntity(EntityId),
    AddComponent(EntityId, String),
    TranslateSelected(crate::math::vec::Vec3),
    ToggleDebugCull,
    SpawnStressTest,
}

pub struct Editor {
    pub registry: ComponentRegistry,
    pub file_dialog_receiver: Option<std::sync::mpsc::Receiver<String>>,
    pub file_dialog_sender: Option<std::sync::mpsc::Sender<String>>,
    pub active_gizmo_axis: Option<u8>,
    pub gizmo_drag_start_mouse: crate::math::vec::Vec2,
    pub gizmo_drag_start_pos: crate::math::vec::Vec3,
    
    // Node Graph state
    pub active_tab: EditorTab,
    pub graph: VisualGraph,
    pub graph_pan: crate::math::vec::Vec2,
    pub dragging_node: Option<u32>,
    pub connecting_from: Option<(u32, u8)>, // (node_id, port_id)
    pub inspector_scroll: f32,
    pub hierarchy_scroll: f32,
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
        registry.register_serializable::<crate::ecs::TransformComponent>();

        registry.register::<crate::ecs::RenderComponent>();
        registry.register_serializable::<crate::ecs::RenderComponent>();

        registry.register::<crate::ecs::PointLightComponent>();
        registry.register_serializable::<crate::ecs::PointLightComponent>();

        registry.register::<crate::ecs::components::CameraComponent>();
        registry.register_serializable::<crate::ecs::components::CameraComponent>();

        registry.register::<crate::ecs::components::HierarchyComponent>();
        registry.register_serializable::<crate::ecs::components::HierarchyComponent>();

        registry.register::<crate::ecs::components::RigidBodyComponent>();
        registry.register::<crate::ecs::components::ColliderComponent>();
        registry.register::<crate::ecs::components::AudioEmitterComponent>();
        registry.register::<crate::ecs::components::AudioListenerComponent>();
        registry.register::<crate::ecs::components::SkeletonComponent>();
        registry.register::<crate::ecs::components::ScriptBehaviorComponent>();
        registry.register::<crate::ecs::components::AnimatorComponent>();

        let (tx, rx) = std::sync::mpsc::channel();
        Self {
            registry,
            file_dialog_receiver: Some(rx),
            file_dialog_sender: Some(tx),
            active_gizmo_axis: None,
            gizmo_drag_start_mouse: crate::math::vec::Vec2::new(0.0, 0.0),
            gizmo_drag_start_pos: crate::math::vec::Vec3::new(0.0, 0.0, 0.0),
            active_tab: EditorTab::Scene3D,
            graph: VisualGraph::new(),
            graph_pan: crate::math::vec::Vec2::new(0.0, 0.0),
            dragging_node: None,
            connecting_from: None,
            inspector_scroll: 0.0,
            hierarchy_scroll: 0.0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        ui_ctx: &mut UiContext,
        world: &mut World,
        _physics: &mut PhysicsSystem,
        selected_entity: &mut Option<EntityId>,
        _bloom_threshold: &mut f32,
        _fps: f32,
        last_visible_meshlets: u32,
        _is_playing: bool,
        screen_w: f32,
        screen_h: f32,
        view: crate::math::mat4::Mat4,
        proj: crate::math::mat4::Mat4,
    ) -> (
        Vec<EditorAction>,
        Option<(u32, u32)>,
        Option<(f32, f32)>,
        bool,
    ) {
        let mut actions = Vec::new();
        let mut raycast_request = None;
        let mut viewport_hovered = false;
        let mut new_viewport_size = None;

        // Check if a file dialog completed
        if let Some(rx) = &self.file_dialog_receiver {
            if let Ok(path) = rx.try_recv() {
                actions.push(EditorAction::SpawnModel(path));
            }
        }

        use crate::ui::context::{
            ButtonBuilder, PanelBuilder, UiRect, SLATE_BASE, SLATE_SECONDARY,
        };

        // Render the stats in top right or left
        let stats_rect = UiRect {
            x: screen_w - 200.0,
            y: 0.0,
            w: 200.0,
            h: 40.0,
        };
        PanelBuilder::new(ui_ctx, 100)
            .rect(stats_rect)
            .style(&SLATE_SECONDARY)
            .begin();
        ui_ctx.begin_vertical_layout(stats_rect);
        
        let mut stats_label = crate::containers::FixedString::<128>::new();
        use core::fmt::Write;
        let _ = write!(stats_label, "FPS: {:.1} | Meshlets: {}", _fps, last_visible_meshlets);
        ui_ctx.label(&stats_label);
        
        ui_ctx.end_vertical_layout();
        ui_ctx.end_panel();

        // 1. Top Bar (Godot style)
        let top_bar_rect = UiRect {
            x: 0.0,
            y: 0.0,
            w: screen_w,
            h: 40.0,
        };
        PanelBuilder::new(ui_ctx, 1)
            .rect(top_bar_rect)
            .style(&SLATE_SECONDARY)
            .begin();

        ui_ctx.begin_horizontal_layout(top_bar_rect);
        let scene_style = if self.active_tab == EditorTab::Scene3D { &SLATE_SECONDARY } else { &SLATE_BASE };
        if ButtonBuilder::new(ui_ctx, 2).text("3D Scene").style(scene_style).build() {
            self.active_tab = EditorTab::Scene3D;
        }
        let graph_style = if self.active_tab == EditorTab::ScriptGraph { &SLATE_SECONDARY } else { &SLATE_BASE };
        if ButtonBuilder::new(ui_ctx, 3).text("Script Graph").style(graph_style).build() {
            self.active_tab = EditorTab::ScriptGraph;
        }
        let profiler_style = if self.active_tab == EditorTab::Profiler { &SLATE_SECONDARY } else { &SLATE_BASE };
        if ButtonBuilder::new(ui_ctx, 4).text("Profiler").style(profiler_style).build() {
            self.active_tab = EditorTab::Profiler;
        }
        if ButtonBuilder::new(ui_ctx, 11)
            .text("Import Model")
            .style(&SLATE_BASE)
            .build()
        {
            if let Some(tx) = &self.file_dialog_sender {
                let tx = tx.clone();
                std::thread::spawn(move || {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Models", &["obj", "gltf", "glb"])
                        .pick_file()
                    {
                        let _ = tx.send(path.to_string_lossy().to_string());
                    }
                });
            }
        }

        // Push play/pause to the center roughly
        let center_offset = (screen_w / 2.0) - ui_ctx.cursor.x - 40.0;
        if center_offset > 0.0 {
            ui_ctx.indent_cursor(center_offset);
        }

        if _is_playing {
            if ButtonBuilder::new(ui_ctx, 5)
                .text("Pause")
                .style(&SLATE_BASE)
                .build()
            {
                actions.push(EditorAction::Pause);
            }
        } else {
            if ButtonBuilder::new(ui_ctx, 6)
                .text("Play")
                .style(&SLATE_BASE)
                .build()
            {
                actions.push(EditorAction::Play);
            }
        }
        ui_ctx.end_horizontal_layout();
        ui_ctx.end_panel();

        if self.active_tab == EditorTab::Scene3D {
        // 2. Left Scene Hierarchy
        let hierarchy_w = 250.0;
        let hierarchy_rect = UiRect {
            x: 0.0,
            y: 40.0,
            w: hierarchy_w,
            h: screen_h - 40.0,
        };

        if hierarchy_rect.contains(ui_ctx.mouse_pos) {
            self.hierarchy_scroll -= ui_ctx.mouse_scroll_y * 30.0;
            if self.hierarchy_scroll < 0.0 {
                self.hierarchy_scroll = 0.0;
            }
        }

        PanelBuilder::new(ui_ctx, 7)
            .rect(hierarchy_rect)
            .style(&SLATE_SECONDARY)
            .begin();
            
        ui_ctx.draw_commands.push(crate::ui::context::DrawCommand::SetScissor { rect: Some(hierarchy_rect) });

        ui_ctx.begin_vertical_layout(hierarchy_rect);
        ui_ctx.cursor.y -= self.hierarchy_scroll;
        ui_ctx.label("Scene Hierarchy");
        ui_ctx.label_color("RED TEXT TEST", crate::ui::UiColor::rgba(255, 0, 0, 255));
        ui_ctx.label(" ");

        if ButtonBuilder::new(ui_ctx, 100)
            .text("Create Empty Entity")
            .style(&SLATE_BASE)
            .build()
        {
            actions.push(EditorAction::SpawnEntity);
        }
        ui_ctx.label(" ");

        if ButtonBuilder::new(ui_ctx, 105)
            .text("Toggle Debug Culling")
            .style(&SLATE_BASE)
            .build()
        {
            actions.push(EditorAction::ToggleDebugCull);
        }
        ui_ctx.label(" ");

        if ButtonBuilder::new(ui_ctx, 106)
            .text("10,000 Object Stress Test")
            .style(&SLATE_BASE)
            .build()
        {
            actions.push(EditorAction::SpawnStressTest);
        }
        ui_ctx.label(" ");

        let mut all_entities = [0; 1024];
        let alive_count = world.get_alive_entities(&mut all_entities);
        let alive = &all_entities[..alive_count];

        for &entity in alive {
            let mut is_root = true;
            if world.has_component::<crate::ecs::components::HierarchyComponent>(entity) {
                let parent = unsafe {
                    world
                        .get_component::<crate::ecs::components::HierarchyComponent>(entity)
                        .parent
                };
                if parent.is_some() {
                    is_root = false;
                }
            }

            if is_root {
                Self::draw_entity_tree(ui_ctx, entity, selected_entity, world, alive, 0);
            }
        }

        ui_ctx.end_vertical_layout();
        ui_ctx.draw_commands.push(crate::ui::context::DrawCommand::SetScissor { rect: None });
        ui_ctx.end_panel();

        // 3. Right Inspector (Unity style)
        let inspector_w = 300.0;
        let inspector_rect = UiRect {
            x: screen_w - inspector_w,
            y: 40.0,
            w: inspector_w,
            h: screen_h - 40.0,
        };

        if inspector_rect.contains(ui_ctx.mouse_pos) {
            self.inspector_scroll -= ui_ctx.mouse_scroll_y * 30.0;
            if self.inspector_scroll < 0.0 {
                self.inspector_scroll = 0.0;
            }
        }

        PanelBuilder::new(ui_ctx, 8)
            .rect(inspector_rect)
            .style(&SLATE_SECONDARY)
            .begin();
            
        ui_ctx.draw_commands.push(crate::ui::context::DrawCommand::SetScissor { rect: Some(inspector_rect) });

        ui_ctx.begin_vertical_layout(inspector_rect);
        ui_ctx.cursor.y -= self.inspector_scroll;
        ui_ctx.label("Inspector");

        if let Some(entity_id) = *selected_entity {
            ui_ctx.label(" "); // spacing
            use core::fmt::Write;
            let mut label = crate::containers::FixedString::<128>::new();
            let _ = write!(label, "Entity ID: {}", entity_id);
            ui_ctx.label(&label);

            ui_ctx.label(" ");
            let mut bloom_exp = true;
            if ui_ctx.collapsing_header("Post Processing", &mut bloom_exp) {
                ui_ctx.drag_float("Bloom Threshold", _bloom_threshold);
            }

            ui_ctx.label(" ");
            self.registry
                .draw_entity(entity_id, world, ui_ctx, _physics);

            ui_ctx.label(" ");
            if ButtonBuilder::new(ui_ctx, 101)
                .text("Delete Entity")
                .style(&SLATE_BASE)
                .build()
            {
                actions.push(EditorAction::DeleteEntity(entity_id));
            }
            ui_ctx.label(" ");
            ui_ctx.label("--- Add Component ---");
            for comp_name in &self.registry.component_names {
                // simple buttons for each component
                if ButtonBuilder::new(ui_ctx, 200 + (comp_name.as_ptr() as u64 % 1000))
                    .text(comp_name)
                    .style(&SLATE_BASE)
                    .build()
                {
                    actions.push(EditorAction::AddComponent(entity_id, comp_name.clone()));
                }
            }
        } else {
            ui_ctx.label("No Entity Selected.");
        }

        ui_ctx.end_vertical_layout();
        ui_ctx.draw_commands.push(crate::ui::context::DrawCommand::SetScissor { rect: None });
        ui_ctx.end_panel();

        // 4. Center Viewport (Unreal style)
        let viewport_rect = UiRect {
            x: hierarchy_w,
            y: 40.0,
            w: screen_w - hierarchy_w - inspector_w,
            h: screen_h - 40.0,
        };

        ui_ctx.image(viewport_rect, 0); // 0 is offscreen_texture_id

        // If mouse is inside viewport and clicked, trigger raycast
        if viewport_rect.contains(ui_ctx.mouse_pos) {
            viewport_hovered = true;
            if ui_ctx.mouse_pressed {
                // Convert mouse pos to Normalized Device Coordinates for the raycast
                let local_x = ui_ctx.mouse_pos.x - viewport_rect.x;
                let local_y = ui_ctx.mouse_pos.y - viewport_rect.y;
                let ndc_x = (local_x / viewport_rect.w) * 2.0 - 1.0;
                let ndc_y = (local_y / viewport_rect.h) * 2.0 - 1.0;
                raycast_request = Some((ndc_x, ndc_y));
            }
        }

        new_viewport_size = Some((viewport_rect.w as u32, viewport_rect.h as u32));

        // --- 3D GIZMOS ---
        if let Some(entity) = selected_entity {
            let transforms =
                world.get_component_array::<crate::ecs::components::TransformComponent>();
            if transforms.has(*entity) {
                let transform = unsafe { transforms.get(*entity) };

                // Helper to project 3D to 2D
                let project = |pos: crate::math::vec::Vec3| -> Option<crate::math::vec::Vec2> {
                    let mut p = crate::math::vec::Vec4::new(pos.x, pos.y, pos.z, 1.0);
                    p = view * p;
                    p = proj * p;
                    if p.w <= 0.0 {
                        return None;
                    }
                    let ndc_x = p.x / p.w;
                    let ndc_y = p.y / p.w;

                    let screen_x = viewport_rect.x + (ndc_x * 0.5 + 0.5) * viewport_rect.w;
                    let screen_y = viewport_rect.y + (ndc_y * 0.5 + 0.5) * viewport_rect.h;
                    Some(crate::math::vec::Vec2::new(screen_x, screen_y))
                };

                let origin_2d = project(transform.position);

                if let Some(o_2d) = origin_2d {
                    // Check if origin_2d is inside the viewport rect
                    if viewport_rect.contains(o_2d) || self.active_gizmo_axis.is_some() {
                        let axis_len = 2.0;
                        let x_pos =
                            transform.position + crate::math::vec::Vec3::new(axis_len, 0.0, 0.0);
                        let y_pos =
                            transform.position + crate::math::vec::Vec3::new(0.0, axis_len, 0.0);
                        let z_pos =
                            transform.position + crate::math::vec::Vec3::new(0.0, 0.0, axis_len);

                        let x_2d = project(x_pos);
                        let y_2d = project(y_pos);
                        let z_2d = project(z_pos);

                        let mut hovering_axis = None;

                        let dist_to_line = |p: crate::math::vec::Vec2,
                                            a: crate::math::vec::Vec2,
                                            b: crate::math::vec::Vec2|
                         -> f32 {
                            let l2 = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y);
                            if l2 == 0.0 {
                                return ((p.x - a.x) * (p.x - a.x) + (p.y - a.y) * (p.y - a.y))
                                    .sqrt();
                            }
                            let t = ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / l2;
                            let t = t.clamp(0.0, 1.0);
                            let proj_x = a.x + t * (b.x - a.x);
                            let proj_y = a.y + t * (b.y - a.y);
                            ((p.x - proj_x) * (p.x - proj_x) + (p.y - proj_y) * (p.y - proj_y))
                                .sqrt()
                        };

                        if let Some(x2) = x_2d {
                            if dist_to_line(ui_ctx.mouse_pos, o_2d, x2) < 15.0 {
                                hovering_axis = Some(0);
                            }
                            let c = if hovering_axis == Some(0) || self.active_gizmo_axis == Some(0)
                            {
                                crate::ui::UiColor::rgb(255, 100, 100)
                            } else {
                                crate::ui::UiColor::rgb(200, 50, 50)
                            };
                            ui_ctx.add_line(
                                o_2d,
                                x2,
                                c,
                                if hovering_axis == Some(0) || self.active_gizmo_axis == Some(0) {
                                    4.0
                                } else {
                                    2.0
                                },
                            );
                        }
                        if let Some(y2) = y_2d {
                            if dist_to_line(ui_ctx.mouse_pos, o_2d, y2) < 15.0 {
                                hovering_axis = Some(1);
                            }
                            let c = if hovering_axis == Some(1) || self.active_gizmo_axis == Some(1)
                            {
                                crate::ui::UiColor::rgb(100, 255, 100)
                            } else {
                                crate::ui::UiColor::rgb(50, 200, 50)
                            };
                            ui_ctx.add_line(
                                o_2d,
                                y2,
                                c,
                                if hovering_axis == Some(1) || self.active_gizmo_axis == Some(1) {
                                    4.0
                                } else {
                                    2.0
                                },
                            );
                        }
                        if let Some(z2) = z_2d {
                            if dist_to_line(ui_ctx.mouse_pos, o_2d, z2) < 15.0 {
                                hovering_axis = Some(2);
                            }
                            let c = if hovering_axis == Some(2) || self.active_gizmo_axis == Some(2)
                            {
                                crate::ui::UiColor::rgb(100, 100, 255)
                            } else {
                                crate::ui::UiColor::rgb(50, 50, 200)
                            };
                            ui_ctx.add_line(
                                o_2d,
                                z2,
                                c,
                                if hovering_axis == Some(2) || self.active_gizmo_axis == Some(2) {
                                    4.0
                                } else {
                                    2.0
                                },
                            );
                        }

                        // Center dot
                        ui_ctx.draw_commands.push(crate::ui::DrawCommand::Quad {
                            rect: crate::ui::UiRect {
                                x: o_2d.x - 4.0,
                                y: o_2d.y - 4.0,
                                w: 8.0,
                                h: 8.0,
                            },
                            color: crate::ui::UiColor::WHITE,
                            rounding: 4.0,
                        });

                        // Interaction
                        if ui_ctx.mouse_pressed && hovering_axis.is_some() {
                            self.active_gizmo_axis = hovering_axis;
                            self.gizmo_drag_start_mouse = ui_ctx.mouse_pos;
                            self.gizmo_drag_start_pos = transform.position;
                            raycast_request = None; // block raycast selection
                        }

                        if !ui_ctx.mouse_down {
                            self.active_gizmo_axis = None;
                        }

                        if let Some(axis) = self.active_gizmo_axis {
                            let delta = crate::math::vec::Vec2::new(
                                ui_ctx.mouse_pos.x - self.gizmo_drag_start_mouse.x,
                                ui_ctx.mouse_pos.y - self.gizmo_drag_start_mouse.y,
                            );

                            let sensitivity = 0.02;
                            let mut t_delta = crate::math::vec::Vec3::new(0.0, 0.0, 0.0);

                            if axis == 0 {
                                let projected_dir = if let Some(x2) = x_2d {
                                    crate::math::vec::Vec2::new(x2.x - o_2d.x, x2.y - o_2d.y)
                                } else {
                                    crate::math::vec::Vec2::new(1.0, 0.0)
                                };
                                let len = (projected_dir.x * projected_dir.x
                                    + projected_dir.y * projected_dir.y)
                                    .sqrt();
                                let norm = if len > 0.0 {
                                    crate::math::vec::Vec2::new(
                                        projected_dir.x / len,
                                        projected_dir.y / len,
                                    )
                                } else {
                                    crate::math::vec::Vec2::new(1.0, 0.0)
                                };
                                let move_amt = (delta.x * norm.x + delta.y * norm.y) * sensitivity;
                                t_delta.x = move_amt;
                            } else if axis == 1 {
                                let projected_dir = if let Some(y2) = y_2d {
                                    crate::math::vec::Vec2::new(y2.x - o_2d.x, y2.y - o_2d.y)
                                } else {
                                    crate::math::vec::Vec2::new(0.0, -1.0)
                                };
                                let len = (projected_dir.x * projected_dir.x
                                    + projected_dir.y * projected_dir.y)
                                    .sqrt();
                                let norm = if len > 0.0 {
                                    crate::math::vec::Vec2::new(
                                        projected_dir.x / len,
                                        projected_dir.y / len,
                                    )
                                } else {
                                    crate::math::vec::Vec2::new(0.0, -1.0)
                                };
                                let move_amt = (delta.x * norm.x + delta.y * norm.y) * sensitivity;
                                t_delta.y = move_amt;
                            } else if axis == 2 {
                                let projected_dir = if let Some(z2) = z_2d {
                                    crate::math::vec::Vec2::new(z2.x - o_2d.x, z2.y - o_2d.y)
                                } else {
                                    crate::math::vec::Vec2::new(1.0, 1.0)
                                };
                                let len = (projected_dir.x * projected_dir.x
                                    + projected_dir.y * projected_dir.y)
                                    .sqrt();
                                let norm = if len > 0.0 {
                                    crate::math::vec::Vec2::new(
                                        projected_dir.x / len,
                                        projected_dir.y / len,
                                    )
                                } else {
                                    crate::math::vec::Vec2::new(1.0, 1.0)
                                };
                                let move_amt = (delta.x * norm.x + delta.y * norm.y) * sensitivity;
                                t_delta.z = move_amt;
                            }

                            let new_pos = self.gizmo_drag_start_pos + t_delta;
                            actions.push(EditorAction::TranslateSelected(new_pos));
                            raycast_request = None; // block raycast when dragging
                        }
                    }
                }
            }
        }
        } else if self.active_tab == EditorTab::ScriptGraph {
            // Full-screen canvas below the top bar
            let canvas_rect = UiRect {
                x: 0.0,
                y: 40.0,
                w: screen_w,
                h: screen_h - 40.0,
            };
            PanelBuilder::new(ui_ctx, 1000)
                .rect(canvas_rect)
                .style(&SLATE_SECONDARY)
                .begin();

            ui_ctx.begin_vertical_layout(canvas_rect);
            ui_ctx.label("Visual Scripting Node Graph");
            
            if ButtonBuilder::new(ui_ctx, 1001).text("Add Node (Update)").style(&SLATE_BASE).build() {
                let id = self.graph.nodes.len() as u32 + 1;
                self.graph.nodes.push(crate::scripting::visual_graph::GraphNode {
                    id,
                    node_type: crate::scripting::visual_graph::NodeType::OnUpdate,
                    x: self.graph_pan.x + 100.0 + (id as f32 * 20.0),
                    y: self.graph_pan.y + 100.0,
                    inputs: vec![],
                    outputs: vec![crate::scripting::visual_graph::NodePort {
                        name: "Flow Out".to_string(),
                        data_type: crate::scripting::visual_graph::PortType::Flow,
                    }],
                    param_value: 0.0,
                });
            }
            if ButtonBuilder::new(ui_ctx, 1002).text("Add Node (Print)").style(&SLATE_BASE).build() {
                let id = self.graph.nodes.len() as u32 + 1;
                self.graph.nodes.push(crate::scripting::visual_graph::GraphNode {
                    id,
                    node_type: crate::scripting::visual_graph::NodeType::Print,
                    x: self.graph_pan.x + 300.0 + (id as f32 * 20.0),
                    y: self.graph_pan.y + 100.0,
                    inputs: vec![crate::scripting::visual_graph::NodePort {
                        name: "Flow In".to_string(),
                        data_type: crate::scripting::visual_graph::PortType::Flow,
                    }],
                    outputs: vec![crate::scripting::visual_graph::NodePort {
                        name: "Flow Out".to_string(),
                        data_type: crate::scripting::visual_graph::PortType::Flow,
                    }],
                    param_value: 0.0,
                });
            }

            let mut hovered_node = None;
            for node in self.graph.nodes.iter_mut().rev() {
                let node_rect = UiRect {
                    x: canvas_rect.x + node.x,
                    y: canvas_rect.y + node.y,
                    w: 120.0,
                    h: 60.0 + (node.inputs.len().max(node.outputs.len()) as f32 * 20.0),
                };
                
                if node_rect.contains(ui_ctx.mouse_pos) {
                    hovered_node = Some(node.id);
                    if ui_ctx.mouse_pressed {
                        self.dragging_node = Some(node.id);
                    }
                    break;
                }
            }

            if ui_ctx.mouse_released {
                self.dragging_node = None;
            }

            if let Some(drag_id) = self.dragging_node {
                if let Some(node) = self.graph.nodes.iter_mut().find(|n| n.id == drag_id) {
                    node.x += ui_ctx.mouse_delta.x;
                    node.y += ui_ctx.mouse_delta.y;
                }
            } else if ui_ctx.mouse_down && hovered_node.is_none() && canvas_rect.contains(ui_ctx.mouse_pos) {
                // Pan canvas
                self.graph_pan.x += ui_ctx.mouse_delta.x;
                self.graph_pan.y += ui_ctx.mouse_delta.y;
            }

            // Draw Nodes
            for node in &self.graph.nodes {
                let node_rect = UiRect {
                    x: canvas_rect.x + node.x,
                    y: canvas_rect.y + node.y,
                    w: 120.0,
                    h: 60.0 + (node.inputs.len().max(node.outputs.len()) as f32 * 20.0),
                };
                
                let is_dragged = self.dragging_node == Some(node.id);
                let bg_color = if is_dragged {
                    crate::ui::context::UiColor::rgb(80, 80, 80)
                } else {
                    crate::ui::context::UiColor::rgb(50, 50, 50)
                };

                ui_ctx.draw_commands.push(crate::ui::context::DrawCommand::Quad {
                    rect: node_rect,
                    color: bg_color,
                    rounding: 4.0,
                });
                
                let mut title = crate::containers::FixedString::<128>::new();
                use core::fmt::Write;
                let _ = write!(title, "{:?}", node.node_type);
                ui_ctx.draw_commands.push(crate::ui::context::DrawCommand::Text {
                    pos: crate::math::vec::Vec2::new(node_rect.x + 10.0, node_rect.y + 10.0),
                    text: title,
                    color: crate::ui::context::UiColor::WHITE,
                    font_size: 16.0,
                });

                // Draw ports (stub)
                for (i, _in_port) in node.inputs.iter().enumerate() {
                    ui_ctx.draw_commands.push(crate::ui::context::DrawCommand::Quad {
                        rect: UiRect { x: node_rect.x - 4.0, y: node_rect.y + 40.0 + (i as f32 * 20.0), w: 8.0, h: 8.0 },
                        color: crate::ui::context::UiColor::rgb(200, 200, 50),
                        rounding: 4.0,
                    });
                }
                for (i, _out_port) in node.outputs.iter().enumerate() {
                    ui_ctx.draw_commands.push(crate::ui::context::DrawCommand::Quad {
                        rect: UiRect { x: node_rect.x + node_rect.w - 4.0, y: node_rect.y + 40.0 + (i as f32 * 20.0), w: 8.0, h: 8.0 },
                        color: crate::ui::context::UiColor::rgb(200, 200, 50),
                        rounding: 4.0,
                    });
                }
            }
            
            ui_ctx.end_vertical_layout();
            ui_ctx.end_panel();
        }

        (
            actions,
            new_viewport_size,
            raycast_request,
            viewport_hovered,
        )
    }

    fn draw_entity_tree(
        ui_ctx: &mut crate::ui::UiContext,
        entity: crate::ecs::EntityId,
        selected_entity: &mut Option<crate::ecs::EntityId>,
        world: &crate::ecs::World,
        all_alive: &[crate::ecs::EntityId],
        depth: u32,
    ) {
        use core::fmt::Write;

        let mut label = crate::containers::FixedString::<128>::new();

        let mut icon = "[O]";
        if world.has_component::<crate::ecs::components::CameraComponent>(entity) {
            icon = "[C]";
        } else if world.has_component::<crate::ecs::components::PointLightComponent>(entity) {
            icon = "[L]";
        } else if world.has_component::<crate::ecs::components::RenderComponent>(entity) {
            icon = "[M]";
        } else if world.has_component::<crate::ecs::components::AudioEmitterComponent>(entity) {
            icon = "[A]";
        }

        let _ = write!(label, "{} Entity {}", icon, entity);
        let is_selected = *selected_entity == Some(entity);

        let depth_offset = depth as f32 * 16.0;
        ui_ctx.indent_cursor(depth_offset);
        if ui_ctx.selectable_label(&label, is_selected) {
            *selected_entity = Some(entity);
        }
        ui_ctx.unindent_cursor(depth_offset);

        for &child in all_alive {
            if world.has_component::<crate::ecs::components::HierarchyComponent>(child) {
                let child_parent = unsafe {
                    world
                        .get_component::<crate::ecs::components::HierarchyComponent>(child)
                        .parent
                };
                if child_parent == Some(entity) {
                    Self::draw_entity_tree(
                        ui_ctx,
                        child,
                        selected_entity,
                        world,
                        all_alive,
                        depth + 1,
                    );
                }
            }
        }
    }
}
