use crate::containers::FixedArray;
use egui::{Rect, Vec2, Stroke, Color32};
use crate::app::slate_theme::EngineTheme;

#[derive(Clone, Copy, PartialEq)]
pub enum SplitDirection {
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy, PartialEq)]
pub enum PanelType {
    None,
    Viewport,
    Hierarchy,
    Inspector,
    NodeGraph,
    Console,
}

#[derive(Clone)]
pub struct DockNode {
    pub id: usize,
    pub parent_id: Option<usize>,
    pub split_dir: Option<SplitDirection>,
    pub split_ratio: f32,
    pub child_a: Option<usize>,
    pub child_b: Option<usize>,
    pub tabs: FixedArray<PanelType, 4>,
    pub active_tab: usize,
}

pub struct DockingManager {
    pub nodes: FixedArray<DockNode, 64>,
    pub root_id: Option<usize>,
    pub dragging_split: Option<usize>,
}

impl Default for DockingManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DockingManager {
    pub fn new() -> Self {
        let mut manager = Self {
            nodes: FixedArray::new(),
            root_id: None,
            dragging_split: None,
        };

        // Create layout
        let n_hierarchy = manager.add_leaf();
        manager.add_tab_to_leaf(n_hierarchy, PanelType::Hierarchy);

        let n_viewport = manager.add_leaf();
        manager.add_tab_to_leaf(n_viewport, PanelType::Viewport); // 3D Viewport in center

        let n_inspector = manager.add_leaf();
        manager.add_tab_to_leaf(n_inspector, PanelType::Inspector);

        let n_bottom = manager.add_leaf();
        manager.add_tab_to_leaf(n_bottom, PanelType::Console);
        manager.add_tab_to_leaf(n_bottom, PanelType::NodeGraph);

        let n_split2 = manager.add_split(SplitDirection::Horizontal, 0.7, n_viewport, n_inspector);
        let n_split1 = manager.add_split(SplitDirection::Vertical, 0.7, n_split2, n_bottom);
        let n_root = manager.add_split(SplitDirection::Horizontal, 0.2, n_hierarchy, n_split1);

        manager.root_id = Some(n_root);
        manager
    }

    fn add_leaf(&mut self) -> usize {
        let id = self.nodes.len();
        self.nodes.push(DockNode {
            id,
            parent_id: None,
            split_dir: None,
            split_ratio: 0.5,
            child_a: None,
            child_b: None,
            tabs: FixedArray::new(),
            active_tab: 0,
        });
        id
    }

    pub fn add_tab_to_leaf(&mut self, leaf_id: usize, panel: PanelType) {
        let node = &mut self.nodes.as_mut_slice()[leaf_id];
        if node.tabs.len() < 4 {
            node.tabs.push(panel);
        }
    }

    fn add_split(&mut self, dir: SplitDirection, ratio: f32, child_a: usize, child_b: usize) -> usize {
        let id = self.nodes.len();
        self.nodes.push(DockNode {
            id,
            parent_id: None,
            split_dir: Some(dir),
            split_ratio: ratio,
            child_a: Some(child_a),
            child_b: Some(child_b),
            tabs: FixedArray::new(),
            active_tab: 0,
        });
        
        self.nodes.as_mut_slice()[child_a].parent_id = Some(id);
        self.nodes.as_mut_slice()[child_b].parent_id = Some(id);
        id
    }

    pub fn draw(&mut self, ui: &mut egui::Ui, theme: &EngineTheme, mut draw_panel: impl FnMut(PanelType, &mut egui::Ui, Rect)) {
        let rect = ui.available_rect_before_wrap();
        
        // Render nodes recursively
        if let Some(root_id) = self.root_id {
            self.draw_node(root_id, rect, ui, theme, &mut draw_panel);
        }
    }

    fn draw_node(&mut self, id: usize, rect: Rect, ui: &mut egui::Ui, theme: &EngineTheme, draw_panel: &mut impl FnMut(PanelType, &mut egui::Ui, Rect)) {
        let node = self.nodes.as_slice()[id].clone();
        
        if let Some(dir) = node.split_dir {
            let split_thickness = 4.0;
            let (rect_a, rect_b, split_rect) = match dir {
                SplitDirection::Horizontal => {
                    let w = rect.width() * node.split_ratio;
                    let r_a = Rect::from_min_size(rect.min, Vec2::new(w - split_thickness * 0.5, rect.height()));
                    let r_b = Rect::from_min_size(rect.min + Vec2::new(w + split_thickness * 0.5, 0.0), Vec2::new(rect.width() - w - split_thickness * 0.5, rect.height()));
                    let s_r = Rect::from_min_size(rect.min + Vec2::new(w - split_thickness * 0.5, 0.0), Vec2::new(split_thickness, rect.height()));
                    (r_a, r_b, s_r)
                },
                SplitDirection::Vertical => {
                    let h = rect.height() * node.split_ratio;
                    let r_a = Rect::from_min_size(rect.min, Vec2::new(rect.width(), h - split_thickness * 0.5));
                    let r_b = Rect::from_min_size(rect.min + Vec2::new(0.0, h + split_thickness * 0.5), Vec2::new(rect.width(), rect.height() - h - split_thickness * 0.5));
                    let s_r = Rect::from_min_size(rect.min + Vec2::new(0.0, h - split_thickness * 0.5), Vec2::new(rect.width(), split_thickness));
                    (r_a, r_b, s_r)
                }
            };

            // Draw children
            if let Some(child_a) = node.child_a {
                self.draw_node(child_a, rect_a, ui, theme, draw_panel);
            }
            if let Some(child_b) = node.child_b {
                self.draw_node(child_b, rect_b, ui, theme, draw_panel);
            }

            // Draw splitter
            ui.painter().rect_filled(split_rect, 0.0, theme.bg_extreme);
            let response = ui.interact(split_rect, ui.id().with(id), egui::Sense::drag());
            
            if response.hovered() {
                ui.output_mut(|o| o.cursor_icon = if dir == SplitDirection::Horizontal { egui::CursorIcon::ResizeHorizontal } else { egui::CursorIcon::ResizeVertical });
            }

            if response.dragged() {
                let delta = response.drag_delta();
                let mut new_ratio = node.split_ratio;
                match dir {
                    SplitDirection::Horizontal => {
                        new_ratio += delta.x / rect.width();
                    },
                    SplitDirection::Vertical => {
                        new_ratio += delta.y / rect.height();
                    }
                }
                new_ratio = new_ratio.clamp(0.05, 0.95);
                self.nodes.as_mut_slice()[id].split_ratio = new_ratio;
            }

        } else {
            // Leaf node: Draw panel background and border
            
            let active_panel = if node.tabs.len() > 0 {
                node.tabs[node.active_tab]
            } else {
                PanelType::None
            };
            
            let bg_color = if active_panel == PanelType::Viewport {
                Color32::TRANSPARENT
            } else {
                theme.bg_panel
            };
            
            ui.painter().rect_filled(rect, 0.0, bg_color);
            if active_panel != PanelType::Viewport {
                ui.painter().rect_stroke(rect, 0.0, Stroke::new(1.0, theme.stroke));
            }
            
            // Draw Panel Header (Tabs)
            let header_height = 24.0;
            let header_rect = Rect::from_min_size(rect.min, Vec2::new(rect.width(), header_height));
            
            if active_panel != PanelType::Viewport {
                ui.painter().rect_filled(header_rect, 0.0, theme.bg_extreme);
                
                let mut x_offset = 0.0;
                let font = egui::FontId::proportional(13.0);
                
                for (i, tab) in node.tabs.as_slice().iter().enumerate() {
                    let title = match tab {
                        PanelType::Viewport => "Viewport",
                        PanelType::Hierarchy => "Hierarchy",
                        PanelType::Inspector => "Inspector",
                        PanelType::NodeGraph => "Node Graph",
                        PanelType::Console => "Console",
                        PanelType::None => "",
                    };
                    
                    let tab_width = 100.0; // Simple fixed width for now
                    let tab_rect = Rect::from_min_size(header_rect.min + Vec2::new(x_offset, 0.0), Vec2::new(tab_width, header_height));
                    
                    // Interaction
                    let response = ui.interact(tab_rect, ui.id().with(id).with(i), egui::Sense::click());
                    if response.clicked() {
                        self.nodes.as_mut_slice()[id].active_tab = i;
                    }
                    
                    let is_active = i == node.active_tab;
                    let text_color = if is_active { theme.text_hovered } else { theme.text_normal };
                    
                    if is_active {
                        let top_line_rect = Rect::from_min_size(tab_rect.min, Vec2::new(tab_width, 2.0));
                        ui.painter().rect_filled(top_line_rect, 0.0, theme.accent);
                    }
                    
                    ui.painter().text(tab_rect.left_center() + Vec2::new(8.0, 0.0), egui::Align2::LEFT_CENTER, title, font.clone(), text_color);
                    x_offset += tab_width;
                }
            }

            // Draw Inner Panel Content
            let mut content_rect = rect;
            if active_panel != PanelType::Viewport {
                content_rect = Rect::from_min_max(rect.min + Vec2::new(0.0, header_height), rect.max);
            }
            
            // Create a child UI for the panel
            let mut child_ui = ui.child_ui_with_id_source(content_rect, *ui.layout(), id);
            
            let padded_rect = content_rect.shrink(4.0);
            let mut padded_ui = child_ui.child_ui_with_id_source(padded_rect, *ui.layout(), id + 100);
            
            draw_panel(active_panel, &mut padded_ui, padded_rect);
        }
    }
}
