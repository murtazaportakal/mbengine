use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use crate::containers::{FixedArray, FixedString};

const MAX_NODES: usize = 1024;
const MAX_CONNECTIONS: usize = 2048;
const MAX_PINS: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PinId(pub NodeId, pub usize);

#[derive(Clone, Debug, PartialEq)]
pub enum NodeType {
    FloatLiteral(f32),
    InputDt,
    GetState(FixedString<32>),
    MathAdd,
    MathMul,
    Vec3Make,
    DecomposeVec3,
    GetTransform,
    SetTransform,
    ComposeTransform,
    DecomposeTransform,
}

impl NodeType {
    pub fn format_name(&self, buf: &mut FixedString<64>) {
        use std::fmt::Write;
        buf.clear();
        match self {
            NodeType::FloatLiteral(v) => { let _ = write!(buf, "Float ({})", v); }
            NodeType::GetState(k) => { let _ = write!(buf, "Get State ('{}')", k.as_str()); }
            _ => { let _ = write!(buf, "{}", self.name()); }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            NodeType::FloatLiteral(_) => "Float",
            NodeType::InputDt => "Delta Time (dt)",
            NodeType::GetState(_) => "Get State",
            NodeType::MathAdd => "Add",
            NodeType::MathMul => "Multiply",
            NodeType::Vec3Make => "Make Vec3",
            NodeType::DecomposeVec3 => "Decompose Vec3",
            NodeType::GetTransform => "Get Transform",
            NodeType::SetTransform => "Set Transform",
            NodeType::ComposeTransform => "Compose Transform",
            NodeType::DecomposeTransform => "Decompose Transform",
        }
    }

    pub fn inputs(&self) -> &'static [&'static str] {
        match self {
            NodeType::FloatLiteral(_) | NodeType::InputDt | NodeType::GetState(_) | NodeType::GetTransform => &[],
            NodeType::MathAdd | NodeType::MathMul => &["A", "B"],
            NodeType::Vec3Make => &["X", "Y", "Z"],
            NodeType::DecomposeVec3 => &["Vec3"],
            NodeType::SetTransform => &["Transform"],
            NodeType::ComposeTransform => &["Translation", "Rotation", "Scale"],
            NodeType::DecomposeTransform => &["Transform"],
        }
    }

    pub fn outputs(&self) -> &'static [&'static str] {
        match self {
            NodeType::SetTransform => &[],
            NodeType::DecomposeVec3 => &["X", "Y", "Z"],
            NodeType::DecomposeTransform => &["Translation", "Rotation", "Scale"],
            _ => &["Out"],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    pub typ: NodeType,
    pub pos: Pos2,
}

#[derive(Clone, Copy, Debug)]
pub struct PinPos {
    pub id: PinId,
    pub pos: Pos2,
    pub is_input: bool,
}

pub struct NodeGraphEditor {
    pub nodes: FixedArray<Node, MAX_NODES>,
    pub connections: FixedArray<(PinId, PinId), MAX_CONNECTIONS>, // (OutputPin, InputPin)
    pub next_node_id: usize,
    pub pan: Vec2,
    pub drag_connection: Option<PinId>, // Currently dragging an output pin
    pub mouse_pos: Pos2,
    pub dragging_node: Option<(NodeId, Vec2)>, // NodeID and offset from mouse
    pub compile_message: Option<(FixedString<128>, std::time::Instant)>,
}

impl NodeGraphEditor {
    pub fn new() -> Self {
        let mut editor = Self {
            nodes: FixedArray::new(),
            connections: FixedArray::new(),
            next_node_id: 1,
            pan: Vec2::ZERO,
            drag_connection: None,
            mouse_pos: Pos2::ZERO,
            dragging_node: None,
            compile_message: None,
        };

        // Initialize with default nodes to match original setup
        let id_dt = editor.add_node(NodeType::InputDt, Pos2::new(100.0, 100.0));
        
        let mut speed_str = FixedString::<32>::new();
        speed_str.push_str("speed");
        let id_speed = editor.add_node(NodeType::GetState(speed_str), Pos2::new(100.0, 200.0));
        
        let id_mul = editor.add_node(NodeType::MathMul, Pos2::new(300.0, 150.0));
        let id_get_t = editor.add_node(NodeType::GetTransform, Pos2::new(100.0, 300.0));
        let id_dec_t = editor.add_node(NodeType::DecomposeTransform, Pos2::new(300.0, 300.0));
        let id_dec_r = editor.add_node(NodeType::DecomposeVec3, Pos2::new(500.0, 300.0));
        let id_add = editor.add_node(NodeType::MathAdd, Pos2::new(700.0, 250.0));
        let id_vec = editor.add_node(NodeType::Vec3Make, Pos2::new(900.0, 300.0));
        let id_com_t = editor.add_node(NodeType::ComposeTransform, Pos2::new(1100.0, 300.0));
        let id_set_t = editor.add_node(NodeType::SetTransform, Pos2::new(1300.0, 300.0));

        editor.connections.push((PinId(id_dt, 0), PinId(id_mul, 0)));
        editor.connections.push((PinId(id_speed, 0), PinId(id_mul, 1)));
        editor.connections.push((PinId(id_get_t, 0), PinId(id_dec_t, 0)));
        editor.connections.push((PinId(id_dec_t, 1), PinId(id_dec_r, 0)));
        editor.connections.push((PinId(id_dec_r, 1), PinId(id_add, 0)));
        editor.connections.push((PinId(id_mul, 0), PinId(id_add, 1)));
        editor.connections.push((PinId(id_dec_r, 0), PinId(id_vec, 0)));
        editor.connections.push((PinId(id_add, 0), PinId(id_vec, 1)));
        editor.connections.push((PinId(id_dec_r, 2), PinId(id_vec, 2)));
        editor.connections.push((PinId(id_dec_t, 0), PinId(id_com_t, 0)));
        editor.connections.push((PinId(id_vec, 0), PinId(id_com_t, 1)));
        editor.connections.push((PinId(id_dec_t, 2), PinId(id_com_t, 2)));
        editor.connections.push((PinId(id_com_t, 0), PinId(id_set_t, 0)));
        
        editor
    }

    pub fn add_node(&mut self, typ: NodeType, pos: Pos2) -> NodeId {
        assert!(self.nodes.len() < MAX_NODES, "Node Graph: MAX_NODES capacity exceeded");
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        self.nodes.push(Node { id, typ, pos });
        id
    }
    
    fn remove_node(&mut self, node_id: NodeId) {
        let mut new_nodes = FixedArray::new();
        for node in self.nodes.as_slice() {
            if node.id != node_id {
                new_nodes.push(node.clone());
            }
        }
        self.nodes = new_nodes;
        
        let mut new_connections = FixedArray::new();
        for &conn in self.connections.as_slice() {
            if conn.0.0 != node_id && conn.1.0 != node_id {
                new_connections.push(conn);
            }
        }
        self.connections = new_connections;
    }

    pub fn draw(&mut self, ui: &mut egui::Ui) {
        let canvas_rect = ui.available_rect_before_wrap();
        let (response, painter) = ui.allocate_painter(canvas_rect.size(), egui::Sense::click_and_drag());
        
        if let Some(pos) = response.hover_pos() {
            self.mouse_pos = pos;
        }

        // --- Draw Background Grid ---
        let grid_size = 50.0;
        let offset = self.pan;
        let stroke = Stroke::new(1.0, Color32::from_rgb(25, 25, 25));
        let (min_x, min_y) = (canvas_rect.min.x, canvas_rect.min.y);
        let (max_x, max_y) = (canvas_rect.max.x, canvas_rect.max.y);
        
        let mut x = min_x + offset.x % grid_size;
        while x < max_x {
            painter.line_segment([Pos2::new(x, min_y), Pos2::new(x, max_y)], stroke);
            x += grid_size;
        }
        
        let mut y = min_y + offset.y % grid_size;
        while y < max_y {
            painter.line_segment([Pos2::new(min_x, y), Pos2::new(max_x, y)], stroke);
            y += grid_size;
        }

        // --- View & Layout Constants ---
        let header_height = 28.0;
        let row_height = 24.0;
        let padding = 12.0;
        let node_width = 180.0;

        let mut pin_positions = FixedArray::<PinPos, MAX_PINS>::new();
        let mut node_rects = FixedArray::<(NodeId, Rect), MAX_NODES>::new();
        let mut nodes_to_remove = FixedArray::<NodeId, 16>::new();

        // --- Pass 1: Layout & Hit testing preparation ---
        for node in self.nodes.as_slice() {
            let num_inputs = node.typ.inputs().len();
            let num_outputs = node.typ.outputs().len();
            let rows = num_inputs.max(num_outputs) as f32;
            let height = header_height + padding * 2.0 + rows * row_height;
            
            let rect = Rect::from_min_size(node.pos + self.pan, Vec2::new(node_width, height));
            node_rects.push((node.id, rect));
            
            for i in 0..num_inputs {
                let pin_pos = rect.min + Vec2::new(10.0, header_height + padding + i as f32 * row_height + row_height * 0.5);
                if pin_positions.len() < MAX_PINS {
                    pin_positions.push(PinPos { id: PinId(node.id, i), pos: pin_pos, is_input: true });
                }
            }
            for i in 0..num_outputs {
                let pin_pos = rect.min + Vec2::new(rect.width() - 10.0, header_height + padding + i as f32 * row_height + row_height * 0.5);
                if pin_positions.len() < MAX_PINS {
                    pin_positions.push(PinPos { id: PinId(node.id, i), pos: pin_pos, is_input: false });
                }
            }
        }

        // --- Pass 2: Interaction State Updates ---
        if response.dragged_by(egui::PointerButton::Middle) {
            self.pan += response.drag_delta();
        }

        let mut hovered_node = None;
        let mut hovered_pin = None;

        for pin in pin_positions.as_slice() {
            if pin.pos.distance(self.mouse_pos) < 8.0 {
                hovered_pin = Some(pin.clone());
            }
        }

        if hovered_pin.is_none() {
            for &(id, rect) in node_rects.as_slice().iter().rev() {
                if rect.contains(self.mouse_pos) {
                    hovered_node = Some((id, rect));
                    break;
                }
            }
        }
        
        if response.clicked_by(egui::PointerButton::Primary) {
            if let Some((id, rect)) = hovered_node {
                let close_rect = Rect::from_min_size(
                    rect.min + Vec2::new(rect.width() - 20.0, 4.0),
                    Vec2::new(16.0, 16.0)
                );
                if close_rect.contains(self.mouse_pos) {
                    if nodes_to_remove.len() < 16 {
                        nodes_to_remove.push(id);
                    }
                }
            }
        }

        if response.drag_started_by(egui::PointerButton::Primary) {
            if let Some(pin) = hovered_pin {
                if !pin.is_input {
                    self.drag_connection = Some(pin.id);
                }
            } else if let Some((node_id, _)) = hovered_node {
                if let Some(node) = self.nodes.as_slice().iter().find(|n| n.id == node_id) {
                    self.dragging_node = Some((node_id, node.pos - self.mouse_pos));
                }
            }
        }

        if response.dragged_by(egui::PointerButton::Primary) {
            if let Some((node_id, offset)) = self.dragging_node {
                if let Some(node) = self.nodes.as_mut_slice().iter_mut().find(|n| n.id == node_id) {
                    node.pos = self.mouse_pos + offset;
                }
            }
        }

        if response.drag_stopped_by(egui::PointerButton::Primary) {
            if let Some(out_pin) = self.drag_connection {
                if let Some(in_pin) = hovered_pin {
                    if in_pin.is_input && in_pin.id.0 != out_pin.0 {
                        let mut new_conns = FixedArray::new();
                        for &c in self.connections.as_slice() {
                            if c.1 != in_pin.id {
                                new_conns.push(c);
                            }
                        }
                        if new_conns.len() < MAX_CONNECTIONS {
                            new_conns.push((out_pin, in_pin.id));
                            self.connections = new_conns;
                        }
                    }
                }
            }
            self.drag_connection = None;
            self.dragging_node = None;
        }

        for &id in nodes_to_remove.as_slice() {
            self.remove_node(id);
        }

        response.context_menu(|ui| {
            if ui.button("Add Float").clicked() {
                self.add_node(NodeType::FloatLiteral(1.0), self.mouse_pos - self.pan);
                ui.close_menu();
            }
            if ui.button("Add Delta Time").clicked() {
                self.add_node(NodeType::InputDt, self.mouse_pos - self.pan);
                ui.close_menu();
            }
            if ui.button("Add Math Add").clicked() {
                self.add_node(NodeType::MathAdd, self.mouse_pos - self.pan);
                ui.close_menu();
            }
            if ui.button("Add Math Mul").clicked() {
                self.add_node(NodeType::MathMul, self.mouse_pos - self.pan);
                ui.close_menu();
            }
        });

        // --- Pass 3: Render Nodes & Connections ---
        let font_id = egui::FontId::proportional(11.0);
        let title_font_id = egui::FontId::proportional(13.0);

        for (i, node) in self.nodes.as_slice().iter().enumerate() {
            let rect = node_rects[i].1;
            
            // Drop Shadow
            let shadow_rect = rect.translate(Vec2::new(4.0, 4.0));
            painter.rect_filled(shadow_rect, 0.0, Color32::from_black_alpha(100));

            // Slate Node Background
            painter.rect_filled(rect, 0.0, Color32::from_rgb(22, 22, 22)); // #161616
            
            // Slate Header Background
            let header_rect = Rect::from_min_size(rect.min, Vec2::new(rect.width(), header_height));
            let is_selected = self.dragging_node.map_or(false, |(id, _)| id == node.id);
            let header_color = if is_selected { 
                Color32::from_rgb(0, 112, 224) // Unreal Blue Accent
            } else { 
                match node.typ {
                    NodeType::FloatLiteral(_) | NodeType::InputDt => Color32::from_rgb(140, 40, 40), // Red/Orange for Constants/Inputs
                    NodeType::GetState(_) => Color32::from_rgb(40, 140, 80), // Teal/Green for State
                    NodeType::MathAdd | NodeType::MathMul | NodeType::Vec3Make | NodeType::DecomposeVec3 => Color32::from_rgb(40, 80, 140), // Blue for Math
                    NodeType::GetTransform | NodeType::SetTransform | NodeType::ComposeTransform | NodeType::DecomposeTransform => Color32::from_rgb(110, 40, 140), // Purple for Transform
                }
            };
            painter.rect_filled(header_rect, 0.0, header_color);
            
            let mut name_buf = FixedString::<64>::new();
            node.typ.format_name(&mut name_buf);
            painter.text(header_rect.left_center() + Vec2::new(8.0, 0.0), egui::Align2::LEFT_CENTER, name_buf.as_str(), title_font_id.clone(), Color32::WHITE);
            
            // Delete button
            let close_rect = Rect::from_min_size(rect.min + Vec2::new(rect.width() - 20.0, (header_height - 16.0) / 2.0), Vec2::new(16.0, 16.0));
            let close_color = if close_rect.contains(self.mouse_pos) { Color32::from_rgb(200, 50, 50) } else { Color32::from_gray(100) };
            painter.text(close_rect.min, egui::Align2::LEFT_TOP, "x", title_font_id.clone(), close_color);
            
            // Slate Structural Border
            let border_color = if is_selected { Color32::from_rgb(200, 130, 0) } else { Color32::from_rgb(10, 10, 10) };
            painter.rect_stroke(rect, 0.0, Stroke::new(1.0, border_color));
            painter.line_segment([header_rect.left_bottom(), header_rect.right_bottom()], Stroke::new(1.0, border_color));

            // Pins and Labels
            let pin_radius = 3.5;
            let inputs = node.typ.inputs();
            for (i, name) in inputs.iter().enumerate() {
                let pin_pos = rect.min + Vec2::new(10.0, header_height + padding + i as f32 * row_height + row_height * 0.5);
                let is_hovered = hovered_pin.map_or(false, |p| p.id == PinId(node.id, i) && p.is_input);
                let color = if is_hovered { Color32::WHITE } else { Color32::from_rgb(180, 180, 180) };
                painter.circle_filled(pin_pos, pin_radius, color);
                painter.circle_stroke(pin_pos, pin_radius, Stroke::new(1.0, Color32::from_rgb(30, 30, 30)));
                painter.text(pin_pos + Vec2::new(8.0, 0.0), egui::Align2::LEFT_CENTER, *name, font_id.clone(), Color32::from_rgb(200, 200, 200));
            }

            let outputs = node.typ.outputs();
            for (i, name) in outputs.iter().enumerate() {
                let pin_pos = rect.min + Vec2::new(rect.width() - 10.0, header_height + padding + i as f32 * row_height + row_height * 0.5);
                let is_hovered = hovered_pin.map_or(false, |p| p.id == PinId(node.id, i) && !p.is_input);
                let color = if is_hovered { Color32::WHITE } else { Color32::from_rgb(180, 180, 180) };
                painter.circle_filled(pin_pos, pin_radius, color);
                painter.circle_stroke(pin_pos, pin_radius, Stroke::new(1.0, Color32::from_rgb(30, 30, 30)));
                painter.text(pin_pos - Vec2::new(8.0, 0.0), egui::Align2::RIGHT_CENTER, *name, font_id.clone(), Color32::from_rgb(200, 200, 200));
            }
        }

        // Draw active bezier connections
        let get_pin_pos = |id: PinId| -> Option<Pos2> {
            for pin in pin_positions.as_slice() {
                if pin.id == id {
                    return Some(pin.pos);
                }
            }
            None
        };

        for &conn in self.connections.as_slice() {
            if let (Some(p1), Some(p2)) = (get_pin_pos(conn.0), get_pin_pos(conn.1)) {
                draw_bezier_connection(painter.clone(), p1, p2);
            }
        }

        if let Some(out_pin) = self.drag_connection {
            if let Some(p1) = get_pin_pos(out_pin) {
                draw_bezier_connection(painter.clone(), p1, self.mouse_pos);
            }
        }
    }

    pub fn compile_to_rhai(&self) -> String {
        use std::fmt::Write;
        // String is allowed here because compilation is not on the hot path (update loop).
        let mut code = String::new();
        let _ = write!(&mut code, "fn init() {{\n    let state = #{{}};\n");
        
        let mut state_vars = std::collections::HashSet::new();
        for node in self.nodes.as_slice() {
            if let NodeType::GetState(name) = &node.typ {
                if state_vars.insert(name.as_str().to_string()) {
                    let _ = write!(&mut code, "    state.{} = 0.0;\n", name.as_str());
                }
            }
        }
        let _ = write!(&mut code, "    return state;\n}}\n\nfn update(state, transform, dt) {{\n");
        
        let set_t = self.nodes.as_slice().iter().find(|n| matches!(n.typ, NodeType::SetTransform));
        
        if let Some(terminal) = set_t {
            let mut computed = std::collections::HashMap::new();
            self.compile_node(terminal.id, &mut code, &mut computed);
        }
        
        let _ = write!(&mut code, "}}\n");
        code
    }

    fn compile_node(&self, id: NodeId, code: &mut String, computed: &mut std::collections::HashMap<PinId, String>) -> String {
        let node = self.nodes.as_slice().iter().find(|n| n.id == id).unwrap();
        
        let mut input_vars = Vec::new();
        for i in 0..node.typ.inputs().len() {
            let in_pin = PinId(id, i);
            if let Some(conn) = self.connections.as_slice().iter().find(|c| c.1 == in_pin) {
                let out_pin = conn.0;
                if let Some(var) = computed.get(&out_pin) {
                    input_vars.push(var.clone());
                } else {
                    let var = self.compile_node(out_pin.0, code, computed);
                    input_vars.push(var);
                }
            } else {
                input_vars.push("0.0".to_string());
            }
        }
        
        let node_var = format!("n{}", id.0);
        use std::fmt::Write;
        
        match &node.typ {
            NodeType::FloatLiteral(v) => {
                let _ = write!(code, "    let {}_0 = {};\n", node_var, v);
                computed.insert(PinId(id, 0), format!("{}_0", node_var));
            }
            NodeType::InputDt => {
                let _ = write!(code, "    let {}_0 = dt;\n", node_var);
                computed.insert(PinId(id, 0), format!("{}_0", node_var));
            }
            NodeType::GetState(name) => {
                let _ = write!(code, "    let {}_0 = state.{};\n", node_var, name.as_str());
                computed.insert(PinId(id, 0), format!("{}_0", node_var));
            }
            NodeType::MathAdd => {
                let _ = write!(code, "    let {}_0 = {} + {};\n", node_var, input_vars[0], input_vars[1]);
                computed.insert(PinId(id, 0), format!("{}_0", node_var));
            }
            NodeType::MathMul => {
                let _ = write!(code, "    let {}_0 = {} * {};\n", node_var, input_vars[0], input_vars[1]);
                computed.insert(PinId(id, 0), format!("{}_0", node_var));
            }
            NodeType::GetTransform => {
                let _ = write!(code, "    let {}_0 = transform;\n", node_var);
                computed.insert(PinId(id, 0), format!("{}_0", node_var));
            }
            NodeType::DecomposeTransform => {
                let _ = write!(code, "    let {}_0 = {}.translation;\n", node_var, input_vars[0]);
                let _ = write!(code, "    let {}_1 = {}.rotation;\n", node_var, input_vars[0]);
                let _ = write!(code, "    let {}_2 = {}.scale;\n", node_var, input_vars[0]);
                computed.insert(PinId(id, 0), format!("{}_0", node_var));
                computed.insert(PinId(id, 1), format!("{}_1", node_var));
                computed.insert(PinId(id, 2), format!("{}_2", node_var));
            }
            NodeType::DecomposeVec3 => {
                let _ = write!(code, "    let {}_0 = {}.x;\n", node_var, input_vars[0]);
                let _ = write!(code, "    let {}_1 = {}.y;\n", node_var, input_vars[0]);
                let _ = write!(code, "    let {}_2 = {}.z;\n", node_var, input_vars[0]);
                computed.insert(PinId(id, 0), format!("{}_0", node_var));
                computed.insert(PinId(id, 1), format!("{}_1", node_var));
                computed.insert(PinId(id, 2), format!("{}_2", node_var));
            }
            NodeType::Vec3Make => {
                let _ = write!(code, "    let {}_0 = vec3({}, {}, {});\n", node_var, input_vars[0], input_vars[1], input_vars[2]);
                computed.insert(PinId(id, 0), format!("{}_0", node_var));
            }
            NodeType::ComposeTransform => {
                let _ = write!(code, "    let mut {}_0 = transform;\n", node_var);
                let _ = write!(code, "    {}_0.translation = {};\n", node_var, input_vars[0]);
                let _ = write!(code, "    {}_0.rotation = {};\n", node_var, input_vars[1]);
                let _ = write!(code, "    {}_0.scale = {};\n", node_var, input_vars[2]);
                computed.insert(PinId(id, 0), format!("{}_0", node_var));
            }
            NodeType::SetTransform => {
                let _ = write!(code, "    transform = {};\n", input_vars[0]);
            }
        }
        
        format!("{}_0", node_var)
    }
}

fn draw_bezier_connection(painter: egui::Painter, p1: Pos2, p2: Pos2) {
    let control_scale = (p2.x - p1.x).abs().max(50.0);
    let cp1 = p1 + Vec2::new(control_scale, 0.0);
    let cp2 = p2 - Vec2::new(control_scale, 0.0);
    
    let shape = egui::Shape::CubicBezier(egui::epaint::CubicBezierShape {
        points: [p1, cp1, cp2, p2],
        closed: false,
        fill: Color32::TRANSPARENT,
        stroke: Stroke::new(1.5, Color32::from_rgb(150, 150, 150)),
    });
    painter.add(shape);
}
