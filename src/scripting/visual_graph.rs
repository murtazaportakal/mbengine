use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum NodeType {
    OnUpdate,
    OnTriggerEnter,
    GetTransform,
    SetTransform,
    Add,
    Multiply,
    Print,
    Branch,
    GetParam,
    SetParam,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PortType {
    Flow,
    Float,
    Vec3,
    Entity,
    Bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodePort {
    pub name: String,
    pub data_type: PortType,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GraphNode {
    pub id: u32,
    pub node_type: NodeType,
    pub x: f32,
    pub y: f32,
    pub inputs: Vec<NodePort>,
    pub outputs: Vec<NodePort>,
    // For simple literals (like Float / Vec3 / String nodes, but let's keep it simple as a single float for now)
    pub param_value: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GraphEdge {
    pub from_node: u32,
    pub from_port: u8,
    pub to_node: u32,
    pub to_port: u8,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct VisualGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

impl VisualGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn to_rhai(&self) -> String {
        // Very basic prototype generator. 
        // A real system does topological sort and data flow generation.
        let mut out = String::new();
        
        out.push_str("fn update(world, entity_id) {\n");

        for node in &self.nodes {
            match node.node_type {
                NodeType::GetTransform => {
                    out.push_str(&format!("    let t{} = world.get_transform(entity_id);\n", node.id));
                }
                NodeType::SetTransform => {
                    out.push_str(&format!("    world.set_transform(entity_id, t{});\n", node.id));
                }
                NodeType::Print => {
                    out.push_str(&format!("    print(\"Node {} executing\");\n", node.id));
                }
                _ => {}
            }
        }

        out.push_str("}\n");
        out
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn from_json(json: &str) -> Option<Self> {
        serde_json::from_str(json).ok()
    }
}
