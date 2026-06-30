use ash::vk;
use crate::renderer::vulkan::VulkanDevice;

/// A lightweight Render Graph that manages dynamic rendering passes automatically.
pub struct RenderGraph {
    nodes: Vec<PassNode>,
}

pub struct PassNode {
    pub name: String,
    pub color_attachments: Vec<vk::RenderingAttachmentInfoKHR>,
    pub depth_attachment: Option<vk::RenderingAttachmentInfoKHR>,
    pub render_area: vk::Rect2D,
    pub execute: Box<dyn Fn(&VulkanDevice, vk::CommandBuffer)>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
        }
    }

    pub fn add_pass(&mut self, node: PassNode) {
        self.nodes.push(node);
    }

    pub fn execute(&self, vulkan: &VulkanDevice, command_buffer: vk::CommandBuffer) {
        for node in &self.nodes {
            let mut rendering_info = vk::RenderingInfoKHR::default()
                .render_area(node.render_area)
                .layer_count(1);
            
            if !node.color_attachments.is_empty() {
                rendering_info = rendering_info.color_attachments(&node.color_attachments);
            }
            if let Some(depth) = &node.depth_attachment {
                rendering_info = rendering_info.depth_attachment(depth);
            }

            unsafe {
                vulkan.device.cmd_begin_rendering(command_buffer, &rendering_info);
            }

            (node.execute)(vulkan, command_buffer);

            unsafe {
                vulkan.device.cmd_end_rendering(command_buffer);
            }
        }
    }
}
