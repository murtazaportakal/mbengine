use ash::vk;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceHandle(pub vk::Image);

#[derive(Clone, Copy, Debug)]
pub struct ResourceState {
    pub layout: vk::ImageLayout,
    pub stage_mask: vk::PipelineStageFlags,
    pub access_mask: vk::AccessFlags,
    pub aspect_mask: vk::ImageAspectFlags,
}

pub struct PassResource {
    pub handle: ResourceHandle,
    pub state: ResourceState,
}

pub struct PassData<'a> {
    pub name: String,
    pub resources: Vec<PassResource>,
    pub execute_fn: Box<dyn FnOnce(vk::CommandBuffer) + 'a>,
}

pub struct RenderGraph<'a> {
    passes: Vec<PassData<'a>>,
}

impl<'a> RenderGraph<'a> {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass<F>(&mut self, name: &str, resources: Vec<PassResource>, execute_fn: F)
    where
        F: FnOnce(vk::CommandBuffer) + 'a,
    {
        self.passes.push(PassData {
            name: name.to_string(),
            resources,
            execute_fn: Box::new(execute_fn),
        });
    }

    pub fn execute(
        self,
        vulkan: &crate::renderer::vulkan::VulkanDevice,
        command_buffer: vk::CommandBuffer,
        resource_tracker: &mut HashMap<ResourceHandle, ResourceState>,
    ) {
        for pass in self.passes {
            for required in &pass.resources {
                let current_state =
                    resource_tracker
                        .get(&required.handle)
                        .cloned()
                        .unwrap_or(ResourceState {
                            layout: vk::ImageLayout::UNDEFINED,
                            stage_mask: vk::PipelineStageFlags::TOP_OF_PIPE,
                            access_mask: vk::AccessFlags::NONE,
                            aspect_mask: required.state.aspect_mask,
                        });

                if current_state.layout != required.state.layout
                    || current_state.access_mask != required.state.access_mask
                    || current_state.stage_mask != required.state.stage_mask
                {
                    let barrier = vk::ImageMemoryBarrier::default()
                        .src_access_mask(current_state.access_mask)
                        .dst_access_mask(required.state.access_mask)
                        .old_layout(current_state.layout)
                        .new_layout(required.state.layout)
                        .image(required.handle.0)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: required.state.aspect_mask,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });

                    unsafe {
                        vulkan.device.cmd_pipeline_barrier(
                            command_buffer,
                            current_state.stage_mask,
                            required.state.stage_mask,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            std::slice::from_ref(&barrier),
                        );
                    }

                    // Update tracker
                    resource_tracker.insert(required.handle, required.state);
                }
            }

            // 2. Execute the pass closure
            (pass.execute_fn)(command_buffer);
        }
    }
}
