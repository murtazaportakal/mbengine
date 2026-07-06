use ash::vk;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceHandle(pub vk::Image);

#[derive(Clone, Copy, Debug)]
pub struct ResourceState {
    pub layout: vk::ImageLayout,
    pub stage_mask: vk::PipelineStageFlags,
    pub access_mask: vk::AccessFlags,
    pub aspect_mask: vk::ImageAspectFlags,
}

/// Zero-allocation flat map for tracking image resource states during
/// render graph execution. Uses a fixed inline array with linear scan —
/// optimal for the small number of render targets the engine tracks (≤5).
pub struct ResourceTracker {
    entries: [(ResourceHandle, ResourceState); Self::MAX_ENTRIES],
    len: usize,
}

impl ResourceTracker {
    /// Maximum number of distinct render targets tracked per frame.
    /// The engine currently uses ~5 (offscreen HDR, SDR, bloom, depth, swapchain).
    pub const MAX_ENTRIES: usize = 16;

    /// Create a new empty tracker with zeroed inline storage.
    pub fn new() -> Self {
        // Initialize with dummy handles — only entries 0..len are valid.
        let dummy = (
            ResourceHandle(vk::Image::null()),
            ResourceState {
                layout: vk::ImageLayout::UNDEFINED,
                stage_mask: vk::PipelineStageFlags::TOP_OF_PIPE,
                access_mask: vk::AccessFlags::NONE,
                aspect_mask: vk::ImageAspectFlags::COLOR,
            },
        );
        Self {
            entries: [dummy; Self::MAX_ENTRIES],
            len: 0,
        }
    }

    /// Look up the current state for a resource handle.
    pub fn get(&self, handle: &ResourceHandle) -> Option<ResourceState> {
        for i in 0..self.len {
            if self.entries[i].0 == *handle {
                return Some(self.entries[i].1);
            }
        }
        None
    }

    /// Insert or update a resource state. Panics if full and inserting a new key
    /// (should never happen — the engine tracks far fewer than MAX_ENTRIES targets).
    pub fn insert(&mut self, handle: ResourceHandle, state: ResourceState) {
        for i in 0..self.len {
            if self.entries[i].0 == handle {
                self.entries[i].1 = state;
                return;
            }
        }
        assert!(
            self.len < Self::MAX_ENTRIES,
            "ResourceTracker overflow: more than {} distinct resources tracked",
            Self::MAX_ENTRIES
        );
        self.entries[self.len] = (handle, state);
        self.len += 1;
    }

    /// Reset the tracker for a new frame without any allocation.
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

impl Default for ResourceTracker {
    fn default() -> Self {
        Self::new()
    }
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

impl<'a> Default for RenderGraph<'a> {
    fn default() -> Self {
        Self::new()
    }
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
        resource_tracker: &mut ResourceTracker,
    ) {
        for pass in self.passes {
            for required in &pass.resources {
                let current_state =
                    resource_tracker
                        .get(&required.handle)
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
