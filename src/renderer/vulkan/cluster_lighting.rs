use crate::renderer::vulkan::VulkanDevice;
use ash::vk;

pub struct ComputeClusterPipeline {
    pub grid_layout: vk::PipelineLayout,
    pub grid_pipeline: vk::Pipeline,
    pub cull_layout: vk::PipelineLayout,
    pub cull_pipeline: vk::Pipeline,
}

impl ComputeClusterPipeline {
    pub fn new(vulkan: &VulkanDevice, vfs: &crate::vfs::Vfs, descriptor_set_layout: vk::DescriptorSetLayout) -> Option<Self> {
        let grid_code = vfs.read_bytes("shaders/cluster_grid.spv").ok()?;
        let grid_module = crate::renderer::vulkan::pipeline::Pipeline::create_shader_module(vulkan, &grid_code)?;
        
        let cull_code = vfs.read_bytes("shaders/cluster_cull.spv").ok()?;
        let cull_module = crate::renderer::vulkan::pipeline::Pipeline::create_shader_module(vulkan, &cull_code)?;

        let entry_name = c"main";
        
        // Grid Pipeline
        let grid_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(grid_module)
            .name(entry_name);
            
        let grid_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(104); // mat4(64) + uvec4(16) + vec2(8) + float(4) + float(4) = 96 (Wait: 64 + 16 = 80 + 8 = 88 + 8 = 96. Let's use 128 to be safe or exactly 96. Push constant max size is 128 usually.)

        let grid_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&grid_push_constant_range));
            
        let grid_layout = unsafe { vulkan.device.create_pipeline_layout(&grid_layout_info, None).ok()? };
        
        let grid_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(grid_stage)
            .layout(grid_layout);
            
        let grid_pipeline = unsafe {
            vulkan.device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&grid_pipeline_info), None).map_err(|e| e.1).ok()?[0]
        };

        // Cull Pipeline
        let cull_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(cull_module)
            .name(entry_name);
            
        let cull_push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(80); // mat4(64) + uint(4) + uvec3(12) = 80

        let cull_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&cull_push_constant_range));
            
        let cull_layout = unsafe { vulkan.device.create_pipeline_layout(&cull_layout_info, None).ok()? };
        
        let cull_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(cull_stage)
            .layout(cull_layout);
            
        let cull_pipeline = unsafe {
            vulkan.device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&cull_pipeline_info), None).map_err(|e| e.1).ok()?[0]
        };

        unsafe {
            vulkan.device.destroy_shader_module(grid_module, None);
            vulkan.device.destroy_shader_module(cull_module, None);
        }

        Some(Self {
            grid_layout,
            grid_pipeline,
            cull_layout,
            cull_pipeline,
        })
    }

    pub fn dispatch(
        &self,
        vulkan: &VulkanDevice,
        cmd: vk::CommandBuffer,
        descriptor_set: vk::DescriptorSet,
        inverse_proj: crate::math::mat4::Mat4,
        view: crate::math::mat4::Mat4,
        screen_width: u32,
        screen_height: u32,
        z_near: f32,
        z_far: f32,
        num_lights: u32,
    ) {
        unsafe {
            // 1. Grid Pass (normally could be only on resize, but cheap enough per-frame for now)
            vulkan.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.grid_pipeline);
            vulkan.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.grid_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );

            #[repr(C)]
            struct GridPC {
                inverse_proj: crate::math::mat4::Mat4,
                screen_dimensions: [u32; 4],
                z_near_far: [f32; 2],
            }
            let grid_pc = GridPC {
                inverse_proj,
                screen_dimensions: [screen_width, screen_height, 0, 0],
                z_near_far: [z_near, z_far],
            };
            vulkan.device.cmd_push_constants(
                cmd,
                self.grid_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(&grid_pc as *const _ as *const u8, std::mem::size_of::<GridPC>()),
            );
            vulkan.device.cmd_dispatch(cmd, 16, 9, 24);

            // Barrier to ensure AABBs are written
            let mem_bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            vulkan.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&mem_bar),
                &[],
                &[],
            );

            // 2. Cull Pass
            vulkan.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.cull_pipeline);
            vulkan.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.cull_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );

            #[repr(C)]
            struct CullPC {
                view_matrix: crate::math::mat4::Mat4,
                num_lights: u32,
                pad: [u32; 3],
            }
            let cull_pc = CullPC {
                view_matrix: view,
                num_lights,
                pad: [0; 3],
            };
            vulkan.device.cmd_push_constants(
                cmd,
                self.cull_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(&cull_pc as *const _ as *const u8, std::mem::size_of::<CullPC>()),
            );
            vulkan.device.cmd_dispatch(cmd, 16, 9, 24);

            // Barrier to ensure lights are culled before rendering
            let draw_bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            vulkan.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&draw_bar),
                &[],
                &[],
            );
        }
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan.device.destroy_pipeline(self.grid_pipeline, None);
            vulkan.device.destroy_pipeline_layout(self.grid_layout, None);
            vulkan.device.destroy_pipeline(self.cull_pipeline, None);
            vulkan.device.destroy_pipeline_layout(self.cull_layout, None);
        }
    }
}
