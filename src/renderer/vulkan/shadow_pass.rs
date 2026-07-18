//! Single-cascade directional shadow map pass.
//!
//! Creates a 2048×2048 D32_SFLOAT depth image and a graphics pipeline that
//! uses `shadow.vert` (no fragment shader — depth-only render pass).
//!
//! The shadow map is bound into the global UBO descriptor set at binding 2
//! (the `shadowMap` sampler) so the existing PBR fragment shader's
//! `ShadowCalculation()` function automatically uses it.
//!
//! # Integration
//! - `RenderBackend` contains a `shadow_pass: Option<ShadowPass>`.
//! - Before the Scene pass each frame, `record_shadow_pass()` is called.
//! - The computed `light_space_matrix` is written into the `GlobalUbo`.
//! - The shadow image view is bound into descriptor set 0, binding 2.

use crate::renderer::vulkan::device::VulkanDevice;
use crate::math::mat4::Mat4;
use crate::math::vec::Vec3;
use ash::vk;

// ── Constants ─────────────────────────────────────────────────────────────────

pub const SHADOW_MAP_SIZE: u32 = 2048;

// ── Push-constant layout (matches shadow.vert) ────────────────────────────────

#[repr(C)]
pub struct ShadowPushConstants {
    pub light_space_matrix: [[f32; 4]; 4],
    pub model_matrix: [[f32; 4]; 4],
}

// ── ShadowPass ────────────────────────────────────────────────────────────────

/// GPU resources and pipeline for the depth-only shadow render pass.
pub struct ShadowPass {
    pub shadow_image: vk::Image,
    pub shadow_view: vk::ImageView,
    pub shadow_memory: vk::DeviceMemory,
    pub shadow_sampler: vk::Sampler,

    pub render_pass: vk::RenderPass,
    pub framebuffer: vk::Framebuffer,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,

    /// Current light-space matrix (recomputed each frame).
    pub light_space_matrix: Mat4,
}

impl ShadowPass {
    /// Create the shadow map image, render pass, framebuffer, and pipeline.
    ///
    /// Returns `None` if `shadow_vert.spv` is not present (graceful degradation).
    pub fn new(vulkan: &VulkanDevice, vfs: &crate::vfs::Vfs) -> Option<Self> {
        // ── Shadow image (D32_SFLOAT) ─────────────────────────────────────────
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(vk::Extent3D { width: SHADOW_MAP_SIZE, height: SHADOW_MAP_SIZE, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let shadow_image = unsafe { vulkan.device.create_image(&image_info, None).ok()? };

        let mem_req = unsafe { vulkan.device.get_image_memory_requirements(shadow_image) };
        let mem_type = vulkan.find_memory_type(
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let shadow_memory = unsafe {
            vulkan.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_req.size)
                    .memory_type_index(mem_type),
                None,
            ).ok()?
        };
        unsafe { vulkan.device.bind_image_memory(shadow_image, shadow_memory, 0).ok()?; }

        let shadow_view = unsafe {
            vulkan.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(shadow_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::D32_SFLOAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            ).ok()?
        };

        // ── Sampler (depth comparison, PCF-ready) ─────────────────────────────
        let shadow_sampler = unsafe {
            vulkan.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                    .compare_enable(false) // fragment shader does manual comparison
                    .max_anisotropy(1.0)
                    .min_lod(0.0)
                    .max_lod(1.0),
                None,
            ).ok()?
        };

        // ── Render pass ───────────────────────────────────────────────────────
        let depth_attachment = vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let depth_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .depth_stencil_attachment(&depth_ref);

        // Dependency: wait for fragment reads before allowing the scene pass to sample the shadow map.
        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dependency_flags(vk::DependencyFlags::BY_REGION);

        let write_to_read_dep = vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .dependency_flags(vk::DependencyFlags::BY_REGION);

        let render_pass = unsafe {
            vulkan.device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(std::slice::from_ref(&depth_attachment))
                    .subpasses(std::slice::from_ref(&subpass))
                    .dependencies(&[dependency, write_to_read_dep]),
                None,
            ).ok()?
        };

        // ── Framebuffer ───────────────────────────────────────────────────────
        let framebuffer = unsafe {
            vulkan.device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(std::slice::from_ref(&shadow_view))
                    .width(SHADOW_MAP_SIZE)
                    .height(SHADOW_MAP_SIZE)
                    .layers(1),
                None,
            ).ok()?
        };

        // ── Shadow pipeline ───────────────────────────────────────────────────
        let vert_code = vfs.read_bytes("shaders/shadow_vert.spv").ok()?;
        let vert_module =
            crate::renderer::vulkan::pipeline::Pipeline::create_shader_module(vulkan, &vert_code)?;

        let entry = c"main";
        let vert_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(entry);

        // Push constants: lightSpaceMatrix (64 bytes) + modelMatrix (64 bytes) = 128 bytes.
        let pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<ShadowPushConstants>() as u32);

        let pipeline_layout = unsafe {
            vulkan.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .push_constant_ranges(std::slice::from_ref(&pc_range)),
                None,
            ).ok()?
        };

        // Vertex input: position only (vec3, binding 0, location 0).
        // Matches the GeometryPool vertex buffer layout (stride = size of Vertex struct).
        let binding_desc = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<crate::renderer::vulkan::pipeline::Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);

        let attr_desc = vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0); // position is the first field in Vertex

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_desc))
            .vertex_attribute_descriptions(std::slice::from_ref(&attr_desc));

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport {
            x: 0.0, y: 0.0,
            width: SHADOW_MAP_SIZE as f32, height: SHADOW_MAP_SIZE as f32,
            min_depth: 0.0, max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width: SHADOW_MAP_SIZE, height: SHADOW_MAP_SIZE },
        };
        let vp_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::FRONT) // cull front faces to reduce peter-panning
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(true)
            .depth_bias_constant_factor(2.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(2.0)
            .line_width(1.0);

        let ms = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let pipeline = unsafe {
            vulkan.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .stages(std::slice::from_ref(&vert_stage))
                    .vertex_input_state(&vertex_input)
                    .input_assembly_state(&input_assembly)
                    .viewport_state(&vp_state)
                    .rasterization_state(&raster)
                    .multisample_state(&ms)
                    .depth_stencil_state(&depth_stencil)
                    .render_pass(render_pass)
                    .subpass(0)
                    .layout(pipeline_layout)],
                None,
            ).map_err(|e| e.1).ok()?[0]
        };

        unsafe { vulkan.device.destroy_shader_module(vert_module, None); }

        Some(Self {
            shadow_image,
            shadow_view,
            shadow_memory,
            shadow_sampler,
            render_pass,
            framebuffer,
            pipeline,
            pipeline_layout,
            light_space_matrix: Mat4::identity(),
        })
    }

    /// Compute a light-space matrix for a directional light.
    ///
    /// Uses a simple orthographic frustum centred on `scene_center` with
    /// `half_size` world units on each side.  Adjust `half_size` to fit
    /// the shadow-casting region.
    pub fn compute_light_space_matrix(
        light_dir: Vec3,
        scene_center: Vec3,
        half_size: f32,
    ) -> Mat4 {
        let light_dir = light_dir.normalize();

        // Place the light "eye" above the scene center along the light direction.
        let eye = scene_center - light_dir * (half_size * 2.0);
        let up = if light_dir.y.abs() > 0.999 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let view = Mat4::look_at(eye, scene_center, up);

        // Orthographic projection: ±half_size in X/Y, [0, 4*half_size] in Z.
        let proj = Mat4::orthographic(
            -half_size, half_size,
            -half_size, half_size,
            0.1,
            half_size * 4.0,
        );
        proj * view
    }

    /// Record the shadow depth pass into `cmd`.
    ///
    /// Iterates all `RenderComponent` entities and renders each into the depth
    /// attachment using push constants for the model matrix.
    pub fn record_shadow_pass(
        &self,
        cmd: vk::CommandBuffer,
        world: &crate::ecs::World,
        world_matrices: &[(u32, crate::math::mat4::Mat4)],
        asset_manager: &crate::asset_manager::AssetManager,
        vulkan: &VulkanDevice,
        geometry_pool: &crate::renderer::vulkan::GeometryPool,
    ) {
        use crate::ecs::{RenderComponent, TransformComponent};

        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
        };

        // Transition shadow image: SHADER_READ_ONLY → DEPTH_ATTACHMENT
        let barrier_in = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .image(self.shadow_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            vulkan.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[], &[],
                std::slice::from_ref(&barrier_in),
            );
        }

        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: SHADOW_MAP_SIZE, height: SHADOW_MAP_SIZE },
            })
            .clear_values(std::slice::from_ref(&clear_depth));

        let lsm = self.light_space_matrix;
        let lsm_raw = [
            [lsm.cols[0].x, lsm.cols[0].y, lsm.cols[0].z, lsm.cols[0].w],
            [lsm.cols[1].x, lsm.cols[1].y, lsm.cols[1].z, lsm.cols[1].w],
            [lsm.cols[2].x, lsm.cols[2].y, lsm.cols[2].z, lsm.cols[2].w],
            [lsm.cols[3].x, lsm.cols[3].y, lsm.cols[3].z, lsm.cols[3].w],
        ];

        unsafe {
            vulkan.device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            vulkan.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            vulkan.device.cmd_bind_vertex_buffers(cmd, 0, &[geometry_pool.vertex_buffer.handle], &[0]);
            vulkan.device.cmd_bind_index_buffer(cmd, geometry_pool.index_buffer.handle, 0, vk::IndexType::UINT32);
        }

        let renders = world.get_component_array::<RenderComponent>();
        let transforms = world.get_component_array::<TransformComponent>();
        let entities = renders.dense_entities_slice();

        for (i, r) in renders.as_slice().iter().enumerate() {
            if !r.visible { continue; }
            let entity = entities[i];
            if !transforms.has(entity) { continue; }

            let t = unsafe { transforms.get(entity) };
            let model = world_matrices.iter()
                .find(|(e, _)| *e == entity)
                .map(|(_, m)| *m)
                .unwrap_or(t.matrix);

            let model_raw = [
                [model.cols[0].x, model.cols[0].y, model.cols[0].z, model.cols[0].w],
                [model.cols[1].x, model.cols[1].y, model.cols[1].z, model.cols[1].w],
                [model.cols[2].x, model.cols[2].y, model.cols[2].z, model.cols[2].w],
                [model.cols[3].x, model.cols[3].y, model.cols[3].z, model.cols[3].w],
            ];

            let pc = ShadowPushConstants {
                light_space_matrix: lsm_raw,
                model_matrix: model_raw,
            };

            if let Some(mesh) = asset_manager.get_mesh(r.mesh_index) {
                unsafe {
                    vulkan.device.cmd_push_constants(
                        cmd, self.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX, 0,
                        std::slice::from_raw_parts(
                            &pc as *const ShadowPushConstants as *const u8,
                            std::mem::size_of::<ShadowPushConstants>(),
                        ),
                    );
                    vulkan.device.cmd_draw_indexed(
                        cmd, mesh.index_count, 1, mesh.index_offset, mesh.vertex_offset as i32, 0,
                    );
                }
            }
        }

        unsafe { vulkan.device.cmd_end_render_pass(cmd); }
        // Note: render pass dependency handles the DEPTH_ATTACHMENT → SHADER_READ transition.
    }

    /// Release all Vulkan resources.
    pub fn shutdown(&self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan.device.destroy_pipeline(self.pipeline, None);
            vulkan.device.destroy_pipeline_layout(self.pipeline_layout, None);
            vulkan.device.destroy_framebuffer(self.framebuffer, None);
            vulkan.device.destroy_render_pass(self.render_pass, None);
            vulkan.device.destroy_sampler(self.shadow_sampler, None);
            vulkan.device.destroy_image_view(self.shadow_view, None);
            vulkan.device.destroy_image(self.shadow_image, None);
            vulkan.device.free_memory(self.shadow_memory, None);
        }
    }
}
