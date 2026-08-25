//! GPU cloth / soft-body simulation pipeline.
//!
//! Wraps the two compute shaders:
//!   - `cloth_integrate.comp` — applies gravity, spring forces, and sphere-collider
//!     responses in a single 2D dispatch over a cloth grid.
//!   - `cloth_solve.comp`     — iteratively enforces structural constraint lengths.
//!
//! # How it works
//! Each cloth entity owns a `ClothGpuInstance`:
//!   - `velocity_buffer`: per-particle `vec4(vel.xyz, inv_mass)`.
//!   - `geometry_buffer`: borrowed from the entity's `GeometryPool` slot
//!     (positions are read/written in place by the integrate shader).
//!   - `descriptor_set_integrate` / `descriptor_set_solve`: pre-built sets bound
//!     per dispatch.
//!
//! # Zero-heap policy
//! All `ClothGpuInstance` objects live in a pre-allocated `Vec` sized at
//! construction.  No per-frame allocations occur in `dispatch_all`.

use crate::renderer::vulkan::{buffer::Buffer, device::VulkanDevice};
use ash::vk;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum number of cloth entities simultaneously simulated.
pub const MAX_CLOTH_INSTANCES: usize = 16;

// ── Push-constant layout (mirrors `cloth_integrate.comp` / `cloth_solve.comp`) ─

/// Push-constant block shared by both cloth shaders.
#[repr(C)]
pub struct ClothPushConstants {
    pub grid_width: u32,
    pub grid_height: u32,
    pub dt: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub num_colliders: u32,
}

// ── Per-instance GPU resources ────────────────────────────────────────────────

/// GPU resources owned by one simulated cloth entity.
pub struct ClothGpuInstance {
    /// Per-particle `vec4(velocity.xyz, inv_mass)`.
    /// `inv_mass == 0.0` → particle is pinned (zero velocity contribution).
    pub velocity_buffer: Buffer,
    /// Sphere-collider SSBO (xyz = centre, w = radius).
    pub collider_buffer: Buffer,
    /// Descriptor set for the *integrate* shader (binds geometry + vel + colliders).
    pub descriptor_set_integrate: vk::DescriptorSet,
    /// Descriptor set for the *solve* shader (binds geometry only).
    pub descriptor_set_solve: vk::DescriptorSet,
    /// Grid dimensions (used to compute dispatch group counts).
    pub grid_width: u32,
    pub grid_height: u32,
    /// Number of solver iterations per frame.
    pub solver_iterations: u32,
    /// Simulation parameters.
    pub stiffness: f32,
    pub damping: f32,
    /// Vulkan buffer of the mesh vertex data (borrowed from GeometryPool).
    /// This is a raw handle — the GeometryPool owns the backing memory.
    pub geometry_buffer: vk::Buffer,
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// Manages the two cloth compute pipelines and their shared descriptor pool.
pub struct ComputeClothPipeline {
    // Integrate pipeline
    pub integrate_pipeline: vk::Pipeline,
    pub integrate_layout: vk::PipelineLayout,
    // Solve pipeline
    pub solve_pipeline: vk::Pipeline,
    pub solve_layout: vk::PipelineLayout,
    // Shared descriptor resources
    pub descriptor_set_layout_integrate: vk::DescriptorSetLayout,
    pub descriptor_set_layout_solve: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
}

impl ComputeClothPipeline {
    /// Create both cloth compute pipelines.
    ///
    /// Returns `None` if either shader cannot be loaded (graceful degradation —
    /// cloth simply will not simulate this session).
    pub fn new(vulkan: &VulkanDevice, vfs: &crate::vfs::Vfs) -> Option<Self> {
        // ── Load shader modules ───────────────────────────────────────────────
        let integrate_code = vfs.read_bytes("shaders/cloth_integrate.spv").ok()?;
        let solve_code = vfs.read_bytes("shaders/cloth_solve.spv").ok()?;

        let integrate_module =
            crate::renderer::vulkan::pipeline::Pipeline::create_shader_module(vulkan, &integrate_code)?;
        let solve_module =
            crate::renderer::vulkan::pipeline::Pipeline::create_shader_module(vulkan, &solve_code)?;

        let entry = c"main";

        // ── Descriptor set layout — integrate ─────────────────────────────────
        // binding 0 — MeshVertices (geometry, read/write)
        // binding 1 — ClothVelocities (vel + inv_mass, read/write)
        // binding 2 — Colliders (read-only)
        let integrate_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let integrate_dsl = unsafe {
            vulkan.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&integrate_bindings),
                None,
            ).ok()?
        };

        // ── Descriptor set layout — solve ─────────────────────────────────────
        // binding 0 — MeshVertices (geometry, read/write)
        let solve_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let solve_dsl = unsafe {
            vulkan.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&solve_bindings),
                None,
            ).ok()?
        };

        // ── Push-constant ranges ──────────────────────────────────────────────
        let pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ClothPushConstants>() as u32);

        // ── Pipeline layouts ──────────────────────────────────────────────────
        let integrate_layout = unsafe {
            vulkan.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&integrate_dsl))
                    .push_constant_ranges(std::slice::from_ref(&pc_range)),
                None,
            ).ok()?
        };
        let solve_layout = unsafe {
            vulkan.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&solve_dsl))
                    .push_constant_ranges(std::slice::from_ref(&pc_range)),
                None,
            ).ok()?
        };

        // ── Compute pipelines ─────────────────────────────────────────────────
        let make_stage = |module| {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry)
        };

        let integrate_pipeline = unsafe {
            vulkan.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(make_stage(integrate_module))
                    .layout(integrate_layout)],
                None,
            ).map_err(|e| e.1).ok()?[0]
        };

        let solve_pipeline = unsafe {
            vulkan.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(make_stage(solve_module))
                    .layout(solve_layout)],
                None,
            ).map_err(|e| e.1).ok()?[0]
        };

        // Cleanup shader modules — no longer needed after pipeline creation.
        unsafe {
            vulkan.device.destroy_shader_module(integrate_module, None);
            vulkan.device.destroy_shader_module(solve_module, None);
        }

        // ── Descriptor pool ───────────────────────────────────────────────────
        // Each instance needs 1 integrate set (3 SSBOs) + 1 solve set (1 SSBO).
        // Total: MAX_CLOTH_INSTANCES * 4 storage buffers, 2 * MAX sets.
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_CLOTH_INSTANCES as u32 * 4)];
        let descriptor_pool = unsafe {
            vulkan.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(MAX_CLOTH_INSTANCES as u32 * 2),
                None,
            ).ok()?
        };

        Some(Self {
            integrate_pipeline,
            integrate_layout,
            solve_pipeline,
            solve_layout,
            descriptor_set_layout_integrate: integrate_dsl,
            descriptor_set_layout_solve: solve_dsl,
            descriptor_pool,
        })
    }

    /// Allocate a pair of descriptor sets for a new cloth instance.
    pub fn allocate_descriptor_sets(
        &self,
        vulkan: &VulkanDevice,
    ) -> Option<(vk::DescriptorSet, vk::DescriptorSet)> {
        let layouts = [
            self.descriptor_set_layout_integrate,
            self.descriptor_set_layout_solve,
        ];
        let sets = unsafe {
            vulkan.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&layouts),
            ).ok()?
        };
        Some((sets[0], sets[1]))
    }

    /// Write buffer references into a `ClothGpuInstance`'s descriptor sets.
    pub fn update_descriptors(
        &self,
        vulkan: &VulkanDevice,
        instance: &ClothGpuInstance,
    ) {
        let geom_info = vk::DescriptorBufferInfo::default()
            .buffer(instance.geometry_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let vel_info = vk::DescriptorBufferInfo::default()
            .buffer(instance.velocity_buffer.handle)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let col_info = vk::DescriptorBufferInfo::default()
            .buffer(instance.collider_buffer.handle)
            .offset(0)
            .range(vk::WHOLE_SIZE);

        let writes = [
            // Integrate set
            vk::WriteDescriptorSet::default()
                .dst_set(instance.descriptor_set_integrate)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&geom_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(instance.descriptor_set_integrate)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&vel_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(instance.descriptor_set_integrate)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&col_info)),
            // Solve set
            vk::WriteDescriptorSet::default()
                .dst_set(instance.descriptor_set_solve)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&geom_info)),
        ];
        unsafe { vulkan.device.update_descriptor_sets(&writes, &[]); }
    }

    /// Record all cloth dispatches for this frame into `cmd`.
    ///
    /// For each instance:
    ///   1. Integrate (forces + collisions) — one 2D dispatch.
    ///   2. Solve (constraint enforcement) — `solver_iterations` 2D dispatches.
    ///   3. Memory barrier so the geometry buffer is visible to the vertex stage.
    pub fn dispatch_all(
        &self,
        vulkan: &VulkanDevice,
        cmd: vk::CommandBuffer,
        instances: &[ClothGpuInstance],
        dt: f32,
    ) {
        if instances.is_empty() {
            return;
        }

        for inst in instances {
            let gx = inst.grid_width.div_ceil(16);
            let gy = inst.grid_height.div_ceil(16);

            let pc = ClothPushConstants {
                grid_width:    inst.grid_width,
                grid_height:   inst.grid_height,
                dt,
                stiffness:     inst.stiffness,
                damping:       inst.damping,
                num_colliders: 0, // collider upload is handled by application
            };

            let pc_bytes = unsafe {
                std::slice::from_raw_parts(
                    &pc as *const ClothPushConstants as *const u8,
                    std::mem::size_of::<ClothPushConstants>(),
                )
            };

            // ── Integrate pass ────────────────────────────────────────────────
            unsafe {
                vulkan.device.cmd_bind_pipeline(
                    cmd, vk::PipelineBindPoint::COMPUTE, self.integrate_pipeline,
                );
                vulkan.device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE, self.integrate_layout, 0,
                    std::slice::from_ref(&inst.descriptor_set_integrate), &[],
                );
                vulkan.device.cmd_push_constants(
                    cmd, self.integrate_layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes,
                );
                vulkan.device.cmd_dispatch(cmd, gx, gy, 1);
            }

            // Barrier: integrate writes → solve reads
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            unsafe {
                vulkan.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&barrier), &[], &[],
                );
            }

            // ── Solve passes ──────────────────────────────────────────────────
            unsafe {
                vulkan.device.cmd_bind_pipeline(
                    cmd, vk::PipelineBindPoint::COMPUTE, self.solve_pipeline,
                );
                vulkan.device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::COMPUTE, self.solve_layout, 0,
                    std::slice::from_ref(&inst.descriptor_set_solve), &[],
                );
                vulkan.device.cmd_push_constants(
                    cmd, self.solve_layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes,
                );
            }

            for _ in 0..inst.solver_iterations {
                unsafe { vulkan.device.cmd_dispatch(cmd, gx, gy, 1); }

                // Barrier between solver iterations
                unsafe {
                    vulkan.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        std::slice::from_ref(&barrier), &[], &[],
                    );
                }
            }

            // Final barrier: solve writes → vertex reads
            let final_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::VERTEX_ATTRIBUTE_READ);
            unsafe {
                vulkan.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::VERTEX_INPUT,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&final_barrier), &[], &[],
                );
            }
        }
    }

    /// Destroy all Vulkan objects owned by this pipeline.
    pub fn shutdown(&self, vulkan: &VulkanDevice) {
        unsafe {
            vulkan.device.destroy_pipeline(self.integrate_pipeline, None);
            vulkan.device.destroy_pipeline(self.solve_pipeline, None);
            vulkan.device.destroy_pipeline_layout(self.integrate_layout, None);
            vulkan.device.destroy_pipeline_layout(self.solve_layout, None);
            vulkan.device.destroy_descriptor_set_layout(self.descriptor_set_layout_integrate, None);
            vulkan.device.destroy_descriptor_set_layout(self.descriptor_set_layout_solve, None);
            vulkan.device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
