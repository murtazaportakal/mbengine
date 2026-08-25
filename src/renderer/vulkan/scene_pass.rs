use ash::vk;
use crate::renderer::vulkan::render_graph::{RenderGraph, ResourceHandle, ResourceState, PassResource};
use crate::renderer::vulkan::{VulkanDevice, Pipeline, GeometryPool, OffscreenTarget};

pub struct ScenePass {
    pub pipeline: Pipeline,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: [vk::DescriptorSet; 2],
    pub global_texture_descriptor_sets: [vk::DescriptorSet; 2],
    pub compute_descriptor_sets: Vec<Option<vk::DescriptorSet>>,
    pub material_descriptor_sets: Vec<Option<vk::DescriptorSet>>,
}

impl ScenePass {
    pub fn new(
        pipeline: Pipeline,
        descriptor_pool: vk::DescriptorPool,
        descriptor_sets: [vk::DescriptorSet; 2],
        global_texture_descriptor_sets: [vk::DescriptorSet; 2],
        compute_descriptor_sets: Vec<Option<vk::DescriptorSet>>,
        material_descriptor_sets: Vec<Option<vk::DescriptorSet>>,
    ) -> Self {
        Self {
            pipeline,
            descriptor_pool,
            descriptor_sets,
            global_texture_descriptor_sets,
            compute_descriptor_sets,
            material_descriptor_sets,
        }
    }

    pub fn update_descriptor_set(
        &self,
        vulkan: &VulkanDevice,
        frame: usize,
        ubo_buffer: vk::Buffer,
        instance_buffer: vk::Buffer,
        anim_bone_matrices_buffer: vk::Buffer,
    ) {
        if ubo_buffer == vk::Buffer::null() || instance_buffer == vk::Buffer::null() || self.descriptor_sets[frame] == vk::DescriptorSet::null() {
            return;
        }

        let ubo_info = vk::DescriptorBufferInfo::default().buffer(ubo_buffer).offset(0).range(vk::WHOLE_SIZE);
        let instance_info = vk::DescriptorBufferInfo::default().buffer(instance_buffer).offset(0).range(vk::WHOLE_SIZE);
        
        let write_ubo = vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[frame]).dst_binding(0).dst_array_element(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(std::slice::from_ref(&ubo_info));
        let write_instance = vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[frame]).dst_binding(3).dst_array_element(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&instance_info));
        
        let mut writes = vec![write_ubo, write_instance];

        let bone_info;
        let write_bones;
        if anim_bone_matrices_buffer != vk::Buffer::null() {
            bone_info = vk::DescriptorBufferInfo::default().buffer(anim_bone_matrices_buffer).offset(0).range(vk::WHOLE_SIZE);
            write_bones = vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[frame]).dst_binding(7).dst_array_element(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&bone_info));
            writes.push(write_bones);
        }

        unsafe { vulkan.device.update_descriptor_sets(&writes, &[]); }
    }

    pub fn update_bindless_texture(&self, vulkan: &VulkanDevice, texture_view: vk::ImageView, sampler: vk::Sampler, index: u32) {
        if self.global_texture_descriptor_sets[0] == vk::DescriptorSet::null() { return; }
        
        let image_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(texture_view)
            .sampler(sampler);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.global_texture_descriptor_sets[0])
            .dst_binding(0)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&image_info));
        unsafe { vulkan.device.update_descriptor_sets(&[write], &[]); }
    }

    pub fn reload_shaders(
        &mut self,
        vulkan: &VulkanDevice,
        asset_manager: &crate::asset_manager::AssetManager,
    ) -> bool {
        let new_pipeline = Pipeline::new(
            vulkan,
            vk::Format::R16G16B16A16_SFLOAT,
            &asset_manager.vfs,
            "shaders/vert.spv",
            "shaders/frag.spv",
        );
        if let Some(p) = new_pipeline {
            self.pipeline.shutdown(vulkan);
            self.pipeline = p;
            
            unsafe { vulkan.device.reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty()).unwrap(); }
            
            let layouts = [self.pipeline.descriptor_set_layout, self.pipeline.descriptor_set_layout];
            let alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool).set_layouts(&layouts);
            
            if let Ok(sets) = unsafe { vulkan.device.allocate_descriptor_sets(&alloc) } {
                self.descriptor_sets = [sets[0], sets[1]];
                return true;
            }
        }
        false
    }

    pub fn record_phase0<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        vulkan: &'a VulkanDevice,
        offscreen_target: &'a OffscreenTarget,
        geometry_pool: &'a GeometryPool,
        indirect_buffer: vk::Buffer,
        draw_count_buffer: vk::Buffer,
        max_instances: u32,
        current_frame: usize,
    ) {
        let clear_values = [
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.05, 0.05, 0.05, 1.0] } },
            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
        ];

        graph.add_pass("Scene", vec![
            PassResource {
                handle: ResourceHandle(offscreen_target.color_image),
                state: ResourceState { layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE, aspect_mask: vk::ImageAspectFlags::COLOR },
            },
            PassResource {
                handle: ResourceHandle(offscreen_target.depth_image),
                state: ResourceState { layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS, access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE, aspect_mask: vk::ImageAspectFlags::DEPTH },
            },
        ], move |cb| unsafe {
            let color_att = vk::RenderingAttachmentInfoKHR::default()
                .image_view(offscreen_target.color_view).image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::STORE).clear_value(clear_values[0]);
            let depth_att = vk::RenderingAttachmentInfoKHR::default()
                .image_view(offscreen_target.depth_view).image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::STORE).clear_value(clear_values[1]);
            let ri = vk::RenderingInfoKHR::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: offscreen_target.width, height: offscreen_target.height } })
                .layer_count(1).color_attachments(std::slice::from_ref(&color_att)).depth_attachment(&depth_att);
            
            vulkan.device.cmd_begin_rendering(cb, &ri);
            vulkan.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle);
            
            if self.descriptor_sets[current_frame] != vk::DescriptorSet::null() {
                vulkan.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.layout, 0, std::slice::from_ref(&self.descriptor_sets[current_frame]), &[]);
            }
            
            let vp = vk::Viewport { x: 0.0, y: 0.0, width: offscreen_target.width as f32, height: offscreen_target.height as f32, min_depth: 0.0, max_depth: 1.0 };
            vulkan.device.cmd_set_viewport(cb, 0, std::slice::from_ref(&vp));
            let sc = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: offscreen_target.width, height: offscreen_target.height } };
            vulkan.device.cmd_set_scissor(cb, 0, std::slice::from_ref(&sc));
            vulkan.device.cmd_bind_vertex_buffers(cb, 0, &[geometry_pool.vertex_buffer.handle], &[0]);
            vulkan.device.cmd_bind_index_buffer(cb, geometry_pool.index_buffer.handle, 0, vk::IndexType::UINT32);
            
            if self.global_texture_descriptor_sets[0] != vk::DescriptorSet::null() {
                vulkan.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.layout, 1, std::slice::from_ref(&self.global_texture_descriptor_sets[0]), &[]);
            }
            
            vulkan.device.cmd_draw_indexed_indirect_count(
                cb, indirect_buffer, 0,
                draw_count_buffer, 0, max_instances,
                std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
            vulkan.device.cmd_end_rendering(cb);
        });
    }

    pub fn record_phase2<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        vulkan: &'a VulkanDevice,
        offscreen_target: &'a OffscreenTarget,
        geometry_pool: &'a GeometryPool,
        indirect_buffer_phase2: vk::Buffer,
        draw_count_buffer_phase2: vk::Buffer,
        max_instances: u32,
        current_frame: usize,
    ) {
        graph.add_pass("Scene_Phase2", vec![
            PassResource {
                handle: ResourceHandle(offscreen_target.color_image),
                state: ResourceState { layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE, aspect_mask: vk::ImageAspectFlags::COLOR },
            },
            PassResource {
                handle: ResourceHandle(offscreen_target.depth_image),
                state: ResourceState { layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS, access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE, aspect_mask: vk::ImageAspectFlags::DEPTH },
            },
        ], move |cb| unsafe {
            let color_att = vk::RenderingAttachmentInfoKHR::default()
                .image_view(offscreen_target.color_view).image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD).store_op(vk::AttachmentStoreOp::STORE);
            let depth_att = vk::RenderingAttachmentInfoKHR::default()
                .image_view(offscreen_target.depth_view).image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD).store_op(vk::AttachmentStoreOp::STORE);
            let ri = vk::RenderingInfoKHR::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: offscreen_target.width, height: offscreen_target.height } })
                .layer_count(1).color_attachments(std::slice::from_ref(&color_att)).depth_attachment(&depth_att);
            
            vulkan.device.cmd_begin_rendering(cb, &ri);
            vulkan.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle);
            
            if self.descriptor_sets[current_frame] != vk::DescriptorSet::null() {
                vulkan.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.layout, 0, std::slice::from_ref(&self.descriptor_sets[current_frame]), &[]);
            }
            
            let vp = vk::Viewport { x: 0.0, y: 0.0, width: offscreen_target.width as f32, height: offscreen_target.height as f32, min_depth: 0.0, max_depth: 1.0 };
            vulkan.device.cmd_set_viewport(cb, 0, std::slice::from_ref(&vp));
            let sc = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: offscreen_target.width, height: offscreen_target.height } };
            vulkan.device.cmd_set_scissor(cb, 0, std::slice::from_ref(&sc));
            vulkan.device.cmd_bind_vertex_buffers(cb, 0, &[geometry_pool.vertex_buffer.handle], &[0]);
            vulkan.device.cmd_bind_index_buffer(cb, geometry_pool.index_buffer.handle, 0, vk::IndexType::UINT32);
            
            if self.global_texture_descriptor_sets[0] != vk::DescriptorSet::null() {
                vulkan.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.layout, 1, std::slice::from_ref(&self.global_texture_descriptor_sets[0]), &[]);
            }
            
            vulkan.device.cmd_draw_indexed_indirect_count(
                cb, indirect_buffer_phase2, 0,
                draw_count_buffer_phase2, 0, max_instances,
                std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
            vulkan.device.cmd_end_rendering(cb);
        });
    }

    pub fn shutdown(&mut self, vulkan: &VulkanDevice) {
        unsafe {
            if self.descriptor_pool != vk::DescriptorPool::null() {
                vulkan.device.destroy_descriptor_pool(self.descriptor_pool, None);
            }
        }
        self.pipeline.shutdown(vulkan);
    }
}
