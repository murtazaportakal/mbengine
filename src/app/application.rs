//! The core game loop and application state.

use crate::ecs::World;
use crate::ecs::{
    CameraComponent, HierarchyComponent, LightComponent, RenderComponent, TransformComponent,
};
use crate::math::vec::Vec3;
use crate::memory::{MemoryConfig, MemorySubsystem};
use crate::platform::{win32, Timer, Window};
use crate::renderer::vulkan::RenderBackend;
use ash::vk;

/// High-level engine coordinator.
///
/// Render state is delegated to `RenderBackend` whose field declaration
/// order guarantees correct Vulkan resource drop order.
pub struct Application {
    pub world: World,
    pub render: RenderBackend,
    pub window: Window,
    pub asset_manager: crate::asset_manager::AssetManager,
    /// Pre-allocated flat map of entity-index → world matrix.
    pub world_matrices: std::collections::HashMap<crate::ecs::EntityId, crate::math::mat4::Mat4>,
    pub input: crate::app::input::Input,
    pub timer: Timer,
    pub memory: MemorySubsystem,
    pub ui_ctx: crate::ui::UiContext,
    pub ui_font: crate::ui::font::Font,
    pub physics: crate::physics::PhysicsSystem,
    #[cfg(feature = "editor")]
    pub selected_entity: Option<crate::ecs::EntityId>,
    #[cfg(feature = "editor")]
    pub editor: crate::app::editor::Editor,
    #[cfg(feature = "editor")]
    pub hot_reloader: Option<crate::app::hot_reload::HotReloader>,
    pub audio_system: crate::audio::AudioSystem,
    pub audio_subsystem: Option<crate::audio::AudioSubsystem>,
    pub script_engine: crate::scripting::engine::ScriptEngine,
    pub is_playing: bool,
}

impl Application {
    /// Initialize the engine subsystems.
    pub fn new(title: &str, width: i32, height: i32) -> Option<Self> {
        crate::utils::shader_compiler::compile_all_shaders();

        // 1. Initialize Memory
        let mut memory = MemorySubsystem::default();
        if !memory.init(MemoryConfig::default()) {
            return None;
        }

        // 2. Initialize Platform Window & Input
        let mut window = Window::new(title, width, height);
        let mut input = crate::app::input::Input::new();
        let timer = Timer::new();

        // 3. Initialize Asset Manager and Font
        let mut asset_manager = crate::asset_manager::AssetManager::new();
        let font_bytes = asset_manager
            .vfs
            .read_bytes("assets/fonts/Roboto-Regular.ttf")
            .unwrap_or_else(|_| vec![]);
        let ui_font = crate::ui::font::Font::load_ascii(&font_bytes, 64.0)
            .unwrap_or(crate::ui::font::Font {
                texture_data: vec![255; 1024 * 1024 * 4],
                width: 1024,
                height: 1024,
                glyphs: [crate::ui::font::GlyphInfo::default(); 128],
                line_height: 64.0,
            });

        // Poll once to consume the WM_SIZE from CreateWindow so the
        // window has its final dimensions.
        window.poll_events(&mut input);

        // 4. Initialize Render Backend (creates VulkanDevice, loads default
        //    textures into VRAM, builds descriptor pools/sets, etc.)
        let mut render = RenderBackend::new(&window, width, height, &mut asset_manager, &ui_font)?;

        // Ensure index 0 is a default white texture so untextured models render properly
        asset_manager.load_solid_color(&render.vulkan, "default_white", 255, 255, 255, 255);

        // If the OS resized the window during creation, recreate the
        // swapchain now so the first frame's acquire_next_image succeeds.
        if window.width as u32 != render.swapchain.extent.width
            || window.height as u32 != render.swapchain.extent.height
        {
            render.recreate_swapchain(&mut window, &mut input);
        }

        // Load cube model into geometry pool
        let cube_model_indices = asset_manager
            .load_model(&render.vulkan, &mut render.geometry_pool, "cube.obj")
            .unwrap_or(&[])
            .to_vec();
        if cube_model_indices.is_empty() {
            crate::log_info!("Failed to load cube.obj. Rendering will fail.");
        }

        if render.pipeline.is_none() {
            crate::log_info!("Shaders not found or failed to compile. Rendering will be skipped.");
        }

        // 4. Initialize ECS
        let mut world = unsafe { World::new(memory.persistent_arena()) };

        unsafe {
            world.register_component::<TransformComponent>(20000);
            world.register_component::<RenderComponent>(20000);
            world.register_component::<CameraComponent>(10);
            world.register_component::<LightComponent>(10);
            world.register_component::<crate::ecs::components::PointLightComponent>(10);
            world.register_component::<HierarchyComponent>(20000);
            world.register_component::<crate::ecs::components::RigidBodyComponent>(20000);
            world.register_component::<crate::ecs::components::ColliderComponent>(20000);
            world.register_component::<crate::ecs::components::AudioListenerComponent>(10);
            world.register_component::<crate::ecs::components::AudioEmitterComponent>(100);
            world.register_component::<crate::ecs::components::SkeletonComponent>(100);
            world.register_component::<crate::ecs::components::AnimatorComponent>(100);
            world.register_component::<crate::ecs::components::ScriptBehaviorComponent>(100);
            world.register_component::<crate::ecs::components::NameComponent>(10000);
        }

        let physics = crate::physics::PhysicsSystem::new();

        // Spawn camera first (entity 0) matching convention
        let camera_entity = world.create_entity();
        let mut cam_transform = TransformComponent::default();
        cam_transform.position = Vec3::new(0.0, 3.0, -8.0);
        cam_transform.rotation.x = -std::f32::consts::FRAC_PI_8;
        unsafe {
            world.add_component(camera_entity, cam_transform);
            let mut cam_comp = CameraComponent::default();
            cam_comp.proj = crate::math::mat4::Mat4::perspective(
                std::f32::consts::FRAC_PI_4,
                width as f32 / height as f32,
                0.1,
                100.0,
            );
            world.add_component(camera_entity, cam_comp);
            world.add_component(camera_entity, crate::ecs::components::AudioListenerComponent);
        }

        // Spawn a directional light
        let light_entity = world.create_entity();
        unsafe {
            world.add_component(
                light_entity,
                LightComponent {
                    direction: Vec3::new(0.5, -1.0, -0.5).normalize(),
                    color: Vec3::new(3.0, 3.0, 3.0),
                },
            );
        }

        // Spawn a single default cube entity at the origin, if loaded.
        // Spawn a single default cube entity at the origin, if loaded.
        if !cube_model_indices.is_empty() {
            let mesh_index = cube_model_indices[0];
            let cube_entity = world.create_entity();
            unsafe {
                world.add_component(
                    cube_entity,
                    TransformComponent {
                        position: Vec3::new(0.0, 0.0, 0.0),
                        rotation: Vec3::new(0.0, 0.0, 0.0),
                        scale: Vec3::new(1.0, 1.0, 1.0),
                        matrix: crate::math::mat4::Mat4::identity(),
                    },
                );
                world.add_component(
                    cube_entity,
                    RenderComponent {
                        visible: true,
                        mesh_index,
                        metallic: 0.0,
                        roughness: 0.8,
                        r: 0.8,
                        g: 0.8,
                        b: 0.8,
                    },
                );
                world.add_component(
                    cube_entity,
                    HierarchyComponent {
                        parent: None,
                        local_matrix: crate::math::mat4::Mat4::identity(),
                    },
                );
            }
        }

        let ui_ctx = crate::ui::UiContext::new();

        let audio_subsystem = if title == "Engine Boot Test" {
            None
        } else {
            crate::audio::AudioSubsystem::new()
        };
        let audio_system =
            crate::audio::AudioSystem::new(audio_subsystem.as_ref(), &asset_manager.vfs);
        let script_engine = crate::scripting::engine::ScriptEngine::new();

        let app = Self {
            world,
            render,
            window,
            asset_manager,
            world_matrices: std::collections::HashMap::with_capacity(10000),
            input,
            timer,
            memory,
            ui_ctx,
            ui_font,
            physics,
            #[cfg(feature = "editor")]
            selected_entity: None,
            #[cfg(feature = "editor")]
            editor: crate::app::editor::Editor::new(),
            #[cfg(feature = "editor")]
            hot_reloader: None,
            audio_system,
            audio_subsystem,
            script_engine,
            is_playing: false,
        };

        Some(app)
    }

    /// The canonical game loop.
    pub fn run(&mut self) {
        crate::log_info!("Application started.");
        let _ = self.timer.tick();

        while self.window.poll_events(&mut self.input) {
            let dt = self.timer.tick();
            let ppp = (self.window.height as f32 / 720.0).max(0.5);
            self.input.ui_scale = ppp;

            self.ui_ctx.begin_frame(
                crate::math::vec::Vec2::new(self.input.mouse_x as f32, self.input.mouse_y as f32),
                self.input.keys[0x01],
                self.input.mouse_scroll_y,
            );

            if self.is_playing {
                #[cfg(feature = "editor")]
                {
                    if let Some(reloader) = &mut self.hot_reloader {
                        reloader.update();
                        reloader.call_game_update(&mut self.world, &mut self.physics, dt as f32);
                    }
                }
                use crate::ecs::System;
                self.audio_system.update(dt as f32, &self.world);
                crate::ecs::animation_system::process_animations(
                    &self.world, &self.asset_manager, dt as f32,
                );
                crate::ecs::scripting_system::process_scripts(
                    &mut self.world, &self.asset_manager, &self.script_engine, &self.physics, dt as f32,
                );
            }

            let mut view_mat = crate::math::mat4::Mat4::identity();
            let mut proj_mat = crate::math::mat4::Mat4::identity();

            let cam_entity = {
                let cameras = self.world.get_component_array::<CameraComponent>();
                cameras.dense_entities_slice().first().copied()
            };
            if let Some(cam_entity) = cam_entity {
                let transforms = self.world.get_component_array::<TransformComponent>();
                if transforms.has(cam_entity) {
                    let transform = unsafe { transforms.get(cam_entity) };
                    let pitch = transform.rotation.x;
                    let yaw = transform.rotation.y;
                    let forward = crate::math::vec::Vec3::new(
                        yaw.sin() * pitch.cos(), pitch.sin(), yaw.cos() * pitch.cos(),
                    ).normalize();
                    let center = transform.position + forward;
                    view_mat = crate::math::mat4::Mat4::look_at(
                        transform.position, center, Vec3::new(0.0, 1.0, 0.0),
                    );
                    let ar = self.render.offscreen_target.width as f32
                        / self.render.offscreen_target.height as f32;
                    proj_mat = crate::math::mat4::Mat4::perspective(
                        std::f32::consts::FRAC_PI_4, ar, 0.1, 100.0,
                    );
                }
            }

            // --- Editor UI draw (editor feature only) ---
            #[cfg(feature = "editor")]
            let (actions, new_viewport_size, raycast_request, viewport_hovered) = self.editor.draw(
                &mut self.ui_ctx, &mut self.world, &mut self.physics,
                &mut self.selected_entity, &mut self.render.bloom_threshold,
                1.0 / dt as f32, self.render.last_visible_meshlets, self.is_playing,
                self.window.width as f32, self.window.height as f32,
                view_mat, proj_mat,
            );
            // In standalone builds there is no editor, so these default to empty.
            #[cfg(not(feature = "editor"))]
            let (actions, new_viewport_size, raycast_request, viewport_hovered):
                (Vec<()>, Option<(u32,u32)>, Option<(f32,f32)>, bool) =
                    (Vec::new(), None, None, true);

            #[cfg(feature = "editor")]
            for action in actions {
                match action {
                    crate::app::editor::EditorAction::Play => self.is_playing = true,
                    crate::app::editor::EditorAction::Pause => self.is_playing = false,
                    crate::app::editor::EditorAction::ToggleDebugCull => {
                        self.render.debug_cull = !self.render.debug_cull;
                    }
                    crate::app::editor::EditorAction::ToggleDebugMeshlets => {
                        self.render.debug_meshlets = !self.render.debug_meshlets;
                    }
                    crate::app::editor::EditorAction::SpawnEntity => {
                        let new_entity = self.world.create_entity();
                        unsafe {
                            self.world.add_component(new_entity, TransformComponent::default());
                            self.world.add_component(new_entity, HierarchyComponent::default());
                        }
                        self.selected_entity = Some(new_entity);
                    }
                    crate::app::editor::EditorAction::SpawnStressTest => {
                        let spacing = 2.0;
                        let grid_size = 100;
                        let start_x = -(grid_size as f32 * spacing) / 2.0;
                        let start_z = -(grid_size as f32 * spacing) / 2.0;
                        
                        for x in 0..grid_size {
                            for z in 0..grid_size {
                                let new_entity = self.world.create_entity();
                                unsafe {
                                    let mut transform = TransformComponent::default();
                                    transform.position = crate::math::vec::Vec3::new(
                                        start_x + x as f32 * spacing,
                                        0.0,
                                        start_z + z as f32 * spacing,
                                    );
                                    self.world.add_component(new_entity, transform);
                                    
                                    let mut render = crate::ecs::components::RenderComponent::default();
                                    render.r = (x as f32) / (grid_size as f32);
                                    render.g = (z as f32) / (grid_size as f32);
                                    render.b = 0.5;
                                    self.world.add_component(new_entity, render);
                                }
                            }
                        }
                    }
                    crate::app::editor::EditorAction::TranslateSelected(pos) => {
                        if let Some(entity) = self.selected_entity {
                            let transforms = self.world.get_component_array_mut::<TransformComponent>();
                            if transforms.has(entity) {
                                let transform = unsafe { transforms.get_mut(entity) };
                                transform.position = pos;
                                let phys = self.world.get_component_array::<crate::ecs::components::RigidBodyComponent>();
                                if phys.has(entity) {
                                    let pc = unsafe { phys.get(entity) };
                                    if let Some(body) = self.physics.rigid_body_set.get_mut(pc.handle) {
                                        body.set_position(
                                            rapier3d::math::Isometry::translation(pos.x, pos.y, pos.z),
                                            true,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    crate::app::editor::EditorAction::DeleteEntity(e) => {
                        let mut to_delete = vec![e];
                        {
                            let hierarchies = self.world.get_component_array::<HierarchyComponent>();
                            let mut i = 0;
                            while i < to_delete.len() {
                                let current = to_delete[i];
                                for (j, hier) in hierarchies.as_slice().iter().enumerate() {
                                    if hier.parent == Some(current) {
                                        let child = hierarchies.dense_entities_slice()[j];
                                        if !to_delete.contains(&child) {
                                            to_delete.push(child);
                                        }
                                    }
                                }
                                i += 1;
                            }
                        }
                        {
                            let colliders = self.world.get_component_array::<crate::ecs::components::ColliderComponent>();
                            let rigid_bodies = self.world.get_component_array::<crate::ecs::components::RigidBodyComponent>();
                            for &entity in &to_delete {
                                if colliders.has(entity) {
                                    let h = unsafe { colliders.get(entity) }.handle;
                                    self.physics.collider_set.remove(h, &mut self.physics.island_manager, &mut self.physics.rigid_body_set, true);
                                }
                                if rigid_bodies.has(entity) {
                                    let h = unsafe { rigid_bodies.get(entity) }.handle;
                                    self.physics.rigid_body_set.remove(h, &mut self.physics.island_manager, &mut self.physics.collider_set, &mut self.physics.impulse_joint_set, &mut self.physics.multibody_joint_set, true);
                                }
                            }
                        }
                        for &entity in &to_delete {
                            self.world.destroy_entity(entity);
                            if self.selected_entity == Some(entity) {
                                self.selected_entity = None;
                            }
                        }
                    }
                    crate::app::editor::EditorAction::AddComponent(e, name) => {
                        self.editor.registry.add_component(&name, e, &mut self.world, &mut self.physics);
                    }
                    crate::app::editor::EditorAction::SpawnModel(path) => {
                        let path_obj = std::path::Path::new(&path);
                        let mut mesh_indices = None;
                        let name = path_obj.file_stem().unwrap().to_string_lossy().to_string();
                        let path_str = path_obj.to_str().unwrap();

                        if path_obj.extension().map_or(false, |ext| ext == "gltf" || ext == "glb") {
                            crate::log_info!("[SpawnModel] Loading GLTF: {}", path_obj.display());
                            mesh_indices = Some(self.asset_manager.load_gltf(&self.render.vulkan, &mut self.render.geometry_pool, &name, path_str).unwrap_or(&[]).to_vec());
                        } else if path_obj.extension().map_or(false, |ext| ext == "obj") {
                            crate::log_info!("[SpawnModel] Loading OBJ: {}", path_obj.display());
                            mesh_indices = Some(self.asset_manager.load_model(&self.render.vulkan, &mut self.render.geometry_pool, path_str).unwrap_or(&[]).to_vec());
                        } else if path_obj.extension().map_or(false, |ext| ext == "mesh") {
                            crate::log_info!("[SpawnModel] Loading Cooked Mesh: {}", path_obj.display());
                            if let Some(idx) = self.asset_manager.load_cooked_mesh(&self.render.vulkan, &mut self.render.geometry_pool, path_str) {
                                mesh_indices = Some(vec![idx]);
                            }
                        } else if path_obj.extension().map_or(false, |ext| ext == "mat") {
                            crate::log_info!("[SpawnModel] Loading Material: {}", path_obj.display());
                            self.asset_manager.load_cooked_material(&self.render.vulkan, &name, path_str);
                            if let Some(entity) = self.selected_entity {
                                let renders = self.world.get_component_array_mut::<RenderComponent>();
                                if renders.has(entity) {
                                    if let Some(mat) = self.asset_manager.materials.get(&name).cloned() {
                                        let mesh_idx = {
                                            let r = unsafe { renders.get_mut(entity) };
                                            r.r = mat.base_color_factor[0];
                                            r.g = mat.base_color_factor[1];
                                            r.b = mat.base_color_factor[2];
                                            r.metallic = mat.metallic_factor;
                                            r.roughness = mat.roughness_factor;
                                            r.mesh_index
                                        };
                                        let mut albedo_idx = 0;
                                        let mut normal_idx = 0;
                                        let mut mr_idx = 0;
                                        let mut emissive_idx = 0;
                                        if let Some(tex) = &mat.albedo_texture {
                                            albedo_idx = self.asset_manager.texture_indices.get(tex).copied().unwrap_or(0) as u32;
                                        }
                                        if let Some(tex) = &mat.normal_texture {
                                            normal_idx = self.asset_manager.texture_indices.get(tex).copied().unwrap_or(0) as u32;
                                        }
                                        if let Some(tex) = &mat.mr_texture {
                                            mr_idx = self.asset_manager.texture_indices.get(tex).copied().unwrap_or(0) as u32;
                                        }
                                        if let Some(tex) = &mat.emissive_texture {
                                            emissive_idx = self.asset_manager.texture_indices.get(tex).copied().unwrap_or(0) as u32;
                                        }
                                        
                                        if let Some(mesh) = self.asset_manager.get_mesh_mut(mesh_idx) {
                                            if mat.albedo_texture.is_some() { mesh.diffuse_texture_idx = albedo_idx; }
                                            if mat.normal_texture.is_some() { mesh.normal_texture_idx = normal_idx; }
                                            if mat.mr_texture.is_some() { mesh.mr_texture_idx = mr_idx; }
                                            if mat.emissive_texture.is_some() { mesh.emissive_texture_idx = emissive_idx; }
                                        }
                                    }
                                }
                            }
                            continue; // Do not spawn a new entity for materials!
                        }

                        crate::log_info!("[SpawnModel] mesh_indices={:?}", mesh_indices);
                        unsafe { self.render.vulkan.device.device_wait_idle().unwrap(); }
                        // We must NOT call recreate_frame_buffers() here.
                        // Destroying the command pool mid-loop corrupts the
                        // validation layer's buffer-reference tracking and
                        // causes VUID-vkDestroyBuffer-buffer-00922 / DEVICE_LOST
                        // on the next submit.  device_wait_idle() is enough to
                        // guarantee that the GPU is not using the geometry pool
                        // while we append new meshes.
                        let textures_before = self.asset_manager.texture_indices.len();
                        let new_entity = self.world.create_entity();

                        crate::log_info!("[SpawnModel] mesh_indices={:?}", mesh_indices);

                        // Bindless texture update
                        {
                            let bs = self.render.global_texture_descriptor_sets[0];
                            if bs != vk::DescriptorSet::null() {
                                let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
                                let mut entries: Vec<(u32, usize)> = Vec::new();
                                for (name, &idx) in &self.asset_manager.texture_indices {
                                    if idx as usize >= textures_before {
                                        if let Some(tex) = self.asset_manager.get_texture(name) {
                                            let slot = image_infos.len();
                                            image_infos.push(vk::DescriptorImageInfo::default()
                                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                                .image_view(tex.view).sampler(tex.sampler));
                                            entries.push((idx, slot));
                                        }
                                    }
                                }
                                let mut writes: Vec<vk::WriteDescriptorSet<'_>> = Vec::new();
                                for (idx, slot) in &entries {
                                    writes.push(vk::WriteDescriptorSet::default()
                                        .dst_set(bs).dst_binding(0).dst_array_element(*idx)
                                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                        .image_info(std::slice::from_ref(&image_infos[*slot])));
                                }
                                if !writes.is_empty() {
                                    unsafe { self.render.vulkan.device.update_descriptor_sets(&writes, &[]); }
                                }
                            }
                        }

                        unsafe {
                            self.world.add_component(new_entity, TransformComponent {
                                position: Vec3::new(0.0, 0.0, 0.0),
                                rotation: Vec3::new(0.0, 0.0, 0.0),
                                scale: Vec3::new(1.0, 1.0, 1.0),
                                matrix: crate::math::mat4::Mat4::identity(),
                            });
                            self.world.add_component(new_entity, HierarchyComponent { parent: None, local_matrix: crate::math::mat4::Mat4::identity() });
                        }

                        if let Some(indices) = mesh_indices {
                            if indices.is_empty() {
                                crate::log_info!("[SpawnModel] ERROR: Mesh indices was EMPTY for {}", path);
                            } else if indices.len() == 1 {
                                let mesh_index = indices[0];
                                unsafe {
                                    let mut metallic = 0.0f32;
                                    let mut roughness = 0.8f32;
                                    let mut r = 1.0f32; let mut g = 1.0f32; let mut b = 1.0f32;
                                    if let Some(mesh) = self.asset_manager.get_mesh(mesh_index) {
                                        metallic = mesh.metallic;
                                        roughness = mesh.roughness;
                                        r = mesh.default_color[0];
                                        g = mesh.default_color[1];
                                        b = mesh.default_color[2];
                                        let hx = (mesh.aabb_max[0]-mesh.aabb_min[0]).abs()*0.5;
                                        let hy = (mesh.aabb_max[1]-mesh.aabb_min[1]).abs()*0.5;
                                        let hz = (mesh.aabb_max[2]-mesh.aabb_min[2]).abs()*0.5;
                                        let hx = hx.max(0.01); let hy = hy.max(0.01); let hz = hz.max(0.01);
                                        let collider = rapier3d::prelude::ColliderBuilder::cuboid(hx,hy,hz)
                                            .translation(rapier3d::math::Vector::new(mesh.aabb_min[0]+hx, mesh.aabb_min[1]+hy, mesh.aabb_min[2]+hz)).build();
                                        let handle = self.physics.collider_set.insert(collider);
                                        self.world.add_component(new_entity, crate::ecs::components::ColliderComponent { handle });
                                    }
                                    self.world.add_component(new_entity, RenderComponent {
                                        visible: true, mesh_index, metallic, roughness, r, g, b,
                                    });
                                }
                            } else {
                                for &mesh_index in &indices {
                                    let child = self.world.create_entity();
                                    unsafe {
                                        self.world.add_component(child, TransformComponent {
                                            position: Vec3::new(0.0, 0.0, 0.0),
                                            rotation: Vec3::new(0.0, 0.0, 0.0),
                                            scale: Vec3::new(1.0, 1.0, 1.0),
                                            matrix: crate::math::mat4::Mat4::identity(),
                                        });
                                        let mut metallic = 0.0f32;
                                        let mut roughness = 0.8f32;
                                        let mut r = 1.0f32; let mut g = 1.0f32; let mut b = 1.0f32;
                                        if let Some(mesh) = self.asset_manager.get_mesh(mesh_index) {
                                            metallic = mesh.metallic;
                                            roughness = mesh.roughness;
                                            r = mesh.default_color[0];
                                            g = mesh.default_color[1];
                                            b = mesh.default_color[2];
                                            let hx = (mesh.aabb_max[0]-mesh.aabb_min[0]).abs()*0.5;
                                            let hy = (mesh.aabb_max[1]-mesh.aabb_min[1]).abs()*0.5;
                                            let hz = (mesh.aabb_max[2]-mesh.aabb_min[2]).abs()*0.5;
                                            let hx = hx.max(0.01); let hy = hy.max(0.01); let hz = hz.max(0.01);
                                            let collider = rapier3d::prelude::ColliderBuilder::cuboid(hx,hy,hz)
                                                .translation(rapier3d::math::Vector::new(mesh.aabb_min[0]+hx, mesh.aabb_min[1]+hy, mesh.aabb_min[2]+hz)).build();
                                            let handle = self.physics.collider_set.insert(collider);
                                            self.world.add_component(child, crate::ecs::components::ColliderComponent { handle });
                                        }
                                        self.world.add_component(child, RenderComponent {
                                            visible: true, mesh_index, metallic, roughness, r, g, b,
                                        });
                                        self.world.add_component(child, HierarchyComponent {
                                            parent: Some(new_entity), local_matrix: crate::math::mat4::Mat4::identity(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Handle asset manager events (e.g. new textures loaded)
            let asset_events = self.asset_manager.poll_changes(&self.render.vulkan);
            if !self.asset_manager.new_textures_since_last_frame.is_empty() {
                unsafe { self.render.vulkan.device.device_wait_idle().unwrap(); }
                self.render.update_texture_descriptors(&self.asset_manager);
                self.asset_manager.new_textures_since_last_frame.clear();
            }
            let mut shader_changed = false;
            for event in asset_events {
                if matches!(event, crate::asset_manager::AssetEvent::ShaderChanged) { shader_changed = true; }
            }
            if shader_changed {
                unsafe { self.render.vulkan.device.device_wait_idle().unwrap(); }
                if let Some(mut old) = self.render.pipeline.take() { old.shutdown(&self.render.vulkan); }
                self.render.pipeline = crate::renderer::vulkan::Pipeline::new(
                    &self.render.vulkan, vk::Format::R16G16B16A16_SFLOAT, &self.asset_manager.vfs,
                    "shaders/vert.spv", "shaders/frag.spv",
                );
                if let Some(pipe) = &self.render.pipeline {
                    unsafe { self.render.vulkan.device.reset_descriptor_pool(self.render.descriptor_pool, vk::DescriptorPoolResetFlags::empty()).unwrap(); }
                    let layouts = [pipe.descriptor_set_layout, pipe.descriptor_set_layout];
                    let alloc = vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.render.descriptor_pool).set_layouts(&layouts);
                    if let Ok(sets) = unsafe { self.render.vulkan.device.allocate_descriptor_sets(&alloc) } {
                        self.render.descriptor_sets = [sets[0], sets[1]];
                        for i in 0..2 {
                            let ubo_size = std::mem::size_of::<crate::renderer::vulkan::pipeline::GlobalUbo>() as u64;
                            let ubo_info = vk::DescriptorBufferInfo::default().buffer(self.render.ubo_buffer).offset(0).range(ubo_size);
                            let env = vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(self.asset_manager.get_texture("env_default").unwrap().view).sampler(self.asset_manager.get_texture("env_default").unwrap().sampler);
                            let shd = vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(self.asset_manager.get_texture("shadow_default").unwrap().view).sampler(self.asset_manager.get_texture("shadow_default").unwrap().sampler);
                            let ib = vk::DescriptorBufferInfo::default().buffer(self.render.instance_buffers[i].handle).offset(0).range(vk::WHOLE_SIZE);
                            let writes = [
                                vk::WriteDescriptorSet::default().dst_set(self.render.descriptor_sets[i]).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(std::slice::from_ref(&ubo_info)),
                                vk::WriteDescriptorSet::default().dst_set(self.render.descriptor_sets[i]).dst_binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(std::slice::from_ref(&env)),
                                vk::WriteDescriptorSet::default().dst_set(self.render.descriptor_sets[i]).dst_binding(2).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(std::slice::from_ref(&shd)),
                                vk::WriteDescriptorSet::default().dst_set(self.render.descriptor_sets[i]).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&ib)),
                            ];
                            unsafe { self.render.vulkan.device.update_descriptor_sets(&writes, &[]); }
                        }
                    }
                    for set in self.render.material_descriptor_sets.iter_mut() { *set = None; }
                }
            }

            // Grow per-mesh descriptor slots when new meshes are loaded
            let mesh_count = self.asset_manager.meshes.len();
            if self.render.material_descriptor_sets.len() < mesh_count {
                self.render.material_descriptor_sets.resize(mesh_count, None);
            }
            if self.render.compute_descriptor_sets.len() < mesh_count {
                self.render.compute_descriptor_sets.resize(mesh_count, None);
            }

            // Viewport resize (editor only — standalone uses full swapchain)
            #[cfg(feature = "editor")]
            if let Some((w, h)) = new_viewport_size {
                if w != self.render.offscreen_target.width || h != self.render.offscreen_target.height {
                    unsafe { self.render.vulkan.device.device_wait_idle().unwrap(); }
                    self.render.vulkan.recreate_frame_buffers();
                    self.render.offscreen_target.shutdown(&self.render.vulkan);
                    self.render.offscreen_target = crate::renderer::vulkan::OffscreenTarget::new(
                        &self.render.vulkan, w, h, vk::Format::R16G16B16A16_SFLOAT).unwrap();
                    self.render.sdr_target.shutdown(&self.render.vulkan);
                    self.render.sdr_target = crate::renderer::vulkan::OffscreenTarget::new(
                        &self.render.vulkan, w, h, vk::Format::R8G8B8A8_UNORM).unwrap();
                    self.render.bloom_target.shutdown(&self.render.vulkan);
                    self.render.bloom_target = crate::renderer::vulkan::bloom::BloomTarget::new(
                        &self.render.vulkan, (w/2).max(1), (h/2).max(1), 6).unwrap();
                    self.render.update_post_process_descriptors();
                    self.render.ui_backend.update_user_texture(
                        &self.render.vulkan, self.render.offscreen_texture_id,
                        self.render.sdr_target.color_view, self.render.sdr_target.sampler);
                }
            }

            // Raycast (editor only — requires selected_entity)
            #[cfg(feature = "editor")]
            if let Some((ndc_x, ndc_y)) = raycast_request {
                let cam_entity = {
                    let cameras = self.world.get_component_array::<CameraComponent>();
                    cameras.dense_entities_slice().first().copied()
                };
                if let Some(cam_entity) = cam_entity {
                    let transforms = self.world.get_component_array::<TransformComponent>();
                    let cameras = self.world.get_component_array::<CameraComponent>();
                    if transforms.has(cam_entity) && cameras.has(cam_entity) {
                        let transform = unsafe { transforms.get(cam_entity) };
                        let pitch = transform.rotation.x;
                        let yaw = transform.rotation.y;
                        let forward = Vec3::new(yaw.sin()*pitch.cos(), pitch.sin(), yaw.cos()*pitch.cos()).normalize();
                        let center = transform.position + forward;
                        let view = crate::math::mat4::Mat4::look_at(transform.position, center, Vec3::new(0.0,1.0,0.0));
                        let ar = self.render.offscreen_target.width as f32 / self.render.offscreen_target.height as f32;
                        let proj = crate::math::mat4::Mat4::perspective(std::f32::consts::FRAC_PI_4, ar, 0.1, 10000.0);
                        if let (Some(inv_proj), Some(inv_view)) = (proj.try_inverse(), view.try_inverse()) {
                            let mut target = inv_proj * crate::math::vec::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
                            if target.w != 0.0 { target.x /= target.w; target.y /= target.w; target.z /= target.w; target.w = 1.0; }
                            let world_target = inv_view * target;
                            let world_dir = Vec3::new(world_target.x-transform.position.x, world_target.y-transform.position.y, world_target.z-transform.position.z).normalize();
                            let ray = rapier3d::prelude::Ray::new(
                                rapier3d::math::Point::new(transform.position.x, transform.position.y, transform.position.z),
                                rapier3d::math::Vector::new(world_dir.x, world_dir.y, world_dir.z),
                            );
                            if let Some((handle, _)) = self.physics.query_pipeline.cast_ray(
                                &self.physics.rigid_body_set, &self.physics.collider_set, &ray, 10000.0, true, rapier3d::prelude::QueryFilter::default(),
                            ) {
                                let colliders = self.world.get_component_array::<crate::ecs::components::ColliderComponent>();
                                for (i, col) in colliders.as_slice().iter().enumerate() {
                                    if col.handle == handle {
                                        let mut sel = colliders.dense_entities_slice()[i];
                                        let hierarchies = self.world.get_component_array::<HierarchyComponent>();
                                        while let Some(parent) = unsafe { hierarchies.get(sel) }.parent { sel = parent; }
                                        self.selected_entity = Some(sel);
                                        break;
                                    }
                                }
                            } else {
                                let mut best_dist = f32::MAX;
                                let mut best_entity = None;
                                let renders = self.world.get_component_array::<RenderComponent>();
                                for entity in renders.dense_entities_slice().iter().copied() {
                                    if let Some(matrix) = self.world_matrices.get(&entity) {
                                        let center = Vec3::new(matrix.cols[3].x, matrix.cols[3].y, matrix.cols[3].z);
                                        let sx = Vec3::new(matrix.cols[0].x, matrix.cols[0].y, matrix.cols[0].z).length();
                                        let sy = Vec3::new(matrix.cols[1].x, matrix.cols[1].y, matrix.cols[1].z).length();
                                        let sz = Vec3::new(matrix.cols[2].x, matrix.cols[2].y, matrix.cols[2].z).length();
                                        let radius = sx.max(sy).max(sz) * 1.5;
                                        let l = center - transform.position;
                                        let tca = l.dot(world_dir);
                                        if tca >= 0.0 {
                                            let d2 = l.length_sq() - tca * tca;
                                            if d2 <= radius * radius {
                                                let thc = (radius * radius - d2).sqrt();
                                                let t = tca - thc;
                                                if t >= 0.0 && t < best_dist {
                                                    best_dist = t;
                                                    best_entity = Some(entity);
                                                }
                                            }
                                        }
                                    }
                                }
                                if let Some(mut sel) = best_entity {
                                    let hierarchies = self.world.get_component_array::<HierarchyComponent>();
                                    while let Some(parent) = unsafe { hierarchies.get(sel) }.parent { sel = parent; }
                                    self.selected_entity = Some(sel);
                                } else {
                                    self.selected_entity = None;
                                }
                            }
                        }
                    }
                }
            }

            if self.input.is_key_pressed(win32::VK_ESCAPE) { break; }
            #[cfg(feature = "editor")]
            if self.input.is_key_pressed(win32::VK_F5) { crate::ecs::serialization::save_scene(&self.world, &self.editor.registry, "scene.json"); }
            #[cfg(feature = "editor")]
            if self.input.is_key_pressed(win32::VK_F9) { crate::ecs::serialization::load_scene(&mut self.world, &self.editor.registry, "scene.json"); }

            // Camera update
            {
                let cam_entity = {
                    let cameras = self.world.get_component_array::<CameraComponent>();
                    cameras.dense_entities_slice().first().copied()
                };
                if let Some(cam_entity) = cam_entity {
                    let transforms = self.world.get_component_array_mut::<TransformComponent>();
                    if transforms.has(cam_entity) {
                        let transform = unsafe { transforms.get_mut(cam_entity) };
                        if viewport_hovered && self.input.is_key_down(win32::VK_RBUTTON) {
                            let s = 0.001;
                            transform.rotation.y += self.input.mouse_dx as f32 * s;
                            transform.rotation.x -= self.input.mouse_dy as f32 * s;
                            let mp = std::f32::consts::FRAC_PI_2 - 0.01;
                            transform.rotation.x = transform.rotation.x.clamp(-mp, mp);
                        }
                        let pitch = transform.rotation.x; let yaw = transform.rotation.y;
                        let forward = Vec3::new(yaw.sin()*pitch.cos(), pitch.sin(), yaw.cos()*pitch.cos()).normalize();
                        let right = forward.cross(Vec3::new(0.0,1.0,0.0)).normalize();
                        let speed = 2.0 * dt as f32;
                        if viewport_hovered {
                            if self.input.is_key_down(win32::VK_W) { transform.position += forward * speed; }
                            if self.input.is_key_down(win32::VK_S) { transform.position -= forward * speed; }
                            if self.input.is_key_down(win32::VK_A) { transform.position -= right * speed; }
                            if self.input.is_key_down(win32::VK_D) { transform.position += right * speed; }
                            if self.input.is_key_down(win32::VK_TAB) { transform.position.y += speed; }
                            if self.input.is_key_down(win32::VK_SHIFT) { transform.position.y -= speed; }
                        }
                    }
                }
            }

            // World matrices
            self.world_matrices.clear();
            let transforms = self.world.get_component_array_mut::<TransformComponent>();
            let entities = transforms.dense_entities();
            for (i, t) in transforms.as_mut_slice().iter_mut().enumerate() {
                let entity = unsafe { *entities.add(i) };
                t.update_matrix();
                self.world_matrices.insert(entity, t.matrix);
            }
            let hierarchies = self.world.get_component_array::<HierarchyComponent>();

            for (i, hier) in hierarchies.as_slice().iter().enumerate() {
                let entity = hierarchies.dense_entities_slice()[i];
                if let Some(parent) = hier.parent {
                    if let Some(pw) = self.world_matrices.get(&parent).copied() {
                        if let Some(cl) = self.world_matrices.get(&entity).copied() {
                            self.world_matrices.insert(entity, pw * cl);
                        }
                    }
                }
            }

            self.ui_ctx.end_frame();
            self.render.render_frame(
                &mut self.window, &self.world, &self.world_matrices,
                &self.asset_manager, &self.ui_ctx, &self.ui_font, &mut self.input,
            );
            self.render.current_frame = (self.render.current_frame + 1) % 2;
            self.memory.frame_arena().reset(false);
            
        // Removed spammy debug

        }

        crate::log_info!("Application shutting down.");
        self.render.vulkan.wait_idle();
        self.asset_manager.shutdown(&self.render.vulkan);
        // RenderBackend::drop handles all Vulkan resource cleanup in correct order
    }
}