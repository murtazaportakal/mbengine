use crate::asset_manager::AssetManager;
use crate::ecs::{
    components::{AnimationState, AnimatorComponent, SkeletonComponent},
    World,
};
use crate::renderer::vulkan::skeleton::SkeletonPose;

// ── Duration helpers ──────────────────────────────────────────────────────────

/// Get the maximum duration of a given state for looping/stopping logic.
fn get_state_duration(
    state: &AnimationState,
    skeleton_name: &str,
    asset_manager: &AssetManager,
) -> f32 {
    match state {
        AnimationState::Clip { clip_name } => {
            asset_manager
                .get_animation_clip(skeleton_name, clip_name.as_str())
                .map(|c| c.duration)
                .unwrap_or(0.0)
        }
        AnimationState::Blend1D { clip_a, clip_b, .. } => {
            let da = asset_manager
                .get_animation_clip(skeleton_name, clip_a.as_str())
                .map(|c| c.duration)
                .unwrap_or(0.0);
            let db = asset_manager
                .get_animation_clip(skeleton_name, clip_b.as_str())
                .map(|c| c.duration)
                .unwrap_or(0.0);
            da.max(db)
        }
        AnimationState::Blend2D {
            clip_bl, clip_br, clip_tl, clip_tr, ..
        } => {
            [clip_bl, clip_br, clip_tl, clip_tr]
                .iter()
                .filter_map(|name| {
                    asset_manager
                        .get_animation_clip(skeleton_name, name.as_str())
                        .map(|c| c.duration)
                })
                .fold(0.0f32, f32::max)
        }
    }
}

// ── Pose samplers ─────────────────────────────────────────────────────────────

/// Sample a clip at time `t`, writing into `out_pose`.
fn sample_clip(
    clip_name: &str,
    time: f32,
    skeleton_name: &str,
    asset_manager: &AssetManager,
    bone_count: usize,
    out_pose: &mut SkeletonPose,
) {
    if let Some(clip) = asset_manager.get_animation_clip(skeleton_name, clip_name) {
        let t = if clip.duration > 0.0 { time % clip.duration } else { 0.0 };
        clip.sample_pose(t, bone_count, out_pose);
    } else {
        *out_pose = SkeletonPose::new(bone_count);
    }
}

/// Sample an `AnimationState` into a `SkeletonPose`.
fn sample_state(
    state: &AnimationState,
    time: f32,
    skeleton_name: &str,
    asset_manager: &AssetManager,
    bone_count: usize,
    out_pose: &mut SkeletonPose,
) {
    match state {
        AnimationState::Clip { clip_name } => {
            sample_clip(clip_name.as_str(), time, skeleton_name, asset_manager, bone_count, out_pose);
        }

        AnimationState::Blend1D { clip_a, clip_b, weight } => {
            let mut pose_a = SkeletonPose::new(bone_count);
            let mut pose_b = SkeletonPose::new(bone_count);
            sample_clip(clip_a.as_str(), time, skeleton_name, asset_manager, bone_count, &mut pose_a);
            sample_clip(clip_b.as_str(), time, skeleton_name, asset_manager, bone_count, &mut pose_b);
            SkeletonPose::blend(&pose_a, &pose_b, *weight, out_pose);
        }

        AnimationState::Blend2D {
            clip_bl, clip_br, clip_tl, clip_tr, param_x, param_y,
        } => {
            // Bilinear interpolation:
            //   lerp( lerp(bl, br, x), lerp(tl, tr, x), y )
            let mut pose_bl = SkeletonPose::new(bone_count);
            let mut pose_br = SkeletonPose::new(bone_count);
            let mut pose_tl = SkeletonPose::new(bone_count);
            let mut pose_tr = SkeletonPose::new(bone_count);

            sample_clip(clip_bl.as_str(), time, skeleton_name, asset_manager, bone_count, &mut pose_bl);
            sample_clip(clip_br.as_str(), time, skeleton_name, asset_manager, bone_count, &mut pose_br);
            sample_clip(clip_tl.as_str(), time, skeleton_name, asset_manager, bone_count, &mut pose_tl);
            sample_clip(clip_tr.as_str(), time, skeleton_name, asset_manager, bone_count, &mut pose_tr);

            // Bottom row blend
            let mut pose_bottom = SkeletonPose::new(bone_count);
            SkeletonPose::blend(&pose_bl, &pose_br, *param_x, &mut pose_bottom);

            // Top row blend
            let mut pose_top = SkeletonPose::new(bone_count);
            SkeletonPose::blend(&pose_tl, &pose_tr, *param_x, &mut pose_top);

            // Vertical blend
            SkeletonPose::blend(&pose_bottom, &pose_top, *param_y, out_pose);
        }
    }
}

// ── Main animation processor ──────────────────────────────────────────────────

/// Process all active animations, advance their timers, evaluate state machines,
/// and compute local-to-model bone matrices for GPU skinning.
pub fn process_animations(world: &World, asset_manager: &AssetManager, dt: f32) {
    // SAFETY: this is the only system that mutates AnimatorComponent and
    // SkeletonComponent this frame; the Scheduler enforces exclusivity.
    let animators_mut = unsafe { &mut *world.get_component_array_mut_ptr::<AnimatorComponent>() };
    let skeletons_mut = unsafe { &mut *world.get_component_array_mut_ptr::<SkeletonComponent>() };
    let entities = animators_mut.dense_entities();

    for (i, animator) in animators_mut.as_mut_slice().iter_mut().enumerate() {
        let entity = unsafe { *entities.add(i) };

        if !animator.is_playing {
            continue;
        }

        if !skeletons_mut.has(entity) {
            continue;
        }

        let skeleton_comp = unsafe { skeletons_mut.get_mut(entity) };
        let dt_scaled = dt * animator.speed;
        animator.current_time += dt_scaled;

        let skeleton_name = skeleton_comp.skeleton_name.as_str();

        // ── State machine evaluation ─────────────────────────────────────────
        // If a state machine is present, check transitions each frame.
        // We must call evaluate_transitions via a reborrow to avoid holding an
        // immutable ref across the mutable crossfade_to call.
        if let Some(sm) = &animator.state_machine {
            if let Some((target_state, blend_dur)) = sm.evaluate_transitions() {
                let new_state = target_state.clone();
                let dur = blend_dur;
                // Commit the new state name for future transition evaluations.
                let new_name = {
                    // Find the name of this target state in the state machine.
                    // This is the reverse lookup: state → name.
                    let sm2 = animator.state_machine.as_ref().unwrap();
                    sm2.states.as_slice()
                        .iter()
                        .find(|(_, s)| std::ptr::eq(s, target_state))
                        .map(|(n, _)| n.clone())
                };
                animator.crossfade_to(new_state, dur);
                if let (Some(sm2), Some(name)) = (animator.state_machine.as_mut(), new_name) {
                    sm2.commit_transition(name.as_str());
                }
            }
        }

        if let Some(skeleton) = asset_manager.get_skeleton(skeleton_name) {
            let duration = get_state_duration(&animator.state, skeleton_name, asset_manager);
            if animator.current_time > duration && duration > 0.0 {
                if animator.is_looping {
                    animator.current_time %= duration;
                } else {
                    animator.current_time = duration;
                    if animator.target_state.is_none() {
                        animator.is_playing = false;
                    }
                }
            }

            let bone_count = skeleton.bone_count();

            // 1. Sample current state
            let mut current_pose = SkeletonPose::new(bone_count);
            sample_state(
                &animator.state,
                animator.current_time,
                skeleton_name,
                asset_manager,
                bone_count,
                &mut current_pose,
            );

            // 2. Handle crossfade
            let mut final_pose = current_pose;

            if let Some(target) = &animator.target_state {
                animator.transition_time += dt_scaled;
                animator.crossfade_current += dt_scaled;

                let mut target_pose = SkeletonPose::new(bone_count);
                sample_state(
                    target,
                    animator.transition_time,
                    skeleton_name,
                    asset_manager,
                    bone_count,
                    &mut target_pose,
                );

                let mut blended_pose = SkeletonPose::new(bone_count);
                let blend_weight =
                    (animator.crossfade_current / animator.crossfade_duration).clamp(0.0, 1.0);

                SkeletonPose::blend(&final_pose, &target_pose, blend_weight, &mut blended_pose);
                final_pose = blended_pose;

                // Check if transition is complete
                if animator.crossfade_current >= animator.crossfade_duration {
                    animator.state = animator.target_state.take().unwrap();
                    animator.current_time = animator.transition_time;
                }
            }

            // 3. Convert to matrices and compute final bone matrices
            let mut local_transforms = crate::containers::FixedArray::<
                crate::math::mat4::Mat4,
                { crate::renderer::vulkan::skeleton::MAX_BONES },
            >::new();
            final_pose.to_matrices(&mut local_transforms);
            skeleton.compute_bone_matrices(&local_transforms, &mut skeleton_comp.computed_matrices);
        }
    }
}
