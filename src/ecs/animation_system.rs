use crate::asset_manager::AssetManager;
use crate::ecs::{
    components::{AnimationState, AnimatorComponent, SkeletonComponent},
    World,
};
use crate::renderer::vulkan::skeleton::SkeletonPose;

/// Get the maximum duration of a given state for looping/stopping logic.
fn get_state_duration(
    state: &AnimationState,
    skeleton_name: &str,
    asset_manager: &AssetManager,
) -> f32 {
    match state {
        AnimationState::Clip { clip_name } => {
            if let Some(clip) = asset_manager.get_animation_clip(skeleton_name, clip_name.as_str())
            {
                clip.duration
            } else {
                0.0
            }
        }
        AnimationState::Blend1D { clip_a, clip_b, .. } => {
            let dur_a = if let Some(clip) =
                asset_manager.get_animation_clip(skeleton_name, clip_a.as_str())
            {
                clip.duration
            } else {
                0.0
            };
            let dur_b = if let Some(clip) =
                asset_manager.get_animation_clip(skeleton_name, clip_b.as_str())
            {
                clip.duration
            } else {
                0.0
            };
            dur_a.max(dur_b)
        }
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
            if let Some(clip) = asset_manager.get_animation_clip(skeleton_name, clip_name.as_str())
            {
                let t = if clip.duration > 0.0 {
                    time % clip.duration
                } else {
                    0.0
                };
                clip.sample_pose(t, bone_count, out_pose);
            } else {
                *out_pose = SkeletonPose::new(bone_count);
            }
        }
        AnimationState::Blend1D {
            clip_a,
            clip_b,
            weight,
        } => {
            let mut pose_a = SkeletonPose::new(bone_count);
            let mut pose_b = SkeletonPose::new(bone_count);

            if let Some(clip) = asset_manager.get_animation_clip(skeleton_name, clip_a.as_str()) {
                let t = if clip.duration > 0.0 {
                    time % clip.duration
                } else {
                    0.0
                };
                clip.sample_pose(t, bone_count, &mut pose_a);
            }
            if let Some(clip) = asset_manager.get_animation_clip(skeleton_name, clip_b.as_str()) {
                let t = if clip.duration > 0.0 {
                    time % clip.duration
                } else {
                    0.0
                };
                clip.sample_pose(t, bone_count, &mut pose_b);
            }

            SkeletonPose::blend(&pose_a, &pose_b, *weight, out_pose);
        }
    }
}

/// Process all active animations, advance their timers, and compute local-to-model
/// bone matrices for GPU skinning.
pub fn process_animations(world: &World, asset_manager: &AssetManager, dt: f32) {
    // Safe because this is the only place updating animation time and computed matrices
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

            // 3. Convert to matrices and compute
            let mut local_transforms = crate::containers::FixedArray::<
                crate::math::mat4::Mat4,
                { crate::renderer::vulkan::skeleton::MAX_BONES },
            >::new();
            final_pose.to_matrices(&mut local_transforms);

            // Compute final bone matrices
            skeleton.compute_bone_matrices(&local_transforms, &mut skeleton_comp.computed_matrices);
        }
    }
}
