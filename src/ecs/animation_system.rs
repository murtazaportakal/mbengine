use crate::asset_manager::AssetManager;
use crate::ecs::{
    components::{AnimationState, AnimatorComponent, SkeletonComponent},
    World,
};


// ── Duration helpers ──────────────────────────────────────────────────────────

/// Get the maximum duration of a given state for looping/stopping logic.
fn get_state_duration(
    state: &AnimationState,
    asset_manager: &AssetManager,
) -> f32 {
    match state {
        AnimationState::Clip { clip_handle } => {
            asset_manager
                .get_animation_clip(*clip_handle)
                .map(|c| c.duration)
                .unwrap_or(0.0)
        }
        AnimationState::Blend1D { clip_a, clip_b, .. } => {
            let da = asset_manager
                .get_animation_clip(*clip_a)
                .map(|c| c.duration)
                .unwrap_or(0.0);
            let db = asset_manager
                .get_animation_clip(*clip_b)
                .map(|c| c.duration)
                .unwrap_or(0.0);
            da.max(db)
        }
        AnimationState::Blend2D {
            clip_bl, clip_br, clip_tl, clip_tr, ..
        } => {
            [clip_bl, clip_br, clip_tl, clip_tr]
                .iter()
                .filter_map(|handle| {
                    asset_manager
                        .get_animation_clip(**handle)
                        .map(|c| c.duration)
                })
                .fold(0.0f32, f32::max)
        }
    }
}

// Pose sampling and CPU skinning have been moved to GPU compute shaders.

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

        if asset_manager.get_skeleton(skeleton_name).is_some() {
            let duration = get_state_duration(&animator.state, asset_manager);
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

            // CPU no longer computes keyframes or bone matrices.
            // This is deferred to `anim_update.comp` on the GPU.
            
            // Just handle crossfade timers
            if animator.target_state.is_some() {
                animator.transition_time += dt_scaled;
                animator.crossfade_current += dt_scaled;

                if animator.crossfade_current >= animator.crossfade_duration {
                    animator.state = animator.target_state.take().unwrap();
                    animator.current_time = animator.transition_time;
                }
            }
        }
    }
}
