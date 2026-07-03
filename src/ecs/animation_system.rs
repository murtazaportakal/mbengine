use crate::ecs::{
    components::{AnimatorComponent, SkeletonComponent},
    World,
};
use crate::asset_manager::AssetManager;

/// Process all active animations, advance their timers, and compute local-to-model
/// bone matrices for GPU skinning.
pub fn process_animations(world: &World, asset_manager: &AssetManager, dt: f32) {
    let animators = world.get_component_array::<AnimatorComponent>();

    let entities = animators.dense_entities_slice().to_vec();

    // Safe because this is the only place updating animation time and computed matrices
    let animators_mut = unsafe { world.get_component_array_mut_unchecked::<AnimatorComponent>() };
    let skeletons_mut = unsafe { world.get_component_array_mut_unchecked::<SkeletonComponent>() };

    for (i, animator) in animators_mut.as_mut_slice().iter_mut().enumerate() {
        let entity = entities[i];

        if !animator.is_playing {
            continue;
        }

        if !skeletons_mut.has(entity) {
            continue;
        }

        let skeleton_comp = unsafe { skeletons_mut.get_mut(entity) };
        
        // Advance time
        animator.current_time += dt * animator.speed;

        // Fetch skeleton and clip
        let skeleton_name = skeleton_comp.skeleton_name.as_str();
        let clip_name = animator.clip_name.as_str();

        if let Some(skeleton) = asset_manager.get_skeleton(skeleton_name) {
            if let Some(clip) = asset_manager.get_animation_clip(skeleton_name, clip_name) {
                
                // Handle looping
                if animator.current_time > clip.duration {
                    if animator.is_looping {
                        animator.current_time %= clip.duration;
                        if animator.current_time.is_nan() {
                            animator.current_time = 0.0;
                        }
                    } else {
                        animator.current_time = clip.duration;
                        animator.is_playing = false;
                    }
                }

                // Sample the clip
                let local_transforms = clip.sample(animator.current_time, skeleton.bone_count());
                
                // Compute final bone matrices
                let bone_matrices = skeleton.compute_bone_matrices(&local_transforms);
                skeleton_comp.computed_matrices = bone_matrices;
            }
        }
    }
}

