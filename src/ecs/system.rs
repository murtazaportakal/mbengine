//! Abstract base for all ECS systems.
//!
//! A system processes entities that match a required component mask.
//! Implementors override `update()` to implement per-frame logic.
//!
//! The World filters entities by comparing each entity's component mask
//! against the system's required mask. Systems can also iterate
//! ComponentArrays directly for cache-optimal dense-array traversal.
//!
//! Thread safety:
//!   Systems run on the main thread by default. Future phases may
//!   add job-system support for parallel system execution.

use super::types::ComponentMask;
use super::world::World;

/// Trait that all ECS systems must implement.
///
/// Each system declares which components it requires via `required_components()`,
/// and implements its per-frame logic in `update()`.
pub trait System {
    /// Called once per frame by the World.
    ///
    /// # Arguments
    /// * `dt` — Delta time in seconds since the last frame.
    /// * `world` — The World instance — use it to query components.
    fn update(&mut self, dt: f32, world: &mut World);

    /// The required component mask for this system.
    /// An entity must have ALL components in the mask to be processed.
    fn required_components(&self) -> ComponentMask;

    /// Set the required component mask.
    fn set_required_components(&mut self, mask: ComponentMask);
}
