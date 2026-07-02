//! Abstract base for all ECS systems.
//!
//! A system processes entities that match a required component mask.
//! Implementors override `update()` to implement per-frame logic.
//!
//! Thread safety:
//!   Systems must be `Send` to allow the Scheduler to run them concurrently
//!   on background threads.

use super::types::ComponentMask;
use super::world::World;

/// Trait that all ECS systems must implement.
pub trait System: Send {
    /// Called once per frame by the Scheduler.
    ///
    /// # Arguments
    /// * `dt` — Delta time in seconds since the last frame.
    /// * `world` — The World instance — use it to query components.
    fn update(&mut self, dt: f32, world: &World);

    /// The read component mask for this system.
    fn read_components(&self) -> ComponentMask;

    /// The write component mask for this system.
    fn write_components(&self) -> ComponentMask;
}
