pub mod app;
pub mod asset_manager;
pub mod audio;
pub mod containers;
pub mod ecs;
pub mod logging;
pub mod math;
pub mod memory;
pub mod physics;
pub mod platform;
pub mod renderer {
    pub mod vulkan;
}
pub mod scripting;
pub mod ui;
pub mod utils;
pub mod vfs;

#[cfg(feature = "standalone")]
pub mod game_logic {
    include!(concat!(env!("OUT_DIR"), "/game_logic.rs"));
}
