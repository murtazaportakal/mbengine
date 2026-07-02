//! Integration test booting the full Application loop.

use engine::app::Application;
use engine::renderer::RenderDevice;

#[test]
fn test_application_boot() {
    let mut app = match Application::new("Engine Boot Test", 800, 600) {
        Some(a) => a,
        None => {
            println!("Vulkan not supported on this CI runner. Skipping.");
            return;
        }
    };

    // Simulate a few frames without calling run() to avoid an infinite loop in CI
    for _ in 0..10 {
        app.window.poll_events(&mut app.input);
        let dt = app.timer.tick();
        if let Some(reloader) = &mut app.hot_reloader {
            reloader.update();
            reloader.call_game_update(&mut app.world, &mut app.physics, dt as f32);
        }
        app.memory.frame_arena().reset(false);
    }

    // Explicitly call the shutdown sequence usually handled in run()
    app.vulkan.wait_idle();
    app.swapchain.shutdown(&app.vulkan);
    app.vulkan.shutdown();

    // Drop will happen implicitly at scope exit, destroying World before MemorySubsystem.
}
