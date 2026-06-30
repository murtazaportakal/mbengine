use engine::app::Application;

fn main() {
    println!("Booting Engine...");
    if let Some(mut app) = Application::new("Engine - Phase 10 (ECS Rendering)", 800, 600) {
        app.run();
    } else {
        eprintln!("Failed to initialize the application (Vulkan might not be supported).");
    }
}
