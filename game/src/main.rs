use engine::app::Application;
use game::game_update;

fn main() {
    println!("Booting Standalone Game...");
    if let Some(mut app) = Application::new("MBEngine Standalone", 1280, 720) {
        app.run(|w, p, dt| game_update(w, p, dt));
    } else {
        eprintln!("Failed to initialize the application.");
    }
}
