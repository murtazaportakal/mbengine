use notify::{Watcher, RecursiveMode};
use crate::ecs::World;
use crate::physics::PhysicsSystem;

pub struct HotReloader {
    library: Option<libloading::Library>,
    dll_path: String,
    rx: std::sync::mpsc::Receiver<notify::Result<notify::Event>>,
    _watcher: notify::RecommendedWatcher,
}

impl HotReloader {
    pub fn new(dll_path: &str) -> Option<Self> {
        println!("[HotReload] Initializing hot reloader for {}", dll_path);
        let (tx, rx) = std::sync::mpsc::channel();
        let watcher_res = notify::recommended_watcher(tx);
        if let Err(e) = &watcher_res {
            eprintln!("[HotReload] Failed to create watcher: {:?}", e);
            return None;
        }
        let mut watcher = watcher_res.unwrap();
        
        let path = std::path::Path::new(dll_path);
        let dir = path.parent().unwrap_or(std::path::Path::new("."));
        if let Err(e) = watcher.watch(dir, RecursiveMode::NonRecursive) {
            eprintln!("[HotReload] Failed to watch directory {:?}: {:?}", dir, e);
            return None;
        }

        let mut reloader = Self {
            library: None,
            dll_path: dll_path.to_string(),
            rx,
            _watcher: watcher,
        };
        
        reloader.reload();
        Some(reloader)
    }

    pub fn reload(&mut self) {
        // Drop the old library so the file handle is released.
        self.library = None;
        
        // On Windows, rustc locks the DLL if it's loaded. 
        // We copy it to a temporary file and load the copy.
        let temp_path = format!("{}_temp.dll", self.dll_path);
        
        // Wait a few MS for the OS file lock to drop from rustc
        std::thread::sleep(std::time::Duration::from_millis(50));
        
        match std::fs::copy(&self.dll_path, &temp_path) {
            Ok(_) => {
                unsafe {
                    match libloading::Library::new(&temp_path) {
                        Ok(lib) => {
                            self.library = Some(lib);
                            println!("[HotReload] Successfully loaded game DLL");
                        }
                        Err(e) => eprintln!("[HotReload] Failed to load DLL: {:?}", e),
                    }
                }
            }
            Err(e) => eprintln!("[HotReload] Failed to copy DLL from {} to {}: {:?}", self.dll_path, temp_path, e),
        }
    }

    pub fn update(&mut self) {
        let mut needs_reload = false;
        while let Ok(event) = self.rx.try_recv() {
            if let Ok(event) = event {
                if event.kind.is_modify() || event.kind.is_create() {
                    for path in event.paths {
                        if path.to_string_lossy().ends_with("game.dll") {
                            needs_reload = true;
                        }
                    }
                }
            }
        }

        if needs_reload {
            self.reload();
        }
    }

    pub fn call_game_update(&self, world: &mut World, physics: &mut PhysicsSystem, dt: f32) {
        if let Some(lib) = &self.library {
            unsafe {
                match lib.get::<libloading::Symbol<unsafe extern "C" fn(&mut World, &mut PhysicsSystem, f32)>>(b"game_update") {
                    Ok(func) => {
                        func(world, physics, dt);
                    }
                    Err(e) => {
                        eprintln!("[HotReload] Failed to find game_update symbol: {:?}", e);
                    }
                }
            }
        }
    }
}
