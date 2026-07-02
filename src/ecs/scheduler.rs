//! The concurrent job scheduler.
//!
//! Evaluates read/write component dependencies of registered systems and
//! builds a dependency graph, organizing them into non-overlapping stages.
//! Executes stages sequentially, but systems within a stage concurrently
//! using `std::thread::scope`.

use super::system::System;
use super::world::World;

/// Groups systems into execution stages based on read/write component conflicts.
pub struct Scheduler {
    /// The registered systems. We use Option so we can temporarily take ownership
    /// of systems to pass them to worker threads, bypassing `&mut` sharing limitations.
    systems: Vec<Option<Box<dyn System>>>,
    /// Each inner Vec represents a stage and contains indices into `self.systems`.
    stages: Vec<Vec<usize>>,
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            stages: Vec::new(),
        }
    }

    /// Add a system to the scheduler.
    pub fn add_system(&mut self, system: Box<dyn System>) {
        self.systems.push(Some(system));
    }

    /// Build the execution graph, sorting systems into stages.
    /// This should be called once after all systems are added.
    pub fn build_graph(&mut self) {
        self.stages.clear();

        for i in 0..self.systems.len() {
            let sys = self.systems[i].as_ref().unwrap();
            let reads = sys.read_components();
            let writes = sys.write_components();

            // Find the earliest stage where this system doesn't conflict.
            let mut target_stage = 0;
            
            for (stage_idx, stage) in self.stages.iter().enumerate() {
                let mut conflict = false;
                for &other_idx in stage.iter() {
                    let other = self.systems[other_idx].as_ref().unwrap();
                    let other_reads = other.read_components();
                    let other_writes = other.write_components();

                    if (writes & other_reads) != 0 || 
                       (reads & other_writes) != 0 || 
                       (writes & other_writes) != 0 {
                        conflict = true;
                        break;
                    }
                }
                
                if conflict {
                    // Conflict in this stage, must go to the next stage or later.
                    target_stage = stage_idx + 1;
                }
            }

            if target_stage >= self.stages.len() {
                self.stages.push(Vec::new());
            }
            self.stages[target_stage].push(i);
        }
    }

    /// Execute all systems concurrently.
    pub fn execute(&mut self, world: &World, dt: f32) {
        // Execute stages sequentially
        for stage in &self.stages {
            // Temporarily take ownership of systems for this stage
            let mut active_systems: Vec<Box<dyn System>> = Vec::with_capacity(stage.len());
            for &idx in stage.iter() {
                active_systems.push(self.systems[idx].take().unwrap());
            }

            std::thread::scope(|s| {
                for sys in active_systems.iter_mut() {
                    // Send to background threads
                    s.spawn(move || {
                        sys.update(dt, world);
                    });
                }
            });

            // Restore systems back into the main array
            let mut idx_iter = stage.iter();
            for sys in active_systems {
                let idx = *idx_iter.next().unwrap();
                self.systems[idx] = Some(sys);
            }
        }
    }
}
