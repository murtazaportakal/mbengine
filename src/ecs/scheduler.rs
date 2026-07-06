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
    /// Pre-allocated scratch buffer for temporarily holding systems during
    /// parallel execution. Sized in `build_graph()` to the largest stage width.
    /// Reused every frame via `clear()` + `push()` — never reallocates on the hot path.
    scratch_systems: Vec<Box<dyn System>>,
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
            scratch_systems: Vec::new(),
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

                    if (writes & other_reads) != 0
                        || (reads & other_writes) != 0
                        || (writes & other_writes) != 0
                    {
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

        // Pre-allocate the scratch buffer to the largest stage width so that
        // `execute()` never heap-allocates during the game loop.
        let max_stage_width = self.stages.iter().map(|s| s.len()).max().unwrap_or(0);
        self.scratch_systems = Vec::with_capacity(max_stage_width);
    }

    /// Execute all systems concurrently.
    ///
    /// Uses `self.scratch_systems` as a pre-allocated scratch buffer — no heap
    /// allocations occur on this path. The buffer was sized in `build_graph()`.
    pub fn execute(&mut self, world: &World, dt: f32) {
        // Execute stages sequentially
        for stage in &self.stages {
            // Reuse the pre-allocated scratch buffer. `clear()` sets len to 0
            // without releasing capacity, and `push()` never exceeds the
            // capacity reserved in `build_graph()`.
            self.scratch_systems.clear();
            for &idx in stage.iter() {
                self.scratch_systems.push(self.systems[idx].take().unwrap());
            }

            std::thread::scope(|s| {
                for sys in self.scratch_systems.iter_mut() {
                    // Send to background threads
                    s.spawn(move || {
                        sys.update(dt, world);
                    });
                }
            });

            // Restore systems back into the main array.
            // `drain(..)` consumes elements in order without deallocating the
            // backing storage, preserving our pre-allocated capacity.
            let mut drain = self.scratch_systems.drain(..);
            for &idx in stage.iter() {
                self.systems[idx] = Some(drain.next().unwrap());
            }
        }
    }
}
