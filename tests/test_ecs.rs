//! ECS Core Smoke Test — verifies all major operations.
//!
//! Tests:
//!   1. MemorySubsystem init
//!   2. World creation + component registration
//!   3. Entity create / destroy / generation recycling
//!   4. Add / get / remove components
//!   5. Dense-array iteration
//!   6. System execution (MovementSystem)
//!   7. Stale handle detection

use engine::ecs::*;
use engine::memory::*;

// ── test components ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy, Debug)]
struct Velocity {
    vx: f32,
    vy: f32,
    vz: f32,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct Health {
    current: i32,
    max: i32,
}

// ── test system ─────────────────────────────────────────────────────────────

struct MovementSystem {
    required_mask: ComponentMask,
    update_count: i32,
}

impl MovementSystem {
    fn new() -> Self {
        Self {
            required_mask: 0,
            update_count: 0,
        }
    }
}

impl System for MovementSystem {
    fn update(&mut self, dt: f32, world: &mut World) {
        let velocities = world.get_component_array::<Velocity>();
        let count = velocities.len();
        let entities = velocities.dense_entities_slice();

        // Collect entity indices that have both Position and Velocity.
        let entity_indices: Vec<u32> = (0..count as usize)
            .filter(|&i| {
                let entity_idx = entities[i];
                world.get_component_array::<Position>().has(entity_idx)
            })
            .map(|i| entities[i])
            .collect();

        for entity_idx in entity_indices {
            unsafe {
                let vel = *world.get_component_array::<Velocity>().get(entity_idx);
                let pos = world
                    .get_component_array_mut::<Position>()
                    .get_mut(entity_idx);
                pos.x += vel.vx * dt;
                pos.y += vel.vy * dt;
                pos.z += vel.vz * dt;
            }
        }

        self.update_count += 1;
    }

    fn required_components(&self) -> ComponentMask {
        self.required_mask
    }

    fn set_required_components(&mut self, mask: ComponentMask) {
        self.required_mask = mask;
    }
}

// ── tests ───────────────────────────────────────────────────────────────────

#[test]
fn test_full_ecs_smoke_test() {
    // Reset the component registry to avoid cross-test contamination.
    reset_component_registry();

    // ── 1. Memory subsystem ────────────────────────────────────────────
    let mut mem = MemorySubsystem::new();
    let init_ok = mem.init_default();
    assert!(init_ok, "MemorySubsystem::init() failed");
    assert!(mem.is_initialised(), "MemorySubsystem not initialised");
    println!(
        "    Arena capacity: {} MB",
        mem.persistent_arena().capacity() / MB
    );

    // ── 2. World + component registration ──────────────────────────────
    let mut world = unsafe { World::new(mem.persistent_arena()) };

    unsafe {
        world.register_component::<Position>(1024);
        world.register_component::<Velocity>(1024);
        world.register_component::<Health>(512);
    }
    println!("    Registered 3 component types");

    // ── 3. Entity create / destroy / recycling ─────────────────────────
    let e1 = world.create_entity();
    let e2 = world.create_entity();
    let e3 = world.create_entity();

    assert!(world.is_alive(e1), "e1 should be alive");
    assert!(world.is_alive(e2), "e2 should be alive");
    assert!(world.is_alive(e3), "e3 should be alive");
    assert_eq!(
        world.alive_entity_count(),
        3,
        "should have 3 alive entities"
    );

    // Destroy e2 and verify.
    world.destroy_entity(e2);
    assert!(!world.is_alive(e2), "e2 should be dead after destroy");
    assert_eq!(
        world.alive_entity_count(),
        2,
        "should have 2 alive entities"
    );

    // Create a new entity — should recycle e2's slot with a new generation.
    let e4 = world.create_entity();
    assert!(world.is_alive(e4), "e4 should be alive");
    assert!(!world.is_alive(e2), "stale e2 handle should still be dead");

    // Verify recycled slot index matches.
    assert_eq!(
        get_entity_index(e4),
        get_entity_index(e2),
        "e4 should reuse e2's slot index"
    );
    assert!(
        get_entity_generation(e4) > get_entity_generation(e2),
        "e4 should have a higher generation than e2"
    );

    println!(
        "    Entity recycling: slot {}, gen {} -> {}",
        get_entity_index(e2),
        get_entity_generation(e2),
        get_entity_generation(e4)
    );

    // ── 4. Add / get / remove components ───────────────────────────────
    unsafe {
        world.add_component(
            e1,
            Position {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
        );
        world.add_component(
            e1,
            Velocity {
                vx: 10.0,
                vy: 0.0,
                vz: 0.0,
            },
        );
        world.add_component(
            e1,
            Health {
                current: 100,
                max: 100,
            },
        );

        world.add_component(
            e3,
            Position {
                x: 5.0,
                y: 5.0,
                z: 5.0,
            },
        );
        world.add_component(
            e3,
            Velocity {
                vx: 0.0,
                vy: -5.0,
                vz: 0.0,
            },
        );

        world.add_component(
            e4,
            Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        );
        world.add_component(
            e4,
            Health {
                current: 50,
                max: 100,
            },
        );
    }

    assert!(
        world.has_component::<Position>(e1),
        "e1 should have Position"
    );
    assert!(
        world.has_component::<Velocity>(e1),
        "e1 should have Velocity"
    );
    assert!(world.has_component::<Health>(e1), "e1 should have Health");

    unsafe {
        let p1 = world.get_component::<Position>(e1);
        assert!(
            p1.x == 1.0 && p1.y == 2.0 && p1.z == 3.0,
            "e1 Position values should match"
        );
    }

    // Remove Velocity from e1.
    unsafe {
        world.remove_component::<Velocity>(e1);
    }
    assert!(
        !world.has_component::<Velocity>(e1),
        "e1 should no longer have Velocity"
    );
    assert!(
        world.has_component::<Position>(e1),
        "e1 should still have Position"
    );

    // Verify e3's velocity survived e1's removal (swap-and-pop correctness).
    unsafe {
        let v3 = world.get_component::<Velocity>(e3);
        assert_eq!(v3.vy, -5.0, "e3 Velocity should be intact after e1 removal");
    }

    println!("    Components: add/get/remove OK");

    // ── 5. Dense-array iteration ───────────────────────────────────────
    let pos_array = world.get_component_array::<Position>();
    let pos_count = pos_array.len();
    assert_eq!(pos_count, 3, "should have 3 Position components");

    let sum_x: f32 = pos_array.as_slice().iter().map(|p| p.x).sum();
    assert_eq!(sum_x, 6.0, "sum of Position.x should be 6.0 (1+5+0)");

    println!("    Iterated {} positions, sumX = {:.1}", pos_count, sum_x);

    // ── 6. System execution ────────────────────────────────────────────
    // Re-add velocity to e1 for the movement test.
    unsafe {
        world.add_component(
            e1,
            Velocity {
                vx: 10.0,
                vy: 0.0,
                vz: 0.0,
            },
        );
    }

    let pos_type_id = get_component_type_id::<Position>();
    let vel_type_id = get_component_type_id::<Velocity>();
    let mask = build_mask(&[pos_type_id, vel_type_id]);

    let mut move_sys = MovementSystem::new();
    move_sys.set_required_components(mask);

    world.register_system(Box::new(move_sys));

    // Snapshot e1's position before update.
    let e1_x_before = unsafe { world.get_component::<Position>(e1).x };

    // Run one frame at dt = 0.016 (60fps).
    world.update_systems(0.016);

    let e1_x_after = unsafe { world.get_component::<Position>(e1).x };
    let expected_delta = 10.0_f32 * 0.016;

    assert!(
        (e1_x_after - e1_x_before - expected_delta).abs() < 0.001,
        "e1 Position.x should have moved by vel.x * dt"
    );

    println!(
        "    e1.x: {:.3} -> {:.3} (delta: {:.3}, expected: {:.3})",
        e1_x_before,
        e1_x_after,
        e1_x_after - e1_x_before,
        expected_delta
    );

    // ── 7. Stale handle detection ──────────────────────────────────────
    let e5 = world.create_entity();
    unsafe {
        world.add_component(
            e5,
            Position {
                x: 99.0,
                y: 99.0,
                z: 99.0,
            },
        );
    }
    let stale_handle = e5;

    world.destroy_entity(e5);
    assert!(
        !world.is_alive(stale_handle),
        "stale handle should not be alive"
    );

    println!("    Stale handle correctly detected");

    // Cleanup: World is dropped first (reverse declaration order),
    // then MemorySubsystem — matching C++ destruction order.
    drop(world);
    mem.shutdown();

    println!("\n=== All ECS tests passed ===");
}
