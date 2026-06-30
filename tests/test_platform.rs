//! Integration tests for Platform and Logging.

use engine::memory::MemorySubsystem;
use engine::platform::{Timer, Window};
use engine::logging::{set_global_logger, global_logger, Severity};
use engine::{log_info, log_error}; // Use macros directly from crate root since they are #[macro_export]

#[test]
fn test_timer() {
    let mut timer = Timer::new();
    
    // Busy wait for a tiny fraction of a second to ensure timer advances
    let mut i = 0;
    while timer.elapsed_seconds() < 0.01 {
        i += 1;
        // Optimization barrier (black_box equivalent) to prevent dead code elimination
        std::hint::black_box(i);
    }
    
    assert!(timer.elapsed_seconds() >= 0.01);
    assert!(timer.elapsed_ms() >= 10.0);
    
    timer.reset();
    assert!(timer.elapsed_seconds() < 0.01);
}

#[test]
fn test_logger() {
    let mut mem = MemorySubsystem::new();
    mem.init_default();
    
    // Initialize logger with capacity 4
    set_global_logger(mem.persistent_arena(), 4);
    
    let logger = global_logger().expect("Logger should be initialized");
    
    // Use the macros
    log_info!("Hello {}", "world");
    log_error!("Testing error {}", 42);
    
    // Validate RingBuffer contents
    let entry1 = logger.pop().unwrap();
    assert_eq!(entry1.severity, Severity::Info);
    assert_eq!(entry1.message.as_str(), "Hello world");
    
    let entry2 = logger.pop().unwrap();
    assert_eq!(entry2.severity, Severity::Error);
    assert_eq!(entry2.message.as_str(), "Testing error 42");
    
    // Empty
    assert!(logger.pop().is_none());
    
    mem.shutdown();
}

#[test]
fn test_window_creation() {
    // Basic smoke test to ensure we can create and pump messages 
    // without crashing or failing FFI calls.
    let mut window = Window::new("Engine Test Window", 800, 600);
    
    assert!(!window.hwnd().is_null());
    assert!(!window.hinstance().is_null());
    
    // Poll a few times
    for _ in 0..10 {
        assert!(window.poll_events());
    }
    
    // Note: We don't manually destroy it here. In a real engine, we'd handle 
    // WM_CLOSE gracefully. For the test, it'll just drop/leak the HWND when the test exits,
    // which is fine since the OS cleans it up when the test executable terminates.
}
