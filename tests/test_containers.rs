//! Integration tests for Custom Containers.

use engine::containers::*;
use engine::memory::*;

// ── FixedArray ──────────────────────────────────────────────────────────────

#[test]
fn test_fixed_array() {
    let mut arr = FixedArray::<i32, 4>::new();
    assert!(arr.is_empty());
    assert_eq!(arr.capacity(), 4);

    arr.push(10);
    arr.push(20);
    assert_eq!(arr.len(), 2);
    assert!(!arr.is_empty());
    assert_eq!(arr.as_slice(), &[10, 20]);

    assert_eq!(arr.pop(), Some(20));
    assert_eq!(arr.len(), 1);

    arr.push(30);
    arr.push(40);
    arr.push(50);
    assert!(arr.is_full());

    arr.clear();
    assert!(arr.is_empty());
}

#[test]
#[should_panic(expected = "FixedArray::push: capacity exceeded")]
fn test_fixed_array_overflow() {
    let mut arr = FixedArray::<i32, 2>::new();
    arr.push(1);
    arr.push(2);
    arr.push(3); // panics
}

// ── DynamicArray ────────────────────────────────────────────────────────────

#[test]
fn test_dynamic_array() {
    let mut mem = MemorySubsystem::new();
    mem.init_default();
    let arena = mem.frame_arena();


    let mut arr = DynamicArray::<i32>::with_capacity(arena, 2);

    assert!(arr.is_empty());
    assert_eq!(arr.capacity(), 2);

    arr.push(arena, 1);
    arr.push(arena, 2);
    assert_eq!(arr.len(), 2);
    assert_eq!(arr.capacity(), 2);

    // This push should trigger a reallocation (capacity doubles to 4)
    arr.push(arena, 3);
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.capacity(), 4);
    assert_eq!(arr.as_slice(), &[1, 2, 3]);

    assert_eq!(arr.pop(), Some(3));
    assert_eq!(arr.len(), 2);

    arr.clear();
    assert!(arr.is_empty());

    drop(arr);
    mem.shutdown();
}

// ── RingBuffer ──────────────────────────────────────────────────────────────

#[test]
fn test_ring_buffer() {
    let mut mem = MemorySubsystem::new();
    mem.init_default();
    let arena = mem.frame_arena();

    // Capacity must be power of 2
    let ring = RingBuffer::<i32>::new(arena, 4);

    assert!(ring.is_empty());
    
    assert_eq!(ring.push(10), Ok(()));
    assert_eq!(ring.push(20), Ok(()));
    assert_eq!(ring.push(30), Ok(()));
    assert_eq!(ring.push(40), Ok(()));
    // Buffer is full (4 items)
    assert_eq!(ring.push(50), Err(50));

    assert_eq!(ring.len(), 4);

    assert_eq!(ring.pop(), Some(10));
    assert_eq!(ring.pop(), Some(20));

    assert_eq!(ring.len(), 2);

    assert_eq!(ring.push(50), Ok(()));
    assert_eq!(ring.push(60), Ok(()));

    assert_eq!(ring.pop(), Some(30));
    assert_eq!(ring.pop(), Some(40));
    assert_eq!(ring.pop(), Some(50));
    assert_eq!(ring.pop(), Some(60));
    assert_eq!(ring.pop(), None);

    drop(ring);
    mem.shutdown();
}

#[test]
#[should_panic(expected = "RingBuffer capacity must be a power of two")]
fn test_ring_buffer_invalid_capacity() {
    let mut mem = MemorySubsystem::new();
    mem.init_default();
    let arena = mem.frame_arena();
    
    let _ring = RingBuffer::<i32>::new(arena, 3);
}

// ── HashMap ─────────────────────────────────────────────────────────────────

#[test]
fn test_hash_map() {
    let mut mem = MemorySubsystem::new();
    mem.init_default();
    let arena = mem.frame_arena();

    let mut map = HashMap::<&str, i32>::with_capacity(arena, 4);
    println!("Map created");

    map.insert(arena, "one", 1);
    map.insert(arena, "two", 2);
    map.insert(arena, "three", 3);
    println!("Inserted 3 items");

    assert_eq!(map.len(), 3);
    assert_eq!(map.get(&"one"), Some(&1));
    assert_eq!(map.get(&"two"), Some(&2));
    assert_eq!(map.get(&"three"), Some(&3));
    println!("Get 3 items OK");

    // Update existing key
    map.insert(arena, "two", 22);
    assert_eq!(map.get(&"two"), Some(&22));
    println!("Update OK");

    // Force a resize (load factor > 90% for capacity 4 is 3.6, so inserting 4th will resize)
    map.insert(arena, "four", 4);
    println!("Insert 4th OK (resized)");
    
    assert_eq!(map.get(&"one"), Some(&1));
    assert_eq!(map.get(&"two"), Some(&22));
    assert_eq!(map.get(&"three"), Some(&3));
    assert_eq!(map.get(&"four"), Some(&4));
    println!("Get after resize OK");

    // Remove
    map.remove(&"three");
    println!("Remove OK");
    assert_eq!(map.get(&"three"), None);
    assert_eq!(map.len(), 3);

    map.clear();
    println!("Clear OK");
    
    drop(map);

    mem.shutdown();
    println!("Shutdown OK");
}

// ── FixedString ─────────────────────────────────────────────────────────────

#[test]
fn test_fixed_string() {
    let mut s = FixedString::<16>::new();
    assert!(s.is_empty());

    s.push_str("Hello");
    assert_eq!(s.len(), 5);
    assert_eq!(s.as_str(), "Hello");

    s.push(',');
    s.push(' ');
    s.push_str("World!");
    assert_eq!(s.as_str(), "Hello, World!");
    
    assert_eq!(s.len(), 13);
    
    let parsed = FixedString::<16>::try_from_str("Rust").unwrap();
    assert_eq!(parsed.as_str(), "Rust");
    
    // Test PartialEq
    let s2 = FixedString::<16>::try_from_str("Hello, World!").unwrap();
    assert_eq!(s, s2);
}

#[test]
#[should_panic(expected = "FixedString::push_str: capacity exceeded")]
fn test_fixed_string_overflow() {
    let mut s = FixedString::<4>::new();
    s.push_str("Hello"); // panics
}
