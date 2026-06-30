//! Shared low-level memory utilities used across all allocators.
//!
//! Centralises alignment math, power-of-two checks, and size helpers
//! so that individual allocator modules don't duplicate them.

// ── size constants ──────────────────────────────────────────────────────────

/// 1 Kilobyte.
pub const KB: usize = 1024;

/// 1 Megabyte.
pub const MB: usize = 1024 * KB;

/// 1 Gigabyte.
pub const GB: usize = 1024 * MB;

/// Default alignment for SIMD-friendly data (SSE/NEON = 16, AVX = 32).
pub const DEFAULT_ALIGNMENT: usize = 16;

/// Typical OS page size. Used as the minimum virtual memory allocation unit.
pub const PAGE_SIZE: usize = 4096;

// ── alignment utilities ─────────────────────────────────────────────────────

/// Compile-time power-of-two check.
#[inline(always)]
pub const fn is_power_of_two(value: usize) -> bool {
    value != 0 && (value & (value - 1)) == 0
}

/// Align `address` forward to the next multiple of `alignment`.
/// `alignment` MUST be a power of two.
#[inline(always)]
pub const fn align_forward(address: usize, alignment: usize) -> usize {
    debug_assert!(is_power_of_two(alignment));
    let mask = alignment - 1;
    (address + mask) & !mask
}

/// Align a byte size up to a multiple of `alignment`.
#[inline(always)]
pub const fn align_size(size: usize, alignment: usize) -> usize {
    debug_assert!(is_power_of_two(alignment));
    let mask = alignment - 1;
    (size + mask) & !mask
}

/// Check whether an address is aligned to the given boundary.
#[inline(always)]
pub const fn is_aligned(address: usize, alignment: usize) -> bool {
    debug_assert!(is_power_of_two(alignment));
    (address & (alignment - 1)) == 0
}

/// Check whether a pointer is aligned to the given boundary.
#[inline(always)]
pub fn is_ptr_aligned<T>(ptr: *const T, alignment: usize) -> bool {
    is_aligned(ptr as usize, alignment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_power_of_two() {
        assert!(!is_power_of_two(0));
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(!is_power_of_two(3));
        assert!(is_power_of_two(4));
        assert!(is_power_of_two(16));
        assert!(is_power_of_two(4096));
        assert!(!is_power_of_two(5));
    }

    #[test]
    fn test_align_forward() {
        assert_eq!(align_forward(0, 16), 0);
        assert_eq!(align_forward(1, 16), 16);
        assert_eq!(align_forward(15, 16), 16);
        assert_eq!(align_forward(16, 16), 16);
        assert_eq!(align_forward(17, 16), 32);
    }

    #[test]
    fn test_align_size() {
        assert_eq!(align_size(0, 16), 0);
        assert_eq!(align_size(1, 16), 16);
        assert_eq!(align_size(16, 16), 16);
        assert_eq!(align_size(17, 16), 32);
    }

    #[test]
    fn test_is_aligned() {
        assert!(is_aligned(0, 16));
        assert!(is_aligned(16, 16));
        assert!(is_aligned(32, 16));
        assert!(!is_aligned(1, 16));
        assert!(!is_aligned(15, 16));
    }
}
