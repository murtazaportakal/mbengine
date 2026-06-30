//! Stack-allocated fixed-capacity string.
//!
//! Replaces `String` for small strings where the maximum capacity is known at
//! compile time, avoiding heap allocations. Guaranteed to be valid UTF-8.

use std::borrow::Borrow;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::str;

/// A fixed-capacity string stored inline on the stack.
#[derive(Clone, Copy)]
pub struct FixedString<const N: usize> {
    buffer: [u8; N],
    len: usize,
}

impl<const N: usize> FixedString<N> {
    /// Create a new, empty FixedString.
    #[inline]
    pub const fn new() -> Self {
        Self {
            buffer: [0; N],
            len: 0,
        }
    }

    /// Try to create a FixedString from a string slice.
    /// Returns `None` if the string slice exceeds the capacity `N`.
    pub fn try_from_str(s: &str) -> Option<Self> {
        if s.len() > N {
            return None;
        }
        let mut fs = Self::new();
        fs.buffer[..s.len()].copy_from_slice(s.as_bytes());
        fs.len = s.len();
        Some(fs)
    }

    /// Number of bytes currently in the string.
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the string is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum number of bytes the string can hold.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Appends a given string slice onto the end of this `FixedString`.
    ///
    /// # Panics
    /// Panics if the capacity is exceeded.
    pub fn push_str(&mut self, s: &str) {
        let new_len = self.len + s.len();
        assert!(new_len <= N, "FixedString::push_str: capacity exceeded");
        
        self.buffer[self.len..new_len].copy_from_slice(s.as_bytes());
        self.len = new_len;
    }

    /// Appends a given character to the end of this `FixedString`.
    ///
    /// # Panics
    /// Panics if the capacity is exceeded.
    pub fn push(&mut self, c: char) {
        let mut b = [0; 4];
        let s = c.encode_utf8(&mut b);
        self.push_str(s);
    }

    /// Truncates this `FixedString`, removing all contents.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Extracts a string slice containing the entire `FixedString`.
    #[inline]
    pub fn as_str(&self) -> &str {
        // Safety: We only ever push valid UTF-8 via push_str and push.
        unsafe { str::from_utf8_unchecked(&self.buffer[..self.len]) }
    }

    /// Extracts a mutable string slice containing the entire `FixedString`.
    #[inline]
    pub fn as_mut_str(&mut self) -> &mut str {
        // Safety: We only ever push valid UTF-8 via push_str and push.
        unsafe { str::from_utf8_unchecked_mut(&mut self.buffer[..self.len]) }
    }
}

impl<const N: usize> Default for FixedString<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Deref for FixedString<N> {
    type Target = str;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl<const N: usize> DerefMut for FixedString<N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_str()
    }
}

impl<const N: usize> AsRef<str> for FixedString<N> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<const N: usize> Borrow<str> for FixedString<N> {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl<const N: usize> fmt::Display for FixedString<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl<const N: usize> fmt::Debug for FixedString<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<const N: usize> PartialEq for FixedString<N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl<const N: usize> Eq for FixedString<N> {}
