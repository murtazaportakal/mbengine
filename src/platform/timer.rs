//! High-resolution timer using OS-specific APIs.
//!
//! Currently uses `QueryPerformanceCounter` on Windows for nanosecond precision.

use crate::platform::win32;

/// A high-resolution timer.
pub struct Timer {
    frequency: f64,
    start_time: i64,
}

impl Timer {
    /// Create a new timer and start it immediately.
    pub fn new() -> Self {
        let mut freq_raw: i64 = 0;
        unsafe {
            win32::QueryPerformanceFrequency(&mut freq_raw);
        }
        let frequency = freq_raw as f64;

        let mut timer = Self {
            frequency,
            start_time: 0,
        };
        timer.reset();
        timer
    }

    /// Reset the timer to the current time.
    #[inline]
    pub fn reset(&mut self) {
        let mut now: i64 = 0;
        unsafe {
            win32::QueryPerformanceCounter(&mut now);
        }
        self.start_time = now;
    }

    /// Get the elapsed time in seconds since the timer was created or last reset.
    #[inline]
    pub fn elapsed_seconds(&self) -> f64 {
        let mut now: i64 = 0;
        unsafe {
            win32::QueryPerformanceCounter(&mut now);
        }
        (now - self.start_time) as f64 / self.frequency
    }
    
    /// Get the elapsed time in milliseconds.
    #[inline]
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed_seconds() * 1000.0
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}
