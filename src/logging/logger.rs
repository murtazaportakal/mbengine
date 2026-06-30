//! Ring-buffer-backed logging subsystem.

use crate::containers::{FixedString, RingBuffer};
use crate::memory::ArenaAllocator;
use std::sync::{Mutex, OnceLock};

/// Severity levels for logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Fatal,
}

impl Severity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Info => "INFO",
            Severity::Warning => "WARN",
            Severity::Error => "ERROR",
            Severity::Fatal => "FATAL",
        }
    }
}

/// A single log entry.
pub struct LogEntry {
    pub severity: Severity,
    pub message: FixedString<256>,
}

/// The core Logger, backed by a RingBuffer.
pub struct Logger {
    buffer: Mutex<RingBuffer<LogEntry>>,
}

impl Logger {
    /// Create a new logger with a specific capacity.
    /// Capacity must be a power of two.
    pub fn new(arena: &mut ArenaAllocator, capacity: usize) -> Self {
        Self {
            buffer: Mutex::new(RingBuffer::new(arena, capacity)),
        }
    }

    /// Push a message into the log buffer.
    /// If the buffer is full, the oldest message is dropped (force-popped).
    pub fn log(&self, severity: Severity, msg: &str) {
        let mut entry = LogEntry {
            severity,
            message: FixedString::new(),
        };

        // Truncate if the message is too long for FixedString<256>
        let len = msg.len().min(256);
        entry.message.push_str(&msg[..len]);

        let buf = self.buffer.lock().unwrap();
        
        // If the ring buffer is full, pop the oldest to make room
        if let Err(_) = buf.push(entry) {
            let _ = buf.pop();
            // Re-create the entry since it was consumed by `Err(entry)`
            let mut entry = LogEntry {
                severity,
                message: FixedString::new(),
            };
            entry.message.push_str(&msg[..len]);
            let _ = buf.push(entry);
        }
    }

    /// Pop the oldest message from the log buffer.
    pub fn pop(&self) -> Option<LogEntry> {
        self.buffer.lock().unwrap().pop()
    }
}

static GLOBAL_LOGGER: OnceLock<Logger> = OnceLock::new();

/// Initialize the global logger with a specific capacity from an arena.
pub fn set_global_logger(arena: &mut ArenaAllocator, capacity: usize) {
    let logger = Logger::new(arena, capacity);
    let _ = GLOBAL_LOGGER.set(logger);
}

/// Helper to get the global logger if initialized.
pub fn global_logger() -> Option<&'static Logger> {
    GLOBAL_LOGGER.get()
}

// ── Macros ──────────────────────────────────────────────────────────────────

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        if let Some(logger) = $crate::logging::global_logger() {
            logger.log($crate::logging::Severity::Info, &format!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        if let Some(logger) = $crate::logging::global_logger() {
            logger.log($crate::logging::Severity::Warning, &format!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        if let Some(logger) = $crate::logging::global_logger() {
            logger.log($crate::logging::Severity::Error, &format!($($arg)*));
        }
    };
}
