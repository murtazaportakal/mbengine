use std::fs;
use std::path::{Path, PathBuf};

/// Virtual File System for loading assets.
/// In debug builds, this maps directly to the physical hard drive.
/// In future release builds, this can map to a packed `.pak` file or a ZIP archive.
pub struct Vfs {
    root_dir: PathBuf,
}

impl Vfs {
    pub fn new(root_dir: impl AsRef<Path>) -> Self {
        Self {
            root_dir: root_dir.as_ref().to_path_buf(),
        }
    }

    /// Read an entire file into a string.
    pub fn read_to_string(&self, path: impl AsRef<Path>) -> std::io::Result<String> {
        let full_path = self.resolve_path(path);
        fs::read_to_string(full_path)
    }

    /// Read an entire file into a byte vector.
    pub fn read_bytes(&self, path: impl AsRef<Path>) -> std::io::Result<Vec<u8>> {
        let full_path = self.resolve_path(path);
        fs::read(full_path)
    }

    /// Resolves a virtual path into a physical path.
    pub fn resolve_path(&self, path: impl AsRef<Path>) -> PathBuf {
        self.root_dir.join(path)
    }
}

impl Default for Vfs {
    fn default() -> Self {
        // Default to the current working directory
        Self::new(".")
    }
}
