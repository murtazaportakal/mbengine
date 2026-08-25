use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VfsMode {
    Filesystem,
    Pak,
}

#[derive(Clone, Debug)]
pub struct PakEntry {
    pub offset: u64,
    pub size: u64,
}

#[allow(dead_code)]
pub struct VfsInner {
    mode: VfsMode,
    pak_file: Option<Mutex<File>>,
    toc: HashMap<String, PakEntry>,
}

/// Virtual File System for loading assets.
/// In debug builds, this maps directly to the physical hard drive.
/// In release builds, this maps to a packed `data.pak` file if present.
#[derive(Clone)]
pub struct Vfs {
    root_dir: PathBuf,
    inner: Arc<VfsInner>,
}

impl Vfs {
    pub fn new(root_dir: impl AsRef<Path>) -> Self {
        let root_dir = root_dir.as_ref().to_path_buf();
        let pak_path = root_dir.join("data.pak");

        let mut pak_file_opt = None;
        let mut toc = HashMap::new();
        #[allow(unused_assignments)]
        let mut mode = VfsMode::Filesystem;

        #[cfg(feature = "standalone")]
        {
            mode = VfsMode::Pak;
        }

        if pak_path.exists() {
            mode = VfsMode::Pak;
            if let Ok(mut f) = File::open(&pak_path) {
                // Read magic
                let mut magic = [0u8; 4];
                if f.read_exact(&mut magic).is_ok() && &magic == b"EPAK" {
                    let mut u32_buf = [0u8; 4];
                    let _ = f.read_exact(&mut u32_buf); // version
                    let _ = f.read_exact(&mut u32_buf);
                    let num_files = u32::from_le_bytes(u32_buf);

                    for _ in 0..num_files {
                        let mut u16_buf = [0u8; 2];
                        if f.read_exact(&mut u16_buf).is_err() {
                            break;
                        }
                        let path_len = u16::from_le_bytes(u16_buf) as usize;

                        let mut path_bytes = vec![0u8; path_len];
                        if f.read_exact(&mut path_bytes).is_err() {
                            break;
                        }
                        let path_str = String::from_utf8(path_bytes).unwrap_or_default();

                        let mut u64_buf = [0u8; 8];
                        if f.read_exact(&mut u64_buf).is_err() {
                            break;
                        }
                        let offset = u64::from_le_bytes(u64_buf);
                        if f.read_exact(&mut u64_buf).is_err() {
                            break;
                        }
                        let size = u64::from_le_bytes(u64_buf);

                        // Normalize path to use forward slashes so it matches queries
                        let normalized_path = path_str.replace("\\", "/");
                        toc.insert(normalized_path, PakEntry { offset, size });
                    }
                    pak_file_opt = Some(Mutex::new(f));
                }
            }
        }

        Self {
            root_dir,
            inner: Arc::new(VfsInner {
                mode,
                pak_file: pak_file_opt,
                toc,
            }),
        }
    }

    /// Read an entire file into a byte vector.
    pub fn read_bytes(&self, path: impl AsRef<Path>) -> std::io::Result<Vec<u8>> {
        let path_str = path.as_ref().to_string_lossy().replace("\\", "/");

        // Try Pak first
        if let Some(pak_mutex) = &self.inner.pak_file {
            if let Some(entry) = self.inner.toc.get(&path_str) {
                if let Ok(mut f) = pak_mutex.lock() {
                    f.seek(SeekFrom::Start(entry.offset))?;
                    let mut buf = vec![0u8; entry.size as usize];
                    f.read_exact(&mut buf)?;
                    return Ok(buf);
                }
            }
        }

        // Fallback to disk
        let full_path = self.resolve_path(path);
        fs::read(full_path)
    }

    /// Read a specific chunk of a file directly into a raw pointer.
    ///
    /// # Safety
    /// The caller must ensure that `dst` points to a valid, appropriately aligned
    /// memory block of at least `len` bytes.
    pub unsafe fn read_chunk_into_ptr(
        &self,
        path: impl AsRef<Path>,
        file_offset: u64,
        dst: *mut u8,
        len: usize,
    ) -> std::io::Result<()> {
        let path_str = path.as_ref().to_string_lossy().replace("\\", "/");

        // Try Pak first
        if let Some(pak_mutex) = &self.inner.pak_file {
            if let Some(entry) = self.inner.toc.get(&path_str) {
                if file_offset + len as u64 > entry.size {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "Chunk exceeds file bounds",
                    ));
                }
                if let Ok(mut f) = pak_mutex.lock() {
                    f.seek(SeekFrom::Start(entry.offset + file_offset))?;
                    let buf = std::slice::from_raw_parts_mut(dst, len);
                    f.read_exact(buf)?;
                    return Ok(());
                }
            }
        }

        // Fallback to disk
        let full_path = self.resolve_path(path);
        let mut f = File::open(full_path)?;
        let metadata = f.metadata()?;
        let file_len = metadata.len();
        if file_offset + len as u64 > file_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Chunk exceeds file bounds",
            ));
        }
        f.seek(SeekFrom::Start(file_offset))?;
        let buf = std::slice::from_raw_parts_mut(dst, len);
        f.read_exact(buf)?;
        Ok(())
    }

    /// Read an entire file into a string.
    pub fn read_to_string(&self, path: impl AsRef<Path>) -> std::io::Result<String> {
        let bytes = self.read_bytes(path)?;
        String::from_utf8(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
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
