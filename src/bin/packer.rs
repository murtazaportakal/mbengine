use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

fn main() -> std::io::Result<()> {
    println!("Starting VFS Packer...");

    let root_dir = PathBuf::from(".");
    let pak_path = root_dir.join("assets.pak");

    let mut out_file = File::create(&pak_path)?;

    // 1. Write Header
    out_file.write_all(b"EPAK")?;
    out_file.write_all(&1u32.to_le_bytes())?; // Version 1

    let dirs_to_pack = vec!["assets", "shaders"]; // Shaders that we use directly

    // Collect files
    let mut files_to_pack = Vec::new();
    for dir in dirs_to_pack {
        let dir_path = root_dir.join(dir);
        if dir_path.exists() {
            collect_files(&dir_path, &mut files_to_pack, &root_dir)?;
        }
    }

    println!("Found {} files to pack.", files_to_pack.len());

    // Write Number of Files
    out_file.write_all(&(files_to_pack.len() as u32).to_le_bytes())?;

    // TOC requires computing sizes first. We will write a dummy TOC, then file data, then update TOC.
    // Or we can pre-read all files (or get metadata sizes) to compute offsets.
    // Let's compute sizes and offsets first.

    // TOC entry size: 2 bytes (len) + string len + 8 bytes (offset) + 8 bytes (size).
    let mut current_offset = 4 + 4 + 4; // Magic + Version + NumFiles
    for (path_str, _path) in &files_to_pack {
        let path_bytes = path_str.as_bytes();
        current_offset += 2 + path_bytes.len() + 16;
    }

    let mut data_offset = current_offset as u64;

    // 2. Write TOC
    for (path_str, path) in &files_to_pack {
        let path_bytes = path_str.as_bytes();
        out_file.write_all(&(path_bytes.len() as u16).to_le_bytes())?;
        out_file.write_all(path_bytes)?;

        let size = fs::metadata(path)?.len();
        out_file.write_all(&data_offset.to_le_bytes())?;
        out_file.write_all(&size.to_le_bytes())?;

        data_offset += size;
    }

    // 3. Write Data
    let mut total_size = 0;
    for (path_str, path) in &files_to_pack {
        println!("Packing: {}", path_str);
        let mut f = File::open(path)?;
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;
        out_file.write_all(&buffer)?;
        total_size += buffer.len();
    }

    println!("Packed {} bytes into assets.pak successfully!", total_size);

    Ok(())
}

fn collect_files(
    dir: &Path,
    files: &mut Vec<(String, PathBuf)>,
    root: &Path,
) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                collect_files(&path, files, root)?;
            } else {
                let relative_path = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace("\\", "/");
                files.push((relative_path, path));
            }
        }
    }
    Ok(())
}
