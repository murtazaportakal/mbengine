use std::path::{Path, PathBuf};
use std::process::Command;

pub fn compile_shader(shader_path: &Path) -> Result<(), String> {
    let glslang_path = PathBuf::from("glslang/bin/glslangValidator.exe");
    if !glslang_path.exists() {
        return Err("glslangValidator.exe not found".to_string());
    }

    let mut spv_path = shader_path.to_path_buf();
    let ext = shader_path.extension().unwrap().to_str().unwrap();
    spv_path.set_file_name(format!("{}.spv", ext));

    let output = Command::new(&glslang_path)
        .arg("-V")
        .arg(shader_path)
        .arg("-o")
        .arg(&spv_path)
        .output()
        .map_err(|e| e.to_string())?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stdout);
        return Err(format!("Shader compilation failed: {}", err));
    }

    println!("Compiled shader: {:?}", spv_path);
    Ok(())
}

pub fn compile_all_shaders() {
    let shaders_dir = PathBuf::from("src/shaders");
    if !shaders_dir.exists() {
        return;
    }

    for entry in std::fs::read_dir(&shaders_dir).unwrap().flatten() {
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "vert" || ext == "frag" {
                let _ = compile_shader(&path);
            }
        }
    }
}
