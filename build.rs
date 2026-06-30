use std::process::Command;
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=shaders/shader.vert");
    println!("cargo:rerun-if-changed=shaders/shader.frag");

    // First try local glslangValidator
    let local_compiler = PathBuf::from("glslang").join("bin").join("glslangValidator.exe");
    
    let (compiler, args) = if local_compiler.exists() {
        (local_compiler, vec!["-V"])
    } else {
        // Fallback to Vulkan SDK glslc
        let vulkan_sdk = match env::var("VULKAN_SDK") {
            Ok(val) => val,
            Err(_) => {
                println!("cargo:warning=Neither local glslangValidator nor VULKAN_SDK found. Skipping shader compilation.");
                return;
            }
        };
        let glslc = PathBuf::from(vulkan_sdk).join("Bin").join("glslc.exe");
        if !glslc.exists() {
            println!("cargo:warning=glslc.exe not found in Vulkan SDK. Skipping shader compilation.");
            return;
        }
        (glslc, vec![])
    };

    // Compile Vertex Shader
    let mut vert_args = args.clone();
    vert_args.extend(vec!["shaders/shader.vert", "-o", "shaders/vert.spv"]);
    let vert_status = Command::new(&compiler)
        .args(&vert_args)
        .status()
        .expect("Failed to execute shader compiler for vertex shader");

    if !vert_status.success() {
        panic!("Failed to compile shader.vert");
    }

    // Compile Fragment Shader
    let mut frag_args = args.clone();
    frag_args.extend(vec!["shaders/shader.frag", "-o", "shaders/frag.spv"]);
    let frag_status = Command::new(&compiler)
        .args(&frag_args)
        .status()
        .expect("Failed to execute shader compiler for fragment shader");

    if !frag_status.success() {
        panic!("Failed to compile shader.frag");
    }
}
