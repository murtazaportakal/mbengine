use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=shaders/shader.vert");
    println!("cargo:rerun-if-changed=shaders/shader.frag");
    println!("cargo:rerun-if-changed=shaders/shadow.vert");
    println!("cargo:rerun-if-changed=shaders/post_process.vert");
    println!("cargo:rerun-if-changed=shaders/post_process.frag");

    // First try local glslangValidator
    let local_compiler = PathBuf::from("glslang")
        .join("bin")
        .join("glslangValidator.exe");

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
            println!(
                "cargo:warning=glslc.exe not found in Vulkan SDK. Skipping shader compilation."
            );
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

    // Compile Shadow Vertex Shader
    let mut shadow_args = args.clone();
    shadow_args.extend(vec!["shaders/shadow.vert", "-o", "shaders/shadow.spv"]);
    let shadow_status = Command::new(&compiler)
        .args(&shadow_args)
        .status()
        .expect("Failed to execute shader compiler for shadow vertex shader");

    if !shadow_status.success() {
        panic!("Failed to compile shadow.vert");
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

    // Compile Post Process Vertex Shader
    let mut pp_vert_args = args.clone();
    pp_vert_args.extend(vec![
        "shaders/post_process.vert",
        "-o",
        "shaders/post_process_vert.spv",
    ]);
    let pp_vert_status = Command::new(&compiler)
        .args(&pp_vert_args)
        .status()
        .expect("Failed to execute shader compiler for post_process.vert");

    if !pp_vert_status.success() {
        panic!("Failed to compile post_process.vert");
    }

    // Compile Post Process Fragment Shader
    let mut pp_frag_args = args.clone();
    pp_frag_args.extend(vec![
        "shaders/post_process.frag",
        "-o",
        "shaders/post_process_frag.spv",
    ]);
    let pp_frag_status = Command::new(&compiler)
        .args(&pp_frag_args)
        .status()
        .expect("Failed to execute shader compiler for post_process.frag");

    if !pp_frag_status.success() {
        panic!("Failed to compile post_process.frag");
    }
}
