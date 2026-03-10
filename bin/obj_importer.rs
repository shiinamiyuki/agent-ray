//! CLI tool that imports one or more OBJ files into a scene JSON + BIN pair.
//!
//! # Usage
//!
//! ```text
//! obj_importer [OPTIONS] --output <SCENE.json> <OBJ_FILE>...
//!
//! Options:
//!   -o, --output <SCENE.json>   Path to the scene JSON file (created if absent)
//!   --overwrite                  Force-create a new scene even if the file exists
//!   -h, --help                  Print this help message
//! ```
//!
//! Geometry data (vertices, normals, UVs, indices) is stored in a companion
//! `.bin` file next to the JSON.  If `--output` points to an existing scene
//! file (and `--overwrite` is not set) the new meshes are **appended** to
//! both the JSON and the binary buffer.

use std::path::Path;
use std::process;

use agent_ray::scene_format::{
    bin_path_for, import_obj_to_scene_desc, load_bin_buffer, load_scene_file, save_scene_file,
    BinBuffer, SceneDescription,
};

fn print_help() {
    eprintln!(
        "\
Usage: obj_importer [OPTIONS] --output <SCENE.json> <OBJ_FILE>...

Import one or more OBJ files into a scene JSON + BIN pair.

Arguments:
  <OBJ_FILE>...              One or more .obj files to import

Options:
  -o, --output <SCENE.json>  Path to the output scene JSON file
  --overwrite                Force-create a new scene even if the file exists
  -h, --help                 Print this help message"
    );
}

struct Args {
    output: String,
    overwrite: bool,
    obj_files: Vec<String>,
}

fn parse_args() -> Option<Args> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return None;
    }

    let mut output: Option<String> = None;
    let mut overwrite = false;
    let mut obj_files: Vec<String> = Vec::new();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return None,
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --output requires a value");
                    return None;
                }
                output = Some(args[i].clone());
            }
            "--overwrite" => {
                overwrite = true;
            }
            other => {
                if other.starts_with('-') {
                    eprintln!("error: unknown option '{other}'");
                    return None;
                }
                obj_files.push(other.to_string());
            }
        }
        i += 1;
    }

    let output = match output {
        Some(o) => o,
        None => {
            eprintln!("error: --output is required");
            return None;
        }
    };

    if obj_files.is_empty() {
        eprintln!("error: at least one OBJ file is required");
        return None;
    }

    Some(Args { output, overwrite, obj_files })
}

fn main() {
    let args = match parse_args() {
        Some(a) => a,
        None => {
            print_help();
            process::exit(1);
        }
    };

    let out_path = Path::new(&args.output);
    let scene_dir = out_path.parent().unwrap_or(Path::new("."));

    // Load existing scene + bin, or create fresh ones.
    let (mut desc, mut buf) = if !args.overwrite && out_path.exists() {
        println!("Loading existing scene '{}'…", out_path.display());
        let d = match load_scene_file(out_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: failed to load '{}': {e}", out_path.display());
                process::exit(1);
            }
        };
        let b = match load_bin_buffer(&d, scene_dir) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: failed to load binary buffer: {e}");
                process::exit(1);
            }
        };
        (d, b)
    } else {
        if args.overwrite && out_path.exists() {
            println!("Overwriting existing scene '{}'.", out_path.display());
        }
        println!("Creating new scene.");
        (SceneDescription::default(), BinBuffer::new())
    };

    // Import each OBJ file.
    for obj in &args.obj_files {
        let obj_path = Path::new(obj);
        println!("Importing '{}'…", obj_path.display());
        if let Err(e) = import_obj_to_scene_desc(&mut desc, &mut buf, obj_path, None) {
            eprintln!("error: failed to import '{}': {e}", obj_path.display());
            process::exit(1);
        }
        println!(
            "  → {} meshes, {} materials, {} objects total  (bin: {:.1} KB)",
            desc.meshes.len(),
            desc.materials.len(),
            desc.objects.len(),
            buf.len() as f64 / 1024.0,
        );
    }

    // Save JSON + BIN.
    if let Err(e) = save_scene_file(out_path, &mut desc, &buf) {
        eprintln!("error: {e}");
        process::exit(1);
    }
    let bin = bin_path_for(out_path);
    println!(
        "Saved scene → '{}' + '{}' ({:.1} KB)",
        out_path.display(),
        bin.display(),
        buf.len() as f64 / 1024.0,
    );
}
