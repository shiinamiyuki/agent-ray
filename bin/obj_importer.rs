//! CLI tool that imports one or more OBJ files into a scene JSON file.
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
//! If `--output` points to an existing scene file (and `--overwrite` is not
//! set) the OBJ meshes and materials are **appended** to that scene.
//! Otherwise a fresh scene with default camera settings is created.

use std::path::Path;
use std::process;

use agent_ray::scene_format::{
    import_obj_to_scene_desc, load_scene_file, save_scene_file, SceneDescription,
};

fn print_help() {
    eprintln!(
        "\
Usage: obj_importer [OPTIONS] --output <SCENE.json> <OBJ_FILE>...

Import one or more OBJ files into a scene JSON file.

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
                // Catch accidental flags.
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

    Some(Args {
        output,
        overwrite,
        obj_files,
    })
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

    // Load existing scene or create a fresh one.
    let mut desc = if !args.overwrite && out_path.exists() {
        println!("Loading existing scene '{}'…", out_path.display());
        match load_scene_file(out_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: failed to load '{}': {e}", out_path.display());
                process::exit(1);
            }
        }
    } else {
        if args.overwrite && out_path.exists() {
            println!("Overwriting existing scene '{}'.", out_path.display());
        }
        println!("Creating new scene.");
        SceneDescription::default()
    };

    // Import each OBJ file.
    for obj in &args.obj_files {
        let obj_path = Path::new(obj);
        println!("Importing '{}'…", obj_path.display());
        if let Err(e) = import_obj_to_scene_desc(&mut desc, obj_path, None) {
            eprintln!("error: failed to import '{}': {e}", obj_path.display());
            process::exit(1);
        }
        println!(
            "  → {} meshes, {} materials, {} objects total",
            desc.meshes.len(),
            desc.materials.len(),
            desc.objects.len(),
        );
    }

    // Save.
    if let Err(e) = save_scene_file(out_path, &desc) {
        eprintln!("error: {e}");
        process::exit(1);
    }
    println!("Saved scene → '{}'", out_path.display());
}
