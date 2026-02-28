# AGENTS.md

## Instructions
First take a look at `design/roadmap.md` to get an overview of the project roadmap and milestones.
Then read through the `design/notes.md` file understand what we have left to do and what we have accomplished so far.
When you have finished the current session, please update the roadmap (if ncessary) and write a brief summary of what you have accomplished in the `design/notes.md` file in the format of 
```markdown
## [Date] - [Session Name]
[Brief summary of accomplishments and progress made during the session]
```

You can update the roadmap by checking off completed items and adding new items if necessary. 

## Building and Running
Just use `cargo build` to build the project. The `dev` profile has incremental build and optimizations enabled, so you should almost never need to use the `release` profile.

## Project Structure
- `design/`: contains design documents and notes.
    - `design/roadmap.md`: outlines the roadmap and milestones for the project.
- `bin`: the CLI entry point for the application as well as some numerical tests (e.g. BSDF).
- `src`: the main source code for the application, including all modules and components.
    - `src/geometry.rs`: defines geometric primitives and operations.
    - `src/accel/`: an two-level SAH BVH implementation for efficient ray tracing.
    - `src/primitives/`: defines various primitives such as meshes, voxels and procedural shapes.
    - `src/surfaces/`: defines various BSDF models for material representation.
    - `src/integrators/`: defines various integrators for rendering, including path tracing and photon mapping.
        - `src/integrators/path_tracer.rs`: implements a MIS path tracer.
        - `src/integrators/photon_mapper.rs`: implements a photon mapper for global illumination.
        - `src/integrators/vol_path_tracer.rs`: implements a volumetric path tracer for participating media.
        - src/integrators/bidirectional_path_tracer.rs: implements a bidirectional path tracer for improved convergence.
    - `src/cameras/`: defines various camera models for scene rendering.
    - `src/lights/`: defines various light sources for scene illumination.
    - `src/scene.rs`: defines the scene structure and loading functionality.
    - `src/utils.rs`: utility functions and helpers for the application.
- `assets`: contains example scenes, textures, and other resources for testing and demonstration purposes.
- `tests`: contains unit tests and integration tests for the application.