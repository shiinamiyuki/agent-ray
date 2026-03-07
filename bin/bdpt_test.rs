use agent_ray::cameras::PinholeCamera;
use agent_ray::film::ToneMapper;
use agent_ray::importer::load_obj_scene;
use agent_ray::integrators::{BdptConfig, BidirectionalPathTracer, Integrator, MisMode};
use agent_ray::lights::{PointLight, PowerLightDistribution};
use agent_ray::scene::Scene;
use glam::{Mat4, Vec3A};
use std::path::Path;
use std::sync::Arc;

fn main() {
    let width: usize = 1600;
    let height: usize = 720;
    let aspect_ratio = width as f32 / height as f32;

    // -----------------------------------------------------------------------
    // Camera
    // -----------------------------------------------------------------------
    let camera = PinholeCamera::from_eye_angle(
        Vec3A::new(4.0, 1.0, -2.2),
        -90.0,
        0.0,
        60.0,
        aspect_ratio,
    );

    // -----------------------------------------------------------------------
    // Scene
    // -----------------------------------------------------------------------
    println!("Loading scene...");
    let obj_path = Path::new("assets/fireplace_room/fireplace_room.obj");
    let objects =
        load_obj_scene(obj_path, Mat4::IDENTITY).expect("Failed to load fireplace_room.obj");
    println!("  {} mesh objects loaded.", objects.len());

    // -----------------------------------------------------------------------
    // Lights
    // -----------------------------------------------------------------------
    let point_light = Arc::new(PointLight::new(
        Vec3A::new(1.4, 2.0, -2.0),
        Vec3A::new(150.0, 120.0, 80.0) * 0.3,
    ));

    let lights: Vec<Arc<dyn agent_ray::lights::Light>> = vec![point_light];
    let light_dist = Box::new(PowerLightDistribution::new(&lights));

    // -----------------------------------------------------------------------
    // Assemble and render.
    // -----------------------------------------------------------------------
    let scene = Scene::new(objects, lights, Some(light_dist));

    let config = BdptConfig {
        spp: 16,
        max_depth: 5,
        rr_depth: 3,
        mis_mode: MisMode::Power,
        mis_beta: 1.0,
        debug_strategy_images: true,
    };
    let integrator = BidirectionalPathTracer::new(config);

    println!(
        "Rendering {}×{} @ {}spp (BDPT)…",
        width, height, integrator.config.spp
    );
    let film = integrator.render(&scene, &camera, width, height);

    // -----------------------------------------------------------------------
    // Tone-map and save.
    // -----------------------------------------------------------------------
    let out = "bdpt_test.png";
    film.to_rgb_image(ToneMapper::Reinhard, 2.2, 1.0)
        .save(out)
        .expect("Failed to save image");
    println!("Saved → {out}");
}
