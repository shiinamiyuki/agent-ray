use agent_ray::cameras::PinholeCamera;
use agent_ray::importer::load_obj_scene;
use agent_ray::integrators::{BdptConfig, BidirectionalPathTracer, Integrator};
use agent_ray::lights::{PointLight, PowerLightDistribution};
use agent_ray::scene::Scene;
use agent_ray::utils::save_image_as_png;
use glam::{Mat4, Vec3A};
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Simple Reinhard tone-mapper + gamma correction
// ---------------------------------------------------------------------------

#[inline]
fn tonemap(linear: Vec3A) -> Vec3A {
    let lum = 0.2126 * linear.x + 0.7152 * linear.y + 0.0722 * linear.z;
    let scale = if lum > 1e-6 {
        (lum / (1.0 + lum)) / lum
    } else {
        1.0 / (1.0 + lum)
    };
    let mapped = (linear * scale).clamp(Vec3A::ZERO, Vec3A::ONE);
    mapped.powf(1.0 / 2.2)
}

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
        spp: 1,
        max_depth: 8,
        rr_depth: 3,
    };
    let integrator = BidirectionalPathTracer::new(config);

    println!(
        "Rendering {}×{} @ {}spp (BDPT)…",
        width, height, integrator.config.spp
    );
    let hdr = integrator.render(&scene, &camera, width, height);

    // -----------------------------------------------------------------------
    // Tone-map and save.
    // -----------------------------------------------------------------------
    let mut pixels = vec![0u8; width * height * 3];
    for (i, radiance) in hdr.iter().enumerate() {
        let srgb = tonemap(*radiance);
        pixels[i * 3] = (srgb.x * 255.0 + 0.5) as u8;
        pixels[i * 3 + 1] = (srgb.y * 255.0 + 0.5) as u8;
        pixels[i * 3 + 2] = (srgb.z * 255.0 + 0.5) as u8;
    }

    let out = "bdpt_test.png";
    save_image_as_png(&pixels, width as u32, height as u32, out).unwrap();
    println!("Saved → {out}");
}
