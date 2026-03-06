use agent_ray::cameras::PinholeCamera;
use agent_ray::importer::load_obj_scene;
use agent_ray::integrators::{Integrator, PathTracer, PathTracerConfig};
use agent_ray::lights::{PointLight, PowerLightDistribution};
use agent_ray::scene::Scene;
use agent_ray::utils::save_image_as_png;
use glam::{Mat4, Vec3A};
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Simple Reinhard tone-mapper + gamma correction
// ---------------------------------------------------------------------------

/// Luminance-based Reinhard operator followed by sRGB gamma (2.2).
///
/// The luminance is compressed with L_out = L / (1 + L) and the colour is
/// rescaled accordingly to preserve hue and saturation.  A minimum luminance
/// floor avoids a division by zero for perfectly black pixels.
#[inline]
fn tonemap(linear: Vec3A) -> Vec3A {
    let lum = 0.2126 * linear.x + 0.7152 * linear.y + 0.0722 * linear.z;
    // Luminance-preserving Reinhard.
    let scale = if lum > 1e-6 {
        (lum / (1.0 + lum)) / lum
    } else {
        1.0 / (1.0 + lum) // degenerate: per-channel fallback
    };
    let mapped = (linear * scale).clamp(Vec3A::ZERO, Vec3A::ONE);
    // Gamma 2.2 encode.
    mapped.powf(1.0 / 2.2)
}

fn main() {
    let width: usize = 1600;
    let height: usize = 720;
    let aspect_ratio = width as f32 / height as f32;

    // -----------------------------------------------------------------------
    // Camera
    //
    // Positioned outside the room entrance, looking roughly inward.
    // These values match the normal-visualisation test and give a good view
    // of the fireplace room geometry.
    // -----------------------------------------------------------------------
    let camera = PinholeCamera::from_eye_angle(
        Vec3A::new(4.0, 1.0, -2.2), // eye
        -90.0,
        0.0,
        60.0,
        aspect_ratio,
    );

    // -----------------------------------------------------------------------
    // Scene: load OBJ via the importer (MTL → PBR BSDFs automatically).
    // -----------------------------------------------------------------------
    println!("Loading scene...");
    let obj_path = Path::new("assets/fireplace_room/fireplace_room.obj");
    let objects =
        load_obj_scene(obj_path, Mat4::IDENTITY).expect("Failed to load fireplace_room.obj");
    println!("  {} mesh objects loaded.", objects.len());

    // -----------------------------------------------------------------------
    // Lights
    //
    // One warm point light placed roughly at the centre of the room, elevated
    // to simulate a hanging bulb or a bright fireplace glow.  Intensity is in
    // watts / sr; at ~5 m distance this gives ~60 W/m² of irradiance, similar
    // to indoor lighting.
    // -----------------------------------------------------------------------
    let point_light = Arc::new(PointLight::new(
        Vec3A::new(1.4, 2.0, -2.0),           // world-space position
        Vec3A::new(150.0, 120.0, 80.0) * 0.3, // warm white (slightly orange-tinted)
    ));

    let lights: Vec<Arc<dyn agent_ray::lights::Light>> = vec![point_light];
    let light_dist = Box::new(PowerLightDistribution::new(&lights));

    // -----------------------------------------------------------------------
    // Assemble the scene and run.
    // -----------------------------------------------------------------------
    let scene = Scene::new(objects, lights, Some(light_dist));

    let config = PathTracerConfig {
        spp: 16,
        max_depth: 2,
        rr_depth: 3,
    };
    let integrator = PathTracer::new(config);

    println!(
        "Rendering {}×{} @ {}spp…",
        width, height, integrator.config.spp
    );
    let hdr = integrator.render(&scene, &camera, width, height);

    // -----------------------------------------------------------------------
    // Tone-map to 8-bit sRGB and save.
    // -----------------------------------------------------------------------
    let mut pixels = vec![0u8; width * height * 3];
    for (i, radiance) in hdr.iter().enumerate() {
        let srgb = tonemap(*radiance);
        pixels[i * 3] = (srgb.x * 255.0 + 0.5) as u8;
        pixels[i * 3 + 1] = (srgb.y * 255.0 + 0.5) as u8;
        pixels[i * 3 + 2] = (srgb.z * 255.0 + 0.5) as u8;
    }

    let out = "pt_test.png";
    save_image_as_png(&pixels, width as u32, height as u32, out).unwrap();
    println!("Saved → {out}");
}
