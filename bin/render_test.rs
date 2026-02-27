use agent_ray::geometry::{Triangle, Intersect, HitInfo};
use agent_ray::cameras::{PinholeCamera, Camera};
use agent_ray::utils::{save_image_as_png};
use glam::{Vec3A, Vec2};
use rayon::prelude::*;

fn main() {
    let width = 800;
    let height = 600;
    let aspect_ratio = width as f32 / height as f32;

    let camera = PinholeCamera::new(
        Vec3A::new(0.0, 0.0, 3.0),
        Vec3A::new(0.0, 0.0, 0.0),
        Vec3A::new(0.0, 1.0, 0.0),
        45.0,
        aspect_ratio,
    );

    let t1 = Triangle::new(
        Vec3A::new(-0.5, -0.5, 0.0),
        Vec3A::new(0.5, -0.5, 0.0),
        Vec3A::new(0.0, 0.5, 0.0),
    );
    let t2 = Triangle::new(
        Vec3A::new(-0.8, -0.2, -0.5),
        Vec3A::new(-0.2, -0.2, -0.5),
        Vec3A::new(-0.5, 0.8, -0.5),
    );
    let shapes: Vec<Box<dyn Intersect + Send + Sync>> = vec![
        Box::new(t1),
        Box::new(t2),
    ];

    let mut pixels = vec![0u8; (width * height * 3) as usize];
    
    pixels.par_chunks_mut((width * 3) as usize).enumerate().for_each(|(y, row)| {
        for x in 0..width as usize {
            let uv = Vec2::new(x as f32 / width as f32, 1.0 - (y as f32 / height as f32));
            let ray = camera.generate_ray(uv);

            let mut closest_hit: Option<HitInfo> = None;
            for shape in &shapes {
                if let Some(hit) = shape.hit(&ray) {
                    if closest_hit.is_none() || (hit.t < closest_hit.as_ref().unwrap().t) {
                        closest_hit = Some(hit);
                    }
                }
            }

            let color = if let Some(hit) = closest_hit {
                let n = (hit.n.normalize() * 0.5) + Vec3A::splat(0.5);
                [
                    (n.x.clamp(0.0, 1.0) * 255.0) as u8,
                    (n.y.clamp(0.0, 1.0) * 255.0) as u8,
                    (n.z.clamp(0.0, 1.0) * 255.0) as u8,
                ]
            } else {
                [20, 20, 20]
            };

            row[x * 3] = color[0];
            row[x * 3 + 1] = color[1];
            row[x * 3 + 2] = color[2];
        }
    });

    save_image_as_png(&pixels, width, height, "test_normals.png").unwrap();
    println!("Rendered normal test image to test_normals.png");
}
