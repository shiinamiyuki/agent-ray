use agent_ray::accel::bvh::{BLASAccel, InstanceBuildInfo, TLAS};
use agent_ray::cameras::{Camera, PinholeCamera};
use agent_ray::primitives::mesh::TriangleMesh;
use agent_ray::utils::save_image_as_png;
use glam::{Mat4, Vec2, Vec3A};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

fn main() {
    let width = 800;
    let height = 600;
    let aspect_ratio = width as f32 / height as f32;

    let camera = PinholeCamera::new(
        Vec3A::new(10.0, 10.0, 30.0),
        Vec3A::new(0.0, 5.0, 0.0),
        Vec3A::new(0.0, 1.0, 0.0),
        45.0,
        aspect_ratio,
    );

    let mesh_path = Path::new("assets/fireplace_room/fireplace_room.obj");
    let meshes = TriangleMesh::load_obj(mesh_path).expect("Failed to load mesh");

    let instance_build_info: Vec<InstanceBuildInfo> = meshes
        .into_iter()
        .map(|m| InstanceBuildInfo {
            blas: Arc::new(BLASAccel::build(Arc::new(m))),
            transform: Mat4::IDENTITY,
        })
        .collect();

    let tlas = TLAS::build(&instance_build_info);

    let mut pixels = vec![0u8; (width * height * 3) as usize];

    pixels
        .par_chunks_mut((width * 3) as usize)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width as usize {
                let uv = Vec2::new(x as f32 / width as f32, 1.0 - (y as f32 / height as f32));
                let ray = camera.generate_ray(uv);

                let color = if let Some(hit) = tlas.intersect(&ray) {
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
