use glam::{Vec2, Vec3A, Vec4};
use std::path::Path;
use anyhow::Result;
use crate::geometry::{AABB, Ray, Triangle, Intersect};
use crate::accel::bvh::{BLASPrimitive, RayHit};

pub struct TriangleMesh {
    pub positions: Vec<Vec3A>,
    pub normals: Option<Vec<Vec3A>>,
    pub tex_coords: Option<Vec<Vec2>>,
    pub tangents: Option<Vec<Vec4>>,
    pub indices: Vec<u32>,
    pub material_slots: Vec<u32>,
}

impl TriangleMesh {
    pub fn load_obj(path: &Path) -> Result<Vec<Self>> {
        let (models, _materials) = tobj::load_obj(
            path,
            &tobj::GPU_LOAD_OPTIONS,
        )?;

        let mut meshes = Vec::new();

        for model in models {
            let mesh = &model.mesh;
            
            let positions: Vec<Vec3A> = mesh.positions
                .chunks_exact(3)
                .map(|p| Vec3A::new(p[0], p[1], p[2]))
                .collect();

            let normals = if !mesh.normals.is_empty() {
                Some(mesh.normals
                    .chunks_exact(3)
                    .map(|n| Vec3A::new(n[0], n[1], n[2]))
                    .collect())
            } else {
                None
            };

            let tex_coords = if !mesh.texcoords.is_empty() {
                Some(mesh.texcoords
                    .chunks_exact(2)
                    .map(|t| Vec2::new(t[0], t[1]))
                    .collect())
            } else {
                None
            };

            // tobj doesn't load tangents by default, so we'll leave it as None for now
            // We can implement tangent calculation later if needed.
            let tangents = None;

            let indices = mesh.indices.clone();
            let material_slots = vec![mesh.material_id.unwrap_or(0) as u32];

            meshes.push(TriangleMesh {
                positions,
                normals,
                tex_coords,
                tangents,
                indices,
                material_slots,
            });
        }

        Ok(meshes)
    }
}

impl BLASPrimitive for TriangleMesh {
    fn primitive_count(&self) -> usize {
        self.indices.len() / 3
    }

    fn primitive_aabb(&self, prim_id: usize) -> AABB {
        let i0 = self.indices[prim_id * 3] as usize;
        let i1 = self.indices[prim_id * 3 + 1] as usize;
        let i2 = self.indices[prim_id * 3 + 2] as usize;

        let v0 = self.positions[i0];
        let v1 = self.positions[i1];
        let v2 = self.positions[i2];

        let min = v0.min(v1).min(v2);
        let max = v0.max(v1).max(v2);

        AABB::new(min, max)
    }

    fn intersect_primitive(&self, prim_id: usize, ray: &Ray, t_max: f32) -> Option<RayHit> {
        let i0 = self.indices[prim_id * 3] as usize;
        let i1 = self.indices[prim_id * 3 + 1] as usize;
        let i2 = self.indices[prim_id * 3 + 2] as usize;

        let v0 = self.positions[i0];
        let v1 = self.positions[i1];
        let v2 = self.positions[i2];

        let triangle = Triangle::new(v0, v1, v2);
        let mut local_ray = *ray;
        local_ray.t_max = t_max;

        triangle.hit(&local_ray).map(|hit| {
            RayHit {
                instance_id: 0,
                prim_id: prim_id as u32,
                n: hit.n,
                uv: hit.uv,
                t: hit.t,
            }
        })
    }
}
