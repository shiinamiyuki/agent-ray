use std::sync::Arc;
use glam::{Mat3, Mat4, Vec2, Vec3A};

use crate::accel::bvh::{BLASAccel, BLASPrimitive, InstanceBuildInfo, TLAS};
use crate::geometry::{Onb, Ray};
use crate::lights::{Light, LightDistribution};
use crate::primitives::mesh::TriangleMesh;
use crate::surfaces::Bsdf;

// ---------------------------------------------------------------------------
// SceneObject
// ---------------------------------------------------------------------------

/// One logical object in the scene: a mesh, its world transform, and
/// the BSDF materials corresponding to its material slots.
pub struct SceneObject {
    pub mesh: Arc<TriangleMesh>,
    pub blas: Arc<BLASAccel>,
    /// Local-to-world transform.
    pub transform: Mat4,
    /// Column-major 3×3 inverse-transpose used for normals.
    normal_mat: Mat3,
    /// BSDFs indexed by material slot.  
    /// `TriangleMesh::material_slots` maps triangle → slot index.
    pub materials: Vec<Arc<dyn Bsdf>>,
}

impl SceneObject {
    pub fn new(mesh: Arc<TriangleMesh>, transform: Mat4, materials: Vec<Arc<dyn Bsdf>>) -> Self {
        let blas = Arc::new(BLASAccel::build(
            Arc::clone(&mesh) as Arc<dyn BLASPrimitive>,
        ));
        let normal_mat = Mat3::from_mat4(transform).inverse().transpose();
        Self { mesh, blas, transform, normal_mat, materials }
    }

    /// Resolve the BSDF for triangle `prim_id`.
    ///
    /// If the mesh has one shared material slot, all triangles use it.
    /// Otherwise `material_slots[prim_id]` selects from `self.materials`.
    pub fn material(&self, prim_id: usize) -> Arc<dyn Bsdf> {
        let slot = if self.mesh.material_slots.len() == 1 {
            self.mesh.material_slots[0] as usize
        } else {
            self.mesh.material_slots[prim_id] as usize
        };
        Arc::clone(&self.materials[slot.min(self.materials.len() - 1)])
    }

    /// Interpolated texture UV coordinates for triangle `prim_id`.
    ///
    /// When the mesh has no `tex_coords`, the raw barycentric `(u, v)` are
    /// returned as a fallback — this is meaningless as a texture address but
    /// keeps the code path simple and prevents a crash.
    pub fn tex_uv(&self, prim_id: usize, bary_uv: Vec2) -> Vec2 {
        if let Some(tex_coords) = &self.mesh.tex_coords {
            let i0 = self.mesh.indices[prim_id * 3] as usize;
            let i1 = self.mesh.indices[prim_id * 3 + 1] as usize;
            let i2 = self.mesh.indices[prim_id * 3 + 2] as usize;
            let w = 1.0 - bary_uv.x - bary_uv.y;
            tex_coords[i0] * w + tex_coords[i1] * bary_uv.x + tex_coords[i2] * bary_uv.y
        } else {
            bary_uv
        }
    }

    /// World-space shading normal for triangle `prim_id`.
    ///
    /// Interpolates vertex normals (bilinear barycentrics) when they are
    /// present; otherwise uses the geometric normal returned by the BVH.
    ///
    /// `bary_uv` are the (u, v) from Möller-Trumbore:
    ///   P = (1-u-v)·v0 + u·v1 + v·v2
    /// `geom_n_obj` is the un-transformed geometric normal in object space.
    pub fn shading_normal(&self, prim_id: usize, bary_uv: Vec2, geom_n_obj: Vec3A) -> Vec3A {
        let n_obj = if let Some(normals) = &self.mesh.normals {
            let i0 = self.mesh.indices[prim_id * 3] as usize;
            let i1 = self.mesh.indices[prim_id * 3 + 1] as usize;
            let i2 = self.mesh.indices[prim_id * 3 + 2] as usize;
            let w = 1.0 - bary_uv.x - bary_uv.y;
            (normals[i0] * w + normals[i1] * bary_uv.x + normals[i2] * bary_uv.y).normalize()
        } else {
            geom_n_obj
        };
        // Normal transforms by the inverse-transpose of the linear part.
        Vec3A::from(self.normal_mat * glam::Vec3::from(n_obj)).normalize()
    }
}

// ---------------------------------------------------------------------------
// ShadingPoint
// ---------------------------------------------------------------------------

/// All per-hit surface data needed by an integrator.
pub struct ShadingPoint {
    /// World-space hit point.
    pub p: Vec3A,
    /// World-space shading normal (unit, faces toward the incoming ray).
    pub n: Vec3A,
    /// Orthonormal basis aligned with `n` (local Z = n).
    pub onb: Onb,
    /// Resolved BSDF for this triangle.
    pub bsdf: Arc<dyn Bsdf>,
    /// Interpolated texture UV coordinates at the hit point.
    pub tex_uv: Vec2,
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// A fully assembled renderable scene.
pub struct Scene {
    pub tlas: TLAS,
    /// Scene objects in TLAS instance order (index == `RayHit::instance_id`).
    pub objects: Vec<Arc<SceneObject>>,
    /// All light sources.
    pub lights: Vec<Arc<dyn Light>>,
    /// Importance distribution over `lights`.
    pub light_dist: Option<Box<dyn LightDistribution>>,
}

impl Scene {
    /// Build a scene from objects, lights, and a light distribution strategy.
    pub fn new(
        objects: Vec<Arc<SceneObject>>,
        lights: Vec<Arc<dyn Light>>,
        light_dist: Option<Box<dyn LightDistribution>>,
    ) -> Self {
        let instance_build_info: Vec<InstanceBuildInfo> = objects
            .iter()
            .map(|o| InstanceBuildInfo {
                blas: Arc::clone(&o.blas),
                transform: o.transform,
            })
            .collect();
        let tlas = TLAS::build(&instance_build_info);
        Self { tlas, objects, lights, light_dist }
    }

    /// Closest-hit query.  Returns `(t, ShadingPoint)` or `None` on a miss.
    pub fn intersect(&self, ray: &Ray) -> Option<(f32, ShadingPoint)> {
        let hit = self.tlas.intersect(ray)?;

        let obj = &self.objects[hit.instance_id as usize];
        let prim_id = hit.prim_id as usize;

        let p = ray.at(hit.t);

        // `hit.n` is in object (BLAS) space — transform to world space.
        let n = obj.shading_normal(prim_id, hit.uv, hit.n);

        // Flip shading normal toward the incoming ray to handle back-faces.
        let wo = -ray.direction;
        let n = if n.dot(wo) < 0.0 { -n } else { n };

        let onb = Onb::from_normal(n);
        let bsdf = obj.material(prim_id);
        let tex_uv = obj.tex_uv(prim_id, hit.uv);

        Some((hit.t, ShadingPoint { p, n, onb, bsdf, tex_uv }))
    }

    /// Shadow / occlusion query.  Returns `true` when the ray is blocked.
    #[inline]
    pub fn occluded(&self, ray: &Ray) -> bool {
        self.tlas.intersect(ray).is_some()
    }
}
