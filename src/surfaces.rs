use crate::{geometry::Onb, prelude::*};
use std::f32::consts::PI;

pub mod lambertian;
pub mod microfacet;

pub use lambertian::Lambertian;
pub use microfacet::{ConductorBsdf, DielectricBsdf};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

pub struct BsdfSample {
    pub wi: Vec3A,
    pub pdf: f32,
    pub f: Vec3A,
}

/// All `eval`/`sample`/`pdf` methods operate in **local space** where the
/// shading normal is the +Z axis.
///
/// `uv` carries the interpolated texture coordinates at the shading point so
/// that texture-mapped parameters (e.g. albedo) can be looked up per call.
pub trait Bsdf: Send + Sync {
    fn eval(&self, wi: Vec3A, wo: Vec3A, uv: Vec2) -> Vec3A;
    fn sample(&self, wo: Vec3A, uv: Vec2, u_sel: f32, u_dir: Vec2) -> Option<BsdfSample>;
    fn pdf(&self, wi: Vec3A, wo: Vec3A) -> f32;
}

// ---------------------------------------------------------------------------
// Local-space trigonometric helpers
// ---------------------------------------------------------------------------

/// cos θ in local space (v.z = cos(angle with normal)).
#[inline] pub fn cos_theta(v: Vec3A) -> f32 { v.z }
#[inline] pub fn cos2_theta(v: Vec3A) -> f32 { v.z * v.z }
#[inline] pub fn abs_cos_theta(v: Vec3A) -> f32 { v.z.abs() }
#[inline] pub fn sin2_theta(v: Vec3A) -> f32 { (1.0 - cos2_theta(v)).max(0.0) }
#[inline] pub fn sin_theta(v: Vec3A) -> f32 { sin2_theta(v).sqrt() }
#[inline] pub fn tan2_theta(v: Vec3A) -> f32 { sin2_theta(v) / cos2_theta(v).max(1e-10) }
#[inline] pub fn same_hemisphere(a: Vec3A, b: Vec3A) -> bool { a.z * b.z > 0.0 }

// ---------------------------------------------------------------------------
// Sampling utilities
// ---------------------------------------------------------------------------

/// Cosine-weighted hemisphere sample.  
/// Returns a direction in local space (normal = +Z) with pdf = cos_theta / π.
pub fn cosine_hemisphere_sample(u: Vec2) -> Vec3A {
    let r = u.x.sqrt();
    let phi = 2.0 * PI * u.y;
    let x = r * phi.cos();
    let y = r * phi.sin();
    let z = (1.0 - x * x - y * y).max(0.0).sqrt();
    Vec3A::new(x, y, z)
}

/// Uniform sample on the unit disk using concentric mapping.
pub fn concentric_disk_sample(u: Vec2) -> Vec2 {
    let u = 2.0 * u - Vec2::ONE;
    if u.x == 0.0 && u.y == 0.0 {
        return Vec2::ZERO;
    }
    let (r, theta) = if u.x.abs() > u.y.abs() {
        (u.x, PI / 4.0 * (u.y / u.x))
    } else {
        (u.y, PI / 2.0 - PI / 4.0 * (u.x / u.y))
    };
    Vec2::new(r * theta.cos(), r * theta.sin())
}

// ---------------------------------------------------------------------------
// Fresnel functions
// ---------------------------------------------------------------------------

/// Schlick approximation to Fresnel reflectance.  
/// `f0`: reflectance at normal incidence (conductor or dielectric).
#[inline]
pub fn fresnel_schlick(cos_theta_i: f32, f0: Vec3A) -> Vec3A {
    let t = (1.0 - cos_theta_i.abs()).clamp(0.0, 1.0).powi(5);
    f0 + (Vec3A::ONE - f0) * t
}

/// Exact Fresnel for a dielectric interface; returns scalar reflectance.
pub fn fresnel_dielectric(cos_theta_i: f32, eta: f32) -> f32 {
    let cos_i = cos_theta_i.clamp(-1.0, 1.0);
    let (cos_i, eta) = if cos_i < 0.0 {
        (-cos_i, 1.0 / eta)
    } else {
        (cos_i, eta)
    };
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t >= 1.0 {
        return 1.0; // total internal reflection
    }
    let cos_t = (1.0 - sin2_t).sqrt();
    let r_par = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let r_per = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    0.5 * (r_par * r_par + r_per * r_per)
}

// ---------------------------------------------------------------------------
// Geometric optics
// ---------------------------------------------------------------------------

/// Reflect `v` about a normal `n` (all in the same coordinate frame).
#[inline]
pub fn reflect(v: Vec3A, n: Vec3A) -> Vec3A {
    -v + n * 2.0 * v.dot(n)
}

/// Refract `v` through a surface with normal `n` and relative IOR `eta`.  
/// Returns `None` on total internal reflection.
pub fn refract(v: Vec3A, n: Vec3A, eta: f32) -> Option<Vec3A> {
    let cos_i = n.dot(v);
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t >= 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin2_t).sqrt();
    Some(eta * (-v) + n * (eta * cos_i - cos_t))
}

// ---------------------------------------------------------------------------
// SurfaceClosure
// ---------------------------------------------------------------------------

/// Pairs a `Bsdf` with a local coordinate frame so callers can work entirely
/// in world space.
pub struct SurfaceClosure {
    pub bsdf: Box<dyn Bsdf>,
    pub onb: Onb,
}

impl SurfaceClosure {
    pub fn new(bsdf: Box<dyn Bsdf>, onb: Onb) -> Self {
        Self { bsdf, onb }
    }

    /// Evaluate the BSDF. `wi` and `wo` are in **world space**.
    pub fn eval(&self, wi: Vec3A, wo: Vec3A, uv: Vec2) -> Vec3A {
        let wi_local = self.onb.to_local(wi);
        let wo_local = self.onb.to_local(wo);
        self.bsdf.eval(wi_local, wo_local, uv)
    }

    /// Sample the BSDF. `wo` is in **world space**; the returned `wi` is also
    /// in world space.
    pub fn sample(&self, wo: Vec3A, uv: Vec2, u_sel: f32, u_dir: Vec2) -> Option<BsdfSample> {
        let wo_local = self.onb.to_local(wo);
        let mut sample = self.bsdf.sample(wo_local, uv, u_sel, u_dir)?;
        sample.wi = self.onb.to_world(sample.wi);
        Some(sample)
    }

    /// PDF of sampling `wi` given `wo`. Both in **world space**.
    pub fn pdf(&self, wi: Vec3A, wo: Vec3A) -> f32 {
        let wi_local = self.onb.to_local(wi);
        let wo_local = self.onb.to_local(wo);
        self.bsdf.pdf(wi_local, wo_local)
    }
}
