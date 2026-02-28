use glam::{Vec2, Vec3A};
use std::f32::consts::FRAC_1_PI;

use crate::surfaces::{abs_cos_theta, cosine_hemisphere_sample, same_hemisphere, Bsdf, BsdfSample};

/// Perfectly diffuse (Lambertian) BSDF.
///
/// `f(wi, wo) = albedo / π`
///
/// Sampled with cosine-weighted hemisphere importance sampling so that
/// `pdf(wi) = cos_theta(wi) / π`.
pub struct Lambertian {
    /// Diffuse reflectance (ρ).  Values should be in [0, 1]^3.
    pub albedo: Vec3A,
}

impl Lambertian {
    pub fn new(albedo: Vec3A) -> Self {
        Self { albedo }
    }
}

impl Bsdf for Lambertian {
    fn eval(&self, wi: Vec3A, wo: Vec3A) -> Vec3A {
        if !same_hemisphere(wi, wo) {
            return Vec3A::ZERO;
        }
        self.albedo * FRAC_1_PI
    }

    fn sample(&self, wo: Vec3A, _u_sel: f32, u_dir: Vec2) -> Option<BsdfSample> {
        // Cosine-weighted sample, placed in the same hemisphere as `wo`.
        let mut wi = cosine_hemisphere_sample(u_dir);
        if wo.z < 0.0 {
            wi.z = -wi.z;
        }
        let pdf = abs_cos_theta(wi) * FRAC_1_PI;
        if pdf == 0.0 {
            return None;
        }
        Some(BsdfSample {
            wi,
            pdf,
            f: self.albedo * FRAC_1_PI,
        })
    }

    fn pdf(&self, wi: Vec3A, wo: Vec3A) -> f32 {
        if !same_hemisphere(wi, wo) {
            return 0.0;
        }
        abs_cos_theta(wi) * FRAC_1_PI
    }
}
