use glam::{Vec2, Vec3A};
use std::f32::consts::PI;

use crate::surfaces::{abs_cos_theta, cos2_theta, fresnel_schlick, reflect, same_hemisphere, Bsdf, BsdfSample};

// ---------------------------------------------------------------------------
// GGX distribution helpers
// ---------------------------------------------------------------------------

/// Isotropic GGX (Trowbridge-Reitz) NDF.
///
/// D(h) = α² / (π · ((α²−1)·cos²θ_h + 1)²)
#[inline]
fn ggx_d(alpha: f32, h: Vec3A) -> f32 {
    let a2 = alpha * alpha;
    let cos2_h = cos2_theta(h);
    let denom = cos2_h * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}

/// Auxiliary function Λ for the Smith height-correlated G2.
///
/// Λ(v) = (√(1 + α²·tan²θ_v) − 1) / 2
#[inline]
fn ggx_lambda(alpha: f32, v: Vec3A) -> f32 {
    let cos2 = cos2_theta(v);
    if cos2 <= 0.0 {
        return f32::INFINITY;
    }
    let a2 = alpha * alpha;
    let tan2 = (1.0 - cos2) / cos2;
    ((1.0 + a2 * tan2).sqrt() - 1.0) * 0.5
}

/// Smith G1 (one-sided masking); kept for reference and single-lobe uses.
#[allow(dead_code)]
#[inline]
fn ggx_g1(alpha: f32, v: Vec3A) -> f32 {
    1.0 / (1.0 + ggx_lambda(alpha, v))
}

/// Height-correlated Smith G2.
///
/// G2(wi, wo) = 1 / (1 + Λ(wi) + Λ(wo))
#[inline]
fn ggx_g2(alpha: f32, wi: Vec3A, wo: Vec3A) -> f32 {
    1.0 / (1.0 + ggx_lambda(alpha, wi) + ggx_lambda(alpha, wo))
}

/// Sample a micro-normal from the isotropic GGX NDF.
///
/// Inverts the GGX CDF: tan²θ_h = α²·ξ₁ / (1−ξ₁)
fn ggx_sample_wh(alpha: f32, wo: Vec3A, u: Vec2) -> Vec3A {
    let a2 = alpha * alpha;
    let cos2_h = (1.0 - u.x) / ((a2 - 1.0) * u.x + 1.0);
    let cos_h = cos2_h.max(0.0).sqrt();
    let sin_h = (1.0 - cos2_h).max(0.0).sqrt();
    let phi = 2.0 * PI * u.y;
    let mut h = Vec3A::new(sin_h * phi.cos(), sin_h * phi.sin(), cos_h);
    // Ensure h is in the same hemisphere as wo
    if h.z * wo.z < 0.0 {
        h = -h;
    }
    h
}

// ---------------------------------------------------------------------------
// ConductorBsdf
// ---------------------------------------------------------------------------

/// Cook-Torrance microfacet BSDF for opaque conductors.
///
/// ```text
/// f(wi,wo) = D(h) · G2(wi,wo) · F(wi·h) / (4 |cosθᵢ| |cosθₒ|)
/// ```
///
/// | Term | Model |
/// |------|-------|
/// | D    | Isotropic GGX (Trowbridge-Reitz) |
/// | G    | Height-correlated Smith G2 |
/// | F    | Schlick Fresnel |
///
/// `roughness` maps to the GGX α parameter (`roughness²` remapping is **not**
/// applied here — pass a perceptual roughness that has already been squared
/// to α if desired).
pub struct ConductorBsdf {
    /// Reflectance at normal incidence (F0).
    pub f0: Vec3A,
    /// GGX α (roughness).  Clamped to [1e-3, 1] to avoid singularities.
    pub roughness: f32,
}

impl ConductorBsdf {
    pub fn new(f0: Vec3A, roughness: f32) -> Self {
        Self {
            f0,
            roughness: roughness.clamp(1e-3, 1.0),
        }
    }
}

impl Bsdf for ConductorBsdf {
    fn eval(&self, wi: Vec3A, wo: Vec3A) -> Vec3A {
        if !same_hemisphere(wi, wo) {
            return Vec3A::ZERO;
        }
        let cos_wi = abs_cos_theta(wi);
        let cos_wo = abs_cos_theta(wo);
        if cos_wi == 0.0 || cos_wo == 0.0 {
            return Vec3A::ZERO;
        }

        let h = (wi + wo).normalize();
        let f = fresnel_schlick(wi.dot(h).abs(), self.f0);
        let d = ggx_d(self.roughness, h);
        let g = ggx_g2(self.roughness, wi, wo);

        f * (d * g / (4.0 * cos_wi * cos_wo))
    }

    fn sample(&self, wo: Vec3A, _u_sel: f32, u_dir: Vec2) -> Option<BsdfSample> {
        if wo.z == 0.0 {
            return None;
        }
        let h = ggx_sample_wh(self.roughness, wo, u_dir);
        let wi = reflect(wo, h);
        if !same_hemisphere(wi, wo) {
            return None;
        }
        let pdf = self.pdf(wi, wo);
        if pdf == 0.0 {
            return None;
        }
        Some(BsdfSample {
            wi,
            pdf,
            f: self.eval(wi, wo),
        })
    }

    fn pdf(&self, wi: Vec3A, wo: Vec3A) -> f32 {
        if !same_hemisphere(wi, wo) {
            return 0.0;
        }
        let h = (wi + wo).normalize();
        let wh_dot_wi = wi.dot(h).abs();
        if wh_dot_wi == 0.0 {
            return 0.0;
        }
        // pdf(wi) = D(h)·|cosθ_h| / (4·|wi·h|)
        ggx_d(self.roughness, h) * abs_cos_theta(h) / (4.0 * wh_dot_wi)
    }
}

// ---------------------------------------------------------------------------
// DielectricBsdf
// ---------------------------------------------------------------------------

/// Cook-Torrance microfacet BSDF for smooth or rough dielectrics (glass-like).
///
/// Handles both reflection and refraction, switching probabilistically based on
/// the Fresnel reflectance.
pub struct DielectricBsdf {
    /// Interior index of refraction (medium on the wi side for transmission).
    pub eta: f32,
    /// GGX α (roughness).
    pub roughness: f32,
}

impl DielectricBsdf {
    pub fn new(eta: f32, roughness: f32) -> Self {
        Self {
            eta,
            roughness: roughness.clamp(1e-3, 1.0),
        }
    }

    /// Schlick Fresnel for a dielectric; `cos_i` is the angle with the half-vector.
    fn fresnel(&self, cos_i: f32) -> f32 {
        crate::surfaces::fresnel_dielectric(cos_i, self.eta)
    }
}

impl Bsdf for DielectricBsdf {
    fn eval(&self, wi: Vec3A, wo: Vec3A) -> Vec3A {
        let reflect_side = same_hemisphere(wi, wo);
        let cos_wo = abs_cos_theta(wo);
        let cos_wi = abs_cos_theta(wi);
        if cos_wo == 0.0 || cos_wi == 0.0 {
            return Vec3A::ZERO;
        }

        if reflect_side {
            // --- reflective lobe ---
            let h = (wi + wo).normalize();
            let f = self.fresnel(wi.dot(h).abs());
            let d = ggx_d(self.roughness, h);
            let g = ggx_g2(self.roughness, wi, wo);
            Vec3A::splat(f * d * g / (4.0 * cos_wi * cos_wo))
        } else {
            // --- transmissive lobe ---
            // Half-vector for refraction (points into the denser medium).
            let eta = if wo.z > 0.0 { self.eta } else { 1.0 / self.eta };
            let h = -(wo + wi * eta).normalize();
            // Ensure h points outwards (same hemisphere as wo).
            let h = if h.z * wo.z < 0.0 { -h } else { h };

            let wo_dot_h = wo.dot(h);
            let wi_dot_h = wi.dot(h);
            if wo_dot_h == 0.0 || wi_dot_h == 0.0 {
                return Vec3A::ZERO;
            }

            let f = 1.0 - self.fresnel(wi_dot_h.abs());
            let d = ggx_d(self.roughness, h);
            let g = ggx_g2(self.roughness, wi, wo);

            let denom = (wo_dot_h + eta * wi_dot_h).powi(2);
            if denom == 0.0 {
                return Vec3A::ZERO;
            }

            // Radiometric correction factor for non-unit IOR.
            let eta2 = eta * eta;
            let value = f * d * g * eta2 * wo_dot_h.abs() * wi_dot_h.abs()
                / (cos_wi * cos_wo * denom);
            Vec3A::splat(value)
        }
    }

    fn sample(&self, wo: Vec3A, u_sel: f32, u_dir: Vec2) -> Option<BsdfSample> {
        if wo.z == 0.0 {
            return None;
        }
        let h = ggx_sample_wh(self.roughness, wo, u_dir);
        let f = self.fresnel(wo.dot(h).abs());

        if u_sel < f {
            // --- sample reflection ---
            let wi = reflect(wo, h);
            if !same_hemisphere(wi, wo) {
                return None;
            }
            let pdf = self.pdf(wi, wo);
            if pdf == 0.0 {
                return None;
            }
            Some(BsdfSample {
                wi,
                pdf,
                f: self.eval(wi, wo),
            })
        } else {
            // --- sample refraction ---
            let eta = if wo.z > 0.0 { 1.0 / self.eta } else { self.eta };
            let wi = crate::surfaces::refract(wo, h, eta)?;
            if same_hemisphere(wi, wo) {
                return None; // TIR should have been caught by fresnel test
            }
            let pdf = self.pdf(wi, wo);
            if pdf == 0.0 {
                return None;
            }
            Some(BsdfSample {
                wi,
                pdf,
                f: self.eval(wi, wo),
            })
        }
    }

    fn pdf(&self, wi: Vec3A, wo: Vec3A) -> f32 {
        let cos_wo = abs_cos_theta(wo);
        let cos_wi = abs_cos_theta(wi);
        if cos_wo == 0.0 || cos_wi == 0.0 {
            return 0.0;
        }

        if same_hemisphere(wi, wo) {
            // Reflection pdf
            let h = (wi + wo).normalize();
            let wh_dot_wi = wi.dot(h).abs();
            if wh_dot_wi == 0.0 {
                return 0.0;
            }
            let f = self.fresnel(wo.dot(h).abs());
            f * ggx_d(self.roughness, h) * abs_cos_theta(h) / (4.0 * wh_dot_wi)
        } else {
            // Refraction pdf
            let eta = if wo.z > 0.0 { self.eta } else { 1.0 / self.eta };
            let h = -(wo + wi * eta).normalize();
            let h = if h.z * wo.z < 0.0 { -h } else { h };

            let wo_dot_h = wo.dot(h);
            let wi_dot_h = wi.dot(h);
            if wo_dot_h == 0.0 || wi_dot_h == 0.0 {
                return 0.0;
            }
            let denom = (wo_dot_h + eta * wi_dot_h).powi(2);
            if denom == 0.0 {
                return 0.0;
            }
            let f = 1.0 - self.fresnel(wo_dot_h.abs());
            f * ggx_d(self.roughness, h) * abs_cos_theta(h) * eta * eta * wi_dot_h.abs() / denom
        }
    }
}
