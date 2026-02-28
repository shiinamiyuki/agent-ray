use glam::{Vec2, Vec3A};
use crate::geometry::Ray;
use crate::lights::{EmissionSample, Light, LightSample};

/// An isotropic point light that radiates equally in all directions.
///
/// Because the source is infinitesimally small, this is a **delta light** —
/// it cannot be hit by a random ray and MIS should not be applied to it.
///
/// Radiance arriving at a surface point `p` is:
///
/// ```text
/// Li = intensity / |pos - p|²
/// ```
///
/// ## BDPT support
///
/// `sample_emission` spawns a ray from the point light in a uniformly random
/// direction on the sphere.  The positional component is a delta distribution
/// (`is_positional_delta() == true`), so the BDPT handles it specially in MIS
/// weight computation.
pub struct PointLight {
    /// Position in world space.
    pub position: Vec3A,
    /// Spectral intensity (watts per steradian: I = Φ / 4π).
    pub intensity: Vec3A,
}

impl PointLight {
    pub fn new(position: Vec3A, intensity: Vec3A) -> Self {
        Self { position, intensity }
    }
}

/// Sample a uniformly random direction on the unit sphere.
///
/// Uses the standard parameterisation:
///   cos θ = 1 − 2ξ₁,  φ = 2πξ₂
///
/// PDF = 1 / (4π).
fn uniform_sphere_sample(u: Vec2) -> Vec3A {
    let cos_theta = 1.0 - 2.0 * u.x;
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * std::f32::consts::PI * u.y;
    Vec3A::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

impl Light for PointLight {
    fn sample(&self, ref_point: Vec3A, _u: Vec2) -> Option<LightSample> {
        let offset = self.position - ref_point;
        let dist2 = offset.length_squared();
        let dist = dist2.sqrt();
        if dist < 1e-6 {
            return None;
        }
        let wi = offset / dist;
        // Inverse-square falloff; pdf = 1 for delta lights.
        let li = self.intensity / dist2;
        Some(LightSample { wi, dist, li, pdf: 1.0 })
    }

    /// Total radiant power: Φ = 4π · I  (isotropic point source).
    fn power(&self) -> Vec3A {
        self.intensity * (4.0 * std::f32::consts::PI)
    }

    fn is_delta(&self) -> bool {
        true
    }

    // -------------------------------------------------------------------
    // BDPT emission support
    // -------------------------------------------------------------------

    fn sample_emission(&self, _u_pos: Vec2, u_dir: Vec2) -> Option<EmissionSample> {
        let dir = uniform_sphere_sample(u_dir);
        Some(EmissionSample {
            ray: Ray::new(self.position, dir, 1e-4, f32::MAX),
            // Le = intensity for all directions (isotropic).
            le: self.intensity,
            // Position PDF is a delta — we store 1.0; the MIS weight function
            // must recognise the delta via `is_positional_delta`.
            pdf_pos: 1.0,
            // Uniform sphere: 1 / (4π).
            pdf_dir: 1.0 / (4.0 * std::f32::consts::PI),
            // Point light has no surface; use the emission direction as a
            // stand-in normal so that angle computations don't degenerate.
            n_light: dir,
        })
    }

    fn pdf_emission_dir(&self, _dir: Vec3A) -> f32 {
        // Uniform on sphere.
        1.0 / (4.0 * std::f32::consts::PI)
    }

    fn is_positional_delta(&self) -> bool {
        true
    }
}
