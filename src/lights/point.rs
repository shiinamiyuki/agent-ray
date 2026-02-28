use glam::{Vec2, Vec3A};
use crate::lights::{Light, LightSample};

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
}
