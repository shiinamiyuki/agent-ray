use glam::{Vec2, Vec3A};
use std::sync::Arc;

pub mod point;
pub use point::PointLight;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Result of sampling a light source from a given surface point.
pub struct LightSample {
    /// Direction from the shading point toward the light (world space, normalised).
    pub wi: Vec3A,
    /// Distance from the shading point to the light sample point.
    /// Delta lights return the exact geometric distance.
    pub dist: f32,
    /// Incident radiance at the shading point for this sample.  
    /// For delta lights this already incorporates the inverse-square falloff;
    /// the caller multiplies by the BSDF and divides by `pdf`.
    pub li: Vec3A,
    /// Solid-angle PDF of this sample.  
    /// For delta lights `pdf == 1.0` — they should be handled without MIS.
    pub pdf: f32,
}

/// Common interface for all light sources.
pub trait Light: Send + Sync {
    /// Sample a direction toward the light as seen from `ref_point`.  
    /// `u` is a 2-D uniform random variable in [0,1)².  
    /// Returns `None` if the light provides no contribution (e.g. shading point
    /// coincides with the source).
    fn sample(&self, ref_point: Vec3A, u: Vec2) -> Option<LightSample>;

    /// Total radiant power (Φ, watts) emitted by the light.  
    /// Used by `PowerLightDistribution` to weight sampling.
    fn power(&self) -> Vec3A;

    /// Whether the light is a *delta distribution* (point, directional, …).  
    /// Delta lights cannot contribute via BSDF-driven path continuation, so
    /// integrators should skip MIS for them.
    fn is_delta(&self) -> bool;
}

// ---------------------------------------------------------------------------
// Light distribution trait
// ---------------------------------------------------------------------------

/// Strategy for choosing which light to sample when a scene contains many lights.
///
/// Callers use the two-step pattern:
/// 1. `let (idx, sel_pdf) = dist.sample_index(u_sel);`  
/// 2. `let ls = lights[idx].sample(p, u_dir)?;`  
/// 3. Combined PDF: `sel_pdf * ls.pdf`
pub trait LightDistribution: Send + Sync {
    /// Draw a light index.  `u` is a uniform random variable in [0, 1).  
    /// Returns `(index, probability_of_choosing_that_index)`.
    fn sample_index(&self, u: f32) -> (usize, f32);

    /// Probability mass assigned to light `idx`.
    fn pmf(&self, idx: usize) -> f32;

    /// Number of lights in the distribution.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Uniform distribution
// ---------------------------------------------------------------------------

/// Selects each light with equal probability 1/N.
pub struct UniformLightDistribution {
    n: usize,
}

impl UniformLightDistribution {
    pub fn new(lights: &[Arc<dyn Light>]) -> Self {
        Self { n: lights.len() }
    }
}

impl LightDistribution for UniformLightDistribution {
    fn sample_index(&self, u: f32) -> (usize, f32) {
        let idx = ((u * self.n as f32) as usize).min(self.n - 1);
        (idx, 1.0 / self.n as f32)
    }

    fn pmf(&self, _idx: usize) -> f32 {
        1.0 / self.n as f32
    }

    fn len(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Power-proportional distribution
// ---------------------------------------------------------------------------

/// Samples lights proportionally to their total emitted *luminance power*,
/// approximated as the luminosity of `Light::power()` using the standard
/// luminous-efficiency approximation Y = 0.2126 R + 0.7152 G + 0.0722 B.
///
/// This is a simple but effective heuristic: bright lights are sampled more
/// often, reducing variance without requiring scene-space information.
pub struct PowerLightDistribution {
    /// Unnormalised CDF (cdf[i] = Σ weight[0..=i], normalised so cdf.last() == 1).
    cdf: Vec<f32>,
    /// Per-light PMF values (normalised weights).
    pmf: Vec<f32>,
}

impl PowerLightDistribution {
    /// Build the distribution from a slice of lights.
    /// Panics if `lights` is empty.
    pub fn new(lights: &[Arc<dyn Light>]) -> Self {
        assert!(!lights.is_empty(), "PowerLightDistribution requires at least one light");

        let weights: Vec<f32> = lights
            .iter()
            .map(|l| {
                let p = l.power();
                // Luminance approximation
                (0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z).max(0.0)
            })
            .collect();

        let total: f32 = weights.iter().sum();
        let (pmf, cdf) = if total == 0.0 {
            // Degenerate case: all lights have zero power → fall back to uniform.
            let n = lights.len();
            let u = 1.0 / n as f32;
            let pmf = vec![u; n];
            let cdf = (1..=n).map(|i| i as f32 * u).collect();
            (pmf, cdf)
        } else {
            let pmf: Vec<f32> = weights.iter().map(|&w| w / total).collect();
            let mut cdf = Vec::with_capacity(pmf.len());
            let mut acc = 0.0f32;
            for &p in &pmf {
                acc += p;
                cdf.push(acc);
            }
            // Clamp last entry to exactly 1 to avoid floating-point overshoot.
            *cdf.last_mut().unwrap() = 1.0;
            (pmf, cdf)
        };

        Self { cdf, pmf }
    }
}

impl LightDistribution for PowerLightDistribution {
    fn sample_index(&self, u: f32) -> (usize, f32) {
        // Binary search for the first cdf entry >= u.
        let idx = self
            .cdf
            .partition_point(|&c| c < u)
            .min(self.cdf.len() - 1);
        (idx, self.pmf[idx])
    }

    fn pmf(&self, idx: usize) -> f32 {
        self.pmf[idx]
    }

    fn len(&self) -> usize {
        self.pmf.len()
    }
}
