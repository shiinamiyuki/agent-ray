use glam::{Vec2, Vec3A};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use crate::cameras::Camera;
use crate::geometry::Ray;
use crate::integrators::Integrator;
use crate::scene::Scene;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the path tracer.
pub struct PathTracerConfig {
    /// Samples per pixel.
    pub spp: u32,
    /// Maximum path length (number of bounces including the camera ray).
    pub max_depth: u32,
    /// Minimum path depth before Russian roulette starts.
    pub rr_depth: u32,
}

impl Default for PathTracerConfig {
    fn default() -> Self {
        Self { spp: 64, max_depth: 8, rr_depth: 3 }
    }
}

// ---------------------------------------------------------------------------
// PathTracer
// ---------------------------------------------------------------------------

pub struct PathTracer {
    pub config: PathTracerConfig,
}

impl PathTracer {
    pub fn new(config: PathTracerConfig) -> Self {
        Self { config }
    }

    /// Estimate incident radiance along `ray` using an iterative path tracer.
    ///
    /// ## Algorithm
    ///
    /// Each bounce:
    /// 1. **Direct lighting** — one light is sampled from the scene's light
    ///    distribution.  Delta lights (e.g. point lights) contribute via
    ///    direct light sampling only; non-delta lights will use MIS (TODO).
    /// 2. **Indirect lighting** — the BSDF is importance-sampled to extend
    ///    the path; throughput is updated as `f·|cosθ| / pdf`.
    /// 3. **Russian roulette** — paths are terminated stochastically after
    ///    `rr_depth` bounces, with survival probability = max component of
    ///    throughput (clamped to 0.95).
    fn li(&self, scene: &Scene, initial_ray: Ray, rng: &mut SmallRng) -> Vec3A {
        let mut radiance = Vec3A::ZERO;
        let mut throughput = Vec3A::ONE;
        let mut ray = initial_ray;

        for depth in 0..self.config.max_depth {
            // ----------------------------------------------------------------
            // Intersect scene
            // ----------------------------------------------------------------
            let Some((_t, hit)) = scene.intersect(&ray) else {
                // Miss — environment radiance (black; extend to env-map later).
                break;
            };

            let wo = -ray.direction; // outgoing: toward previous vertex / camera
            let wo_local = hit.onb.to_local(wo);

            // ----------------------------------------------------------------
            // Direct lighting — sample one light
            // ----------------------------------------------------------------
            if let Some(light_dist) = &scene.light_dist {
                let u_sel: f32 = rng.random();
                let (light_idx, sel_pmf) = light_dist.sample_index(u_sel);
                let light = &scene.lights[light_idx];

                let u_light = Vec2::new(rng.random(), rng.random());
                if let Some(ls) = light.sample(hit.p, u_light) {
                    // Offset origin along the shading normal in the light's
                    // half-space to avoid self-intersection.
                    let offset = hit.n * 1e-4 * ls.wi.dot(hit.n).signum();
                    // Shorten the ray slightly so it stops just in front of
                    // the light sample point.
                    let shadow_ray =
                        Ray::new(hit.p + offset, ls.wi, 0.0, ls.dist * (1.0 - 1e-4));

                    if !scene.occluded(&shadow_ray) {
                        let wi_local = hit.onb.to_local(ls.wi);
                        let cos_wi = wi_local.z.abs();

                        if cos_wi > 0.0 {
                            let f = hit.bsdf.eval(wi_local, wo_local, hit.tex_uv);

                            if light.is_delta() {
                                // Delta light: direct sampling, no MIS.
                                radiance += throughput * f * cos_wi * ls.li / sel_pmf;
                            } else {
                                // Non-delta: MIS with BSDF sampling.
                                // pdf_light is in solid angle; we balance using
                                // the power heuristic.
                                let pdf_light = ls.pdf * sel_pmf;
                                let pdf_bsdf = hit.bsdf.pdf(wi_local, wo_local);
                                let mis_w = power_heuristic(pdf_light, pdf_bsdf);
                                radiance +=
                                    throughput * f * cos_wi * ls.li * mis_w / pdf_light;
                            }
                        }
                    }
                }
            }

            // ----------------------------------------------------------------
            // Sample BSDF to extend the path
            // ----------------------------------------------------------------
            let u_sel: f32 = rng.random();
            let u_dir = Vec2::new(rng.random(), rng.random());

            let Some(bs) = hit.bsdf.sample(wo_local, hit.tex_uv, u_sel, u_dir) else {
                break;
            };

            if bs.pdf == 0.0 || bs.f == Vec3A::ZERO {
                break;
            }

            let cos_wi = bs.wi.z.abs(); // local space, n = +Z
            throughput *= bs.f * cos_wi / bs.pdf;

            // ----------------------------------------------------------------
            // Russian roulette
            // ----------------------------------------------------------------
            if depth + 1 >= self.config.rr_depth {
                let survival = throughput.max_element().min(0.95);
                if rng.random::<f32>() > survival {
                    break;
                }
                throughput /= survival;
            }

            // ----------------------------------------------------------------
            // Spawn next ray
            // ----------------------------------------------------------------
            let wi_world = hit.onb.to_world(bs.wi);
            let offset = hit.n * 1e-4 * wi_world.dot(hit.n).signum();
            ray = Ray::new(hit.p + offset, wi_world, 1e-4, f32::MAX);
        }

        radiance
    }
}

impl Integrator for PathTracer {
    fn render(
        &self,
        scene: &Scene,
        camera: &dyn Camera,
        width: usize,
        height: usize,
    ) -> Vec<Vec3A> {
        let spp = self.config.spp;
        let mut pixels = vec![Vec3A::ZERO; width * height];

        // Process each row in parallel; per-pixel RNG seeded deterministically.
        pixels
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..width {
                    // Seed incorporates pixel position so each pixel gets an
                    // independent sequence, yielding good coverage.
                    let seed = (y as u64).wrapping_mul(2654435761)
                        ^ (x as u64).wrapping_mul(805459861);
                    let mut rng = SmallRng::seed_from_u64(seed);

                    let mut accum = Vec3A::ZERO;
                    for _ in 0..spp {
                        // Jittered sub-pixel sample.
                        let jx: f32 = rng.random();
                        let jy: f32 = rng.random();
                        let u = (x as f32 + jx) / width as f32;
                        let v = 1.0 - (y as f32 + jy) / height as f32;
                        let ray = camera.generate_ray(glam::Vec2::new(u, v));
                        accum += self.li(scene, ray, &mut rng);
                    }
                    row[x] = accum / spp as f32;
                }
            });

        pixels
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Balance heuristic (β=2) for MIS.
///
/// w(p_a, p_b) = p_a² / (p_a² + p_b²)
#[inline]
fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    a2 / (a2 + b2)
}
