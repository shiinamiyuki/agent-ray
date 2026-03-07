//! # Gradient-Domain Path Tracer (G-PT)
//!
//! Implements a gradient-domain path tracer using **random replay shift
//! mapping** as described in:
//!
//! - Kettunen et al., *"Gradient-Domain Path Tracing"*, SIGGRAPH 2015.
//! - Manzi et al., *"Gradient-Domain Path Tracing"*, TOG 34(4), 2015.
//!
//! ## Overview
//!
//! For each pixel we trace a **base path** through the standard path tracing
//! loop, recording every random number consumed by the sampler.  We then
//! **replay** those same random numbers four times, once for each of the
//! neighbouring pixels (x±1, y±1), with only the initial camera-sample
//! jitter shifted by ±1 pixel.  Because the same random sequence drives
//! the BSDF / light / RR decisions the offset paths are highly correlated
//! with the base path, producing low-variance finite-difference gradients:
//!
//!     ΔIx(x,y) = I(x+1,y) − I(x,y)
//!     ΔIy(x,y) = I(x,y+1) − I(x,y)
//!
//! After all samples are accumulated we solve a **screened Poisson equation**
//! to reconstruct the final image from the primal (noisy) image and the
//! gradient fields:
//!
//!     argmin_I  α ‖I − I_primal‖² + ‖∇I − G‖²
//!
//! using a simple SOR (Successive Over-Relaxation) iterative solver.

use std::sync::Arc;

use glam::{Vec2, Vec3A};
use rayon::prelude::*;

use crate::cameras::Camera;
use crate::film::Film;
use crate::geometry::Ray;
use crate::integrators::Integrator;
use crate::sampler::{IndependentSampler, Sampler};
use crate::scene::Scene;

// ===========================================================================
// Configuration
// ===========================================================================

/// Configuration for the gradient-domain path tracer.
pub struct GdptConfig {
    /// Samples per pixel.
    pub spp: u32,
    /// Maximum path depth (number of bounces).
    pub max_depth: u32,
    /// Minimum depth before Russian roulette kicks in.
    pub rr_depth: u32,
    /// Screening weight α for the Poisson reconstruction.
    /// Higher values trust the primal image more; lower values trust the
    /// gradients more.  Typical range: 0.1 – 1.0.
    pub alpha: f32,
    /// Number of SOR iterations for the Poisson reconstruction.
    pub poisson_iterations: u32,
    /// SOR relaxation factor (ω).  ω ∈ (1, 2) for over-relaxation;
    /// ω = 1 is plain Gauss-Seidel.
    pub sor_omega: f32,
}

impl Default for GdptConfig {
    fn default() -> Self {
        Self {
            spp: 64,
            max_depth: 8,
            rr_depth: 3,
            alpha: 0.2,
            poisson_iterations: 200,
            sor_omega: 1.6,
        }
    }
}

// ===========================================================================
// ReplaySampler — records & replays random number sequences
// ===========================================================================

/// A sampler wrapper that can operate in two modes:
///
/// 1. **Record** — forwards to an inner `IndependentSampler` and records
///    every `f32` produced (the "tape").
/// 2. **Replay** — plays back the recorded tape, ignoring the inner RNG.
///
/// The first two values on the tape correspond to the camera jitter; when
/// replaying for an offset pixel we substitute shifted jitter values and
/// replay the rest identically.
struct ReplaySampler {
    inner: IndependentSampler,
    tape: Vec<f32>,
    cursor: usize,
    recording: bool,
}

impl ReplaySampler {
    /// Create a new sampler in **record** mode.
    fn new(inner: IndependentSampler) -> Self {
        Self {
            inner,
            tape: Vec::with_capacity(128),
            cursor: 0,
            recording: true,
        }
    }

    /// Switch to **replay** mode and reset the cursor to the beginning.
    fn start_replay(&mut self) {
        self.recording = false;
        self.cursor = 0;
    }

    /// Access the recorded tape (e.g. to patch camera jitter values).
    fn tape_mut(&mut self) -> &mut Vec<f32> {
        &mut self.tape
    }

    /// Reset for a brand-new sample (record mode).
    fn reset_for_new_sample(&mut self) {
        self.tape.clear();
        self.cursor = 0;
        self.recording = true;
        self.inner.start_next_sample();
    }
}

impl Sampler for ReplaySampler {
    fn next_1d(&mut self) -> f32 {
        if self.recording {
            let v = self.inner.next_1d();
            self.tape.push(v);
            v
        } else {
            let v = if self.cursor < self.tape.len() {
                self.tape[self.cursor]
            } else {
                // Fallback: if the offset path consumes more randoms than the
                // base (shouldn't happen with random replay, but be safe).
                self.inner.next_1d()
            };
            self.cursor += 1;
            v
        }
    }

    fn start_next_sample(&mut self) {
        // In record mode, delegate; in replay mode, just reset cursor.
        if self.recording {
            self.inner.start_next_sample();
        }
        self.cursor = 0;
    }

    fn clone_for_pixel(&self, pixel_x: u32, pixel_y: u32, _sample_index: u32) -> Box<dyn Sampler> {
        Box::new(ReplaySampler::new(IndependentSampler::seeded_for_pixel(
            pixel_x, pixel_y,
        )))
    }
}

// ===========================================================================
// Gradient-Domain Path Tracer
// ===========================================================================

pub struct GradientDomainPathTracer {
    pub config: GdptConfig,
}

impl GradientDomainPathTracer {
    pub fn new(config: GdptConfig) -> Self {
        Self { config }
    }

    /// Standard iterative path trace — identical to `PathTracer::li` but
    /// uses the `Sampler` trait so that a `ReplaySampler` can be passed in.
    fn li(&self, scene: &Scene, initial_ray: Ray, sampler: &mut dyn Sampler) -> Vec3A {
        let mut radiance = Vec3A::ZERO;
        let mut throughput = Vec3A::ONE;
        let mut ray = initial_ray;

        for depth in 0..self.config.max_depth {
            // Intersect
            let Some((_t, hit)) = scene.intersect(&ray) else {
                break;
            };

            let wo = -ray.direction;
            let wo_local = hit.onb.to_local(wo);

            // Direct lighting — sample one light
            if let Some(light_dist) = &scene.light_dist {
                let u_sel: f32 = sampler.next_1d();
                let (light_idx, sel_pmf) = light_dist.sample_index(u_sel);
                let light = &scene.lights[light_idx];

                let u_light = sampler.next_2d();
                if let Some(ls) = light.sample(hit.p, u_light) {
                    let offset = hit.n * 1e-4 * ls.wi.dot(hit.n).signum();
                    let shadow_ray =
                        Ray::new(hit.p + offset, ls.wi, 0.0, ls.dist * (1.0 - 1e-4));

                    if !scene.occluded(&shadow_ray) {
                        let wi_local = hit.onb.to_local(ls.wi);
                        let cos_wi = wi_local.z.abs();

                        if cos_wi > 0.0 {
                            let f = hit.bsdf.eval(wi_local, wo_local, hit.tex_uv);

                            if light.is_delta() {
                                radiance += throughput * f * cos_wi * ls.li / sel_pmf;
                            } else {
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

            // Sample BSDF to extend path
            let u_sel: f32 = sampler.next_1d();
            let u_dir = sampler.next_2d();

            let Some(bs) = hit.bsdf.sample(wo_local, hit.tex_uv, u_sel, u_dir) else {
                break;
            };

            if bs.pdf == 0.0 || bs.f == Vec3A::ZERO {
                break;
            }

            let cos_wi = bs.wi.z.abs();
            throughput *= bs.f * cos_wi / bs.pdf;

            // Russian roulette
            if depth + 1 >= self.config.rr_depth {
                let survival = throughput.max_element().min(0.95);
                if sampler.next_1d() > survival {
                    break;
                }
                throughput /= survival;
            }

            // Spawn next ray
            let wi_world = hit.onb.to_world(bs.wi);
            let offset = hit.n * 1e-4 * wi_world.dot(hit.n).signum();
            ray = Ray::new(hit.p + offset, wi_world, 1e-4, f32::MAX);
        }

        radiance
    }

    /// Trace a base path at `(px, py)` with `sampler` in **record mode**,
    /// then replay 4 offset paths and return `(base_radiance, dx, dy)`.
    ///
    /// `dx` = average of (I(x+1,y) − I_base) and (I_base − I(x−1,y))
    /// `dy` = average of (I(x,y+1) − I_base) and (I_base − I(x,y−1))
    ///
    /// The forward/backward differences are averaged into a central-difference
    /// gradient at each pixel, which is the standard G-PT approach.
    fn trace_with_gradients(
        &self,
        scene: &Scene,
        camera: &dyn Camera,
        sampler: &mut ReplaySampler,
        px: usize,
        py: usize,
        width: usize,
        height: usize,
    ) -> (Vec3A, Vec3A, Vec3A) {
        // ------------------------------------------------------------------
        // 1. Record base path
        // ------------------------------------------------------------------
        sampler.reset_for_new_sample();

        // The first two random numbers on the tape are the camera jitter.
        let jitter = sampler.next_2d();
        let u_base = (px as f32 + jitter.x) / width as f32;
        let v_base = 1.0 - (py as f32 + jitter.y) / height as f32;
        let base_ray = camera.generate_ray(Vec2::new(u_base, v_base));

        let base_radiance = self.li(scene, base_ray, sampler);

        // At this point the tape contains [jitter.x, jitter.y, ...rest...].
        // For offset paths we patch tape[0..2] and replay.

        let original_jx = sampler.tape_mut()[0];
        let original_jy = sampler.tape_mut()[1];

        // ------------------------------------------------------------------
        // 2. Offset: x+1  (shift camera sample by +1 pixel in x)
        // ------------------------------------------------------------------
        let offset_right = if px + 1 < width {
            sampler.tape_mut()[0] = original_jx;
            sampler.tape_mut()[1] = original_jy;
            sampler.start_replay();
            let _jx = sampler.next_1d();
            let _jy = sampler.next_1d();
            let u_off = ((px + 1) as f32 + original_jx) / width as f32;
            let v_off = 1.0 - (py as f32 + original_jy) / height as f32;
            let off_ray = camera.generate_ray(Vec2::new(u_off, v_off));
            self.li(scene, off_ray, sampler)
        } else {
            base_radiance
        };

        // ------------------------------------------------------------------
        // 3. Offset: y+1  (shift camera sample by +1 pixel in y)
        // ------------------------------------------------------------------
        let offset_down = if py + 1 < height {
            sampler.tape_mut()[0] = original_jx;
            sampler.tape_mut()[1] = original_jy;
            sampler.start_replay();
            let _jx = sampler.next_1d();
            let _jy = sampler.next_1d();
            let u_off = (px as f32 + original_jx) / width as f32;
            let v_off = 1.0 - ((py + 1) as f32 + original_jy) / height as f32;
            let off_ray = camera.generate_ray(Vec2::new(u_off, v_off));
            self.li(scene, off_ray, sampler)
        } else {
            base_radiance
        };

        // ------------------------------------------------------------------
        // 4. Forward-difference gradients
        //
        //   dx(x,y) = I(x+1,y) − I(x,y)
        //   dy(x,y) = I(x,y+1) − I(x,y)
        //
        // Boundary pixels default to zero (offset == base).
        // ------------------------------------------------------------------

        let dx = offset_right - base_radiance;
        let dy = offset_down - base_radiance;

        (base_radiance, dx, dy)
    }
}

impl Integrator for GradientDomainPathTracer {
    fn render(
        &self,
        scene: &Scene,
        camera: &dyn Camera,
        width: usize,
        height: usize,
    ) -> Arc<Film> {
        let spp = self.config.spp;

        // Primal (base) image and gradient films.
        let primal_film = Arc::new(Film::new(width, height));
        let dx_film = Arc::new(Film::new(width, height));
        let dy_film = Arc::new(Film::new(width, height));

        println!(
            "[G-PT] Tracing {}×{} @ {} spp (max_depth={}, rr_depth={})…",
            width, height, spp, self.config.max_depth, self.config.rr_depth,
        );

        // -----------------------------------------------------------------
        // Pass 1: trace base + offset paths, accumulate primal & gradients
        // -----------------------------------------------------------------
        (0..height).into_par_iter().for_each(|y| {
            for x in 0..width {
                let inner =
                    IndependentSampler::seeded_for_pixel(x as u32, y as u32);
                let mut sampler = ReplaySampler::new(inner);

                let mut primal_accum = Vec3A::ZERO;
                let mut dx_accum = Vec3A::ZERO;
                let mut dy_accum = Vec3A::ZERO;

                for _s in 0..spp {
                    let (base, dx, dy) = self.trace_with_gradients(
                        scene, camera, &mut sampler, x, y, width, height,
                    );

                    primal_accum += base;
                    dx_accum += dx;
                    dy_accum += dy;
                }

                let inv_spp = 1.0 / spp as f32;
                primal_film.add_sample(x, y, primal_accum * inv_spp);
                dx_film.add_sample(x, y, dx_accum * inv_spp);
                dy_film.add_sample(x, y, dy_accum * inv_spp);
            }
        });

        println!("[G-PT] Path tracing complete. Starting Poisson reconstruction…");

        // -----------------------------------------------------------------
        // Pass 2: screened Poisson reconstruction via SOR
        // -----------------------------------------------------------------
        let reconstructed = screened_poisson_reconstruct(
            &primal_film,
            &dx_film,
            &dy_film,
            width,
            height,
            self.config.alpha,
            self.config.poisson_iterations,
            self.config.sor_omega,
        );

        println!("[G-PT] Reconstruction complete.");

        reconstructed
    }
}

// ===========================================================================
// Screened Poisson reconstruction
// ===========================================================================

/// Solve the screened Poisson equation:
///
///     argmin_I  α ‖I − I_primal‖² + ‖∇I − G‖²
///
/// where the gradient operator ∇ uses forward differences and `G = (dx, dy)`.
///
/// The Euler-Lagrange equation for each pixel `(x,y)` (for each colour
/// channel independently) is:
///
///     α · I(x,y)
///   − [I(x+1,y) − I(x,y) − dx(x,y)]           // ∂/∂I(x,y) of ‖∇_x I − dx‖²
///   − [I(x,y+1) − I(x,y) − dy(x,y)]           
///   + [I(x,y) − I(x−1,y) − dx(x−1,y)]         // backward contribution
///   + [I(x,y) − I(x,y−1) − dy(x,y−1)]
///   = α · I_primal(x,y)
///
/// Rearranging into the standard Ax = b form:
///
///   (α + N_neighbours) · I(x,y) − Σ_neighbours I(n)
///       = α · I_primal(x,y)
///         + dx(x,y) − dx(x−1,y)
///         + dy(x,y) − dy(x,y−1)
///
/// where `N_neighbours` is the number of valid neighbours (2 at corners,
/// 3 at edges, 4 in the interior).
///
/// This is a sparse linear system with a 5-point stencil, solved iteratively
/// with **Successive Over-Relaxation (SOR)**.
fn screened_poisson_reconstruct(
    primal: &Film,
    dx_film: &Film,
    dy_film: &Film,
    width: usize,
    height: usize,
    alpha: f32,
    iterations: u32,
    omega: f32,
) -> Arc<Film> {
    let n = width * height;

    // Work buffers — three channels stored as flat Vec<f32>.
    // Initialise to the primal image as a warm start.
    let mut img_r = vec![0.0f32; n];
    let mut img_g = vec![0.0f32; n];
    let mut img_b = vec![0.0f32; n];

    // Pre-load primal and gradient data into flat arrays for fast access.
    let mut primal_r = vec![0.0f32; n];
    let mut primal_g = vec![0.0f32; n];
    let mut primal_b = vec![0.0f32; n];
    let mut gdx_r = vec![0.0f32; n];
    let mut gdx_g = vec![0.0f32; n];
    let mut gdx_b = vec![0.0f32; n];
    let mut gdy_r = vec![0.0f32; n];
    let mut gdy_g = vec![0.0f32; n];
    let mut gdy_b = vec![0.0f32; n];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let p = primal.get_pixel(x, y);
            primal_r[idx] = p.x;
            primal_g[idx] = p.y;
            primal_b[idx] = p.z;

            // Warm-start with the primal
            img_r[idx] = p.x;
            img_g[idx] = p.y;
            img_b[idx] = p.z;

            let dx = dx_film.get_pixel(x, y);
            gdx_r[idx] = dx.x;
            gdx_g[idx] = dx.y;
            gdx_b[idx] = dx.z;

            let dy = dy_film.get_pixel(x, y);
            gdy_r[idx] = dy.x;
            gdy_g[idx] = dy.y;
            gdy_b[idx] = dy.z;
        }
    }

    // Pre-compute the RHS of the linear system:
    //   b(x,y) = α * primal(x,y) + dx(x,y) - dx(x-1,y) + dy(x,y) - dy(x,y-1)
    //
    // The divergence of the gradient field (dx(x,y) − dx(x−1,y)) represents
    // the net flow of gradients into pixel (x,y).
    let mut rhs_r = vec![0.0f32; n];
    let mut rhs_g = vec![0.0f32; n];
    let mut rhs_b = vec![0.0f32; n];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;

            // Divergence of gradient field: div G = dx(x,y) - dx(x-1,y) + dy(x,y) - dy(x,y-1)
            let mut div_r = gdx_r[idx];
            let mut div_g = gdx_g[idx];
            let mut div_b = gdx_b[idx];

            if x > 0 {
                let left = y * width + (x - 1);
                div_r -= gdx_r[left];
                div_g -= gdx_g[left];
                div_b -= gdx_b[left];
            }

            div_r += gdy_r[idx];
            div_g += gdy_g[idx];
            div_b += gdy_b[idx];

            if y > 0 {
                let up = (y - 1) * width + x;
                div_r -= gdy_r[up];
                div_g -= gdy_g[up];
                div_b -= gdy_b[up];
            }

            rhs_r[idx] = alpha * primal_r[idx] + div_r;
            rhs_g[idx] = alpha * primal_g[idx] + div_g;
            rhs_b[idx] = alpha * primal_b[idx] + div_b;
        }
    }

    // SOR iterations
    for iter in 0..iterations {
        // Red-black ordering for better convergence.  Even iterations:
        // process pixels where (x+y) is even; odd iterations: (x+y) is odd.
        // We do both in each "iteration" for simplicity.
        for parity in 0..2u32 {
            for y in 0..height {
                for x in 0..width {
                    if ((x + y) as u32 & 1) != parity {
                        continue;
                    }

                    let idx = y * width + x;

                    // Count neighbours and sum their current values.
                    let mut n_neigh = 0u32;
                    let mut sum_r = 0.0f32;
                    let mut sum_g = 0.0f32;
                    let mut sum_b = 0.0f32;

                    if x > 0 {
                        let i = y * width + (x - 1);
                        sum_r += img_r[i];
                        sum_g += img_g[i];
                        sum_b += img_b[i];
                        n_neigh += 1;
                    }
                    if x + 1 < width {
                        let i = y * width + (x + 1);
                        sum_r += img_r[i];
                        sum_g += img_g[i];
                        sum_b += img_b[i];
                        n_neigh += 1;
                    }
                    if y > 0 {
                        let i = (y - 1) * width + x;
                        sum_r += img_r[i];
                        sum_g += img_g[i];
                        sum_b += img_b[i];
                        n_neigh += 1;
                    }
                    if y + 1 < height {
                        let i = (y + 1) * width + x;
                        sum_r += img_r[i];
                        sum_g += img_g[i];
                        sum_b += img_b[i];
                        n_neigh += 1;
                    }

                    let diag = alpha + n_neigh as f32;
                    let inv_diag = 1.0 / diag;

                    // Gauss-Seidel update
                    let gs_r = (rhs_r[idx] + sum_r) * inv_diag;
                    let gs_g = (rhs_g[idx] + sum_g) * inv_diag;
                    let gs_b = (rhs_b[idx] + sum_b) * inv_diag;

                    // SOR relaxation
                    img_r[idx] = (1.0 - omega) * img_r[idx] + omega * gs_r;
                    img_g[idx] = (1.0 - omega) * img_g[idx] + omega * gs_g;
                    img_b[idx] = (1.0 - omega) * img_b[idx] + omega * gs_b;
                }
            }
        }

        if (iter + 1) % 50 == 0 || iter + 1 == iterations {
            println!("[G-PT] Poisson SOR iteration {}/{}", iter + 1, iterations);
        }
    }

    // Write result into a Film.
    let result = Arc::new(Film::new(width, height));
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let v = Vec3A::new(
                img_r[idx].max(0.0),
                img_g[idx].max(0.0),
                img_b[idx].max(0.0),
            );
            result.set_pixel(x, y, v);
        }
    }

    result
}

// ===========================================================================
// Utilities
// ===========================================================================

/// Power heuristic (β=2) for MIS.
#[inline]
fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    a2 / (a2 + b2)
}
