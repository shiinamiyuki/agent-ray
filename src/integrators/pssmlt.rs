//! # Primary Sample Space Metropolis Light Transport (PSSMLT)
//!
//! Implements the PSSMLT algorithm as described in:
//!
//! - Kelemen et al., *"A Simple and Robust Mutation Strategy for the
//!   Metropolis Light Transport Algorithm"*, Computer Graphics Forum
//!   (Eurographics) 21(3), 2002.
//!
//! ## Overview
//!
//! PSSMLT performs Markov Chain Monte Carlo (MCMC) in the *primary sample
//! space* — the unit hypercube `[0,1)^N` that parameterises one complete
//! light path.  The algorithm maintains multiple independent chains, each
//! consisting of:
//!
//! 1. A current random-number vector **u** ∈ `[0,1)^N` (N is lazily grown).
//! 2. The corresponding image contribution (pixel + radiance).
//!
//! At every iteration a **mutation** is proposed:
//!
//! - **Large step (independent):** a fresh i.i.d. uniform vector.
//! - **Small step (dependent):** `u_new = fract(u_old + σ · Gaussian)`.
//!
//! The proposal is accepted or rejected with probability
//! `min(1, f(u_new) / f(u_old))` where `f` is the scalar luminance of the
//! path contribution.  Accepted paths replace the current state; both
//! accepted and rejected paths deposit their contribution to the film
//! weighted by the Metropolis acceptance logic.
//!
//! ## Normalisation
//!
//! The raw MCMC image has unknown total brightness.  To recover the correct
//! brightness we accumulate the luminance of every *large-step* sample into
//! a global counter `b`.  The final image is scaled by
//!
//!     `b / (n_large_steps · n_pixels)`
//!
//! where `n_large_steps` is the total number of large-step proposals across
//! all chains.
//!
//! ## Bootstrap
//!
//! Before rendering we trace `n_bootstrap` paths (each with its own fresh
//! random vector) and compute their luminance.  We then *resample*
//! `n_chains` starting states from those bootstrap paths, weighting by
//! luminance (importance resampling).  This seeds each chain with a bright
//! path, reducing the initial burn-in waste.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use glam::{Vec2, Vec3A};
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg32;
use rayon::prelude::*;

use crate::cameras::Camera;
use crate::film::Film;
use crate::geometry::Ray;
use crate::integrators::Integrator;
use crate::sampler::Sampler;
use crate::scene::Scene;

// ===========================================================================
// Configuration
// ===========================================================================

/// Configuration for the PSSMLT integrator.
pub struct PssmltConfig {
    /// Desired number of samples per pixel.
    pub spp: u32,
    /// Maximum path depth (number of bounces including the camera ray).
    pub max_depth: u32,
    /// Minimum path depth before Russian roulette starts.
    pub rr_depth: u32,
    /// Number of bootstrap paths used to seed the chains.
    pub n_bootstrap: u32,
    /// Number of independent Markov chains.
    pub n_chains: u32,
    /// Probability of choosing a large (independent) mutation.
    pub large_step_prob: f32,
    /// Standard deviation σ for the small (dependent) Gaussian mutation.
    pub sigma: f32,
    /// Maximum image-space mutation size for small steps, as a fraction of
    /// the resolution (0–1).  The first two random dimensions (pixel x/y)
    /// are perturbed with `σ_image = image_mutation_size` instead of the
    /// global `sigma`.  Set to 1.0 to disable (same σ everywhere).
    pub image_mutation_size: f32,
}

impl Default for PssmltConfig {
    fn default() -> Self {
        Self {
            spp: 64,
            max_depth: 8,
            rr_depth: 3,
            n_bootstrap: 100_000,
            n_chains: 128,
            large_step_prob: 0.3,
            sigma: 0.01,
            image_mutation_size: 0.02,
        }
    }
}

// ===========================================================================
// MCMCSampler — primary-sample-space random vector with mutations
// ===========================================================================

/// A sampler that maintains a lazy N-dimensional vector of random numbers
/// in `[0,1)`.  Each `next_1d()` call either reads the current element
/// (existing dimension) or extends the vector (new dimension).
///
/// Two mutation strategies are supported:
///
/// - **Large step (independent):** the entire vector is replaced with fresh
///   i.i.d. uniform variates.
/// - **Small step (dependent):** each component is perturbed via
///   `fract(old + σ · N(0,1))`.
pub struct MCMCSampler {
    /// The current accepted state.
    state: Vec<f32>,
    /// The proposed state (written lazily during path evaluation).
    proposal: Vec<f32>,
    /// Read cursor into `proposal` — incremented by each `next_1d()` call.
    cursor: usize,
    /// Whether the current proposal is a large (independent) step.
    is_large_step: bool,
    /// Standard deviation for small-step Gaussian perturbation.
    sigma: f32,
    /// Standard deviation for image-coordinate dimensions (dims 0 and 1).
    sigma_image: f32,
    /// Internal RNG for generating mutations and accept/reject decisions.
    rng: Pcg32,
}

impl MCMCSampler {
    /// Create a new sampler with the given perturbation σ, image σ, and RNG seed.
    pub fn new(sigma: f32, sigma_image: f32, seed: u64) -> Self {
        Self {
            state: Vec::new(),
            proposal: Vec::new(),
            cursor: 0,
            is_large_step: true,
            sigma,
            sigma_image,
            rng: Pcg32::seed_from_u64(seed),
        }
    }

    /// Begin a new proposal.  `large` controls the mutation strategy.
    pub fn start_proposal(&mut self, large: bool) {
        self.is_large_step = large;
        self.cursor = 0;
        self.proposal.clear();
    }

    /// Accept the current proposal: copy it into the accepted state.
    pub fn accept(&mut self) {
        std::mem::swap(&mut self.state, &mut self.proposal);
        // Trim state to the actually-used dimensions (cursor).
        self.state.truncate(self.cursor);
    }

    /// Reject the current proposal: the state stays unchanged.
    pub fn reject(&mut self) {
        // Nothing to do — `state` is untouched.
    }

    /// Sample a standard-normal variate (Box-Muller transform).
    fn gaussian(&mut self) -> f32 {
        let u1: f32 = self.rng.random();
        let u2: f32 = self.rng.random();
        // Ensure u1 > 0 to avoid log(0).
        let u1 = u1.max(1e-10);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Seed the accepted state directly (used during bootstrap).
    pub fn seed_state(&mut self, state: Vec<f32>) {
        self.state = state;
    }
}

impl Sampler for MCMCSampler {
    fn next_1d(&mut self) -> f32 {
        let dim = self.cursor;
        self.cursor += 1;

        let value = if self.is_large_step {
            // Independent mutation: fresh uniform sample.
            self.rng.random::<f32>()
        } else if dim < self.state.len() {
            // Dependent mutation: perturb existing component.
            let old = self.state[dim];
            // Dims 0–1 are image coordinates — use the image-specific σ.
            let s = if dim < 2 { self.sigma_image } else { self.sigma };
            let noise = s * self.gaussian();
            // fract wraps into [0, 1).
            let v = old + noise;
            v - v.floor()
        } else {
            // New dimension that didn't exist in the old state — sample fresh.
            self.rng.random::<f32>()
        };

        // Append to proposal.
        self.proposal.push(value);

        value
    }

    fn clone_for_pixel(&self, _px: u32, _py: u32, _sample_index: u32) -> Box<dyn Sampler> {
        // MCMC sampler is not cloneable in the usual sense; panic
        panic!("MCMCSampler is not cloneable");
    }
}

// ===========================================================================
// PSSMLT integrator
// ===========================================================================

pub struct Pssmlt {
    pub config: PssmltConfig,
}

impl Pssmlt {
    pub fn new(config: PssmltConfig) -> Self {
        Self { config }
    }

    // -----------------------------------------------------------------------
    // Path tracer core (same as PathTracer::li but accepts &mut MCMCSampler)
    // -----------------------------------------------------------------------

    /// Evaluate the full image contribution for the path defined by the
    /// current state of the MCMC sampler.
    ///
    /// Returns `(pixel_x, pixel_y, radiance)` where `pixel_x/y` are integer
    /// pixel coordinates.  The radiance is the *un-normalised* path
    /// contribution.
    fn evaluate_path(
        &self,
        scene: &Scene,
        camera: &dyn Camera,
        width: usize,
        height: usize,
        sampler: &mut MCMCSampler,
    ) -> (usize, usize, Vec3A) {
        // --- Generate camera ray from the random vector ---
        let jx = sampler.next_1d();
        let jy = sampler.next_1d();
        let px = (jx * width as f32).min(width as f32 - 1.0);
        let py = (jy * height as f32).min(height as f32 - 1.0);
        let ix = px as usize;
        let iy = py as usize;
        let u = px / width as f32;
        let v = 1.0 - py / height as f32;
        let ray = camera.generate_ray(Vec2::new(u, v));

        let radiance = self.li(scene, ray, sampler);
        (ix, iy, radiance)
    }

    /// Standard iterative path tracer (same algorithm as `PathTracer::li`).
    fn li(&self, scene: &Scene, initial_ray: Ray, sampler: &mut MCMCSampler) -> Vec3A {
        let mut radiance = Vec3A::ZERO;
        let mut throughput = Vec3A::ONE;
        let mut ray = initial_ray;

        for depth in 0..self.config.max_depth {
            let Some((_t, hit)) = scene.intersect(&ray) else {
                break;
            };

            let wo = -ray.direction;
            let wo_local = hit.onb.to_local(wo);

            // --- Direct lighting: sample one light ---
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

            // --- Sample BSDF to extend the path ---
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

            // --- Russian roulette ---
            if depth + 1 >= self.config.rr_depth {
                let survival = throughput.max_element().min(0.95);
                if sampler.next_1d() > survival {
                    break;
                }
                throughput /= survival;
            }

            // --- Spawn next ray ---
            let wi_world = hit.onb.to_world(bs.wi);
            let offset = hit.n * 1e-4 * wi_world.dot(hit.n).signum();
            ray = Ray::new(hit.p + offset, wi_world, 1e-4, f32::MAX);
        }

        radiance
    }
}

// ---------------------------------------------------------------------------
// Integrator trait implementation
// ---------------------------------------------------------------------------

/// Atomic f64 accumulator for the normalisation constant.
struct AtomicF64 {
    bits: AtomicU64,
}

impl AtomicF64 {
    fn new(val: f64) -> Self {
        Self {
            bits: AtomicU64::new(val.to_bits()),
        }
    }

    fn load(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Relaxed))
    }

    fn fetch_add(&self, val: f64) {
        let mut current = self.bits.load(Ordering::Relaxed);
        loop {
            let new = (f64::from_bits(current) + val).to_bits();
            match self.bits.compare_exchange_weak(
                current,
                new,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}

/// Per-chain statistics accumulated locally (no atomics in the hot loop)
/// and merged into global counters after the chain finishes.
#[derive(Default)]
struct ChainStats {
    large_proposed: u64,
    large_accepted: u64,
    small_proposed: u64,
    small_accepted: u64,
}

/// Global statistics aggregated across all chains (atomic).
struct GlobalStats {
    large_proposed: AtomicU64,
    large_accepted: AtomicU64,
    small_proposed: AtomicU64,
    small_accepted: AtomicU64,
}

impl GlobalStats {
    fn new() -> Self {
        Self {
            large_proposed: AtomicU64::new(0),
            large_accepted: AtomicU64::new(0),
            small_proposed: AtomicU64::new(0),
            small_accepted: AtomicU64::new(0),
        }
    }

    /// Merge a finished chain's local counters into the global totals.
    fn merge(&self, local: &ChainStats) {
        self.large_proposed.fetch_add(local.large_proposed, Ordering::Relaxed);
        self.large_accepted.fetch_add(local.large_accepted, Ordering::Relaxed);
        self.small_proposed.fetch_add(local.small_proposed, Ordering::Relaxed);
        self.small_accepted.fetch_add(local.small_accepted, Ordering::Relaxed);
    }

    fn print_summary(&self) {
        let lp = self.large_proposed.load(Ordering::Relaxed);
        let la = self.large_accepted.load(Ordering::Relaxed);
        let sp = self.small_proposed.load(Ordering::Relaxed);
        let sa = self.small_accepted.load(Ordering::Relaxed);
        let large_ratio = if lp > 0 { la as f64 / lp as f64 } else { 0.0 };
        let small_ratio = if sp > 0 { sa as f64 / sp as f64 } else { 0.0 };
        let total_p = lp + sp;
        let total_a = la + sa;
        let overall_ratio = if total_p > 0 { total_a as f64 / total_p as f64 } else { 0.0 };
        println!("[PSSMLT] Stats:");
        println!("  Large steps: {}/{} accepted ({:.2}%)", la, lp, large_ratio * 100.0);
        println!("  Small steps: {}/{} accepted ({:.2}%)", sa, sp, small_ratio * 100.0);
        println!("  Overall:     {}/{} accepted ({:.2}%)", total_a, total_p, overall_ratio * 100.0);
    }
}

impl Integrator for Pssmlt {
    fn render(
        &self,
        scene: &Scene,
        camera: &dyn Camera,
        width: usize,
        height: usize,
    ) -> Arc<Film> {
        let n_pixels = width * height;
        let mutations_per_chain =
            (self.config.spp as u64 * n_pixels as u64) / self.config.n_chains as u64;

        // ===================================================================
        // Phase 1 — Bootstrap
        // ===================================================================
        println!(
            "[PSSMLT] Bootstrap: tracing {} paths...",
            self.config.n_bootstrap
        );

        // Trace bootstrap paths in parallel, collecting (luminance, state).
        let bootstrap_results: Vec<(f32, Vec<f32>)> = (0..self.config.n_bootstrap)
            .into_par_iter()
            .map(|i| {
                let mut sampler = MCMCSampler::new(self.config.sigma, self.config.image_mutation_size, i as u64 * 37 + 1);
                sampler.start_proposal(true); // large step = fresh
                let (_ix, _iy, rad) = self.evaluate_path(scene, camera, width, height, &mut sampler);
                let lum = luminance(rad);
                // The proposal IS the state for bootstrap paths.
                let state = sampler.proposal.clone();
                (lum, state)
            })
            .collect();

        // Build CDF over luminances for importance resampling.
        let total_lum: f64 = bootstrap_results.iter().map(|(l, _)| *l as f64).sum();
        let _avg_bootstrap_lum = if self.config.n_bootstrap > 0 {
            total_lum / self.config.n_bootstrap as f64
        } else {
            0.0
        };

        if total_lum <= 0.0 {
            eprintln!("[PSSMLT] Warning: all bootstrap paths have zero luminance. Scene might be fully occluded.");
            return Arc::new(Film::new(width, height));
        }

        // Prefix-sum CDF (unnormalised).
        let mut cdf = Vec::with_capacity(self.config.n_bootstrap as usize);
        let mut running = 0.0_f64;
        for (l, _) in &bootstrap_results {
            running += *l as f64;
            cdf.push(running);
        }

        // Resample n_chains starting states.
        let mut seed_rng = Pcg32::seed_from_u64(0xDEAD_BEEF);
        let chain_seeds: Vec<(Vec<f32>, u64)> = (0..self.config.n_chains)
            .map(|chain_idx| {
                let u: f64 = seed_rng.random::<f64>() * total_lum;
                let idx = cdf.partition_point(|&c| c < u).min(cdf.len() - 1);
                let state = bootstrap_results[idx].1.clone();
                // Give each chain a unique RNG seed.
                let chain_seed = 0xCAFE_0000u64 + chain_idx as u64;
                (state, chain_seed)
            })
            .collect();

        // ===================================================================
        // Phase 2 — Run chains
        // ===================================================================
        println!(
            "[PSSMLT] Running {} chains × {} mutations ({} total, {} spp)...",
            self.config.n_chains,
            mutations_per_chain,
            self.config.n_chains as u64 * mutations_per_chain,
            self.config.spp,
        );

        let film = Arc::new(Film::new(width, height));
        // Global accumulator: sum of luminances of all large-step samples.
        let b_accum = AtomicF64::new(0.0);
        // Global counter of large-step proposals.
        let n_large = AtomicU64::new(0);
        // Per-chain stats merged after each chain completes.
        let stats = GlobalStats::new();

        chain_seeds.into_par_iter().for_each(|(init_state, chain_seed)| {
            let mut sampler = MCMCSampler::new(self.config.sigma, self.config.image_mutation_size, chain_seed);
            sampler.seed_state(init_state);
            let mut local_stats = ChainStats::default();

            // Evaluate the initial state to get the current contribution.
            sampler.start_proposal(false);
            // Replay the seeded state: we do a "null mutation" (small step
            // with the exact same state) so that `proposal` is populated.
            // Actually, we need to evaluate the path with the initial state.
            // The simplest way: do a large-step proposal that ignores the
            // state, but we already have the state, so we use a trick:
            // set state, start a small-step proposal, evaluate — the
            // proposal will read from state for existing dimensions.
            let (mut cur_px, mut cur_py, mut cur_rad) =
                self.evaluate_path(scene, camera, width, height, &mut sampler);
            // Accept this initial "proposal" to make state consistent.
            sampler.accept();
            let mut cur_lum = luminance(cur_rad);

            for _mutation in 0..mutations_per_chain {
                // Decide mutation strategy.
                let is_large: bool = sampler.rng.random::<f32>() < self.config.large_step_prob;

                sampler.start_proposal(is_large);
                let (prop_px, prop_py, prop_rad) =
                    self.evaluate_path(scene, camera, width, height, &mut sampler);
                let prop_lum = luminance(prop_rad);

                // Accumulate large-step luminance for normalisation.
                if is_large {
                    b_accum.fetch_add(prop_lum as f64);
                    n_large.fetch_add(1, Ordering::Relaxed);
                    local_stats.large_proposed += 1;
                } else {
                    local_stats.small_proposed += 1;
                }

                // Metropolis accept/reject.
                let accept_prob = if cur_lum > 0.0 {
                    (prop_lum / cur_lum).min(1.0)
                } else {
                    1.0
                };

                // Deposit contributions weighted by the Metropolis estimator:
                //   proposed:  a(y→x) / f(y)   ← proposal gets accept_prob / prop_lum
                //   current:   (1 − a(y→x)) / f(x)   ← current gets (1−accept_prob) / cur_lum
                //
                // These are splatted to the film; the per-pixel scale is later
                // applied via the normalisation constant.
                if prop_lum > 0.0 {
                    let w_prop = accept_prob / prop_lum;
                    film.add_sample(prop_px, prop_py, prop_rad * w_prop);
                }
                if cur_lum > 0.0 {
                    let w_cur = (1.0 - accept_prob) / cur_lum;
                    film.add_sample(cur_px, cur_py, cur_rad * w_cur);
                }

                // Accept or reject.
                if sampler.rng.random::<f32>() < accept_prob {
                    sampler.accept();
                    cur_px = prop_px;
                    cur_py = prop_py;
                    cur_rad = prop_rad;
                    cur_lum = prop_lum;
                    if is_large {
                        local_stats.large_accepted += 1;
                    } else {
                        local_stats.small_accepted += 1;
                    }
                } else {
                    sampler.reject();
                }
            }

            // Merge per-chain stats into global counters (one atomic add per chain).
            stats.merge(&local_stats);
        });

        // ===================================================================
        // Phase 3 — Normalise
        // ===================================================================
        let b = b_accum.load();
        let n_large_total = n_large.load(Ordering::Relaxed);

        // The normalisation constant is:
        //   b_avg = b / n_large_total          (average luminance of large-step samples)
        //   scale = b_avg / n_pixels
        //         = b / (n_large_total * n_pixels)
        //
        // But the film already has the sum over all mutations_per_chain * n_chains
        // mutations of the weighted contributions. Each mutation deposits a
        // total weight of 1/f (split between current and proposed). The total
        // number of mutations is n_chains * mutations_per_chain. So the raw
        // film value at pixel (x,y) is:
        //
        //   raw(x,y) = Σ [ w · L(x,y) ]
        //
        // and the correct pixel value is:
        //
        //   I(x,y) = b_avg * raw(x,y) / (n_chains * mutations_per_chain / n_pixels)
        //          = b_avg * raw(x,y) * n_pixels / (n_chains * mutations_per_chain)
        //
        // Simplifying:
        //   I(x,y) = raw(x,y) * b * n_pixels / (n_large_total * n_chains * mutations_per_chain)
        //
        // But n_chains * mutations_per_chain = spp * n_pixels, so:
        //   I(x,y) = raw(x,y) * b / (n_large_total * spp)

        let total_mutations = self.config.n_chains as u64 * mutations_per_chain;
        let scale = if n_large_total > 0 && total_mutations > 0 {
            (b / n_large_total as f64) / (total_mutations as f64 / n_pixels as f64)
        } else {
            1.0
        };

        println!(
            "[PSSMLT] Normalisation: b={:.4}, n_large={}, scale={:.6e}",
            b, n_large_total, scale
        );
        stats.print_summary();

        // Apply scale to the film by reading and writing each pixel.
        let scale_f32 = scale as f32;
        let scaled_film = Arc::new(Film::new(width, height));
        for y in 0..height {
            for x in 0..width {
                let v = film.get_pixel(x, y) * scale_f32;
                scaled_film.set_pixel(x, y, v);
            }
        }

        scaled_film
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// CIE luminance of an RGB triple.
#[inline]
fn luminance(rgb: Vec3A) -> f32 {
    (0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z).max(0.0)
}

/// Power heuristic (β=2) for MIS.
#[inline]
fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    a2 / (a2 + b2)
}
