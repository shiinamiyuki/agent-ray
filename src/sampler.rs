//! # Sampler abstraction
//!
//! This module provides a `Sampler` trait that encapsulates random number
//! generation for Monte Carlo rendering.  Integrators consume samples via
//! `next_1d()` and `next_2d()` without knowing the underlying sampling
//! strategy, making it straightforward to swap in blue-noise, Sobol, or
//! stratified samplers later.
//!
//! ## Provided implementations
//!
//! - [`IndependentSampler`]: wraps a `SmallRng` for plain i.i.d. uniform
//!   samples.  This is the default and reproduces the existing behaviour.

use glam::Vec2;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A source of `[0, 1)` uniform samples for Monte Carlo integration.
///
/// Implementations may be purely random ([`IndependentSampler`]) or
/// quasi-random (blue-noise, Sobol, etc.).  The trait is object-safe so
/// integrators can accept `&mut dyn Sampler`.
pub trait Sampler: Send {
    /// Return a single uniform `f32` in `[0, 1)`.
    fn next_1d(&mut self) -> f32;

    /// Return a uniform 2D sample in `[0, 1)²`.
    ///
    /// The default implementation calls `next_1d` twice; quasi-random
    /// samplers should override this to return properly stratified pairs.
    fn next_2d(&mut self) -> Vec2 {
        Vec2::new(self.next_1d(), self.next_1d())
    }

    /// Prepare the sampler for a new pixel sample.
    ///
    /// Called once per sample-per-pixel.  Quasi-random samplers use this to
    /// reset their dimension counter; independent samplers can ignore it.
    fn start_next_sample(&mut self) {}

    /// Create a new sampler instance seeded for the given pixel.
    ///
    /// Each pixel should receive an independent (or at least decorrelated)
    /// sample sequence.  The `sample_index` parameter can be used by
    /// deterministic sequences to offset into their global table.
    fn clone_for_pixel(&self, pixel_x: u32, pixel_y: u32, sample_index: u32) -> Box<dyn Sampler>;
}

// ---------------------------------------------------------------------------
// IndependentSampler — i.i.d. uniform samples via SmallRng
// ---------------------------------------------------------------------------

/// Plain independent (white-noise) sampler backed by a fast PRNG.
///
/// This is the simplest possible sampler: every call to `next_1d` returns
/// an i.i.d. uniform variate.  It reproduces the behaviour of the original
/// integrators that used `SmallRng` directly.
pub struct IndependentSampler {
    rng: SmallRng,
}

impl IndependentSampler {
    /// Create a sampler from an explicit seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Deterministic seed derived from pixel coordinates.
    ///
    /// Uses the same hash the integrators were already using, so renders
    /// are bitwise identical when switching from raw `SmallRng` to
    /// `IndependentSampler`.
    pub fn seeded_for_pixel(pixel_x: u32, pixel_y: u32) -> Self {
        let seed = (pixel_y as u64).wrapping_mul(2654435761)
            ^ (pixel_x as u64).wrapping_mul(805459861);
        Self::new(seed)
    }
}

impl Sampler for IndependentSampler {
    #[inline]
    fn next_1d(&mut self) -> f32 {
        self.rng.random()
    }

    fn clone_for_pixel(&self, pixel_x: u32, pixel_y: u32, _sample_index: u32) -> Box<dyn Sampler> {
        Box::new(IndependentSampler::seeded_for_pixel(pixel_x, pixel_y))
    }
}
