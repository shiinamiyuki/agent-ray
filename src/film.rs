//! # Film — HDR pixel accumulation buffer with atomic splatting
//!
//! The `Film` stores a 2D grid of linear-HDR RGB pixels.  Each channel is
//! held in an [`AtomicF32`] so that multiple threads can splat contributions
//! concurrently without mutexes.
//!
//! ## Usage
//!
//! ```ignore
//! let film = Film::new(1280, 720);
//! // From many threads:
//! film.add_sample(x, y, Vec3A::new(0.8, 0.6, 0.3));
//! // After rendering:
//! let ldr = film.to_rgb_image(ToneMapper::Reinhard, 2.2);
//! ldr.save("output.png").unwrap();
//! ```

use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

use glam::Vec3A;

// ---------------------------------------------------------------------------
// AtomicF32 — lock-free atomic floating-point accumulator
// ---------------------------------------------------------------------------

/// A single `f32` that can be atomically incremented from many threads.
///
/// Internally stores the bits as a `AtomicU32` and uses a compare-and-swap
/// loop to implement `fetch_add`.
#[repr(transparent)]
pub struct AtomicF32 {
    bits: AtomicU32,
}

impl AtomicF32 {
    /// Create a new `AtomicF32` initialised to `val`.
    #[inline]
    pub fn new(val: f32) -> Self {
        Self {
            bits: AtomicU32::new(val.to_bits()),
        }
    }

    /// Read the current value.
    #[inline]
    pub fn load(&self) -> f32 {
        f32::from_bits(self.bits.load(Ordering::Relaxed))
    }

    /// Store a value.
    #[inline]
    pub fn store(&self, val: f32) {
        self.bits.store(val.to_bits(), Ordering::Relaxed);
    }

    /// Atomically add `val` to the stored value (CAS loop).
    #[inline]
    pub fn fetch_add(&self, val: f32) {
        let mut current = self.bits.load(Ordering::Relaxed);
        loop {
            let new = (f32::from_bits(current) + val).to_bits();
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

impl Default for AtomicF32 {
    fn default() -> Self {
        Self::new(0.0)
    }
}

// ---------------------------------------------------------------------------
// Pixel
// ---------------------------------------------------------------------------

/// A single pixel with three atomic f32 channels (R, G, B).
pub struct AtomicPixel {
    pub r: AtomicF32,
    pub g: AtomicF32,
    pub b: AtomicF32,
}

impl AtomicPixel {
    #[inline]
    pub fn new() -> Self {
        Self {
            r: AtomicF32::new(0.0),
            g: AtomicF32::new(0.0),
            b: AtomicF32::new(0.0),
        }
    }

    /// Atomically add an RGB contribution.
    #[inline]
    pub fn add(&self, rgb: Vec3A) {
        self.r.fetch_add(rgb.x);
        self.g.fetch_add(rgb.y);
        self.b.fetch_add(rgb.z);
    }

    /// Read the current accumulated value.
    #[inline]
    pub fn load(&self) -> Vec3A {
        Vec3A::new(self.r.load(), self.g.load(), self.b.load())
    }

    /// Overwrite with a specific value (non-atomic write, use only when
    /// exclusive access is guaranteed).
    #[inline]
    pub fn store(&self, rgb: Vec3A) {
        self.r.store(rgb.x);
        self.g.store(rgb.y);
        self.b.store(rgb.z);
    }
}

impl Default for AtomicPixel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tone-mapping
// ---------------------------------------------------------------------------

/// Selects how linear HDR values are mapped to `[0, 1]` before gamma
/// encoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ToneMapper {
    /// Clamp to `[0, 1]` — no compression.
    Clamp,
    /// Luminance-preserving Reinhard: $L_{out} = L / (1 + L)$.
    Reinhard,
    /// Extended Reinhard with a configurable white point:
    /// $L_{out} = L (1 + L/L_w^2) / (1 + L)$.
    ReinhardExtended {
        /// The luminance value mapped to pure white.
        white_point: f32,
    },
}

/// Apply the selected tone-mapper to a single linear-HDR pixel.
#[inline]
fn tonemap_pixel(linear: Vec3A, tm: ToneMapper) -> Vec3A {
    match tm {
        ToneMapper::Clamp => linear.clamp(Vec3A::ZERO, Vec3A::ONE),
        ToneMapper::Reinhard => {
            let lum = 0.2126 * linear.x + 0.7152 * linear.y + 0.0722 * linear.z;
            let scale = if lum > 1e-6 {
                (lum / (1.0 + lum)) / lum
            } else {
                1.0 / (1.0 + lum)
            };
            (linear * scale).clamp(Vec3A::ZERO, Vec3A::ONE)
        }
        ToneMapper::ReinhardExtended { white_point } => {
            let lw2 = white_point * white_point;
            let lum = 0.2126 * linear.x + 0.7152 * linear.y + 0.0722 * linear.z;
            let numerator = lum * (1.0 + lum / lw2);
            let mapped_lum = numerator / (1.0 + lum);
            let scale = if lum > 1e-6 {
                mapped_lum / lum
            } else {
                1.0
            };
            (linear * scale).clamp(Vec3A::ZERO, Vec3A::ONE)
        }
    }
}

// ---------------------------------------------------------------------------
// Film
// ---------------------------------------------------------------------------

/// A 2D accumulation buffer for linear-HDR pixel values.
///
/// All mutation methods (`add_sample`, `splat`) are safe to call from
/// multiple threads concurrently thanks to [`AtomicPixel`].
pub struct Film {
    pub width: usize,
    pub height: usize,
    pixels: Vec<AtomicPixel>,
}

// SAFETY: AtomicPixel is built entirely on AtomicU32, which is Send + Sync.
unsafe impl Sync for Film {}

impl Film {
    /// Create a new black film of the given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        let mut pixels = Vec::with_capacity(width * height);
        for _ in 0..width * height {
            pixels.push(AtomicPixel::new());
        }
        Self { width, height, pixels }
    }

    /// Number of pixels.
    #[inline]
    pub fn len(&self) -> usize {
        self.width * self.height
    }

    /// Flat index for `(x, y)`.
    #[inline]
    pub fn pixel_index(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.width && y < self.height);
        y * self.width + x
    }

    /// Atomically add a weighted sample to pixel `(x, y)`.
    ///
    /// This is the main entry point for integrators that accumulate per-pixel
    /// contributions (path tracing row-parallel loops).
    #[inline]
    pub fn add_sample(&self, x: usize, y: usize, value: Vec3A) {
        let idx = self.pixel_index(x, y);
        self.pixels[idx].add(value);
    }

    /// Atomically add a contribution by flat pixel index.
    ///
    /// Useful for splatting (e.g. BDPT light-tracing strategy) when the
    /// coordinates have already been resolved to an index.
    #[inline]
    pub fn add_splat(&self, index: usize, value: Vec3A) {
        debug_assert!(index < self.pixels.len());
        self.pixels[index].add(value);
    }

    /// Read the accumulated value at `(x, y)`.
    #[inline]
    pub fn get_pixel(&self, x: usize, y: usize) -> Vec3A {
        let idx = self.pixel_index(x, y);
        self.pixels[idx].load()
    }

    /// Read the accumulated value by flat index.
    #[inline]
    pub fn get_pixel_by_index(&self, index: usize) -> Vec3A {
        self.pixels[index].load()
    }

    /// Overwrite a pixel (use only when no other thread is writing).
    #[inline]
    pub fn set_pixel(&self, x: usize, y: usize, value: Vec3A) {
        let idx = self.pixel_index(x, y);
        self.pixels[idx].store(value);
    }

    /// Reset all pixels to black.
    pub fn clear(&self) {
        for p in &self.pixels {
            p.store(Vec3A::ZERO);
        }
    }

    /// Return a flat `Vec<Vec3A>` snapshot of the HDR buffer (row-major).
    ///
    /// Useful for passing to code that expects the old `Vec<Vec3A>` format.
    pub fn to_hdr_vec(&self) -> Vec<Vec3A> {
        self.pixels.iter().map(|p| p.load()).collect()
    }

    /// Tone-map and gamma-encode the HDR buffer into an 8-bit sRGB image.
    ///
    /// `scale` is multiplied into every pixel before tone-mapping — use it
    /// to apply `1 / spp` normalisation.
    ///
    /// Returns an `image::RgbImage` that can be saved directly via
    /// `.save("output.png")`.
    pub fn to_rgb_image(
        &self,
        tone_mapper: ToneMapper,
        gamma: f32,
        scale: f32,
    ) -> image::RgbImage {
        let inv_gamma = 1.0 / gamma;
        let mut img = image::RgbImage::new(self.width as u32, self.height as u32);
        for y in 0..self.height {
            for x in 0..self.width {
                let linear = self.get_pixel(x, y) * scale;
                let mapped = tonemap_pixel(linear, tone_mapper);
                let encoded = mapped.powf(inv_gamma);
                let r = (encoded.x * 255.0 + 0.5).min(255.0) as u8;
                let g = (encoded.y * 255.0 + 0.5).min(255.0) as u8;
                let b = (encoded.z * 255.0 + 0.5).min(255.0) as u8;
                img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
            }
        }
        img
    }

    /// Save the HDR buffer as a ZIP-compressed OpenEXR file (linear, no
    /// tone-mapping).
    ///
    /// `scale` is multiplied into every pixel before writing — typically
    /// `1.0 / spp` for normalisation.
    pub fn save_exr(&self, path: impl AsRef<Path>, scale: f32) -> Result<(), Box<dyn std::error::Error>> {
        use exr::prelude::*;

        let w = self.width;
        let h = self.height;

        // Build flat channel vectors (row-major, top-to-bottom).
        let mut r_data = Vec::with_capacity(w * h);
        let mut g_data = Vec::with_capacity(w * h);
        let mut b_data = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let v = self.get_pixel(x, y) * scale;
                r_data.push(f16::from_f32(v.x));
                g_data.push(f16::from_f32(v.y));
                b_data.push(f16::from_f32(v.z));
            }
        }

        let layer = Layer::new(
            (w, h),
            LayerAttributes::named("beauty"),
            Encoding {
                compression: Compression::ZIP16,
                ..Encoding::default()
            },
            AnyChannels::sort(smallvec::smallvec![
                AnyChannel::new("R", FlatSamples::F16(r_data)),
                AnyChannel::new("G", FlatSamples::F16(g_data)),
                AnyChannel::new("B", FlatSamples::F16(b_data)),
            ]),
        );

        let image = Image::from_layer(layer);
        image.write().to_file(path)?;
        Ok(())
    }
}
