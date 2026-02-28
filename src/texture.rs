//! Texture types for spatially-varying material parameters.
//!
//! All textures return values in **linear** colour space.  `ImageTexture`
//! applies the standard sRGB → linear conversion when loading so that
//! every downstream computation (BSDF evaluation, lighting) is physically
//! correct.
//!
//! # Sampling convention
//!
//! `uv` coordinates use the OpenGL convention (origin at bottom-left) and
//! wrap with **repeat** semantics.  Fractional parts are used for bilinear
//! filtering across the texel grid.

use std::path::Path;
use std::sync::Arc;

use glam::{Vec2, Vec3A};
use anyhow::Result;

// ---------------------------------------------------------------------------
// Texture trait
// ---------------------------------------------------------------------------

/// A 2-D spatially-varying value that maps texture coordinates to a colour.
pub trait Texture: Send + Sync {
    fn sample(&self, uv: Vec2) -> Vec3A;
}

// ---------------------------------------------------------------------------
// ConstantTexture
// ---------------------------------------------------------------------------

/// A solid-colour texture; ignores UV coordinates.
pub struct ConstantTexture(pub Vec3A);

impl ConstantTexture {
    pub fn new(color: Vec3A) -> Arc<Self> {
        Arc::new(Self(color))
    }
}

impl Texture for ConstantTexture {
    #[inline]
    fn sample(&self, _uv: Vec2) -> Vec3A {
        self.0
    }
}

// ---------------------------------------------------------------------------
// ImageTexture
// ---------------------------------------------------------------------------

/// A raster texture loaded from an image file (PNG, JPEG, …).
///
/// # Loading
///
/// Pixels are decoded as 8-bit-per-channel RGB.  Each channel is converted
/// from sRGB to linear using the piecewise IEC 61966-2-1 transfer function
/// before being stored, so sampling always yields linear-light values.
///
/// # Sampling
///
/// UV coordinates are repeated (tiled) and bilinear interpolation is applied
/// across the four nearest texels.
pub struct ImageTexture {
    /// Linear-light texel data, row-major (top row first).
    data: Vec<Vec3A>,
    width: u32,
    height: u32,
}

impl ImageTexture {
    /// Load an image from `path` and decode it into a linear-light texture.
    ///
    /// Returns an `Arc<ImageTexture>` so it can be shared across many
    /// material instances without duplication.
    pub fn load(path: &Path) -> Result<Arc<Self>> {
        let img = image::open(path)?.into_rgb8();
        let width = img.width();
        let height = img.height();

        let data: Vec<Vec3A> = img
            .pixels()
            .map(|p| {
                Vec3A::new(
                    srgb_to_linear(p[0] as f32 / 255.0),
                    srgb_to_linear(p[1] as f32 / 255.0),
                    srgb_to_linear(p[2] as f32 / 255.0),
                )
            })
            .collect();

        Ok(Arc::new(Self { data, width, height }))
    }
}

impl Texture for ImageTexture {
    /// Bilinear sample with repeat wrap.
    fn sample(&self, uv: Vec2) -> Vec3A {
        let w = self.width as f32;
        let h = self.height as f32;

        // Repeat wrap: take fractional part, ensure positive.
        let u = uv.x.fract().rem_euclid(1.0) * w - 0.5;
        let v = uv.y.fract().rem_euclid(1.0) * h - 0.5;

        let x0 = u.floor() as i32;
        let y0 = v.floor() as i32;
        let tx = u - u.floor();
        let ty = v - v.floor();

        let c00 = self.fetch(x0,     y0);
        let c10 = self.fetch(x0 + 1, y0);
        let c01 = self.fetch(x0,     y0 + 1);
        let c11 = self.fetch(x0 + 1, y0 + 1);

        // Bilinear blend.
        c00 * ((1.0 - tx) * (1.0 - ty))
            + c10 * (tx * (1.0 - ty))
            + c01 * ((1.0 - tx) * ty)
            + c11 * (tx * ty)
    }
}

impl ImageTexture {
    /// Fetch a single texel, wrapping coordinates with repeat.
    #[inline]
    fn fetch(&self, x: i32, y: i32) -> Vec3A {
        let w = self.width as i32;
        let h = self.height as i32;
        let x = x.rem_euclid(w) as u32;
        let y = y.rem_euclid(h) as u32;
        self.data[(y * self.width + x) as usize]
    }
}

// ---------------------------------------------------------------------------
// sRGB helpers
// ---------------------------------------------------------------------------

/// IEC 61966-2-1 sRGB → linear transfer function.
///
/// Converts a normalised `[0, 1]` sRGB value to linear light.
#[inline]
fn srgb_to_linear(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}
