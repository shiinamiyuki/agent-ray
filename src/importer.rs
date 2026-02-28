//! OBJ/MTL scene importer.
//!
//! # Material conversion strategy
//!
//! Blinn-Phong MTL parameters are mapped onto our PBR BSDFs using the
//! following priority, applied in order:
//!
//! 1.  **Transparent** (`d < 0.5`):
//!     → `DielectricBsdf { eta = Ni, roughness = ns_to_roughness(Ns) }`.
//!     A dissolve factor below 0.5 indicates the object is more transmissive
//!     than opaque; a dielectric BSDF is the most physically appropriate model.
//!
//! 2.  **Conductor / glossy** (`luminance(Ks) > 0.04`):
//!     → `ConductorBsdf { f0 = Ks, roughness = ns_to_roughness(Ns) }`.
//!     A non-trivial specular color signals a metallic or strongly glossy
//!     response.  We treat `Ks` as the Schlick F0 normal-incidence reflectance.
//!
//! 3.  **Diffuse fallback**:
//!     → `Lambertian { albedo = Kd }`.
//!     Used when the material is opaque with negligible specular response.
//!
//! ## Roughness conversion
//!
//! MTL shininess (`Ns ∈ [0, 1000]`) is converted to a GGX α via the
//! Phong-to-NDF relation:
//!
//! ```text
//! α = sqrt(2 / (Ns + 2))
//! ```
//!
//! This maps `Ns = 0` → `α = 1` (very rough / diffuse) and
//! `Ns = 1000` → `α ≈ 0.044` (near-mirror).  The value is clamped to
//! `[1e-3, 1.0]` before being passed to the microfacet constructors which
//! also apply their own clamp.
//!
//! ## Missing MTL defaults
//!
//! | Parameter | Default             | Meaning                   |
//! |-----------|---------------------|---------------------------|
//! | `Kd`      | `(0.5, 0.5, 0.5)`  | mid-grey diffuse albedo   |
//! | `Ks`      | `(0.0, 0.0, 0.0)`  | no specular contribution  |
//! | `Ns`      | `0.0`              | fully diffuse shininess   |
//! | `d`       | `1.0`              | fully opaque              |
//! | `Ni`      | `1.5`              | borosilicate-glass IOR    |
//!
//! ## Texture maps
//!
//! `map_Kd` (diffuse colour texture) is loaded as an [`ImageTexture`] and
//! used as the albedo of the `Lambertian` BSDF.  Paths are resolved relative
//! to the OBJ file's directory; Windows-style backslash separators are
//! normalised automatically.  Texels are decoded from sRGB to linear on load.
//! If the file cannot be opened a warning is printed and the constant `Kd`
//! colour is used as the fallback.

use std::path::Path;
use std::sync::Arc;

use glam::{Mat4, Vec3A};
use anyhow::Result;

use crate::prelude::*;
use crate::primitives::mesh::TriangleMesh;
use crate::scene::SceneObject;
use crate::surfaces::{Bsdf, Lambertian, ConductorBsdf, DielectricBsdf};
use crate::texture::{ConstantTexture, ImageTexture};

// ---------------------------------------------------------------------------
// Roughness helper
// ---------------------------------------------------------------------------

/// Convert a Phong shininess exponent to a GGX α roughness value.
///
/// Uses the Phong-NDF equivalence: `α = sqrt(2 / (Ns + 2))`.
/// Clamped to `[1e-3, 1.0]` so that microfacet models remain well-defined.
fn ns_to_roughness(ns: f32) -> f32 {
    (2.0 / (ns + 2.0)).sqrt().clamp(1e-3, 1.0)
}

/// BT.709 luminance of an RGB colour.
fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

// ---------------------------------------------------------------------------
// MTL → BSDF conversion
// ---------------------------------------------------------------------------

/// Convert a single `tobj::Material` to one of our PBR BSDFs.
///
/// `base_dir` is the directory containing the OBJ file; it is used to
/// resolve relative texture paths stored in the MTL.
///
/// See the module-level doc for the full priority rules.
fn convert_material(mat: &tobj::Material, base_dir: &Path) -> Arc<dyn Bsdf> {
    // Resolve MTL parameters with safe defaults.
    let ks = mat.specular.unwrap_or([0.0, 0.0, 0.0]);
    let ns = mat.shininess.unwrap_or(0.0).max(0.0);
    let d = mat.dissolve.unwrap_or(1.0); // 1 = fully opaque
    let ni = mat.optical_density.unwrap_or(1.5).max(1.0);
    let roughness = ns_to_roughness(ns);

    if d < 0.5 {
        // Priority 1 – transmissive: use dielectric BSDF.
        return Arc::new(DielectricBsdf::new(ni, roughness));
    }

    if luminance(ks) > 0.04 {
        // Priority 2 – non-trivial specular color: treat as conductor.
        let f0 = Vec3A::new(ks[0], ks[1], ks[2]);
        return Arc::new(ConductorBsdf::new(f0, roughness));
    }

    // Priority 3 – diffuse. Try to load map_Kd first.
    //
    // MTL files exported from Windows apps use backslash separators; we
    // normalise to forward slashes so the path is valid on Linux/macOS.
    if let Some(ref tex_name) = mat.diffuse_texture {
        let tex_rel: std::path::PathBuf = tex_name.replace('\\', "/").into();
        let tex_path = base_dir.join(&tex_rel);
        match ImageTexture::load(&tex_path) {
            Ok(tex) => return Arc::new(Lambertian::with_texture(tex)),
            Err(e) => {
                eprintln!(
                    "[importer] warning: could not load texture '{}': {e}",
                    tex_path.display()
                );
                // Fall through to constant-colour fallback below.
            }
        }
    }

    // Constant-colour diffuse fallback.
    let kd = mat.diffuse.unwrap_or([0.5, 0.5, 0.5]);
    let albedo = Vec3A::new(kd[0], kd[1], kd[2]);
    Arc::new(Lambertian::new(albedo))
}

/// Returns a default grey diffuse BSDF used when no MTL file is present.
fn default_material() -> Arc<dyn Bsdf> {
    Arc::new(Lambertian::with_texture(ConstantTexture::new(Vec3A::splat(0.5))))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load an OBJ file and convert it into a list of [`SceneObject`]s ready for
/// use in a [`crate::scene::Scene`].
///
/// # Transform
///
/// `transform` is the local-to-world matrix applied to every object in the
/// file.  Pass [`Mat4::IDENTITY`] to keep the original OBJ coordinates.
///
/// # Material assignment
///
/// Each OBJ model maps to one [`SceneObject`].  All MTL materials referenced
/// by the file are converted (see module-level doc) and stored in the object's
/// material list; each mesh's `material_slots` index then selects the correct
/// BSDF at shading time.
///
/// If the MTL file cannot be found or parsed the entire scene falls back to a
/// uniform mid-grey Lambertian material.
pub fn load_obj_scene(path: &Path, transform: Mat4) -> Result<Vec<Arc<SceneObject>>> {
    let (models, materials_result) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)?;

    // Directory containing the OBJ file; used to resolve relative texture paths.
    let base_dir = path.parent().unwrap_or(Path::new("."));

    // Convert all MTL materials up-front.  On error fall back to a single
    // default material so the scene still renders.
    let bsdfs: Vec<Arc<dyn Bsdf>> = match materials_result {
        Ok(mats) if !mats.is_empty() => {
            mats.iter().map(|m| convert_material(m, base_dir)).collect()
        }
        _ => vec![default_material()],
    };

    let mut objects = Vec::with_capacity(models.len());

    for model in &models {
        let m = &model.mesh;

        // ---- vertex data ------------------------------------------------
        let positions: Vec<Vec3A> = m
            .positions
            .chunks_exact(3)
            .map(|p| Vec3A::new(p[0], p[1], p[2]))
            .collect();

        let normals = if !m.normals.is_empty() {
            Some(
                m.normals
                    .chunks_exact(3)
                    .map(|n| Vec3A::new(n[0], n[1], n[2]))
                    .collect(),
            )
        } else {
            None
        };

        let tex_coords = if !m.texcoords.is_empty() {
            Some(
                m.texcoords
                    .chunks_exact(2)
                    .map(|t| Vec2::new(t[0], t[1]))
                    .collect(),
            )
        } else {
            None
        };

        // ---- material slot mapping --------------------------------------
        //
        // `tobj` with GPU_LOAD_OPTIONS splits meshes by material, so each
        // model typically has a single material_id.  We store that index as
        // the lone entry in `material_slots`; `SceneObject::material()` uses
        // it to index directly into `bsdfs`.
        //
        // If `material_id` is absent (no MTL) we use slot 0, which is always
        // the default grey diffuse created above in that case.
        let mat_id = m.material_id.unwrap_or(0) as u32;
        // Clamp to valid range defensively – `SceneObject::material()` also
        // clamps, but being explicit here documents intent.
        let mat_id = mat_id.min(bsdfs.len().saturating_sub(1) as u32);
        let material_slots = vec![mat_id];

        let mesh = Arc::new(TriangleMesh {
            positions,
            normals,
            tex_coords,
            tangents: None,
            indices: m.indices.clone(),
            material_slots,
        });

        objects.push(Arc::new(SceneObject::new(mesh, transform, bsdfs.clone())));
    }

    Ok(objects)
}
