//! Scene serialization and deserialization using serde / serde_json.
//!
//! This module defines a JSON-friendly **description** layer that mirrors the
//! runtime types in [`crate::scene`], [`crate::cameras`], [`crate::lights`],
//! and [`crate::surfaces`].  Every description type derives
//! `Serialize` + `Deserialize` and can be round-tripped through JSON.
//!
//! # File format overview
//!
//! ```json
//! {
//!   "materials": [
//!     { "Lambertian": { "albedo": [0.8, 0.7, 0.6] } },
//!     { "Conductor":  { "f0": [1.0, 0.86, 0.57], "roughness": 0.1 } },
//!     { "Dielectric": { "eta": 1.5, "roughness": 0.0 } }
//!   ],
//!   "meshes": [ ... ],
//!   "objects": [
//!     { "mesh_index": 0, "material_indices": [0], "transform": null }
//!   ],
//!   "lights": [
//!     { "Point": { "position": [1.0, 2.0, 3.0], "intensity": [100, 80, 60] } }
//!   ],
//!   "camera": {
//!     "PinholeLookAt": {
//!       "eye": [0, 1, 5], "target": [0, 0, 0], "up": [0, 1, 0],
//!       "vfov": 60.0, "aspect_ratio": 1.7778
//!     }
//!   }
//! }
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! use agent_ray::scene_format::{SceneDescription, load_scene_file, save_scene_file};
//!
//! // Save
//! let desc = SceneDescription::default();
//! save_scene_file("my_scene.json", &desc).unwrap();
//!
//! // Load
//! let desc = load_scene_file("my_scene.json").unwrap();
//! let (scene, camera) = desc.build().unwrap();
//! ```

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use glam::{Mat4, Vec3A};
use serde::{Deserialize, Serialize};

use crate::cameras::PinholeCamera;
use crate::lights::{Light, PointLight, PowerLightDistribution};
use crate::primitives::mesh::TriangleMesh;
use crate::scene::{Scene, SceneObject};
use crate::surfaces::{Bsdf, ConductorBsdf, DielectricBsdf, Lambertian};
use crate::texture::ConstantTexture;

// Re-export the helpers so the binary can use them.
pub use self::obj_import::import_obj_to_scene_desc;

// =========================================================================
// Serializable descriptor types
// =========================================================================

// ---- small helpers for glam types ----------------------------------------

/// A 3-component colour / vector stored as `[f32; 3]` for JSON readability.
type Vec3 = [f32; 3];

/// A 4×4 column-major matrix stored as `[f32; 16]`.
type Mat4x4 = [f32; 16];

fn vec3a_to_arr(v: Vec3A) -> Vec3 {
    [v.x, v.y, v.z]
}

fn arr_to_vec3a(a: Vec3) -> Vec3A {
    Vec3A::new(a[0], a[1], a[2])
}

fn mat4_to_arr(m: Mat4) -> Mat4x4 {
    m.to_cols_array()
}

fn arr_to_mat4(a: Mat4x4) -> Mat4 {
    Mat4::from_cols_array(&a)
}

// ---- materials -----------------------------------------------------------

/// Serializable material description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialDesc {
    /// Perfectly diffuse (Lambertian) BSDF with a constant albedo colour.
    Lambertian {
        albedo: Vec3,
    },
    /// Cook-Torrance conductor (metallic) BSDF.
    Conductor {
        /// Normal-incidence reflectance (F0).
        f0: Vec3,
        /// GGX roughness α.
        roughness: f32,
    },
    /// Cook-Torrance dielectric (glass) BSDF.
    Dielectric {
        /// Interior index of refraction.
        eta: f32,
        /// GGX roughness α.
        roughness: f32,
    },
}

impl MaterialDesc {
    /// Convert to a runtime BSDF.
    pub fn build(&self) -> Arc<dyn Bsdf> {
        match self {
            MaterialDesc::Lambertian { albedo } => {
                Arc::new(Lambertian::with_texture(ConstantTexture::new(arr_to_vec3a(*albedo))))
            }
            MaterialDesc::Conductor { f0, roughness } => {
                Arc::new(ConductorBsdf::new(arr_to_vec3a(*f0), *roughness))
            }
            MaterialDesc::Dielectric { eta, roughness } => {
                Arc::new(DielectricBsdf::new(*eta, *roughness))
            }
        }
    }
}

// ---- mesh ----------------------------------------------------------------

/// Serializable triangle-mesh data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshDesc {
    /// Flat array of vertex positions: `[x0, y0, z0, x1, y1, z1, …]`.
    pub positions: std::vec::Vec<f32>,
    /// Optional flat array of vertex normals (same layout as `positions`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normals: Option<std::vec::Vec<f32>>,
    /// Optional flat array of texture coordinates: `[u0, v0, u1, v1, …]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tex_coords: Option<std::vec::Vec<f32>>,
    /// Triangle indices (3 per triangle).
    pub indices: std::vec::Vec<u32>,
    /// Material slot mapping.  If length == 1, all triangles share the same
    /// slot; otherwise one entry per triangle.
    #[serde(default = "default_material_slots")]
    pub material_slots: std::vec::Vec<u32>,
}

fn default_material_slots() -> std::vec::Vec<u32> {
    vec![0]
}

impl MeshDesc {
    /// Build from a runtime `TriangleMesh`.
    pub fn from_mesh(mesh: &TriangleMesh) -> Self {
        let positions: std::vec::Vec<f32> = mesh
            .positions
            .iter()
            .flat_map(|v| [v.x, v.y, v.z])
            .collect();

        let normals = mesh.normals.as_ref().map(|ns| {
            ns.iter().flat_map(|n| [n.x, n.y, n.z]).collect()
        });

        let tex_coords = mesh.tex_coords.as_ref().map(|uvs| {
            uvs.iter().flat_map(|uv| [uv.x, uv.y]).collect()
        });

        Self {
            positions,
            normals,
            tex_coords,
            indices: mesh.indices.clone(),
            material_slots: mesh.material_slots.clone(),
        }
    }

    /// Convert to a runtime `TriangleMesh`.
    pub fn build(&self) -> Arc<TriangleMesh> {
        let positions: std::vec::Vec<Vec3A> = self
            .positions
            .chunks_exact(3)
            .map(|p| Vec3A::new(p[0], p[1], p[2]))
            .collect();

        let normals = self.normals.as_ref().map(|ns| {
            ns.chunks_exact(3)
                .map(|n| Vec3A::new(n[0], n[1], n[2]))
                .collect()
        });

        let tex_coords = self.tex_coords.as_ref().map(|uvs| {
            uvs.chunks_exact(2)
                .map(|uv| glam::Vec2::new(uv[0], uv[1]))
                .collect()
        });

        Arc::new(TriangleMesh {
            positions,
            normals,
            tex_coords,
            tangents: None,
            indices: self.indices.clone(),
            material_slots: self.material_slots.clone(),
        })
    }
}

// ---- scene object --------------------------------------------------------

/// Serializable scene object: links a mesh to its materials and transform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectDesc {
    /// Index into `SceneDescription::meshes`.
    pub mesh_index: usize,
    /// Indices into `SceneDescription::materials` for each material slot.
    pub material_indices: std::vec::Vec<usize>,
    /// Optional 4×4 local-to-world transform (column-major).
    /// `None` means identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<Mat4x4>,
}

// ---- lights --------------------------------------------------------------

/// Serializable light description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LightDesc {
    /// Isotropic point light.
    Point {
        position: Vec3,
        /// Spectral intensity (watts per steradian).
        intensity: Vec3,
    },
}

impl LightDesc {
    /// Build from a runtime `PointLight`.
    pub fn from_point_light(light: &PointLight) -> Self {
        LightDesc::Point {
            position: vec3a_to_arr(light.position),
            intensity: vec3a_to_arr(light.intensity),
        }
    }

    /// Convert to a runtime `Light`.
    pub fn build(&self) -> Arc<dyn Light> {
        match self {
            LightDesc::Point {
                position,
                intensity,
            } => Arc::new(PointLight::new(
                arr_to_vec3a(*position),
                arr_to_vec3a(*intensity),
            )),
        }
    }
}

// ---- camera --------------------------------------------------------------

/// Serializable camera description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CameraDesc {
    /// Pinhole camera defined by a look-at configuration.
    PinholeLookAt {
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        /// Vertical field-of-view in degrees.
        vfov: f32,
        aspect_ratio: f32,
    },
    /// Pinhole camera defined by eye position and Euler angles.
    PinholeEyeAngle {
        eye: Vec3,
        /// Horizontal rotation in degrees.
        yaw: f32,
        /// Vertical tilt in degrees.
        pitch: f32,
        /// Vertical field-of-view in degrees.
        vfov: f32,
        aspect_ratio: f32,
    },
}

impl CameraDesc {
    /// Convert to a runtime `PinholeCamera`.
    pub fn build(&self) -> PinholeCamera {
        match self {
            CameraDesc::PinholeLookAt {
                eye,
                target,
                up,
                vfov,
                aspect_ratio,
            } => PinholeCamera::from_lookat(
                arr_to_vec3a(*eye),
                arr_to_vec3a(*target),
                arr_to_vec3a(*up),
                *vfov,
                *aspect_ratio,
            ),
            CameraDesc::PinholeEyeAngle {
                eye,
                yaw,
                pitch,
                vfov,
                aspect_ratio,
            } => PinholeCamera::from_eye_angle(
                arr_to_vec3a(*eye),
                *yaw,
                *pitch,
                *vfov,
                *aspect_ratio,
            ),
        }
    }
}

// =========================================================================
// Top-level scene description
// =========================================================================

/// A complete, JSON-serializable scene description.
///
/// Contains all the data needed to reconstruct a renderable [`Scene`] plus a
/// [`Camera`](crate::cameras::Camera).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneDescription {
    /// Named material palette.  Objects reference materials by index.
    pub materials: std::vec::Vec<MaterialDesc>,
    /// Shared mesh data.  Objects reference meshes by index so that the same
    /// mesh can be instanced with different transforms / materials.
    pub meshes: std::vec::Vec<MeshDesc>,
    /// Scene objects linking meshes → materials, with optional transforms.
    pub objects: std::vec::Vec<ObjectDesc>,
    /// Light sources.
    pub lights: std::vec::Vec<LightDesc>,
    /// Camera.
    pub camera: CameraDesc,
}

impl Default for SceneDescription {
    fn default() -> Self {
        Self {
            materials: vec![MaterialDesc::Lambertian {
                albedo: [0.5, 0.5, 0.5],
            }],
            meshes: Vec::new(),
            objects: Vec::new(),
            lights: Vec::new(),
            camera: CameraDesc::PinholeLookAt {
                eye: [0.0, 1.0, 5.0],
                target: [0.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
                vfov: 60.0,
                aspect_ratio: 16.0 / 9.0,
            },
        }
    }
}

impl SceneDescription {
    /// Build a renderable [`Scene`] and [`PinholeCamera`] from this
    /// description.
    ///
    /// The returned scene uses a [`PowerLightDistribution`] when lights are
    /// present.
    pub fn build(&self) -> Result<(Scene, PinholeCamera)> {
        // ---- materials ---------------------------------------------------
        let bsdfs: std::vec::Vec<Arc<dyn Bsdf>> =
            self.materials.iter().map(|m| m.build()).collect();

        // ---- meshes ------------------------------------------------------
        let meshes: std::vec::Vec<Arc<TriangleMesh>> =
            self.meshes.iter().map(|m| m.build()).collect();

        // ---- objects -----------------------------------------------------
        let mut scene_objects: std::vec::Vec<Arc<SceneObject>> =
            std::vec::Vec::with_capacity(self.objects.len());

        for obj in &self.objects {
            let mesh = meshes
                .get(obj.mesh_index)
                .with_context(|| {
                    format!(
                        "object references mesh index {} but only {} meshes exist",
                        obj.mesh_index,
                        meshes.len()
                    )
                })?
                .clone();

            let mats: std::vec::Vec<Arc<dyn Bsdf>> = obj
                .material_indices
                .iter()
                .map(|&i| {
                    bsdfs
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| bsdfs.last().unwrap().clone())
                })
                .collect();

            let transform = obj
                .transform
                .map(arr_to_mat4)
                .unwrap_or(Mat4::IDENTITY);

            scene_objects.push(Arc::new(SceneObject::new(mesh, transform, mats)));
        }

        // ---- lights ------------------------------------------------------
        let lights: std::vec::Vec<Arc<dyn Light>> =
            self.lights.iter().map(|l| l.build()).collect();

        let light_dist: Option<Box<dyn crate::lights::LightDistribution>> = if lights.is_empty() {
            None
        } else {
            Some(Box::new(PowerLightDistribution::new(&lights)))
        };

        // ---- camera ------------------------------------------------------
        let camera = self.camera.build();

        // ---- assemble ----------------------------------------------------
        let scene = Scene::new(scene_objects, lights, light_dist);

        Ok((scene, camera))
    }

    // ---- from runtime types (for saving) ---------------------------------

    /// Create a `SceneDescription` from runtime scene components.
    ///
    /// This captures meshes, materials, objects, lights, and camera into the
    /// serializable format.  Materials are captured as constant-colour values;
    /// texture-mapped materials lose the texture path information and use a
    /// default grey albedo — to preserve texture references, build the
    /// `SceneDescription` directly or extend the format.
    pub fn from_scene(
        objects: &[Arc<SceneObject>],
        lights: &[Arc<dyn Light>],
        camera_desc: CameraDesc,
    ) -> Self {
        // De-duplicate meshes by Arc pointer identity.
        let mut mesh_map: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut meshes: std::vec::Vec<MeshDesc> = Vec::new();
        let mut materials: std::vec::Vec<MaterialDesc> = Vec::new();
        let mut scene_objects: std::vec::Vec<ObjectDesc> = Vec::new();

        for obj in objects {
            // ---- mesh (deduplicate by Arc pointer) -----------------------
            let mesh_ptr = Arc::as_ptr(&obj.mesh) as usize;
            let mesh_idx = *mesh_map.entry(mesh_ptr).or_insert_with(|| {
                let idx = meshes.len();
                meshes.push(MeshDesc::from_mesh(&obj.mesh));
                idx
            });

            // ---- materials -----------------------------------------------
            // Each object contributes its own material set.  We append them
            // to the global palette and record their starting index.
            let mat_start = materials.len();
            // We can't introspect the dyn Bsdf back to its concrete type in
            // a fully general way; store a grey Lambertian placeholder.
            for _ in &obj.materials {
                materials.push(MaterialDesc::Lambertian {
                    albedo: [0.5, 0.5, 0.5],
                });
            }
            let mat_indices: std::vec::Vec<usize> =
                (mat_start..mat_start + obj.materials.len()).collect();

            // ---- transform -----------------------------------------------
            let transform = if obj.transform == Mat4::IDENTITY {
                None
            } else {
                Some(mat4_to_arr(obj.transform))
            };

            scene_objects.push(ObjectDesc {
                mesh_index: mesh_idx,
                material_indices: mat_indices,
                transform,
            });
        }

        // ---- lights (best-effort: only PointLight is supported) ----------
        // We cannot downcast trait objects in general.  The caller should use
        // `LightDesc` values directly if precise round-tripping is needed.
        // Here we provide a sensible default: empty light list (the user can
        // populate lights in the JSON directly).
        let light_descs: std::vec::Vec<LightDesc> = Vec::new();
        // Note: if the caller constructs the SceneDescription manually they
        // can fill in `lights` precisely.  We leave this empty since we can't
        // downcast `dyn Light`.
        let _ = lights; // suppress unused warning

        Self {
            materials,
            meshes,
            objects: scene_objects,
            lights: light_descs,
            camera: camera_desc,
        }
    }
}

// =========================================================================
// File I/O
// =========================================================================

/// Save a [`SceneDescription`] to a JSON file.
pub fn save_scene_file<P: AsRef<Path>>(path: P, desc: &SceneDescription) -> Result<()> {
    let json = serde_json::to_string_pretty(desc)
        .context("failed to serialize scene description to JSON")?;
    std::fs::write(path.as_ref(), json)
        .with_context(|| format!("failed to write scene file '{}'", path.as_ref().display()))?;
    Ok(())
}

/// Load a [`SceneDescription`] from a JSON file.
pub fn load_scene_file<P: AsRef<Path>>(path: P) -> Result<SceneDescription> {
    let data = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("failed to read scene file '{}'", path.as_ref().display()))?;
    let desc: SceneDescription = serde_json::from_str(&data)
        .context("failed to deserialize scene description from JSON")?;
    Ok(desc)
}

/// Convenience: load a scene file and immediately build the runtime
/// [`Scene`] + [`PinholeCamera`].
pub fn load_and_build_scene<P: AsRef<Path>>(path: P) -> Result<(Scene, PinholeCamera)> {
    let desc = load_scene_file(path)?;
    desc.build()
}

// =========================================================================
// OBJ → SceneDescription import
// =========================================================================

mod obj_import {
    use std::path::Path;

    use anyhow::Result;

    use super::{MaterialDesc, MeshDesc, ObjectDesc, SceneDescription};

    // ---- roughness helper (mirrors importer.rs) --------------------------

    fn ns_to_roughness(ns: f32) -> f32 {
        (2.0 / (ns + 2.0)).sqrt().clamp(1e-3, 1.0)
    }

    fn luminance(c: [f32; 3]) -> f32 {
        0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    }

    // ---- MTL → MaterialDesc conversion -----------------------------------

    fn convert_material_desc(mat: &tobj::Material) -> MaterialDesc {
        let ks = mat.specular.unwrap_or([0.0, 0.0, 0.0]);
        let ns = mat.shininess.unwrap_or(0.0).max(0.0);
        let d = mat.dissolve.unwrap_or(1.0);
        let ni = mat.optical_density.unwrap_or(1.5).max(1.0);
        let roughness = ns_to_roughness(ns);

        if d < 0.5 {
            return MaterialDesc::Dielectric { eta: ni, roughness };
        }

        if luminance(ks) > 0.04 {
            return MaterialDesc::Conductor {
                f0: ks,
                roughness,
            };
        }

        let kd = mat.diffuse.unwrap_or([0.5, 0.5, 0.5]);
        MaterialDesc::Lambertian { albedo: kd }
    }

    // ---- public API ------------------------------------------------------

    /// Import one OBJ file into a [`SceneDescription`], appending meshes,
    /// materials, and objects.
    ///
    /// `transform` is an optional column-major 4×4 matrix; `None` means
    /// identity.
    ///
    /// The function mutates `desc` in-place so that it can be called
    /// repeatedly to merge multiple OBJ files into a single scene.
    pub fn import_obj_to_scene_desc(
        desc: &mut SceneDescription,
        obj_path: &Path,
        transform: Option<[f32; 16]>,
    ) -> Result<()> {
        let (models, materials_result) =
            tobj::load_obj(obj_path, &tobj::GPU_LOAD_OPTIONS)?;

        // Convert MTL materials to MaterialDesc.
        let mat_descs: Vec<MaterialDesc> = match materials_result {
            Ok(mats) if !mats.is_empty() => {
                mats.iter().map(|m| convert_material_desc(m)).collect()
            }
            _ => vec![MaterialDesc::Lambertian {
                albedo: [0.5, 0.5, 0.5],
            }],
        };

        // Record the starting indices in the global palette so we can offset
        // per-mesh material ids correctly.
        let mat_base = desc.materials.len();
        desc.materials.extend(mat_descs.iter().cloned());

        for model in &models {
            let m = &model.mesh;

            // ---- vertex data --------------------------------------------
            let positions: Vec<f32> = m.positions.clone();

            let normals = if !m.normals.is_empty() {
                Some(m.normals.clone())
            } else {
                None
            };

            let tex_coords = if !m.texcoords.is_empty() {
                Some(m.texcoords.clone())
            } else {
                None
            };

            let mat_id = m.material_id.unwrap_or(0) as u32;
            let mat_id = mat_id.min(mat_descs.len().saturating_sub(1) as u32);
            let material_slots = vec![mat_id];

            let mesh_idx = desc.meshes.len();
            desc.meshes.push(MeshDesc {
                positions,
                normals,
                tex_coords,
                indices: m.indices.clone(),
                material_slots,
            });

            // Material indices: all materials from this OBJ file, offset
            // by `mat_base`.
            let material_indices: Vec<usize> =
                (0..mat_descs.len()).map(|i| mat_base + i).collect();

            desc.objects.push(ObjectDesc {
                mesh_index: mesh_idx,
                material_indices,
                transform,
            });
        }

        Ok(())
    }
}
