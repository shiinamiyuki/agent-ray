//! Scene serialization and deserialization using serde / serde_json.
//!
//! Geometry data (vertex positions, normals, texture coordinates, and
//! triangle indices) is stored in a companion **binary buffer file**
//! (`.bin`) for compactness and fast I/O.  The JSON scene file contains
//! only lightweight metadata — materials, object descriptions, lights,
//! camera, and byte-range references into the `.bin` file.
//!
//! # File layout
//!
//! A scene named `my_scene` consists of two files:
//!
//! | File              | Contents                                      |
//! |-------------------|-----------------------------------------------|
//! | `my_scene.json`   | JSON with materials, objects, lights, camera   |
//! | `my_scene.bin`    | Raw little-endian `f32` / `u32` buffers        |
//!
//! # JSON format overview
//!
//! ```json
//! {
//!   "buffer_file": "my_scene.bin",
//!   "materials": [ ... ],
//!   "meshes": [
//!     {
//!       "positions": { "offset": 0, "length": 1200 },
//!       "normals":   { "offset": 1200, "length": 1200 },
//!       "indices":   { "offset": 2400, "length": 480 },
//!       "material_slots": [0]
//!     }
//!   ],
//!   "objects": [ ... ],
//!   "lights": [ ... ],
//!   "camera": { ... }
//! }
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! use agent_ray::scene_format::{load_and_build_scene, SceneDescription, BinBuffer,
//!                                save_scene_file};
//!
//! // Save
//! let mut desc = SceneDescription::default();
//! let buf = BinBuffer::new();
//! save_scene_file("my_scene.json", &mut desc, &buf).unwrap();
//!
//! // Load
//! let (scene, camera) = load_and_build_scene("my_scene.json").unwrap();
//! ```

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use glam::{Mat4, Vec3A};
use serde::{Deserialize, Serialize};

use crate::cameras::PinholeCamera;
use crate::lights::{Light, PointLight, PowerLightDistribution};
use crate::primitives::mesh::TriangleMesh;
use crate::scene::{Scene, SceneObject};
use crate::surfaces::{Bsdf, ConductorBsdf, DielectricBsdf, Lambertian};
use crate::texture::ConstantTexture;

// Re-export the OBJ-import helper so callers (e.g. the CLI binary) can
// reach it via `scene_format::import_obj_to_scene_desc`.
pub use self::obj_import::import_obj_to_scene_desc;

// =========================================================================
// Small helpers for glam ↔ JSON-friendly arrays
// =========================================================================

type Vec3 = [f32; 3];
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

// =========================================================================
// Binary buffer
// =========================================================================

/// A byte-range reference into the companion `.bin` file.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BufferRef {
    /// Byte offset from the start of the `.bin` file.
    pub offset: u64,
    /// Length in bytes.
    pub length: u64,
}

/// An in-memory accumulator for binary geometry data.
///
/// Append slices of `f32` / `u32` via [`append_f32s`](Self::append_f32s) /
/// [`append_u32s`](Self::append_u32s); each call returns a [`BufferRef`]
/// recording where the data was written.  When finished, call
/// [`write_to_file`](Self::write_to_file) to flush to disk.
#[derive(Debug, Clone, Default)]
pub struct BinBuffer {
    data: Vec<u8>,
}

impl BinBuffer {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Pre-load an existing `.bin` file so that new data can be appended.
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)
            .with_context(|| format!("failed to read binary buffer '{}'", path.display()))?;
        Ok(Self { data })
    }

    /// Current size in bytes.
    pub fn len(&self) -> u64 {
        self.data.len() as u64
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Append `f32` values (little-endian) and return a [`BufferRef`].
    pub fn append_f32s(&mut self, values: &[f32]) -> BufferRef {
        let offset = self.data.len() as u64;
        let byte_len = values.len() * 4;
        self.data.reserve(byte_len);
        for &v in values {
            self.data.extend_from_slice(&v.to_le_bytes());
        }
        BufferRef { offset, length: byte_len as u64 }
    }

    /// Append `u32` values (little-endian) and return a [`BufferRef`].
    pub fn append_u32s(&mut self, values: &[u32]) -> BufferRef {
        let offset = self.data.len() as u64;
        let byte_len = values.len() * 4;
        self.data.reserve(byte_len);
        for &v in values {
            self.data.extend_from_slice(&v.to_le_bytes());
        }
        BufferRef { offset, length: byte_len as u64 }
    }

    /// Flush accumulated bytes to a file.
    pub fn write_to_file(&self, path: &Path) -> Result<()> {
        let mut f = std::fs::File::create(path)
            .with_context(|| format!("failed to create binary buffer '{}'", path.display()))?;
        f.write_all(&self.data)
            .with_context(|| format!("failed to write binary buffer '{}'", path.display()))?;
        Ok(())
    }

    /// Read a range back as `Vec<f32>`.
    pub fn read_f32s(&self, r: &BufferRef) -> Result<Vec<f32>> {
        let start = r.offset as usize;
        let end = start + r.length as usize;
        if end > self.data.len() {
            bail!(
                "buffer ref [{}, +{}) exceeds buffer size {}",
                r.offset, r.length, self.data.len()
            );
        }
        Ok(self.data[start..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Read a range back as `Vec<u32>`.
    pub fn read_u32s(&self, r: &BufferRef) -> Result<Vec<u32>> {
        let start = r.offset as usize;
        let end = start + r.length as usize;
        if end > self.data.len() {
            bail!(
                "buffer ref [{}, +{}) exceeds buffer size {}",
                r.offset, r.length, self.data.len()
            );
        }
        Ok(self.data[start..end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

// =========================================================================
// Materials
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialDesc {
    Lambertian { albedo: Vec3 },
    Conductor { f0: Vec3, roughness: f32 },
    Dielectric { eta: f32, roughness: f32 },
}

impl MaterialDesc {
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

// =========================================================================
// Mesh (binary-backed)
// =========================================================================

/// Mesh descriptor — vertex / index data lives in the `.bin` file, only
/// byte-range references are serialized into JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshDesc {
    /// `f32×3` per vertex.
    pub positions: BufferRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normals: Option<BufferRef>,
    /// `f32×2` per vertex.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tex_coords: Option<BufferRef>,
    /// `u32`, 3 per triangle.
    pub indices: BufferRef,
    /// Material slot mapping (kept inline — always tiny).
    #[serde(default = "default_material_slots")]
    pub material_slots: Vec<u32>,
}

fn default_material_slots() -> Vec<u32> {
    vec![0]
}

impl MeshDesc {
    /// Write a runtime `TriangleMesh` into `buf` and return a `MeshDesc`.
    pub fn from_mesh(mesh: &TriangleMesh, buf: &mut BinBuffer) -> Self {
        let pos_flat: Vec<f32> = mesh.positions.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
        let positions = buf.append_f32s(&pos_flat);

        let normals = mesh.normals.as_ref().map(|ns| {
            let flat: Vec<f32> = ns.iter().flat_map(|n| [n.x, n.y, n.z]).collect();
            buf.append_f32s(&flat)
        });

        let tex_coords = mesh.tex_coords.as_ref().map(|uvs| {
            let flat: Vec<f32> = uvs.iter().flat_map(|uv| [uv.x, uv.y]).collect();
            buf.append_f32s(&flat)
        });

        let indices = buf.append_u32s(&mesh.indices);

        Self { positions, normals, tex_coords, indices, material_slots: mesh.material_slots.clone() }
    }

    /// Reconstruct a `TriangleMesh` from the binary buffer.
    pub fn build(&self, buf: &BinBuffer) -> Result<Arc<TriangleMesh>> {
        let pos_flat = buf.read_f32s(&self.positions)?;
        let positions: Vec<Vec3A> = pos_flat.chunks_exact(3).map(|p| Vec3A::new(p[0], p[1], p[2])).collect();

        let normals = match &self.normals {
            Some(r) => {
                let flat = buf.read_f32s(r)?;
                Some(flat.chunks_exact(3).map(|n| Vec3A::new(n[0], n[1], n[2])).collect())
            }
            None => None,
        };

        let tex_coords = match &self.tex_coords {
            Some(r) => {
                let flat = buf.read_f32s(r)?;
                Some(flat.chunks_exact(2).map(|uv| glam::Vec2::new(uv[0], uv[1])).collect())
            }
            None => None,
        };

        let indices = buf.read_u32s(&self.indices)?;

        Ok(Arc::new(TriangleMesh {
            positions,
            normals,
            tex_coords,
            tangents: None,
            indices,
            material_slots: self.material_slots.clone(),
        }))
    }
}

// =========================================================================
// Object / Light / Camera descriptors
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectDesc {
    pub mesh_index: usize,
    pub material_indices: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<Mat4x4>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LightDesc {
    Point { position: Vec3, intensity: Vec3 },
}

impl LightDesc {
    pub fn from_point_light(light: &PointLight) -> Self {
        LightDesc::Point {
            position: vec3a_to_arr(light.position),
            intensity: vec3a_to_arr(light.intensity),
        }
    }

    pub fn build(&self) -> Arc<dyn Light> {
        match self {
            LightDesc::Point { position, intensity } => {
                Arc::new(PointLight::new(arr_to_vec3a(*position), arr_to_vec3a(*intensity)))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CameraDesc {
    PinholeLookAt {
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        vfov: f32,
        aspect_ratio: f32,
    },
    PinholeEyeAngle {
        eye: Vec3,
        yaw: f32,
        pitch: f32,
        vfov: f32,
        aspect_ratio: f32,
    },
}

impl CameraDesc {
    pub fn build(&self) -> PinholeCamera {
        match self {
            CameraDesc::PinholeLookAt { eye, target, up, vfov, aspect_ratio } => {
                PinholeCamera::from_lookat(
                    arr_to_vec3a(*eye), arr_to_vec3a(*target), arr_to_vec3a(*up),
                    *vfov, *aspect_ratio,
                )
            }
            CameraDesc::PinholeEyeAngle { eye, yaw, pitch, vfov, aspect_ratio } => {
                PinholeCamera::from_eye_angle(
                    arr_to_vec3a(*eye), *yaw, *pitch, *vfov, *aspect_ratio,
                )
            }
        }
    }
}

// =========================================================================
// Top-level SceneDescription
// =========================================================================

/// A complete, JSON-serializable scene description.
///
/// Geometry lives in a companion `.bin` file referenced by `buffer_file`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneDescription {
    /// Relative path to the companion binary buffer file.
    pub buffer_file: String,
    pub materials: Vec<MaterialDesc>,
    pub meshes: Vec<MeshDesc>,
    pub objects: Vec<ObjectDesc>,
    pub lights: Vec<LightDesc>,
    pub camera: CameraDesc,
}

impl Default for SceneDescription {
    fn default() -> Self {
        Self {
            buffer_file: String::new(),
            materials: vec![MaterialDesc::Lambertian { albedo: [0.5, 0.5, 0.5] }],
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
    /// Build a renderable [`Scene`] + [`PinholeCamera`].
    ///
    /// `scene_dir` is the directory containing the `.json` file; it is used
    /// to resolve the relative `buffer_file` path.
    pub fn build_with_dir(&self, scene_dir: &Path) -> Result<(Scene, PinholeCamera)> {
        let buf = if self.meshes.is_empty() {
            BinBuffer::new()
        } else {
            let bin_path = scene_dir.join(&self.buffer_file);
            BinBuffer::from_file(&bin_path)?
        };

        let bsdfs: Vec<Arc<dyn Bsdf>> = self.materials.iter().map(|m| m.build()).collect();

        let meshes: Vec<Arc<TriangleMesh>> = self
            .meshes.iter()
            .map(|m| m.build(&buf))
            .collect::<Result<_>>()?;

        let mut scene_objects: Vec<Arc<SceneObject>> = Vec::with_capacity(self.objects.len());
        for obj in &self.objects {
            let mesh = meshes.get(obj.mesh_index)
                .with_context(|| format!(
                    "object references mesh index {} but only {} meshes exist",
                    obj.mesh_index, meshes.len(),
                ))?.clone();

            let mats: Vec<Arc<dyn Bsdf>> = obj.material_indices.iter()
                .map(|&i| bsdfs.get(i).cloned().unwrap_or_else(|| bsdfs.last().unwrap().clone()))
                .collect();

            let transform = obj.transform.map(arr_to_mat4).unwrap_or(Mat4::IDENTITY);
            scene_objects.push(Arc::new(SceneObject::new(mesh, transform, mats)));
        }

        let lights: Vec<Arc<dyn Light>> = self.lights.iter().map(|l| l.build()).collect();
        let light_dist: Option<Box<dyn crate::lights::LightDistribution>> = if lights.is_empty() {
            None
        } else {
            Some(Box::new(PowerLightDistribution::new(&lights)))
        };

        let camera = self.camera.build();
        let scene = Scene::new(scene_objects, lights, light_dist);
        Ok((scene, camera))
    }

    /// Convenience: build assuming the `.bin` file is in the current directory.
    pub fn build(&self) -> Result<(Scene, PinholeCamera)> {
        self.build_with_dir(Path::new("."))
    }

    /// Capture runtime scene components into a serializable description.
    ///
    /// Geometry data is written into `buf`.  `buffer_file` is left empty —
    /// it is set automatically by [`save_scene_file`].
    pub fn from_scene(
        objects: &[Arc<SceneObject>],
        lights: &[Arc<dyn Light>],
        camera_desc: CameraDesc,
        buf: &mut BinBuffer,
    ) -> Self {
        let mut mesh_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut meshes: Vec<MeshDesc> = Vec::new();
        let mut materials: Vec<MaterialDesc> = Vec::new();
        let mut scene_objects: Vec<ObjectDesc> = Vec::new();

        for obj in objects {
            let mesh_ptr = Arc::as_ptr(&obj.mesh) as usize;
            let mesh_idx = *mesh_map.entry(mesh_ptr).or_insert_with(|| {
                let idx = meshes.len();
                meshes.push(MeshDesc::from_mesh(&obj.mesh, buf));
                idx
            });

            let mat_start = materials.len();
            for _ in &obj.materials {
                materials.push(MaterialDesc::Lambertian { albedo: [0.5, 0.5, 0.5] });
            }
            let mat_indices: Vec<usize> = (mat_start..mat_start + obj.materials.len()).collect();

            let transform = if obj.transform == Mat4::IDENTITY {
                None
            } else {
                Some(mat4_to_arr(obj.transform))
            };

            scene_objects.push(ObjectDesc { mesh_index: mesh_idx, material_indices: mat_indices, transform });
        }

        let _ = lights;

        Self {
            buffer_file: String::new(),
            materials,
            meshes,
            objects: scene_objects,
            lights: Vec::new(),
            camera: camera_desc,
        }
    }
}

// =========================================================================
// File I/O
// =========================================================================

/// Derive the `.bin` path from a `.json` scene path.
pub fn bin_path_for(json_path: &Path) -> PathBuf {
    json_path.with_extension("bin")
}

/// Save a [`SceneDescription`] and its companion binary buffer to disk.
///
/// Writes two files: `path` (JSON) and `path.with_extension("bin")`.
/// `desc.buffer_file` is set automatically before serialization.
pub fn save_scene_file<P: AsRef<Path>>(
    path: P,
    desc: &mut SceneDescription,
    buf: &BinBuffer,
) -> Result<()> {
    let json_path = path.as_ref();
    let bin_file = bin_path_for(json_path);

    // Store only the filename so the pair is relocatable.
    desc.buffer_file = bin_file
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();

    buf.write_to_file(&bin_file)?;

    let json = serde_json::to_string_pretty(desc)
        .context("failed to serialize scene description to JSON")?;
    std::fs::write(json_path, json)
        .with_context(|| format!("failed to write scene file '{}'", json_path.display()))?;

    Ok(())
}

/// Load a [`SceneDescription`] from a JSON file.
///
/// The companion `.bin` file is loaded lazily by
/// [`SceneDescription::build_with_dir`].
pub fn load_scene_file<P: AsRef<Path>>(path: P) -> Result<SceneDescription> {
    let data = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("failed to read scene file '{}'", path.as_ref().display()))?;
    let desc: SceneDescription = serde_json::from_str(&data)
        .context("failed to deserialize scene description from JSON")?;
    Ok(desc)
}

/// Load the binary buffer associated with a scene file.
///
/// Resolves `desc.buffer_file` relative to `scene_dir`.
pub fn load_bin_buffer(desc: &SceneDescription, scene_dir: &Path) -> Result<BinBuffer> {
    if desc.buffer_file.is_empty() || desc.meshes.is_empty() {
        return Ok(BinBuffer::new());
    }
    BinBuffer::from_file(&scene_dir.join(&desc.buffer_file))
}

/// Convenience: load + build in one call.
pub fn load_and_build_scene<P: AsRef<Path>>(path: P) -> Result<(Scene, PinholeCamera)> {
    let json_path = path.as_ref();
    let scene_dir = json_path.parent().unwrap_or(Path::new("."));
    let desc = load_scene_file(json_path)?;
    desc.build_with_dir(scene_dir)
}

// =========================================================================
// OBJ → SceneDescription import
// =========================================================================

mod obj_import {
    use std::path::Path;
    use anyhow::Result;
    use super::{BinBuffer, MaterialDesc, MeshDesc, ObjectDesc, SceneDescription};

    fn ns_to_roughness(ns: f32) -> f32 {
        (2.0 / (ns + 2.0)).sqrt().clamp(1e-3, 1.0)
    }

    fn luminance(c: [f32; 3]) -> f32 {
        0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    }

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
            return MaterialDesc::Conductor { f0: ks, roughness };
        }
        let kd = mat.diffuse.unwrap_or([0.5, 0.5, 0.5]);
        MaterialDesc::Lambertian { albedo: kd }
    }

    /// Import one OBJ file into a [`SceneDescription`], writing geometry
    /// data into `buf`.
    ///
    /// `transform` is an optional column-major 4×4 matrix; `None` ≡ identity.
    pub fn import_obj_to_scene_desc(
        desc: &mut SceneDescription,
        buf: &mut BinBuffer,
        obj_path: &Path,
        transform: Option<[f32; 16]>,
    ) -> Result<()> {
        let (models, materials_result) = tobj::load_obj(obj_path, &tobj::GPU_LOAD_OPTIONS)?;

        let mat_descs: Vec<MaterialDesc> = match materials_result {
            Ok(mats) if !mats.is_empty() => mats.iter().map(|m| convert_material_desc(m)).collect(),
            _ => vec![MaterialDesc::Lambertian { albedo: [0.5, 0.5, 0.5] }],
        };

        let mat_base = desc.materials.len();
        desc.materials.extend(mat_descs.iter().cloned());

        for model in &models {
            let m = &model.mesh;

            let positions = buf.append_f32s(&m.positions);
            let normals = if !m.normals.is_empty() { Some(buf.append_f32s(&m.normals)) } else { None };
            let tex_coords = if !m.texcoords.is_empty() { Some(buf.append_f32s(&m.texcoords)) } else { None };
            let indices = buf.append_u32s(&m.indices);

            let mat_id = m.material_id.unwrap_or(0) as u32;
            let mat_id = mat_id.min(mat_descs.len().saturating_sub(1) as u32);

            let mesh_idx = desc.meshes.len();
            desc.meshes.push(MeshDesc {
                positions,
                normals,
                tex_coords,
                indices,
                material_slots: vec![mat_id],
            });

            let material_indices: Vec<usize> = (0..mat_descs.len()).map(|i| mat_base + i).collect();
            desc.objects.push(ObjectDesc {
                mesh_index: mesh_idx,
                material_indices,
                transform,
            });
        }

        Ok(())
    }
}
