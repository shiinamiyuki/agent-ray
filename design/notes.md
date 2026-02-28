## 2026-02-27 - Initial Geometry Implementation
Implemented basic geometric primitives required for ray tracing:
- `Ray`: origin, normalized direction, and interval $[t_{min}, t_{max}]$.
- `AABB`: Axis-aligned bounding box with intersection and union.
- `Sphere`: Geometric sphere with analytical intersection.
- `Triangle`: Triangle primitive using the Möller-Trumbore intersection algorithm.
- `Intersect` trait: Common interface for intersection tests.
- `HitInfo`: Stores intersection data like point, normal, distance, and UVs.

## 2026-02-27 - Parallel Utilities
Implemented parallel execution helpers in `src/utils.rs` using `rayon`:
- `parallel_for`: Parallel loop for 1D range with thread ID access.
- `parallel_for_2d`: Parallel loop for 2D range (useful for image processing).

## 2026-02-27 - Camera & Rendering Test
- Implemented `PinholeCamera` in `src/cameras.rs`.
- Added `save_image_as_png` utility in `src/utils.rs`.
- Set up `bin/render_test.rs` with a simple scene of two triangles to test ray-triangle intersection and normal visualization.
- Verified parallel rendering using `rayon`.
- Cleaned up `src/main.rs` and moved the test rendering to the `bin/` directory.


## 2026-02-27 - Mesh Implementation
- Implemented `TriangleMesh` in [src/primitives/mesh.rs](src/primitives/mesh.rs).
- Added `positions`, `normals`, `tex_coords`, `tangents`, and `indices` to the mesh representation.
- Integrated `tobj` for loading triangle meshes from OBJ files.
- Added `primitives` and `mesh` modules to [src/lib.rs](src/lib.rs).

## 2026-02-27 - Scene & Path Tracer
Added `src/scene.rs`, `src/integrators/mod.rs`, and `src/integrators/path_tracer.rs`.

**`scene.rs`**:
- `SceneObject`: wraps `Arc<TriangleMesh>` + `BLASAccel` + world transform + a `Vec<Arc<dyn Bsdf>>` material list. Normal matrix (`inv_transpose(M₃×₃)`) is pre-computed for correct world-space normal transforms. `material(prim_id)` respects the per-face vs shared `material_slots` layout from `TriangleMesh`. `shading_normal` interpolates vertex normals (barycentric) when present, falls back to geometric normal.
- `ShadingPoint`: `p`, `n` (face-forwarded world-space shading normal), `Onb`, `Arc<dyn Bsdf>`.
- `Scene`: holds TLAS, `Vec<Arc<SceneObject>>` (index == TLAS instance_id), lights, and an optional `LightDistribution`. `intersect` resolves the full `ShadingPoint` from a `RayHit`. `occluded` is a fast any-hit shadow query.

**`integrators/mod.rs`**:
- `Integrator` trait: `render(scene, camera, width, height) -> Vec<Vec3A>` (linear HDR, row-major).

**`integrators/path_tracer.rs`** — `PathTracer`:
- `PathTracerConfig`: `spp`, `max_depth`, `rr_depth`.
- Iterative path tracer (no recursion) with:
  - **Direct lighting**: one light sampled per bounce via `LightDistribution`. Delta lights (point) contribute directly; non-delta lights use the MIS power heuristic (β=2) with BSDF sampling.
  - **Indirect lighting**: BSDF importance sampling extends the path; throughput = `f·|cosθ|/pdf`.
  - **Russian roulette**: stochastic path termination after `rr_depth` bounces; survival probability = `min(max_component(throughput), 0.95)`.
  - **Jittered sub-pixel sampling**: each sample uses a uniform jitter within the pixel footprint.
  - **Per-pixel deterministic RNG**: `SmallRng` seeded by a hash of `(x, y)` for reproducible results.
- Render loop uses `rayon` parallel row iteration.

## 2026-02-27 - Light Sources & Distribution
Added `src/lights.rs` and `src/lights/point.rs`:

- **`Light` trait** (`src/lights.rs`): `sample(ref_point, u) -> Option<LightSample>`, `power() -> Vec3A`, `is_delta() -> bool`.
- **`LightSample`**: carries `wi` (direction to light), `dist`, `li` (incident radiance), and `pdf` (1.0 for delta lights).
- **`LightDistribution` trait**: `sample_index(u) -> (idx, pmf)` + `pmf(idx)` + `len()`. Intended for the two-step MIS pattern used by path tracers.
- **`UniformLightDistribution`**: selects each light with probability 1/N.
- **`PowerLightDistribution`**: CDF over lights weighted by luminance of `Light::power()` (Y = 0.2126 R + 0.7152 G + 0.0722 B); binary-search inversion.
- **`PointLight`** (`src/lights/point.rs`): isotropic delta light; `Li = intensity / dist²`; `power = 4π · intensity`.

## 2026-02-27 - OBJ/MTL Scene Importer
Added `src/importer.rs` with `load_obj_scene(path, transform) -> Result<Vec<Arc<SceneObject>>>`.

**MTL → BSDF conversion** (priority order):
1. **Transparent** (`d < 0.5`) → `DielectricBsdf { eta = Ni, roughness }`.
2. **Conductor / glossy** (`luminance(Ks) > 0.04`) → `ConductorBsdf { f0 = Ks, roughness }`.
3. **Diffuse fallback** → `Lambertian { albedo = Kd }`.

**Roughness** derived from Phong shininess via `α = sqrt(2 / (Ns + 2))`, mapping `Ns=0 → α=1` (rough) and `Ns=1000 → α≈0.044` (near-mirror).  Defaults: Kd=0.5 grey, Ks=0, Ns=0, d=1, Ni=1.5.  If the MTL file is missing the whole scene falls back to a uniform mid-grey Lambertian.

The function constructs one `SceneObject` per OBJ model, with the full converted BSDF list shared across the object and `material_slots` pointing to the correct index from the MTL.

## 2026-02-27 - Texture System
Added `src/texture.rs` and wired textures through the entire shading pipeline.

**`src/texture.rs`**:
- `Texture` trait: `sample(&self, uv: Vec2) -> Vec3A` (linear).
- `ConstantTexture(Vec3A)` — solid colour, UV-independent.
- `ImageTexture` — loads any image format via the `image` crate; pixels are decoded from sRGB to linear using the piecewise IEC 61966-2-1 transfer function on load. Bilinear filtering with repeat-wrap UV.

**`Bsdf` trait** (`surfaces.rs`): added `uv: Vec2` to `eval` and `sample` so that texture-mapped parameters can be evaluated per shading-point. `SurfaceClosure` updated accordingly.

**`Lambertian`**: albedo field changed from `Vec3A` to `Arc<dyn Texture>`; `::new(Vec3A)` still works (wraps into `ConstantTexture`); new `::with_texture(Arc<dyn Texture>)` constructor added.

**`ShadingPoint`**: added `tex_uv: Vec2`. `SceneObject::tex_uv()` interpolates mesh texture coordinates barycentrically (falls back to raw barycentric UV if mesh has none). `Scene::intersect` resolves and stores it.

**Importer** (`src/importer.rs`): `convert_material` now accepts the OBJ `base_dir` and loads `map_Kd` images via `ImageTexture::load`. Windows backslash separators in MTL paths are normalised. If the texture file cannot be opened a warning is printed and the constant `Kd` colour is used as fallback.

## 2026-02-27 - Path Tracer Test Binary
Added `bin/pt_test.rs`:
- Loads `assets/fireplace_room/fireplace_room.obj` via `load_obj_scene` (full PBR material conversion).
- Places a warm point light at `(0, 8, 0)` with intensity `(1500, 1200, 800)` W/sr to approximate indoor lighting.
- Runs `PathTracer` at 1280×720 with 128 spp, max depth 8, RR after depth 3.
- Applies luminance-preserving Reinhard tone mapping followed by γ 2.2 encoding and saves `pt_test.png`.
- Also re-exported `PathTracerConfig` from `integrators` module for cleaner imports.

## 2026-02-27 - BSDF Implementation
Refactored `src/surfaces.rs` into a module with common utilities and added two BSDF implementations in `src/surfaces/`:

- **Common utilities in `surfaces.rs`**:
  - `Bsdf` trait (now `Send + Sync`) and `BsdfSample` struct.
  - `SurfaceClosure` that converts world↔local via `Onb` (also fixed a pre-existing bug where `eval` was calling `to_world` instead of `to_local`).
  - Local-space trig helpers: `cos_theta`, `abs_cos_theta`, `sin2_theta`, `tan2_theta`, `same_hemisphere`.
  - Sampling helpers: `cosine_hemisphere_sample`, `concentric_disk_sample`.
  - Fresnel functions: `fresnel_schlick` (Schlick approximation) and `fresnel_dielectric` (exact polarised).
  - Geometric optics: `reflect` and `refract` helpers.

- **`surfaces/lambertian.rs`** — `Lambertian` BSDF:
  - Diffuse BSDF: `f = albedo / π`.
  - Cosine-weighted hemisphere importance sampling: `pdf = |cos θᵢ| / π`.

- **`surfaces/microfacet.rs`** — GGX microfacet BSDFs:
  - Shared GGX helpers: isotropic NDF (`ggx_d`), Smith Λ (`ggx_lambda`), height-correlated G2 (`ggx_g2`), NDF sampling (`ggx_sample_wh`).
  - `ConductorBsdf`: Cook-Torrance model (D·G2·F_Schlick) for opaque metals. Samples by reflecting off a GGX-distributed micro-normal.
  - `DielectricBsdf`: Cook-Torrance model for glass/dielectrics. Probabilistically selects the reflective or transmissive lobe using the Fresnel weight, with correct radiometric correction for the transmissive term.

## 2026-02-27 - Two-Level SAH BVH
Implemented a full two-level BVH acceleration structure in `src/accel/bvh.rs`:
- **SAH builder** (`sah_split`): shared between BLAS and TLAS; performs a forward-backward AABB prefix/suffix scan on all three axes to test every exact split point, picks the axis+position with the lowest SAH cost, and only splits when the cost beats a pure leaf.
- **Parallel recursive builder** (`build_recursive`): uses `rayon::join` down to a configurable depth (`MAX_PARALLEL_DEPTH = 8`) for parallel subtree construction. Subtrees are assembled into flat `Vec<BVHNode>` arrays with child indices rebased on merge — no mutation of shared state during parallel phase.
- **Flat node layout**: `BVHNode` stores an AABB plus a `count` discriminant (0 = internal, >0 = leaf); leaves reference a contiguous range in a separate `prim_indices` / `instance_indices` array.
- **`BLASPrimitive` trait**: users implement `primitive_count`, `primitive_aabb`, and `intersect_primitive`; the backing geometry is never reordered — only an index array is permuted.
- **`BLASAccel`**: wraps an `Arc<dyn BLASPrimitive>` with a BVH; exposes `build`, `aabb`, and `intersect`.
- **`Instance`** / **`TLAS`**: TLAS is a BVH over instances; each instance carries a local-to-world `Mat4`. Ray–instance intersection transforms the ray into local space (accounting for the direction-length scale factor so world t-values remain correct).
- Added `AABB::empty()`, `AABB::surface_area()`, `AABB::centroid()`, and `AABB::transform(Mat4)` to `src/geometry.rs`.