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

## 2026-02-28 - Bidirectional Path Tracer
Implemented a full bidirectional path tracer with MIS in `src/integrators/bidirectional_path_tracer.rs`.

**Design:**
- Builds two subpaths per sample: one from the camera, one from a randomly chosen light.
- Enumerates all `(s, t)` connection strategies and weights them via the **power heuristic** (β=2) using Veach's recursive ratio formulation.
- All PDFs are converted to **area measure** so strategies with different vertex counts are directly comparable.
- Delta distributions (point lights, specular BSDFs, pinhole camera) are handled by zeroing their MIS contribution.

**Supported strategies:**
- `s ≥ 2, t = 1`: NEE (next-event estimation) — direct light sampling, equivalent to the unidirectional PT.
- `s = 1, t ≥ 2`: Light tracing — splatted to the image via `Camera::sample_we`.
- `s ≥ 2, t ≥ 2`: General connection through two surface vertices.
- `t = 0` and `s = 0`: Documented as TODOs (require area-light emission / full camera We).

**Trait extensions for BDPT:**
- `Light` trait: added `sample_emission`, `pdf_emission_dir`, `is_positional_delta` with default no-op implementations so existing lights still compile.
- `PointLight`: implements `sample_emission` (uniform sphere sampling) and `pdf_emission_dir`.
- `Camera` trait: added `sample_we`, `pdf_we`, `film_area`, `origin` with default no-op implementations.
- `PinholeCamera`: implements `sample_we` (image-plane projection with `We = 1/(A·cos⁴θ)`) and `pdf_we`.
- `Onb`: added `Clone + Copy` derive.

**Extensibility:** any new `Bsdf`, `Light`, or `Camera` implementation works automatically; BDPT-specific light/camera methods have sensible defaults that gracefully disable unsupported strategies.

**Test:** `bin/bdpt_test.rs` renders the fireplace room scene and produces `bdpt_test.png`. Visually verified correct at 1spp.

## 2026-02-28 - BDPT Debugging & MIS Fix

**Debugging infrastructure added:**
- `MisMode` enum: `Power` (β configurable) and `Uniform` (equal weight per strategy, for sanity-checking).
- `BdptConfig` extended with `mis_mode`, `mis_beta`, `debug_strategy_images` fields.
- Per-strategy image dump: when `debug_strategy_images = true`, writes `bdpt_s{s}_t{t}.png` for every active strategy.

**Bug found and fixed — off-by-one in `pdf_rev` storage:**
- Root cause: in both `generate_camera_subpath` and `generate_light_subpath`, the reverse area-measure PDF computed at vertex `z[j]` (representing $p^\leftarrow(x_{j-1})$) was stored at `vertices.last_mut()` (= `z[j]`) instead of `vertices[prev_idx]` (= `z[j-1]`).
- This meant `z[i].pdf_rev` held $p^\leftarrow(x_{i-1})$ instead of $p^\leftarrow(x_i)$, causing the MIS weight walk to use wrong ratios, under-counting the denominator $\sum r_i$, and over-weighting every strategy → brighter image.
- Fix: store at `vertices[prev_idx].pdf_rev` (matching the PBRT convention where creating `v[j]` retroactively sets `v[j-1].pdfRev`).

## 2026-03-06 - BDPT Path Length Constraint Fix

**Bug found and fixed — `max_depth` not limiting total path length in strategy enumeration:**
- Root cause: the render loop enumerated all `(s, t)` combinations with `s ∈ [0, s_max]` and `t ∈ [0, t_max]` without constraining the total number of path vertices. With `max_depth=1`, each subpath could have up to 2 vertices (1 endpoint + 1 surface hit), allowing strategy `(s=2, t=2)` which produces a 4-vertex path (3 edges, 2 surface bounces) — an indirect lighting path.
- Fix: added constraint `s + t ≤ max_depth + 2` in the strategy enumeration loop. The full path has `s + t` vertices and `s + t - 2` interior surface bounces, so this ensures the number of bounces doesn't exceed `max_depth`. For `max_depth=1`, only strategies with `s + t ≤ 3` are evaluated (direct lighting only).

## 2026-03-06 - BDPT MIS Weight Delta-Endpoint Fix

**Bug found and fixed — MIS weight function treating subpath-origin deltas the same as surface deltas:**
- Root cause: the `mis_weight` power-heuristic walk used `v.is_delta` to decide whether to skip ratio terms for alternative strategies. For the pinhole camera (`z[0].is_delta = true`) and point light (`y[0].is_delta = true`), this zeroed out the ratio for strategies `(s=1, t=2)` and `(s=2, t=1)` respectively. Both strategies then received MIS weight = 1.0, effectively doubling the energy for direct-lighting paths.
- The key insight: Camera and Light endpoint vertices have **dedicated sampling routines** (`sample_we` for camera, `Light::sample` / NEE for lights) that handle their delta nature. These strategies ARE valid and must compete in MIS. Only delta *surface* vertices (specular BSDFs) are genuinely unreachable as generic connection endpoints.
- Fix: introduced `is_mis_delta()` helper that returns `false` for `Camera` and `Light` vertex types (their strategies exist), and `v.is_delta` only for `Surface` vertices. Also set `prev_delta = true` for `s=0` and `t=0` strategies since those are not yet implemented, correctly excluding them from the MIS sum.
- Removed incorrect `assert!(sum_ri > 0.0)` — `sum_ri == 0` is legitimate when the only alternative strategies involve unimplemented paths (s=0, t=0).