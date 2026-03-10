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

## 2026-03-07 - Sampler Trait Abstraction

Introduced a `Sampler` trait (`src/sampler.rs`) to decouple integrators from the raw RNG, enabling future use of blue-noise, Sobol, or stratified samplers.

**`src/sampler.rs`:**
- `Sampler` trait (object-safe, `Send`): `next_1d() -> f32`, `next_2d() -> Vec2`, `start_next_sample()`, `clone_for_pixel(x, y, sample_index) -> Box<dyn Sampler>`.
- `IndependentSampler`: wraps `SmallRng` for plain i.i.d. uniform samples. `seeded_for_pixel(x, y)` reproduces the same deterministic hash the integrators were using, so renders are bitwise identical.

**Integrator refactoring:**
- `PathTracer`: `li()` now takes `&mut dyn Sampler` instead of `&mut SmallRng`. All `rng.random()` / `rng.random::<f32>()` calls replaced with `sampler.next_1d()` / `sampler.next_2d()`. Render loop creates an `IndependentSampler` per pixel.
- `BidirectionalPathTracer`: `generate_camera_subpath` and `generate_light_subpath` now take `&mut dyn Sampler`. All internal RNG usage replaced with sampler calls. Render loop updated similarly.
- Removed direct `rand::rngs::SmallRng` / `rand::{RngExt, SeedableRng}` imports from both integrators.

## 2026-03-07 - Film Class with Atomic Float Accumulation

Added `src/film.rs` — a proper film/framebuffer abstraction with lock-free atomic pixel accumulation.

**`AtomicF32`:** a `#[repr(transparent)]` wrapper around `AtomicU32` that stores `f32` bits and provides `fetch_add` via a CAS loop. No mutex overhead.

**`AtomicPixel`:** three `AtomicF32` channels (R, G, B) with `add(Vec3A)` and `load() -> Vec3A`.

**`Film`:**
- `new(width, height)` — creates a black framebuffer.
- `add_sample(x, y, value)` / `add_splat(index, value)` — atomic accumulation safe from any thread.
- `get_pixel(x, y)` / `get_pixel_by_index(i)` — read accumulated values.
- `to_hdr_vec() -> Vec<Vec3A>` — snapshot for legacy code paths.
- `to_rgb_image(tone_mapper, gamma, scale) -> image::RgbImage` — tone-map + gamma-encode to LDR in one call.

**`ToneMapper` enum:** `Clamp`, `Reinhard` (luminance-preserving), `ReinhardExtended { white_point }`.

**Integrator refactoring:**
- `Integrator::render` now returns `Arc<Film>` instead of `Vec<Vec3A>`.
- `PathTracer`: accumulates into `Film` via `add_sample`; parallel iteration uses `into_par_iter` over rows.
- `BidirectionalPathTracer`: replaced all `Mutex<Vec3A>` splat/strategy buffers with `Film` instances. Splat film for s=1 light tracing and per-strategy debug films all use lock-free atomics.
- `dump_strategy_images` now takes `&[Film]` and calls `film.to_rgb_image(Reinhard, 2.2, inv_spp)`.

**Bin test cleanup:**
- `pt_test.rs` / `bdpt_test.rs`: removed manual `tonemap()` functions and byte-buffer encoding; replaced with `film.to_rgb_image(ToneMapper::Reinhard, 2.2, 1.0).save(path)`.

## 2026-03-07 - OpenEXR Export

Added `Film::save_exr(path, scale)` for writing linear HDR framebuffers as compressed OpenEXR files.

- Uses the `exr` crate directly (already a transitive dependency of `image`) for full control over encoding.
- Writes half-float (f16) RGB channels with **ZIP16** compression — standard for VFX beauty passes.
- `scale` parameter allows `1/spp` normalisation before writing.
- Added `exr` and `smallvec` as direct dependencies in `Cargo.toml`.
- Integrated into `pt_test.rs` — now saves both `pt_test.png` and `pt_test.exr`.
- Compression reduces file size by ~2.6× vs uncompressed (13 MB → 5 MB for 1600×720).

## 2026-03-07 - Gradient-Domain Path Tracer

Implemented a gradient-domain path tracer (G-PT) using random replay shift mapping.

**New files:**
- `src/integrators/gradient_domain_pt.rs` — full G-PT integrator with:
  - `GdptConfig`: spp, max_depth, rr_depth, screening weight α, Poisson iteration count, SOR ω.
  - `ReplaySampler`: wraps `IndependentSampler` with a record/replay tape mechanism. In record mode, every `next_1d()` call is logged; in replay mode, the tape is played back identically so that offset paths share the same random decisions as the base path.
  - `trace_with_gradients()`: traces one base path and two offset paths (x+1, y+1) per sample using the replay sampler, computing forward-difference gradients ΔIx = I(x+1,y) − I(x,y) and ΔIy = I(x,y+1) − I(x,y).
  - Screened Poisson reconstruction via SOR: solves argmin_I α‖I − I_primal‖² + ‖∇I − G‖² using red-black SOR on the 5-point Laplacian stencil. Pre-computes RHS (α·primal + div G) and iterates with configurable ω and iteration count.
- `bin/gdpt_test.rs` — test binary loading the fireplace room scene and rendering with G-PT at 16 spp.

**Modified files:**
- `src/integrators/mod.rs` — registered `gradient_domain_pt` module, exported `GradientDomainPathTracer` and `GdptConfig`.
- `Cargo.toml` — added `gdpt_test` binary target.

**Design notes:**
- Random replay shift mapping is always invertible (no reconnection needed), making the implementation simpler than half-vector copy or manifold exploration shifts.
- Only forward differences (x+1, y+1) are traced; backward shifts could be added later for symmetric gradient estimation at the cost of 2 additional path traces per sample.
- The Poisson solver uses warm-start from the primal image and red-black ordering for faster convergence.

## 2026-03-08 - BDPT s=1 Camera Importance Bug Fix
Fixed a bug in the `s=1` (light tracing to camera) strategy in the bidirectional path tracer that caused slightly brighter images compared to the `s=2,t=1` (NEE) strategy under uniform MIS weights.

**Root cause:** In `connect_bdpt_s1`, the geometric coupling between a light-subpath surface vertex and the camera was missing the `|cos θ_cam|` factor (the angle between the camera forward and the connection direction). The camera's importance `We = 1/(A·cos⁴θ)` does NOT absorb this factor — the measurement equation is `I_j = ∫ We · Li · |cos θ| dω`, where `|cos θ|` is separate. The missing factor caused the s=1 contribution to be scaled by `1/cos θ` (always ≥ 1), making it brighter, especially toward image edges.

**Fix:** Changed `CameraWeSample::we` to return `We · |cos θ| = 1/(A·cos³θ)` instead of the raw `1/(A·cos⁴θ)`. This folds the measurement cosine into the importance value, matching the camera's solid-angle sampling PDF (`pdf_we`). The `connect_bdpt_s1` formula now correctly computes `contribution = throughput · f · cos_surface/dist² · we` without needing a separate cos θ_cam factor.

## 2026-03-08 - BDPT MIS Weight Bugs Fixed
Fixed two bugs in the MIS weight computation that caused the combined Power-heuristic image to be brighter than the reference.

**Bug 1 — Wrong `pdf_connect_light` in `connect_bdpt_s1`:**
`pdf_connect_light` is the reverse PDF at y[t-1] — the area-measure probability of generating y[t-1] from the camera side. For the s=1 strategy, shifting to (s+1=2, t-1) means y[t-1] becomes z[1], the first camera hit, whose PDF is the camera's directional PDF (`camera.pdf_we`) converted to area at y. The code was incorrectly using the BSDF PDF at y and converting in the wrong direction (to the camera instead of to y), making the ratio too small and the s=1 MIS weight too high.

**Bug 2 — Off-by-one in MIS weight walk loops:**
The camera and light walk loops in `mis_weight` used ranges `(1..s).rev()` and `(1..t).rev()`, which re-processed the connection vertex already handled by the first step (z[s-1] or y[t-1]). Changed to `(0..s-1).rev()` and `(0..t-1).rev()` so each walk step processes the correct next vertex (z[s-2], z[s-3], ... and y[t-2], y[t-3], ...). For max_depth=1 this only added zero terms (pdf_rev=0 at leaf vertices), but for longer paths it would corrupt MIS weights.

## 2026-03-08 - PSSMLT (Primary Sample Space Metropolis Light Transport)

Implemented a full PSSMLT integrator in `src/integrators/pssmlt.rs`.

**New files:**
- `src/integrators/pssmlt.rs` — complete PSSMLT implementation with:
  - `PssmltConfig`: spp, max_depth, rr_depth, n_bootstrap, n_chains, large_step_prob, sigma.
  - `MCMCSampler`: a lazy N-dimensional primary-sample-space sampler implementing the `Sampler` trait. Supports two mutation strategies:
    - **Large step (independent):** replaces the entire random vector with fresh i.i.d. uniform variates.
    - **Small step (dependent):** perturbs each component via `fract(old + σ · N(0,1))` using Box-Muller Gaussian noise.
    - The dimension N is dynamic — each `next_1d()` call either reads an existing dimension or extends the vector lazily.
    - `start_proposal(large)` / `accept()` / `reject()` manage the proposal–state lifecycle.
  - **Bootstrap phase:** traces `n_bootstrap` independent paths in parallel, computes their luminance, builds a CDF, and importance-resamples `n_chains` starting states weighted by luminance.
  - **Chain phase:** runs `n_chains` independent Markov chains in parallel (via rayon). Each chain performs `mutations_per_chain` iterations where `n_chains × mutations_per_chain / n_pixels = spp`. At each iteration, a large or small step is chosen randomly according to `large_step_prob`. Proposals are accepted/rejected via the Metropolis ratio `min(1, f_new/f_old)`. Both accepted and rejected paths deposit contributions to the film using the standard Metropolis estimator weighting.
  - **Normalisation:** large-step luminances are accumulated into a global atomic counter. The final image is scaled by `b_avg × n_pixels / total_mutations` to recover correct absolute brightness.
  - `AtomicF64` helper for lock-free double-precision accumulation (CAS loop on `AtomicU64` bits).
- `bin/pssmlt_test.rs` — test binary rendering the fireplace room scene with PSSMLT at 16 spp, 256 chains.

**Modified files:**
- `src/integrators/mod.rs` — registered `pssmlt` module, exported `Pssmlt` and `PssmltConfig`.
- `Cargo.toml` — added `pssmlt_test` binary target.
- `design/roadmap.md` — checked off PSSMLT milestone.

## 2026-03-09 - Scene Serialization (serde + serde_json)
Implemented JSON-based scene serialization and deserialization in `src/scene_format.rs`.

**New module: `src/scene_format.rs`**
- `SceneDescription` — top-level serializable struct containing materials, meshes, objects, lights, and camera.
- `MaterialDesc` — enum with `Lambertian { albedo }`, `Conductor { f0, roughness }`, `Dielectric { eta, roughness }` variants.
- `MeshDesc` — flat-array representation of `TriangleMesh` (positions, normals, tex_coords, indices, material_slots) with `from_mesh()` / `build()` round-trip methods.
- `ObjectDesc` — links a mesh index to material indices with an optional 4×4 transform.
- `LightDesc` — enum starting with `Point { position, intensity }` (extensible for future light types).
- `CameraDesc` — enum with `PinholeLookAt` and `PinholeEyeAngle` variants.
- `SceneDescription::build()` — reconstructs a runtime `Scene` + `PinholeCamera` from the description, using `PowerLightDistribution` when lights are present.
- `SceneDescription::from_scene()` — captures runtime scene objects into the serializable format (mesh deduplication by `Arc` pointer identity; materials stored as placeholder Lambertian since `dyn Bsdf` cannot be introspected).
- `save_scene_file()` / `load_scene_file()` — pretty-printed JSON file I/O.
- `load_and_build_scene()` — convenience one-shot load + build.

**Modified files:**
- `Cargo.toml` — added `serde = { version = "1", features = ["derive"] }` and `serde_json = "1"`.
- `src/lib.rs` — registered `scene_format` module.
- `design/roadmap.md` — checked off scene serialization milestone.

## 2026-03-09 - OBJ Importer CLI
Added `bin/obj_importer.rs` — a CLI tool that imports OBJ files into a scene JSON file.

**New file: `bin/obj_importer.rs`**
- Accepts one or more `.obj` files and an `--output <SCENE.json>` path.
- If the scene file exists, new meshes/materials/objects are **appended** to it.
- If the scene file does not exist, a fresh `SceneDescription` with default camera is created.
- `--overwrite` flag forces creation of a new scene even when the file already exists.
- Proper MTL→MaterialDesc conversion (Lambertian / Conductor / Dielectric) using the same priority rules as `importer.rs`.

**New in `src/scene_format.rs`**
- `obj_import` sub-module with `import_obj_to_scene_desc()`: loads an OBJ file via `tobj`, converts MTL materials directly to `MaterialDesc` enums, and appends meshes + objects + materials to a mutable `SceneDescription`.
- Re-exported as `scene_format::import_obj_to_scene_desc`.

**Modified files:**
- `Cargo.toml` — added `obj_importer` binary target.

## 2026-03-09 - Binary-backed scene storage
Replaced inline JSON mesh data with a companion `.bin` file for geometry buffers.  The fireplace room scene JSON dropped from ~90+ MB to ~38 KB; geometry lives in a 15 MB `.bin`.

**Redesigned `src/scene_format.rs`**
- `BinBuffer` — in-memory accumulator backed by a `Vec<u8>`.  Provides `append_f32s()` / `append_u32s()` that return a `BufferRef { offset, length }`, plus `read_f32s()` / `read_u32s()` for the inverse.  Supports `from_file()` to pre-load an existing `.bin` for append workflows.
- `BufferRef` — serializable `{ offset, length }` byte range into the `.bin`.
- `MeshDesc` now stores `BufferRef` fields (positions, normals, tex_coords, indices) instead of inline `Vec<f32>` / `Vec<u32>`.  `material_slots` remains inline (always small).
- `MeshDesc::from_mesh()` takes `&mut BinBuffer`, writes geometry, returns refs.
- `MeshDesc::build()` takes `&BinBuffer`, reads geometry back.
- `SceneDescription` gained a `buffer_file: String` field (relative path to the `.bin`).
- `SceneDescription::build_with_dir(scene_dir)` resolves the `.bin` relative to the JSON and loads it.
- `save_scene_file()` now takes `&mut SceneDescription` + `&BinBuffer` and writes both the JSON and the `.bin`.
- `load_bin_buffer()` — load the companion `.bin` for an existing scene (used by append workflow).
- `bin_path_for()` — helper to derive `.bin` path from `.json` path.
- `import_obj_to_scene_desc()` now takes `&mut BinBuffer` and writes geometry directly to it.

**Updated `bin/obj_importer.rs`**
- Now loads/creates both the JSON and the `BinBuffer`.
- On append: pre-loads the existing `.bin` via `load_bin_buffer()`, new geometry is appended after the existing data.
- Prints binary buffer size in output.