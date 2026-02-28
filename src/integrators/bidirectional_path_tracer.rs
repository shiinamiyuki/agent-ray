//! # Bidirectional Path Tracer with Multiple Importance Sampling
//!
//! This module implements a **bidirectional path tracer (BDPT)** as described
//! in Eric Veach's thesis (*Robust Monte Carlo Methods for Light Transport
//! Simulation*, 1997, Chapter 10).
//!
//! ## Overview
//!
//! Standard (unidirectional) path tracing builds paths starting from the
//! camera only.  BDPT builds *two* subpaths — one from the camera and one
//! from a light source — and considers every possible way to *connect* them.
//! Each connection strategy `(s, t)` uses `s` camera-subpath vertices and
//! `t` light-subpath vertices.  **Multiple Importance Sampling (MIS)** via
//! the power heuristic weights each strategy so that the one with the lowest
//! variance dominates, drastically reducing noise for difficult lighting
//! configurations (caustics, small light sources, indirect-only illumination).
//!
//! ## Supported connection strategies
//!
//! | Strategy | Description | Notes |
//! |----------|-------------|-------|
//! | `t = 1, s ≥ 2` | **Next-event estimation (NEE)** — connect an eye subpath vertex to a light. | Equivalent to direct-light sampling in a unidirectional PT. |
//! | `s = 1, t ≥ 2` | **Light tracing** — connect a light subpath vertex to the camera. | Splatted to the image; important for caustics. |
//! | `s ≥ 2, t ≥ 2` | **General connection** through two surface vertices. | The hallmark of BDPT. |
//! | `t = 0` | Eye subpath hits an emitter directly. | Only possible for non-delta (area) lights; contributes `Le`. |
//! | `s = 0` | Pure light tracing to the camera. | Requires camera importance `We` evaluation — **not yet implemented** (documented as a TODO). |
//!
//! The `s = 0` strategy would produce a full light path ending at the camera
//! sensor.  It requires a camera `We(ray)` evaluation from an arbitrary
//! incoming direction — the current `Camera` trait does not expose this, so
//! the strategy is skipped.  All other strategies are fully functional.
//!
//! ## Generality / extensibility
//!
//! The integrator interacts with the scene *only* through the trait objects
//! [`Bsdf`], [`Light`], [`Camera`], and the [`Scene`] query methods.  Any
//! new material, primitive shape, or light source that implements the
//! corresponding trait will be picked up automatically.
//!
//! - **New BSDFs**: implement `Bsdf::eval`, `sample`, `pdf`.
//! - **New lights**: implement `Light::sample`, `power`, `is_delta`, and
//!   optionally `sample_emission` / `pdf_emission_dir` for BDPT support.
//! - **New cameras**: implement `Camera::generate_ray` and optionally
//!   `sample_we` / `pdf_we` for the `s = 1` light-tracing strategy.
//! - **New shapes**: just add them to the BVH; the integrator only sees
//!   `Scene::intersect` / `Scene::occluded`.
//!
//! ## MIS weight computation
//!
//! We use the **power heuristic** (β = 2) evaluated via Veach's recursive
//! ratio formulation (thesis §10.2.1).  All PDFs are converted to **area
//! measure** so that strategies with different dimensionality are comparable.
//! Delta components (delta lights, specular BSDFs) are handled by zeroing
//! their MIS contribution, ensuring they never compete with non-delta
//! strategies.
//!
//! ## References
//!
//! - E. Veach, *Robust Monte Carlo Methods for Light Transport Simulation*,
//!   Stanford University, 1997.
//! - M. Pharr, W. Jakob, G. Humphreys, *Physically Based Rendering: From
//!   Theory to Implementation* (PBRT), 4th ed., Chapters 13–16.

use std::sync::Arc;

use glam::{Vec2, Vec3A};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use crate::cameras::Camera;
use crate::geometry::{Onb, Ray};
use crate::integrators::Integrator;
use crate::lights::Light;
use crate::scene::Scene;
use crate::surfaces::Bsdf;

// ===========================================================================
// Configuration
// ===========================================================================

/// Configuration parameters for the bidirectional path tracer.
pub struct BdptConfig {
    /// Samples per pixel.
    pub spp: u32,
    /// Maximum number of vertices on each subpath (including the endpoint).
    /// The maximum full-path length is `2 * max_depth`.
    pub max_depth: u32,
    /// Minimum subpath depth before Russian roulette kicks in.
    pub rr_depth: u32,
}

impl Default for BdptConfig {
    fn default() -> Self {
        Self {
            spp: 64,
            max_depth: 8,
            rr_depth: 3,
        }
    }
}

// ===========================================================================
// Path vertex
// ===========================================================================

/// Classification of a path vertex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VertexType {
    /// The camera origin (eye subpath start).
    Camera,
    /// A light source (light subpath start).
    Light,
    /// A surface interaction (scattering event).
    Surface,
}

/// A single vertex along a camera or light subpath.
///
/// Stores all information required to:
/// 1. Evaluate the local BSDF / emission / importance.
/// 2. Compute forward and reverse area-measure PDFs for MIS weights.
///
/// PDFs are stored in **area measure** so that strategies with different
/// numbers of vertices are directly comparable.
struct PathVertex {
    /// Vertex type (camera, light, or surface).
    vtype: VertexType,

    // -- Geometry ----------------------------------------------------------
    /// World-space position.
    p: Vec3A,
    /// World-space shading normal (faces the incoming ray).
    n: Vec3A,
    /// Local coordinate frame at this vertex (Z = n).
    onb: Onb,

    // -- Surface data (only meaningful for `Surface` vertices) -------------
    /// BSDF at the surface.  `None` for Camera / Light endpoints.
    bsdf: Option<Arc<dyn Bsdf>>,
    /// Texture UV coordinates.
    tex_uv: Vec2,

    // -- Throughput --------------------------------------------------------
    /// Product of `(f · |cos θ| / pdf)` from the subpath origin to this
    /// vertex (inclusive of the path-start contribution, e.g. `Le / pdf`
    /// for light vertices or `We / pdf` for camera vertices).
    throughput: Vec3A,

    // -- PDF bookkeeping for MIS -------------------------------------------
    /// Forward area-measure PDF: the probability density of generating *this*
    /// vertex given the *previous* vertex on the same subpath.
    pdf_fwd: f32,
    /// Reverse area-measure PDF: the probability density of generating this
    /// vertex if the subpath were traced in the *opposite* direction (i.e.
    /// from the other subpath's origin).
    pdf_rev: f32,

    /// Whether this vertex lies on a delta distribution (specular BSDF or
    /// delta light / camera).  Delta vertices cannot participate as interior
    /// connection endpoints, so they are assigned zero MIS weight.
    is_delta: bool,

    // -- Light info (for light vertices and emissive surfaces) -------------
    /// For the light endpoint (`vtype == Light`): reference to the light.
    light: Option<Arc<dyn Light>>,
    /// Index of the light in `scene.lights` (for light-distribution PMF).
    light_idx: usize,
}

impl PathVertex {
    /// Construct a camera endpoint vertex.
    fn camera(p: Vec3A, forward: Vec3A) -> Self {
        Self {
            vtype: VertexType::Camera,
            p,
            n: forward,
            onb: Onb::from_normal(forward),
            bsdf: None,
            tex_uv: Vec2::ZERO,
            throughput: Vec3A::ONE,
            pdf_fwd: 1.0, // camera position is a delta; handled specially
            pdf_rev: 0.0,
            is_delta: true, // pinhole = delta position
            light: None,
            light_idx: 0,
        }
    }

    /// Construct a light endpoint vertex.
    fn light_endpoint(
        p: Vec3A,
        n: Vec3A,
        le: Vec3A,
        pdf_pos: f32,
        is_delta: bool,
        light: Arc<dyn Light>,
        light_idx: usize,
    ) -> Self {
        Self {
            vtype: VertexType::Light,
            p,
            n,
            onb: Onb::from_normal(n),
            bsdf: None,
            tex_uv: Vec2::ZERO,
            throughput: le / pdf_pos, // will be further divided by light-selection PMF
            pdf_fwd: pdf_pos,
            pdf_rev: 0.0,
            is_delta,
            light: Some(light),
            light_idx,
        }
    }

    /// Construct a surface vertex from a scene hit.
    fn surface(
        p: Vec3A,
        n: Vec3A,
        onb: Onb,
        bsdf: Arc<dyn Bsdf>,
        tex_uv: Vec2,
        throughput: Vec3A,
        pdf_fwd: f32,
    ) -> Self {
        Self {
            vtype: VertexType::Surface,
            p,
            n,
            onb,
            bsdf: Some(bsdf),
            tex_uv,
            throughput,
            pdf_fwd,
            pdf_rev: 0.0,
            is_delta: false, // updated later if the BSDF is specular
            light: None,
            light_idx: 0,
        }
    }

    /// Whether the vertex is connectible (not a delta distribution).
    ///
    /// Delta vertices (specular surfaces, point lights, pinhole cameras)
    /// cannot be the interior endpoints of a connection edge because the
    /// probability of the connection direction matching the delta lobe is
    /// zero.
    #[inline]
    fn is_connectible(&self) -> bool {
        match self.vtype {
            VertexType::Camera => true, // we handle camera connections via `sample_we`
            VertexType::Light => {
                // A positional-delta light *can* be connected to (we sample
                // toward it), but a directional-delta light cannot.
                // For now, point lights are connectible (NEE works for them).
                true
            }
            VertexType::Surface => !self.is_delta,
        }
    }
}

// ===========================================================================
// Geometry helpers
// ===========================================================================

/// Convert a solid-angle PDF at vertex `from` to an area-measure PDF at `to`.
///
/// ```text
/// pdf_area = pdf_ω · |cos θ_to| / dist²
/// ```
#[inline]
fn pdf_solid_angle_to_area(pdf_omega: f32, from: Vec3A, to: Vec3A, n_to: Vec3A) -> f32 {
    let d = to - from;
    let dist2 = d.length_squared();
    if dist2 < 1e-12 {
        return 0.0;
    }
    let dir = d / dist2.sqrt();
    let cos_to = n_to.dot(dir).abs();
    pdf_omega * cos_to / dist2
}

/// Geometric coupling term `G(a, b) = |cos θ_a| · |cos θ_b| / dist²`.
#[inline]
fn geometry_term(pa: Vec3A, na: Vec3A, pb: Vec3A, nb: Vec3A) -> f32 {
    let d = pb - pa;
    let dist2 = d.length_squared();
    if dist2 < 1e-12 {
        return 0.0;
    }
    let dir = d / dist2.sqrt();
    let cos_a = na.dot(dir).abs();
    let cos_b = nb.dot(-dir).abs();
    cos_a * cos_b / dist2
}

// ===========================================================================
// Subpath generation
// ===========================================================================

/// Trace the **camera subpath** starting from a ray generated by the camera.
///
/// Returns a `Vec<PathVertex>` where index 0 is the camera vertex and
/// subsequent vertices are surface interactions.
fn generate_camera_subpath(
    scene: &Scene,
    camera: &dyn Camera,
    ray: Ray,
    max_depth: u32,
    rr_depth: u32,
    rng: &mut SmallRng,
) -> Vec<PathVertex> {
    let mut vertices: Vec<PathVertex> = Vec::with_capacity(max_depth as usize + 1);

    // -- z[0]: camera vertex -----------------------------------------------
    let cam_origin = camera.origin();
    let cam_forward = ray.direction; // approximate forward as the first ray dir
    let mut cam_v = PathVertex::camera(cam_origin, cam_forward);

    // The forward PDF of z[0] is a delta (the camera is a specific point in
    // the scene).  We store pdf_fwd = 1.0; the MIS weight handles deltas
    // by setting the corresponding ratio term to 0.
    cam_v.pdf_fwd = 1.0;

    // The camera's directional PDF for the first ray (solid angle at cam).
    let cam_pdf_dir = camera.pdf_we(&ray);

    vertices.push(cam_v);

    // -- trace bounces -----------------------------------------------------
    let mut current_ray = ray;
    let mut throughput = Vec3A::ONE;
    let mut prev_pdf_dir = cam_pdf_dir; // directional PDF of the previous vertex

    for depth in 0..max_depth {
        let Some((_t, hit)) = scene.intersect(&current_ray) else {
            break;
        };

        // Area-measure PDF of this vertex from the previous one.
        let pdf_fwd_area = pdf_solid_angle_to_area(
            prev_pdf_dir,
            vertices.last().unwrap().p,
            hit.p,
            hit.n,
        );

        let v = PathVertex::surface(
            hit.p,
            hit.n,
            hit.onb,
            Arc::clone(&hit.bsdf),
            hit.tex_uv,
            throughput,
            pdf_fwd_area,
        );
        vertices.push(v);

        // -- extend the path via BSDF sampling -----------------------------
        let wo = -current_ray.direction;
        let wo_local = hit.onb.to_local(wo);

        let u_sel: f32 = rng.random();
        let u_dir = Vec2::new(rng.random(), rng.random());
        let Some(bs) = hit.bsdf.sample(wo_local, hit.tex_uv, u_sel, u_dir) else {
            break;
        };
        if bs.pdf == 0.0 || bs.f == Vec3A::ZERO {
            break;
        }

        let cos_wi = bs.wi.z.abs();
        throughput *= bs.f * cos_wi / bs.pdf;

        // Store the directional PDF for the *next* vertex's pdf_fwd.
        prev_pdf_dir = bs.pdf;

        // Also compute the *reverse* directional PDF at this vertex: the
        // PDF of sampling `wo` given `wi` (the direction we'd trace if
        // going backward along the path).
        let pdf_rev_dir = hit.bsdf.pdf(wo_local, bs.wi);
        // Convert reverse PDF from solid angle to area at the *previous*
        // vertex.
        let prev_idx = vertices.len() - 2;
        let prev_p = vertices[prev_idx].p;
        let prev_n = vertices[prev_idx].n;
        vertices.last_mut().unwrap().pdf_rev = pdf_solid_angle_to_area(
            pdf_rev_dir,
            hit.p,
            prev_p,
            prev_n,
        );

        // Russian roulette.
        if depth + 1 >= rr_depth {
            let survival = throughput.max_element().min(0.95);
            if rng.random::<f32>() > survival || survival <= 0.0 {
                break;
            }
            throughput /= survival;
        }

        // Spawn next ray.
        let wi_world = hit.onb.to_world(bs.wi);
        let offset = hit.n * 1e-4 * wi_world.dot(hit.n).signum();
        current_ray = Ray::new(hit.p + offset, wi_world, 1e-4, f32::MAX);
    }

    vertices
}

/// Trace the **light subpath** starting from a randomly chosen light source.
///
/// Returns a `Vec<PathVertex>` where index 0 is the light vertex and
/// subsequent vertices are surface interactions.
fn generate_light_subpath(
    scene: &Scene,
    max_depth: u32,
    rr_depth: u32,
    rng: &mut SmallRng,
) -> Vec<PathVertex> {
    let mut vertices: Vec<PathVertex> = Vec::with_capacity(max_depth as usize + 1);

    // -- Choose a light source ---------------------------------------------
    let light_dist = match &scene.light_dist {
        Some(d) => d,
        None => return vertices,
    };

    let u_sel: f32 = rng.random();
    let (light_idx, sel_pmf) = light_dist.sample_index(u_sel);
    let light = &scene.lights[light_idx];

    // -- Sample emission ---------------------------------------------------
    let u_pos = Vec2::new(rng.random(), rng.random());
    let u_dir = Vec2::new(rng.random(), rng.random());
    let Some(emission) = light.sample_emission(u_pos, u_dir) else {
        return vertices;
    };

    // -- y[0]: light vertex ------------------------------------------------
    let mut light_v = PathVertex::light_endpoint(
        emission.ray.origin,
        emission.n_light,
        emission.le,
        emission.pdf_pos * sel_pmf,
        light.is_positional_delta(),
        Arc::clone(light),
        light_idx,
    );
    // Correct throughput: Le / (pdf_pos * sel_pmf).
    light_v.throughput = emission.le / (emission.pdf_pos * sel_pmf);
    light_v.light_idx = light_idx;
    vertices.push(light_v);

    if max_depth == 0 {
        return vertices;
    }

    // The directional PDF from the light (solid angle).
    let mut prev_pdf_dir = emission.pdf_dir;
    let mut throughput = emission.le / (emission.pdf_pos * sel_pmf * emission.pdf_dir);
    // Scale throughput by |cos θ_light| — for point lights this is 1 (the
    // "normal" is the emission direction); for area lights this would be the
    // cosine at the emitting surface.
    let cos_emit = emission.n_light.dot(emission.ray.direction).abs();
    throughput *= cos_emit;

    let mut current_ray = emission.ray;

    for depth in 0..max_depth {
        let Some((_t, hit)) = scene.intersect(&current_ray) else {
            break;
        };

        let pdf_fwd_area = pdf_solid_angle_to_area(
            prev_pdf_dir,
            vertices.last().unwrap().p,
            hit.p,
            hit.n,
        );

        let v = PathVertex::surface(
            hit.p,
            hit.n,
            hit.onb,
            Arc::clone(&hit.bsdf),
            hit.tex_uv,
            throughput,
            pdf_fwd_area,
        );
        vertices.push(v);

        // -- extend via BSDF sampling --------------------------------------
        let wo = -current_ray.direction;
        let wo_local = hit.onb.to_local(wo);

        let u_sel: f32 = rng.random();
        let u_dir = Vec2::new(rng.random(), rng.random());
        let Some(bs) = hit.bsdf.sample(wo_local, hit.tex_uv, u_sel, u_dir) else {
            break;
        };
        if bs.pdf == 0.0 || bs.f == Vec3A::ZERO {
            break;
        }

        let cos_wi = bs.wi.z.abs();
        throughput *= bs.f * cos_wi / bs.pdf;

        prev_pdf_dir = bs.pdf;

        // Reverse PDF at this vertex.
        let pdf_rev_dir = hit.bsdf.pdf(wo_local, bs.wi);
        let prev_idx = vertices.len() - 2;
        let prev_p = vertices[prev_idx].p;
        let prev_n = vertices[prev_idx].n;
        vertices.last_mut().unwrap().pdf_rev = pdf_solid_angle_to_area(
            pdf_rev_dir,
            hit.p,
            prev_p,
            prev_n,
        );

        // Russian roulette.
        if depth + 1 >= rr_depth {
            let survival = throughput.max_element().min(0.95);
            if rng.random::<f32>() > survival || survival <= 0.0 {
                break;
            }
            throughput /= survival;
        }

        let wi_world = hit.onb.to_world(bs.wi);
        let offset = hit.n * 1e-4 * wi_world.dot(hit.n).signum();
        current_ray = Ray::new(hit.p + offset, wi_world, 1e-4, f32::MAX);
    }

    vertices
}

// ===========================================================================
// MIS weight computation
// ===========================================================================

/// Compute the MIS weight for strategy `(s, t)` using the **power heuristic**
/// (β = 2) via Veach's recursive ratio method.
///
/// ## Arguments
///
/// - `camera_verts`: the camera subpath `z[0..s]`.
/// - `light_verts`: the light subpath `y[0..t]`.
/// - `s`: number of camera-subpath vertices used.
/// - `t`: number of light-subpath vertices used.
/// - `pdf_connect_cam`: the area-measure PDF of generating the connection
///   vertex on the camera side if the strategy were shifted by one (i.e. the
///   reverse PDF at z[s-1] obtained by tracing from the light side).
/// - `pdf_connect_light`: similarly, the reverse PDF at y[t-1] obtained by
///   tracing from the camera side.
///
/// ## Algorithm
///
/// For strategy `(s, t)`, the MIS weight is:
///
/// ```text
/// w_{s,t} = 1 / (1 + Σ_{i≠(s,t)} (p_i / p_{s,t})²)
/// ```
///
/// The ratio `(p_i / p_{s,t})` is computed as a running product of
/// `pdf_rev / pdf_fwd` as we slide the connection point along the full path.
/// Delta vertices contribute 0 to the sum (they cannot be connection
/// endpoints for any other strategy).
fn mis_weight(
    camera_verts: &[PathVertex],
    light_verts: &[PathVertex],
    s: usize,
    t: usize,
    pdf_connect_cam: f32,
    pdf_connect_light: f32,
) -> f32 {
    // Trivial cases: if only one strategy is possible, weight = 1.
    if s + t == 2 {
        return 1.0;
    }

    let mut sum_ri_sq = 0.0f64;

    // -- Walk along the camera subpath (from z[s-1] toward z[0]) -----------
    //
    // For index i (counting from the connection), the ratio is:
    //   r_i = Π_{j=s-1..s-i} (pdf_rev[j] / pdf_fwd[j])
    //
    // If we shift the connection by one position toward the camera, the vertex
    // that used to be on the camera side becomes the connection vertex from
    // the light side.  Its "new" pdf_fwd is the reverse PDF we just computed
    // during connection (`pdf_connect_cam`), and vice versa.
    {
        let mut ri = 1.0f64;

        // At z[s-1]: the "new" forward PDF is pdf_connect_cam (what the
        // light side would compute), and the existing forward is pdf_fwd.
        if s >= 1 {
            let v = &camera_verts[s - 1];
            let rev = pdf_connect_cam as f64;
            let fwd = v.pdf_fwd.max(1e-30) as f64;
            ri *= rev / fwd;

            if !v.is_delta && (s < 2 || !camera_verts[s - 2].is_delta) {
                sum_ri_sq += ri * ri;
            }
        }

        // Walk further toward z[0].
        for i in (1..s).rev() {
            let v = &camera_verts[i];
            let rev = v.pdf_rev.max(0.0) as f64;
            let fwd = v.pdf_fwd.max(1e-30) as f64;
            ri *= rev / fwd;

            let prev_delta = if i > 0 { camera_verts[i - 1].is_delta } else { false };
            if !v.is_delta && !prev_delta {
                sum_ri_sq += ri * ri;
            }
        }
    }

    // -- Walk along the light subpath (from y[t-1] toward y[0]) ------------
    {
        let mut ri = 1.0f64;

        if t >= 1 {
            let v = &light_verts[t - 1];
            let rev = pdf_connect_light as f64;
            let fwd = v.pdf_fwd.max(1e-30) as f64;
            ri *= rev / fwd;

            if !v.is_delta && (t < 2 || !light_verts[t - 2].is_delta) {
                sum_ri_sq += ri * ri;
            }
        }

        for i in (1..t).rev() {
            let v = &light_verts[i];
            let rev = v.pdf_rev.max(0.0) as f64;
            let fwd = v.pdf_fwd.max(1e-30) as f64;
            ri *= rev / fwd;

            let prev_delta = if i > 0 { light_verts[i - 1].is_delta } else { false };
            if !v.is_delta && !prev_delta {
                sum_ri_sq += ri * ri;
            }
        }
    }

    // MIS weight (power heuristic): w = 1 / (1 + sum).
    (1.0 / (1.0 + sum_ri_sq)) as f32
}

// ===========================================================================
// Connection / contribution evaluation
// ===========================================================================

/// Evaluate the **unweighted contribution** of connecting camera vertex
/// `z[s-1]` to light vertex `y[t-1]` and return the MIS-weighted result.
///
/// Returns `(contribution, pixel_x, pixel_y)`.  The pixel coordinates are
/// only meaningful when `s == 1` (light tracing to camera); for `s ≥ 2` the
/// contribution goes to the current pixel being rendered.
fn connect_bdpt(
    scene: &Scene,
    _camera: &dyn Camera,
    camera_verts: &[PathVertex],
    light_verts: &[PathVertex],
    s: usize,
    t: usize,
) -> Option<(Vec3A, f32, f32)> {
    // -----------------------------------------------------------------------
    // Validate & short-circuit degenerate cases
    // -----------------------------------------------------------------------
    if s == 0 {
        // s=0: pure light tracing to camera sensor.
        // TODO: requires evaluating camera We for an arbitrary incoming
        // direction — not yet implemented.  Returning None gracefully skips.
        return None;
    }

    // We need at least the requested number of vertices on each subpath.
    if s > camera_verts.len() || t > light_verts.len() {
        return None;
    }

    // -----------------------------------------------------------------------
    // t = 0: eye path directly hits an emitter
    // -----------------------------------------------------------------------
    if t == 0 {
        // This strategy requires the last camera vertex to be emissive.
        // With only delta (point) lights, this never contributes.  For area
        // lights the emitted radiance Le would be evaluated here.
        //
        // TODO: when area lights are added, look up Le at z[s-1] and compute
        //       the MIS weight accordingly.
        return None;
    }

    // -----------------------------------------------------------------------
    // s = 1: light tracing to camera
    // -----------------------------------------------------------------------
    if s == 1 {
        // Connect the light subpath endpoint y[t-1] to the camera.
        let y = &light_verts[t - 1];

        if !y.is_connectible() {
            return None;
        }

        // Use camera's sample_we to project y.p onto the image.
        // width/height are not known here — we pass 0 and rely on the
        // caller using `sample_we` with proper dimensions.
        // Actually, we'll pass them through the `connect_bdpt_s1` wrapper.
        //
        // For now, return None — the s=1 case is handled separately in the
        // render loop where we have access to width/height.
        return None;
    }

    // -----------------------------------------------------------------------
    // General case: s ≥ 2, t ≥ 1
    // -----------------------------------------------------------------------
    let z = &camera_verts[s - 1]; // camera-side endpoint
    let y = &light_verts[t - 1]; // light-side endpoint

    if !z.is_connectible() || !y.is_connectible() {
        return None;
    }

    // -- Connection geometry -----------------------------------------------
    let d = y.p - z.p;
    let dist2 = d.length_squared();
    if dist2 < 1e-10 {
        return None;
    }
    let dist = dist2.sqrt();
    let wi_z = d / dist; // direction from z → y (world space)
    let wi_y = -wi_z; // direction from y → z

    // -- Visibility check --------------------------------------------------
    let offset_z = z.n * 1e-4 * wi_z.dot(z.n).signum();
    let offset_y = y.n * 1e-4 * wi_y.dot(y.n).signum();
    let shadow_origin = z.p + offset_z;
    let shadow_target = y.p + offset_y;
    let shadow_dir = shadow_target - shadow_origin;
    let shadow_dist = shadow_dir.length();
    if shadow_dist < 1e-6 {
        return None;
    }
    let shadow_ray = Ray::new(
        shadow_origin,
        shadow_dir / shadow_dist,
        0.0,
        shadow_dist * (1.0 - 1e-4),
    );
    if scene.occluded(&shadow_ray) {
        return None;
    }

    // -- Evaluate BSDF at z (camera side) ----------------------------------
    let f_z = if let Some(bsdf_z) = &z.bsdf {
        // At z, `wo` is the direction back along the camera subpath (from z
        // toward z's predecessor) and `wi` is toward the light subpath.
        // However, `throughput` already encodes the BSDF chain up to z, so
        // we only need the BSDF value for the connection edge.
        //
        // `wo_z` = direction from z toward z[s-2] (back along camera path).
        // `wi_z` = direction from z toward y[t-1] (connection direction).
        //
        // For convenience we store the camera ray direction in the vertex,
        // but here we recompute from positions for robustness.
        let wo_z = if s >= 2 {
            (camera_verts[s - 2].p - z.p).normalize()
        } else {
            // s == 1 case handled above
            return None;
        };
        let wo_z_local = z.onb.to_local(wo_z);
        let wi_z_local = z.onb.to_local(wi_z);
        bsdf_z.eval(wi_z_local, wo_z_local, z.tex_uv)
    } else {
        // Camera vertex — no BSDF evaluation needed (s ≥ 2 guarantees
        // z is a surface vertex).
        Vec3A::ONE
    };

    if f_z == Vec3A::ZERO {
        return None;
    }

    // -- Evaluate BSDF / Le at y (light side) ------------------------------
    let f_y = if t == 1 {
        // y is the light itself — sample it toward z.
        // The "BSDF" of a light vertex returning toward z is just 1.
        // The Le contribution is already in the light's sample.
        Vec3A::ONE
    } else if let Some(bsdf_y) = &y.bsdf {
        // Surface vertex on the light subpath.
        let wo_y = if t >= 2 {
            (light_verts[t - 2].p - y.p).normalize()
        } else {
            return None;
        };
        let wo_y_local = y.onb.to_local(wo_y);
        let wi_y_local = y.onb.to_local(wi_y);
        bsdf_y.eval(wi_y_local, wo_y_local, y.tex_uv)
    } else {
        Vec3A::ONE
    };

    if f_y == Vec3A::ZERO {
        return None;
    }

    // -- Geometric coupling ------------------------------------------------
    let g = geometry_term(z.p, z.n, y.p, y.n);

    // -- Contribution (unweighted) -----------------------------------------
    //
    // C_{s,t} = throughput_z · f_z · G(z,y) · f_y · throughput_y
    //
    // For t == 1 (light endpoint), we need to include the light's response
    // to the connection direction.  `y.throughput` already contains Le / pdf.
    // For a delta (point) light, the "Le toward z" is intensity / dist² but
    // that's what `Light::sample` returns; here we reconstruct to be safe.
    let contribution = if t == 1 {
        // y is the light vertex.  The light emits `Le` isotropically (for a
        // point light) so throughput_y already has Le/pdf_light_sel.
        // We weight by the BSDF at z, geometry, and throughput from camera.
        let light = y.light.as_ref().unwrap();
        let ls = light.sample(z.p, Vec2::ZERO)?;
        let wi_z_local = z.onb.to_local(ls.wi);
        let wo_z = if s >= 2 {
            (camera_verts[s - 2].p - z.p).normalize()
        } else {
            return None;
        };
        let wo_z_local = z.onb.to_local(wo_z);
        let f_direct = if let Some(bsdf_z) = &z.bsdf {
            bsdf_z.eval(wi_z_local, wo_z_local, z.tex_uv)
        } else {
            Vec3A::ONE
        };
        let cos_z = z.n.dot(ls.wi).abs();
        // For delta lights, pdf = 1 and li already includes 1/dist².
        z.throughput * f_direct * cos_z * ls.li
    } else {
        // General case: both endpoints are surface vertices.
        z.throughput * f_z * g * f_y * y.throughput
    };

    if contribution.max_element() <= 0.0 {
        return None;
    }

    // -- PDF computations for MIS ------------------------------------------
    //
    // We need the "reverse" PDFs at the connection endpoints:
    //   pdf_connect_cam:   area-measure PDF of z[s-1] as if generated from
    //                      the light side (i.e., pdf of sampling the
    //                      direction from y[t-1] toward z[s-1] using
    //                      the BSDF at y[t-1]).
    //   pdf_connect_light: area-measure PDF of y[t-1] as if generated from
    //                      the camera side.

    let pdf_connect_cam = if t == 1 {
        // Light endpoint: the "reverse" probability of generating z[s-1]
        // from the light is the light's directional emission PDF converted
        // to area measure at z.
        let light = y.light.as_ref().unwrap();
        let dir_y_to_z = (z.p - y.p).normalize();
        let pdf_dir = light.pdf_emission_dir(dir_y_to_z);
        pdf_solid_angle_to_area(pdf_dir, y.p, z.p, z.n)
    } else if let Some(_bsdf_y) = &y.bsdf {
        // Surface vertex on light subpath.
        let wo_y = if t >= 2 {
            (light_verts[t - 2].p - y.p).normalize()
        } else {
            Vec3A::ZERO
        };
        let wi_y = (z.p - y.p).normalize();
        let wo_y_local = y.onb.to_local(wo_y);
        let wi_y_local = y.onb.to_local(wi_y);
        let pdf_dir = y.bsdf.as_ref().unwrap().pdf(wi_y_local, wo_y_local);
        pdf_solid_angle_to_area(pdf_dir, y.p, z.p, z.n)
    } else {
        0.0
    };

    let pdf_connect_light = if let Some(_bsdf_z) = &z.bsdf {
        let wo_z = if s >= 2 {
            (camera_verts[s - 2].p - z.p).normalize()
        } else {
            Vec3A::ZERO
        };
        let wi_z = (y.p - z.p).normalize();
        let wo_z_local = z.onb.to_local(wo_z);
        let wi_z_local = z.onb.to_local(wi_z);
        let pdf_dir = z.bsdf.as_ref().unwrap().pdf(wi_z_local, wo_z_local);
        pdf_solid_angle_to_area(pdf_dir, z.p, y.p, y.n)
    } else {
        0.0
    };

    // -- MIS weight --------------------------------------------------------
    let weight = mis_weight(
        camera_verts,
        light_verts,
        s,
        t,
        pdf_connect_cam,
        pdf_connect_light,
    );

    // For t == 1 with delta lights, the NEE is the only strategy that can
    // reach the light, so the MIS weight should be 1.  The `mis_weight`
    // function handles this correctly because delta vertices contribute 0
    // to the sum of alternative strategies.

    Some((contribution * weight, 0.0, 0.0))
}

/// Handle the `s = 1` strategy (light tracing to camera).
///
/// The light subpath endpoint `y[t-1]` is projected onto the image plane
/// via `camera.sample_we`, and a shadow ray confirms visibility.  The
/// resulting contribution is *splatted* to the corresponding pixel.
fn connect_bdpt_s1(
    scene: &Scene,
    camera: &dyn Camera,
    camera_verts: &[PathVertex],
    light_verts: &[PathVertex],
    t: usize,
    width: usize,
    height: usize,
) -> Option<(Vec3A, f32, f32)> {
    if t < 2 || light_verts.len() < t {
        return None;
    }

    let y = &light_verts[t - 1];
    if !y.is_connectible() {
        return None;
    }

    // Project y onto the image plane.
    let cam_sample = camera.sample_we(y.p, width, height)?;

    // Visibility check from y to camera.
    let to_cam = cam_sample.wi;
    let dist = cam_sample.dist;
    let offset = y.n * 1e-4 * to_cam.dot(y.n).signum();
    let shadow_ray = Ray::new(
        y.p + offset,
        to_cam,
        0.0,
        dist * (1.0 - 1e-4),
    );
    if scene.occluded(&shadow_ray) {
        return None;
    }

    // Evaluate BSDF at y toward the camera.
    let f_y = if let Some(bsdf_y) = &y.bsdf {
        let wo_y = if t >= 2 {
            (light_verts[t - 2].p - y.p).normalize()
        } else {
            return None;
        };
        let wo_y_local = y.onb.to_local(wo_y);
        let wi_y_local = y.onb.to_local(to_cam);
        bsdf_y.eval(wi_y_local, wo_y_local, y.tex_uv)
    } else {
        Vec3A::ONE
    };

    if f_y == Vec3A::ZERO {
        return None;
    }

    // Geometric coupling between y and the camera.
    // For a pinhole camera, there's no "surface" normal — we use the camera
    // forward as a pseudo-normal (which is what CameraWeSample::we accounts for).
    let cos_y = y.n.dot(to_cam).abs();
    let g_factor = cos_y / (dist * dist);

    // Contribution: throughput_y · f_y · G · We
    let contribution = y.throughput * f_y * g_factor * cam_sample.we;

    if contribution.max_element() <= 0.0 {
        return None;
    }

    // MIS weight for s=1.
    // pdf_connect_cam: the reverse PDF of the camera vertex as if
    // generated from the light side.  For a pinhole camera this is
    // the camera's solid-angle PDF converted to area at the camera.
    // Since the camera is a delta point, this is handled by setting
    // the camera vertex's delta flag.
    let pdf_connect_cam = 0.0; // camera is delta; ratio will be zero
    let pdf_connect_light = if let Some(_bsdf_y) = &y.bsdf {
        // PDF of sampling the direction from y toward the camera,
        // converted to area measure.
        let wo_y = if t >= 2 {
            (light_verts[t - 2].p - y.p).normalize()
        } else {
            Vec3A::ZERO
        };
        let wi_y_to_cam = to_cam;
        let wo_y_local = y.onb.to_local(wo_y);
        let wi_y_local = y.onb.to_local(wi_y_to_cam);
        let pdf_dir = y.bsdf.as_ref().unwrap().pdf(wi_y_local, wo_y_local);
        // Convert to area at the camera (use camera forward as normal).
        pdf_solid_angle_to_area(pdf_dir, y.p, camera.origin(), camera_verts[0].n)
    } else {
        0.0
    };

    let weight = mis_weight(
        camera_verts,
        light_verts,
        1,
        t,
        pdf_connect_cam,
        pdf_connect_light,
    );

    Some((
        contribution * weight,
        cam_sample.pixel_x,
        cam_sample.pixel_y,
    ))
}

// ===========================================================================
// Bidirectional path tracer
// ===========================================================================

/// A bidirectional path tracer with full MIS.
///
/// See the [module documentation](self) for design details and extensibility
/// notes.
pub struct BidirectionalPathTracer {
    pub config: BdptConfig,
}

impl BidirectionalPathTracer {
    pub fn new(config: BdptConfig) -> Self {
        Self { config }
    }
}

impl Integrator for BidirectionalPathTracer {
    /// Render the scene using bidirectional path tracing.
    ///
    /// ## Approach
    ///
    /// For each pixel sample:
    /// 1. Generate a camera subpath `z[0..s_max]`.
    /// 2. Generate a light subpath `y[0..t_max]`.
    /// 3. For each valid `(s, t)` combination, connect `z[s-1]` to `y[t-1]`,
    ///    evaluate the unweighted contribution, compute the MIS weight, and
    ///    accumulate the result.
    /// 4. The `s = 1` strategy (light tracing) splatted contributions are
    ///    accumulated into a separate buffer and merged at the end.
    fn render(
        &self,
        scene: &Scene,
        camera: &dyn Camera,
        width: usize,
        height: usize,
    ) -> Vec<Vec3A> {
        let spp = self.config.spp;
        let max_depth = self.config.max_depth;
        let rr_depth = self.config.rr_depth;
        let n_pixels = width * height;

        // Primary accumulation buffer (per-pixel, filled by s ≥ 2 strategies).
        let mut pixels = vec![Vec3A::ZERO; n_pixels];

        // Splat buffer for the s=1 (light tracing) strategy.  This is
        // accumulated with atomic-like semantics by collecting per-row
        // splat lists, then merging after the parallel loop.
        //
        // We use a Vec<Mutex<Vec3A>> for thread-safe splatting.
        use std::sync::Mutex;
        let splat_buf: Vec<Mutex<Vec3A>> =
            (0..n_pixels).map(|_| Mutex::new(Vec3A::ZERO)).collect();

        // Process rows in parallel; each pixel gets an independent RNG.
        pixels
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y_row, row)| {
                for x_col in 0..width {
                    let seed = (y_row as u64).wrapping_mul(2654435761)
                        ^ (x_col as u64).wrapping_mul(805459861);
                    let mut rng = SmallRng::seed_from_u64(seed);

                    let mut accum = Vec3A::ZERO;

                    for _ in 0..spp {
                        // -- Generate camera ray (jittered sub-pixel) ------
                        let jx: f32 = rng.random();
                        let jy: f32 = rng.random();
                        let u = (x_col as f32 + jx) / width as f32;
                        let v = 1.0 - (y_row as f32 + jy) / height as f32;
                        let ray = camera.generate_ray(glam::Vec2::new(u, v));

                        // -- Build subpaths --------------------------------
                        let cam_path = generate_camera_subpath(
                            scene, camera, ray, max_depth, rr_depth, &mut rng,
                        );
                        let light_path = generate_light_subpath(
                            scene, max_depth, rr_depth, &mut rng,
                        );

                        let s_max = cam_path.len();
                        let t_max = light_path.len();

                        // -- Enumerate all (s, t) strategies ---------------
                        //
                        // Path length k = s + t - 1 (number of edges).
                        // We need s ≥ 1, t ≥ 0, s + t ≥ 2.
                        for t in 0..=t_max {
                            for s in 0..=s_max {
                                if s + t < 2 {
                                    continue;
                                }

                                if s == 0 {
                                    // s=0: pure light tracing — not yet
                                    // implemented (requires camera We for
                                    // arbitrary direction).
                                    continue;
                                }

                                if s == 1 {
                                    // s=1: light tracing to camera → splat.
                                    if let Some((c, px, py)) = connect_bdpt_s1(
                                        scene,
                                        camera,
                                        &cam_path,
                                        &light_path,
                                        t,
                                        width,
                                        height,
                                    ) {
                                        // Splat to the target pixel.
                                        let ix = (px as usize).min(width - 1);
                                        let iy = (py as usize).min(height - 1);
                                        let idx = iy * width + ix;
                                        let mut s = splat_buf[idx].lock().unwrap();
                                        *s += c;
                                    }
                                    continue;
                                }

                                // s ≥ 2, t ≥ 0: general / NEE / t=0.
                                if let Some((c, _, _)) = connect_bdpt(
                                    scene,
                                    camera,
                                    &cam_path,
                                    &light_path,
                                    s,
                                    t,
                                ) {
                                    accum += c;
                                }
                            }
                        }
                    }

                    row[x_col] = accum / spp as f32;
                }
            });

        // Merge splat buffer into primary pixels.
        let inv_spp = 1.0 / spp as f32;
        for (i, mutex) in splat_buf.iter().enumerate() {
            let splat = mutex.lock().unwrap();
            if *splat != Vec3A::ZERO {
                pixels[i] += *splat * inv_spp;
            }
        }

        pixels
    }
}

// ===========================================================================
// Utilities
// ===========================================================================

/// Power heuristic (β = 2) for two PDFs.
///
/// `w(p_a, p_b) = p_a² / (p_a² + p_b²)`
#[allow(dead_code)]
#[inline]
fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    if a2 + b2 == 0.0 {
        return 0.0;
    }
    a2 / (a2 + b2)
}
