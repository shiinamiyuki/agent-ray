use std::sync::Arc;

use crate::cameras::Camera;
use crate::film::Film;
use crate::scene::Scene;

pub mod path_tracer;
pub mod bidirectional_path_tracer;
pub mod gradient_domain_pt;
pub use path_tracer::{PathTracer, PathTracerConfig};
pub use bidirectional_path_tracer::{BidirectionalPathTracer, BdptConfig, MisMode};
pub use gradient_domain_pt::{GradientDomainPathTracer, GdptConfig};

/// Common interface for all rendering integrators.
///
/// `render` returns a [`Film`] holding the accumulated linear-HDR image.
/// Callers can then call `film.to_rgb_image(…)` for tone-mapping and
/// gamma-correction, or `film.to_hdr_vec()` for raw pixel data.
pub trait Integrator {
    fn render(
        &self,
        scene: &Scene,
        camera: &dyn Camera,
        width: usize,
        height: usize,
    ) -> Arc<Film>;
}
