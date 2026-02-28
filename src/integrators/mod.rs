use glam::Vec3A;
use crate::cameras::Camera;
use crate::scene::Scene;

pub mod path_tracer;
pub use path_tracer::{PathTracer, PathTracerConfig};

/// Common interface for all rendering integrators.
///
/// `render` returns a linear-HDR image as a flat `Vec<Vec3A>`, row-major
/// (`pixel[y * width + x]`).  Callers are responsible for tone-mapping and
/// gamma-correction before display.
pub trait Integrator {
    fn render(
        &self,
        scene: &Scene,
        camera: &dyn Camera,
        width: usize,
        height: usize,
    ) -> Vec<Vec3A>;
}
