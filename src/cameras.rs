use glam::Vec3A;
use crate::geometry::Ray;

// ---------------------------------------------------------------------------
// Camera We sample (for BDPT light-tracing connections)
// ---------------------------------------------------------------------------

/// Result of evaluating the camera importance function for a world-space point.
///
/// Used by the bidirectional path tracer when connecting light subpath vertices
/// to the camera (the `s = 1` strategy, a.k.a. light tracing).
pub struct CameraWeSample {
    /// Continuous pixel coordinates (x in `[0, width)`, y in `[0, height)`).
    pub pixel_x: f32,
    pub pixel_y: f32,
    /// Direction from the surface point toward the camera (world space, unit).
    pub wi: Vec3A,
    /// Distance from the surface point to the camera.
    pub dist: f32,
    /// Camera importance function `We(p_cam → p_surface)`.  Multiplied by the
    /// path throughput and the geometric coupling to produce a pixel value.
    pub we: f32,
    /// Solid-angle PDF of this camera "sample" (used for MIS weights).
    pub pdf: f32,
}

/// Common interface for all cameras.
///
/// **Adding a new camera model:**  
/// 1. Implement `generate_ray` for unidirectional path tracing.  
/// 2. For full BDPT support, implement `sample_we`, `pdf_we`, and
///    `film_area`.  The default implementations return `None` / `0.0`,
///    which causes the BDPT to skip the `s ≤ 1` strategies (light tracing
///    to camera) for that camera.
pub trait Camera: Send + Sync {
    /// Generate a primary ray for normalised image coordinates `uv ∈ [0,1]²`.
    fn generate_ray(&self, uv: glam::Vec2) -> Ray;

    // -------------------------------------------------------------------
    // BDPT interface (optional — default = unsupported)
    // -------------------------------------------------------------------

    /// Evaluate the camera importance function for a world-space point.
    ///
    /// Given a point `p` visible from the camera, compute the pixel it maps to,
    /// the importance `We`, and the solid-angle PDF.
    ///
    /// Returns `None` if the point is behind the camera or outside the image.
    fn sample_we(
        &self,
        _p: Vec3A,
        _width: usize,
        _height: usize,
    ) -> Option<CameraWeSample> {
        None
    }

    /// Solid-angle PDF of the camera direction for a given ray.
    ///
    /// This returns the same PDF stored in `CameraWeSample::pdf`, but can be
    /// called when you already have the ray and don't need the full projection.
    fn pdf_we(&self, _ray: &Ray) -> f32 {
        0.0
    }

    /// World-space area of the virtual film (image plane) at unit distance
    /// from the camera.  Used internally for `We` / PDF computation.
    fn film_area(&self) -> f32 {
        0.0
    }

    /// Camera origin (eye position).  Required for shadow rays in BDPT.
    fn origin(&self) -> Vec3A {
        Vec3A::ZERO
    }
}

pub struct PinholeCamera {
    pub origin: Vec3A,
    pub lower_left_corner: Vec3A,
    pub horizontal: Vec3A,
    pub vertical: Vec3A,
    /// Unit forward direction (from eye toward scene center).
    forward: Vec3A,
    /// World-space area of the image plane at distance 1 from the pinhole.
    area: f32,
}

impl PinholeCamera {
    /// Create a camera from an explicit look-at configuration.
    pub fn from_lookat(
        lookfrom: Vec3A,
        lookat: Vec3A,
        vup: Vec3A,
        vfov: f32, // vertical field-of-view in degrees
        aspect_ratio: f32,
    ) -> Self {
        let theta = vfov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (lookfrom - lookat).normalize();
        let u = vup.cross(w).normalize();
        let v = w.cross(u);

        let origin = lookfrom;
        let horizontal = viewport_width * u;
        let vertical = viewport_height * v;
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - w;

        let forward = -w;
        let area = viewport_width * viewport_height;

        Self {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
            forward,
            area,
        }
    }

    /// Create a camera from an eye position and Euler angles.
    ///
    /// - `yaw`   – horizontal rotation in degrees (0° looks along −Z, positive rotates right).
    /// - `pitch` – vertical tilt in degrees (0° is horizontal, positive tilts up).
    ///
    /// World-up is assumed to be +Y.
    pub fn from_eye_angle(
        eye: Vec3A,
        yaw: f32,   // degrees
        pitch: f32, // degrees
        vfov: f32,  // vertical field-of-view in degrees
        aspect_ratio: f32,
    ) -> Self {
        let (sin_y, cos_y) = yaw.to_radians().sin_cos();
        let (sin_p, cos_p) = pitch.to_radians().sin_cos();
        let forward = Vec3A::new(sin_y * cos_p, sin_p, -cos_y * cos_p);
        Self::from_lookat(eye, eye + forward, Vec3A::Y, vfov, aspect_ratio)
    }
}

impl Camera for PinholeCamera {
    fn generate_ray(&self, uv: glam::Vec2) -> Ray {
        let direction = self.lower_left_corner + uv.x * self.horizontal + uv.y * self.vertical - self.origin;
        Ray::new(self.origin, direction, 0.001, f32::MAX)
    }

    fn origin(&self) -> Vec3A {
        self.origin
    }

    fn film_area(&self) -> f32 {
        self.area
    }

    /// Evaluate camera importance for a world-space point.
    ///
    /// For a pinhole camera the importance function is:
    ///
    /// ```text
    /// We(ω) = 1 / (A_film · cos⁴ θ)
    /// ```
    ///
    /// where `A_film` is the image-plane area at unit distance and `θ` is the
    /// angle between `ω` and the optical axis.
    ///
    /// The solid-angle PDF of this "sample" is:
    ///
    /// ```text
    /// pdf(ω) = dist² / (A_film · cos³ θ)
    /// ```
    fn sample_we(
        &self,
        p: Vec3A,
        width: usize,
        height: usize,
    ) -> Option<CameraWeSample> {
        // Direction from p to the camera.
        let to_cam = self.origin - p;
        let dist2 = to_cam.length_squared();
        let dist = dist2.sqrt();
        if dist < 1e-6 {
            return None;
        }
        let wi = to_cam / dist;

        // Direction from camera to p (for projection).
        let dir = -wi;

        // cos θ between the ray direction and camera forward.
        let cos_theta = dir.dot(self.forward);
        if cos_theta <= 1e-6 {
            return None; // point is behind the camera
        }

        // Project onto the image plane.
        // The ray direction d can be written as:
        //   d = forward + (u − 0.5) · H + (v − 0.5) · V
        // where H, V are the horizontal/vertical image plane vectors.
        // Dividing by cos θ recovers the image-plane coordinates.
        let right = self.horizontal.normalize();
        let up = self.vertical.normalize();
        let vw = self.horizontal.length();
        let vh = self.vertical.length();

        let d_scaled = dir / cos_theta; // project onto image plane at distance 1
        let u = d_scaled.dot(right) / vw + 0.5;
        let v = d_scaled.dot(up) / vh + 0.5;

        // Bounds check.
        if u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 {
            return None;
        }

        // Convert to pixel coordinates (origin top-left).
        let pixel_x = u * width as f32;
        let pixel_y = (1.0 - v) * height as f32;
        if pixel_x < 0.0 || pixel_x >= width as f32 || pixel_y < 0.0 || pixel_y >= height as f32 {
            return None;
        }

        let cos2 = cos_theta * cos_theta;
        let cos4 = cos2 * cos2;
        let cos3 = cos2 * cos_theta;

        let we = 1.0 / (self.area * cos4);
        let pdf = dist2 / (self.area * cos3);

        Some(CameraWeSample {
            pixel_x,
            pixel_y,
            wi,
            dist,
            we,
            pdf,
        })
    }

    fn pdf_we(&self, ray: &Ray) -> f32 {
        let cos_theta = ray.direction.dot(self.forward);
        if cos_theta <= 0.0 {
            return 0.0;
        }
        1.0 / (self.area * cos_theta * cos_theta * cos_theta)
    }
}
