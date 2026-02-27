use glam::Vec3A;
use crate::geometry::Ray;

pub trait Camera: Send + Sync {
    fn generate_ray(&self, uv: glam::Vec2) -> Ray;
}

pub struct PinholeCamera {
    pub origin: Vec3A,
    pub lower_left_corner: Vec3A,
    pub horizontal: Vec3A,
    pub vertical: Vec3A,
}

impl PinholeCamera {
    pub fn new(
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

        Self {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
        }
    }
}

impl Camera for PinholeCamera {
    fn generate_ray(&self, uv: glam::Vec2) -> Ray {
        let direction = self.lower_left_corner + uv.x * self.horizontal + uv.y * self.vertical - self.origin;
        Ray::new(self.origin, direction, 0.001, f32::MAX)
    }
}
