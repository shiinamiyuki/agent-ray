use glam::{Mat4, Vec3A};

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3A,
    pub direction: Vec3A,
    pub t_min: f32,
    pub t_max: f32,
}

impl Ray {
    pub fn new(origin: Vec3A, direction: Vec3A, t_min: f32, t_max: f32) -> Self {
        let direction = direction.normalize();
        Self {
            origin,
            direction,
            t_min,
            t_max,
        }
    }

    pub fn at(&self, t: f32) -> Vec3A {
        self.origin + self.direction * t
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AABB {
    pub min: Vec3A,
    pub max: Vec3A,
}

impl AABB {
    pub fn new(min: Vec3A, max: Vec3A) -> Self {
        Self { min, max }
    }

    /// An empty (inverted-infinite) AABB that acts as an identity for `union`.
    pub fn empty() -> Self {
        Self {
            min: Vec3A::splat(f32::INFINITY),
            max: Vec3A::splat(f32::NEG_INFINITY),
        }
    }

    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Surface area of the bounding box — used by the SAH cost function.
    pub fn surface_area(&self) -> f32 {
        let d = (self.max - self.min).max(Vec3A::ZERO);
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Geometric center of the bounding box.
    pub fn centroid(&self) -> Vec3A {
        (self.min + self.max) * 0.5
    }

    /// Transform the AABB by a 4×4 matrix, returning the world-space AABB of
    /// all 8 transformed corners.
    pub fn transform(&self, m: &Mat4) -> Self {
        let corners = [
            Vec3A::new(self.min.x, self.min.y, self.min.z),
            Vec3A::new(self.max.x, self.min.y, self.min.z),
            Vec3A::new(self.min.x, self.max.y, self.min.z),
            Vec3A::new(self.max.x, self.max.y, self.min.z),
            Vec3A::new(self.min.x, self.min.y, self.max.z),
            Vec3A::new(self.max.x, self.min.y, self.max.z),
            Vec3A::new(self.min.x, self.max.y, self.max.z),
            Vec3A::new(self.max.x, self.max.y, self.max.z),
        ];
        let mut result = AABB::empty();
        for c in corners {
            let tc = m.transform_point3a(c);
            result.min = result.min.min(tc);
            result.max = result.max.max(tc);
        }
        result
    }

    pub fn intersect(&self, ray: &Ray) -> bool {
        let inv_dir = 1.0 / ray.direction;
        let t0 = (self.min - ray.origin) * inv_dir;
        let t1 = (self.max - ray.origin) * inv_dir;

        let t_min_v = t0.min(t1);
        let t_max_v = t0.max(t1);

        let t_min = t_min_v.max_element().max(ray.t_min);
        let t_max = t_max_v.min_element().min(ray.t_max);

        t_min <= t_max
    }
}

pub struct HitInfo {
    pub t: f32,
    pub p: Vec3A,
    pub n: Vec3A,
    pub uv: glam::Vec2,
}

pub trait Intersect {
    fn hit(&self, ray: &Ray) -> Option<HitInfo>;
}

pub struct Sphere {
    pub center: Vec3A,
    pub radius: f32,
}

impl Sphere {
    pub fn new(center: Vec3A, radius: f32) -> Self {
        Self { center, radius }
    }
}

impl Intersect for Sphere {
    fn hit(&self, ray: &Ray) -> Option<HitInfo> {
        let oc = ray.origin - self.center;
        let b = oc.dot(ray.direction);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = b * b - c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrt_d = discriminant.sqrt();
        let mut t = -b - sqrt_d;
        if t < ray.t_min || t > ray.t_max {
            t = -b + sqrt_d;
            if t < ray.t_min || t > ray.t_max {
                return None;
            }
        }

        let p = ray.at(t);
        let n = (p - self.center) / self.radius;

        Some(HitInfo {
            t,
            p,
            n,
            uv: glam::Vec2::ZERO, // TODO: UV calculation
        })
    }
}

pub struct Triangle {
    pub v0: Vec3A,
    pub edge1: Vec3A,
    pub edge2: Vec3A,
}

impl Triangle {
    pub fn new(v0: Vec3A, v1: Vec3A, v2: Vec3A) -> Self {
        Self {
            v0,
            edge1: v1 - v0,
            edge2: v2 - v0,
        }
    }
}

impl Intersect for Triangle {
    fn hit(&self, ray: &Ray) -> Option<HitInfo> {
        let h = ray.direction.cross(self.edge2);
        let a = self.edge1.dot(h);

        if a.abs() < 1e-8 {
            return None;
        }

        let f = 1.0 / a;
        let s = ray.origin - self.v0;
        let u = f * s.dot(h);

        if u < 0.0 || u > 1.0 {
            return None;
        }

        let q = s.cross(self.edge1);
        let v = f * ray.direction.dot(q);

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = f * self.edge2.dot(q);
        if t < ray.t_min || t > ray.t_max {
            return None;
        }

        Some(HitInfo {
            t,
            p: ray.at(t),
            n: self.edge1.cross(self.edge2).normalize(),
            uv: glam::Vec2::new(u, v),
        })
    }
}
