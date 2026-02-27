use glam::{Vec2, Vec3A, Vec4};
use std::path::Path;
use anyhow::Result;

pub struct TriangleMesh {
    pub positions: Vec<Vec3A>,
    pub normals: Option<Vec<Vec3A>>,
    pub tex_coords: Option<Vec<Vec2>>,
    pub tangents: Option<Vec<Vec4>>,
    pub indices: Vec<u32>,
}

impl TriangleMesh {
    pub fn load_obj(path: &Path) -> Result<Vec<Self>> {
        let (models, _materials) = tobj::load_obj(
            path,
            &tobj::GPU_LOAD_OPTIONS,
        )?;

        let mut meshes = Vec::new();

        for model in models {
            let mesh = &model.mesh;
            
            let positions: Vec<Vec3A> = mesh.positions
                .chunks_exact(3)
                .map(|p| Vec3A::new(p[0], p[1], p[2]))
                .collect();

            let normals = if !mesh.normals.is_empty() {
                Some(mesh.normals
                    .chunks_exact(3)
                    .map(|n| Vec3A::new(n[0], n[1], n[2]))
                    .collect())
            } else {
                None
            };

            let tex_coords = if !mesh.texcoords.is_empty() {
                Some(mesh.texcoords
                    .chunks_exact(2)
                    .map(|t| Vec2::new(t[0], t[1]))
                    .collect())
            } else {
                None
            };

            // tobj doesn't load tangents by default, so we'll leave it as None for now
            // We can implement tangent calculation later if needed.
            let tangents = None;

            let indices = mesh.indices.clone();

            meshes.push(TriangleMesh {
                positions,
                normals,
                tex_coords,
                tangents,
                indices,
            });
        }

        Ok(meshes)
    }
}
