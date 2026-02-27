use std::sync::Arc;

use crate::geometry::{AABB, Ray};
use crate::prelude::*;

// ---------------------------------------------------------------------------
// RayHit
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    pub instance_id: u32,
    pub prim_id: u32,
    pub uv: Vec2,
    pub t: f32,
}
impl RayHit {
    pub const INVALID_ID: u32 = u32::MAX;
}

// ---------------------------------------------------------------------------
// BVH node (shared between BLAS and TLAS)
// ---------------------------------------------------------------------------

/// Flat BVH node used by both BLAS and TLAS.
///
/// - **Leaf**:     `count > 0`; elements `[left_or_start .. left_or_start + count)` in the
///                associated index array.
/// - **Internal**: `count == 0`; left child at `left_or_start`, right child at `right_or_end`
///                within the same node array.
#[derive(Clone, Debug)]
struct BVHNode {
    aabb: AABB,
    left_or_start: u32,
    right_or_end: u32,
    count: u32,
}

impl BVHNode {
    fn leaf(aabb: AABB, start: u32, count: u32) -> Self {
        Self { aabb, left_or_start: start, right_or_end: 0, count }
    }
    fn internal(aabb: AABB, left: u32, right: u32) -> Self {
        Self { aabb, left_or_start: left, right_or_end: right, count: 0 }
    }
    #[inline]
    fn is_leaf(&self) -> bool {
        self.count > 0
    }
}

// ---------------------------------------------------------------------------
// SAH builder — shared between BLAS and TLAS
// ---------------------------------------------------------------------------

/// Maximum number of primitives in a leaf node.
const MAX_LEAF_SIZE: usize = 4;

/// Depth threshold below which the builder spawns rayon tasks.
const MAX_PARALLEL_DEPTH: usize = 8;

/// Try all split points on all three axes using a forward-backward AABB scan
/// and pick the split that minimises the SAH cost.
///
/// `indices` is modified in-place: after a successful split the first
/// `split_pos` entries belong to the left child and the remainder to the right.
///
/// Returns `Some(split_pos)` when splitting is better than a leaf, `None`
/// otherwise.
fn sah_split(
    aabb_fn: &(impl Fn(usize) -> AABB + Sync),
    indices: &mut Vec<usize>,
) -> Option<usize> {
    let n = indices.len();
    debug_assert!(n >= 2);

    // Precompute per-primitive AABBs indexed locally 0..n.
    let prim_aabbs: Vec<AABB> = indices.iter().map(|&i| aabb_fn(i)).collect();

    let parent_aabb = prim_aabbs.iter().cloned().fold(AABB::empty(), |a, b| a.union(&b));
    let parent_sa = parent_aabb.surface_area();

    if parent_sa < f32::EPSILON {
        return None;
    }

    // SAH cost constants: C_trav = 1, C_isect = 1.
    const C_TRAV: f32 = 1.0;
    let leaf_cost = n as f32;

    let mut best_cost = leaf_cost;
    let mut best: Option<(usize, usize)> = None; // (axis, split_pos)

    // `sorted` is a permutation of 0..n; reused across axes.
    let mut sorted: Vec<usize> = (0..n).collect();

    for axis in 0usize..3 {
        sorted.sort_unstable_by(|&a, &b| {
            prim_aabbs[a].centroid()[axis]
                .partial_cmp(&prim_aabbs[b].centroid()[axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Forward scan: prefix[i] = union of AABBs sorted[0..=i].
        let mut prefix = vec![AABB::empty(); n];
        prefix[0] = prim_aabbs[sorted[0]];
        for i in 1..n {
            prefix[i] = prefix[i - 1].union(&prim_aabbs[sorted[i]]);
        }

        // Backward scan: suffix[i] = union of AABBs sorted[i..n].
        let mut suffix = vec![AABB::empty(); n];
        suffix[n - 1] = prim_aabbs[sorted[n - 1]];
        for i in (0..n - 1).rev() {
            suffix[i] = suffix[i + 1].union(&prim_aabbs[sorted[i]]);
        }

        // Test all n-1 split positions.
        for k in 1..n {
            let cost = C_TRAV
                + (prefix[k - 1].surface_area() * k as f32
                    + suffix[k].surface_area() * (n - k) as f32)
                    / parent_sa;
            if cost < best_cost {
                best_cost = cost;
                best = Some((axis, k));
            }
        }
    }

    let (best_axis, split_k) = best?;

    // Re-sort along the winning axis if needed (may already be sorted when best_axis == 2).
    if best_axis != 2 {
        sorted.sort_unstable_by(|&a, &b| {
            prim_aabbs[a].centroid()[best_axis]
                .partial_cmp(&prim_aabbs[b].centroid()[best_axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Apply permutation to indices.
    let old = indices.clone();
    for (i, &si) in sorted.iter().enumerate() {
        indices[i] = old[si];
    }

    Some(split_k)
}

// ---------------------------------------------------------------------------
// Recursive parallel BVH builder
// ---------------------------------------------------------------------------

/// Shift all child / leaf-start references in `nodes` by the given offsets.
fn rebase_nodes(nodes: &mut [BVHNode], node_offset: u32, prim_offset: u32) {
    for node in nodes.iter_mut() {
        if node.is_leaf() {
            node.left_or_start += prim_offset;
        } else {
            node.left_or_start += node_offset;
            node.right_or_end += node_offset;
        }
    }
}

/// Merge two subtree results under a common root node.
fn merge(
    aabb: AABB,
    (mut left_nodes, left_prims): (Vec<BVHNode>, Vec<usize>),
    (mut right_nodes, right_prims): (Vec<BVHNode>, Vec<usize>),
) -> (Vec<BVHNode>, Vec<usize>) {
    let left_node_count = left_nodes.len() as u32;
    let left_prim_count = left_prims.len() as u32;

    // Left subtree starts at merged[1]; right at merged[1 + left_node_count].
    rebase_nodes(&mut left_nodes, 1, 0);
    rebase_nodes(&mut right_nodes, 1 + left_node_count, left_prim_count);

    let root = BVHNode::internal(aabb, 1, 1 + left_node_count);

    let mut nodes = Vec::with_capacity(1 + left_nodes.len() + right_nodes.len());
    nodes.push(root);
    nodes.extend(left_nodes);
    nodes.extend(right_nodes);

    let mut prims = Vec::with_capacity(left_prims.len() + right_prims.len());
    prims.extend(left_prims);
    prims.extend(right_prims);

    (nodes, prims)
}

/// Build a BVH sub-tree over `indices`.
///
/// Returns `(nodes, ordered_indices)` where `nodes[0]` is the sub-tree root
/// and leaf `left_or_start` values are offsets into `ordered_indices`.
fn build_recursive(
    aabb_fn: &(impl Fn(usize) -> AABB + Sync),
    mut indices: Vec<usize>,
    depth: usize,
) -> (Vec<BVHNode>, Vec<usize>) {
    let n = indices.len();

    let aabb = indices
        .iter()
        .map(|&i| aabb_fn(i))
        .fold(AABB::empty(), |a, b| a.union(&b));

    // Attempt an SAH split when above the leaf threshold.
    let split_pos = if n <= MAX_LEAF_SIZE {
        None
    } else {
        sah_split(aabb_fn, &mut indices)
    };

    let Some(split) = split_pos else {
        let count = n as u32;
        return (vec![BVHNode::leaf(aabb, 0, count)], indices);
    };

    let right_indices = indices.split_off(split);
    let left_indices = indices;

    let build_left = || build_recursive(aabb_fn, left_indices, depth + 1);
    let build_right = || build_recursive(aabb_fn, right_indices, depth + 1);

    let (left_result, right_result) = if depth < MAX_PARALLEL_DEPTH {
        rayon::join(build_left, build_right)
    } else {
        (build_left(), build_right())
    };

    merge(aabb, left_result, right_result)
}

// ---------------------------------------------------------------------------
// BVH traversal helper
// ---------------------------------------------------------------------------

/// Iterative stack-based BVH traversal shared by BLAS and TLAS.
///
/// `intersect_leaf` receives `(prim_id, current_t_max)` and should return a
/// `RayHit` with `t <= current_t_max` if the primitive is hit, or `None`.
#[inline]
fn traverse_bvh(
    nodes: &[BVHNode],
    prim_indices: &[usize],
    ray: &Ray,
    mut intersect_leaf: impl FnMut(usize, f32) -> Option<RayHit>,
) -> Option<RayHit> {
    if nodes.is_empty() {
        return None;
    }

    let mut stack = [0u32; 64];
    let mut sp = 0usize;
    let mut best: Option<RayHit> = None;
    let mut t_max = ray.t_max;

    stack[sp] = 0;
    sp += 1;

    while sp > 0 {
        sp -= 1;
        let node = &nodes[stack[sp] as usize];

        // Quick AABB rejection incorporating the current best t_max.
        let probe = Ray { t_max, ..*ray };
        if !node.aabb.intersect(&probe) {
            continue;
        }

        if node.is_leaf() {
            let start = node.left_or_start as usize;
            let end = start + node.count as usize;
            for &prim_id in &prim_indices[start..end] {
                if let Some(hit) = intersect_leaf(prim_id, t_max) {
                    if hit.t < t_max {
                        t_max = hit.t;
                        best = Some(hit);
                    }
                }
            }
        } else {
            debug_assert!(sp + 2 <= stack.len(), "BVH traversal stack overflow");
            stack[sp] = node.right_or_end;
            sp += 1;
            stack[sp] = node.left_or_start;
            sp += 1;
        }
    }

    best
}

// ---------------------------------------------------------------------------
// BLAS
// ---------------------------------------------------------------------------

/// Geometry interface required by `BLASAccel`.
///
/// Implementors expose per-primitive AABBs and ray–primitive intersection.
/// Primitives are identified by a zero-based `prim_id`.
pub trait BLASPrimitive: Send + Sync {
    fn primitive_count(&self) -> usize;
    fn primitive_aabb(&self, prim_id: usize) -> AABB;
    /// Return a hit with `t <= t_max`, or `None`.
    fn intersect_primitive(&self, prim_id: usize, ray: &Ray, t_max: f32) -> Option<RayHit>;
}

/// Bottom-Level Acceleration Structure — a SAH BVH over a set of primitives.
///
/// The backing geometry is not reordered; only an index array is permuted by
/// the builder.
pub struct BLASAccel {
    nodes: Vec<BVHNode>,
    /// Re-ordered primitive indices produced by the SAH builder.
    prim_indices: Vec<usize>,
    overall_aabb: AABB,
    data: Arc<dyn BLASPrimitive>,
}

impl BLASAccel {
    pub fn build(data: Arc<dyn BLASPrimitive>) -> Self {
        let n = data.primitive_count();

        if n == 0 {
            return Self {
                nodes: Vec::new(),
                prim_indices: Vec::new(),
                overall_aabb: AABB::empty(),
                data,
            };
        }

        let indices: Vec<usize> = (0..n).collect();
        let aabb_fn = |id: usize| data.primitive_aabb(id);
        let (nodes, prim_indices) = build_recursive(&aabb_fn, indices, 0);
        let overall_aabb = nodes[0].aabb;

        Self { nodes, prim_indices, overall_aabb, data }
    }

    pub fn aabb(&self) -> AABB {
        self.overall_aabb
    }

    pub fn intersect(&self, ray: &Ray) -> Option<RayHit> {
        traverse_bvh(&self.nodes, &self.prim_indices, ray, |prim_id, t_max| {
            self.data.intersect_primitive(prim_id, ray, t_max)
        })
    }
}

// ---------------------------------------------------------------------------
// TLAS
// ---------------------------------------------------------------------------

/// One instance: a BLAS positioned in world space by an affine transform.
pub struct Instance {
    pub blas: Arc<BLASAccel>,
    /// Local-to-world transform.
    pub transform: Mat4,
    /// World-to-local transform (inverse of `transform`).
    pub inv_transform: Mat4,
    /// World-space AABB — pre-computed for TLAS building.
    world_aabb: AABB,
}

impl Instance {
    pub fn new(blas: Arc<BLASAccel>, transform: Mat4) -> Self {
        let inv_transform = transform.inverse();
        let world_aabb = blas.aabb().transform(&transform);
        Self { blas, transform, inv_transform, world_aabb }
    }
}

/// Input descriptor used to build a TLAS.
pub struct InstanceBuildInfo {
    pub blas: Arc<BLASAccel>,
    pub transform: Mat4,
}

/// Top-Level Acceleration Structure — a SAH BVH over a set of instances.
pub struct TLAS {
    nodes: Vec<BVHNode>,
    /// Re-ordered instance indices produced by the SAH builder.
    instance_indices: Vec<usize>,
    instances: Vec<Instance>,
}

impl TLAS {
    pub fn build(instance_build_info: &[InstanceBuildInfo]) -> Self {
        let instances: Vec<Instance> = instance_build_info
            .iter()
            .map(|info| Instance::new(Arc::clone(&info.blas), info.transform))
            .collect();

        let n = instances.len();
        if n == 0 {
            return Self { nodes: Vec::new(), instance_indices: Vec::new(), instances };
        }

        let indices: Vec<usize> = (0..n).collect();
        let aabb_fn = |id: usize| instances[id].world_aabb;
        let (nodes, instance_indices) = build_recursive(&aabb_fn, indices, 0);

        Self { nodes, instance_indices, instances }
    }

    pub fn intersect(&self, ray: &Ray) -> Option<RayHit> {
        traverse_bvh(&self.nodes, &self.instance_indices, ray, |inst_id, t_max| {
            let inst = &self.instances[inst_id];

            // Transform the ray into the instance's local (BLAS) space.
            //
            // Because Ray::new normalises the direction, the t parameter scales
            // by `dir_scale`.  Given world ray P_w(t) = origin_w + t·dir_w and
            // M = local-to-world, the local parametrisation is
            //   P_l(t) = M⁻¹·origin_w + t·(M⁻¹·dir_w)
            // which shares the same t.  After normalising the local direction
            // to get a unit-direction ray we have:
            //   P_l(s) = origin_l + s·dir_l_norm,  where s = t·dir_scale
            // so  world_t = local_s / dir_scale.
            let local_origin = inst.inv_transform.transform_point3a(ray.origin);
            let local_dir_unnorm = inst.inv_transform.transform_vector3a(ray.direction);
            let dir_scale = local_dir_unnorm.length();
            if dir_scale < f32::EPSILON {
                return None;
            }

            let local_ray = Ray::new(
                local_origin,
                local_dir_unnorm,
                ray.t_min * dir_scale,
                t_max * dir_scale,
            );

            inst.blas.intersect(&local_ray).map(|mut hit| {
                hit.t /= dir_scale;
                hit.instance_id = inst_id as u32;
                hit
            })
        })
    }
}
