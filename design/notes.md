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

## 2026-02-27 - Two-Level SAH BVH
Implemented a full two-level BVH acceleration structure in `src/accel/bvh.rs`:
- **SAH builder** (`sah_split`): shared between BLAS and TLAS; performs a forward-backward AABB prefix/suffix scan on all three axes to test every exact split point, picks the axis+position with the lowest SAH cost, and only splits when the cost beats a pure leaf.
- **Parallel recursive builder** (`build_recursive`): uses `rayon::join` down to a configurable depth (`MAX_PARALLEL_DEPTH = 8`) for parallel subtree construction. Subtrees are assembled into flat `Vec<BVHNode>` arrays with child indices rebased on merge — no mutation of shared state during parallel phase.
- **Flat node layout**: `BVHNode` stores an AABB plus a `count` discriminant (0 = internal, >0 = leaf); leaves reference a contiguous range in a separate `prim_indices` / `instance_indices` array.
- **`BLASPrimitive` trait**: users implement `primitive_count`, `primitive_aabb`, and `intersect_primitive`; the backing geometry is never reordered — only an index array is permuted.
- **`BLASAccel`**: wraps an `Arc<dyn BLASPrimitive>` with a BVH; exposes `build`, `aabb`, and `intersect`.
- **`Instance`** / **`TLAS`**: TLAS is a BVH over instances; each instance carries a local-to-world `Mat4`. Ray–instance intersection transforms the ray into local space (accounting for the direction-length scale factor so world t-values remain correct).
- Added `AABB::empty()`, `AABB::surface_area()`, `AABB::centroid()`, and `AABB::transform(Mat4)` to `src/geometry.rs`.