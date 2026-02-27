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
