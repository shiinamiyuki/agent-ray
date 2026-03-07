## Plan: Gradient-Domain Path Tracer

Implement a gradient-domain path tracer (G-PT) that traces a **base path** per pixel and uses **random replay shift mapping** to generate offset paths at neighboring pixels (±x, ±y). The finite differences ΔIx, ΔIy are accumulated into gradient buffers. A **screened Poisson reconstruction** then solves for the final image from the primal image + gradients, yielding lower error at equal sample counts.

### Steps

1. **Add `GradientDomainPathTracer` integrator file** at `src/integrators/gradient_domain_pt.rs` and register it in `src/integrators/mod.rs`. Define a `GdptConfig` struct (spp, max_depth, rr_depth, reconstruction weight α).

2. **Implement random replay shift mapping** inside the new integrator. Extend or wrap the existing `Sampler` trait with a `ReplaySampler` that records the random number sequence consumed by a base path and replays it for the offset path with a **½-pixel shift** on the primary sample (camera jitter). Each bounce replays the same random numbers. For random replay, the shift is always invertible.

3. **Trace base + 4 offset paths per sample.** For each pixel (x,y) and sample s: trace a base path x̄ via the existing iterative `li`-style loop; then replay with camera-sample shifts to produce offset paths for (x+1,y), (x-1,y), (x,y+1), (x,y-1). Compute gradients ΔIx = I_offset^{x+1} - I_base (and similarly for the other directions). Accumulate the base image into a `Film` ("primal") and the two gradient channels into separate `Film`s (`dx_film`, `dy_film`).

4. **Implement shift validity checking.** for random replay, always valid, no need to check

5. **Implement screened Poisson reconstruction** as a post-process on the three films. Solve argmin_I  α‖I - I_primal‖² + ‖∇I - G‖² using an iterative **Gauss-Seidel / SOR solver** over the pixel grid. This is a sparse linear system with a 5-point Laplacian stencil; ~50–200 iterations typically suffice. Output the reconstructed image as the final `Film`.

6. **Add a binary entry point** `bin/gdpt_test.rs` (similar to `bin/pt_test.rs`) that loads the fireplace scene, runs `GradientDomainPathTracer`, and saves the primal, gradient, and reconstructed images for visual verification.

### Further Considerations

1. **Shift mapping strategy**: Just random replay for now. in the future, we should use hybrid shift mapping: reconnect for diffuse surface and replay for glossy surfaces.

2. **Reconstruction solver convergence**: The screened Poisson solve could optionally use **FFT-based** reconstruction (exact, faster) via a DCT, but iterative Gauss-Seidel is simpler with no external dependency. Should we add an FFT path later? Recommend starting with iterative SOR and adding DCT as a follow-up.

3. **Film extensions**: The current `Film` stores only RGB per pixel. Gradient-domain PT needs ~5 film buffers (primal + dx + dy + reconstructed, optionally throughput weight). The existing `Film` API is sufficient since we can just allocate multiple `Film` instances — no structural changes needed.
