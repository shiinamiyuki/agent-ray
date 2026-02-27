use rayon::prelude::*;

/// Executes a parallel for loop over a range of indices, using Rayon's dynamic chunking.
/// `count`: Total number of iterations.
/// `_chunk_size`: Suggested chunk size for each task (currently handled by Rayon's internal scheduler).
/// `f`: Closure to execute for each index, receiving the index and the thread ID.
pub fn parallel_for<F>(count: usize, _chunk_size: usize, f: F)
where
    F: Fn(usize, usize) + Sync + Send,
{
    (0..count).into_par_iter().for_each(|idx| {
        let tid = rayon::current_thread_index().unwrap_or(0);
        f(idx, tid);
    });
}

/// Executes a parallel for loop over a 2D range (e.g., for image tiles or pixels).
pub fn parallel_for_2d<F>(width: usize, height: usize, f: F)
where
    F: Fn(usize, usize, usize) + Sync + Send,
{
    (0..height).into_par_iter().for_each(|y| {
        let tid = rayon::current_thread_index().unwrap_or(0);
        for x in 0..width {
            f(x, y, tid);
        }
    });
}

pub fn save_image_as_png(pixels: &[u8], width: u32, height: u32, path: &str) -> std::io::Result<()> {
    use image::{save_buffer, ColorType};
    save_buffer(path, pixels, width, height, ColorType::Rgb8).expect("Failed to save image");
    Ok(())
}
