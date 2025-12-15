# rcm_metrics_cuda.cu code walkthrough

Path: `change_detection/cuda/rcm_metrics_cuda.cu` (line numbers from `nl -ba`).

## Section summaries
- **Headers & helpers (1–23)**: CUDA/STD includes, PNG loader `lodepng.h`, and `CUDA_CHECK` macro to wrap CUDA calls and throw with file/line context on error.
- **Metric structs (25–41)**: Plain aggregates for directional and symmetric RCM scores and a container `RCMResults`.
- **GPU kernel (43–52)**: `build_intersection` maps each pixel’s labels `(a,b)` to a flat index `a * n_b + b` and atomically increments the intersection matrix in global memory.
- **Directional metrics (54–94)**: `compute_directional` runs on CPU over a dense matrix: sums rows, normalizes by total, counts parts per row, computes per-row maxima and the fragmentation numerator/denominator, clamps overlap/fragmentation, and combines into composite (`0.5*frag + 0.5*(1-overlap)`).
- **Symmetric metrics (96–113)**: `compute_rcm_cpu` calls `compute_directional` on the matrix and its transpose, then averages forward/backward to produce symmetric scores.
- **Mask loader (115–138)**: `load_mask` uses lodepng to read 8- or 16-bit grayscale PNGs, converts to `uint32_t` labels, and returns width/height/data.
- **Relabeling (140–153)**: `relabel_inplace` builds a dense mapping from original labels to `0..K-1` (unordered_map) and rewrites the buffers to keep the intersection matrix compact.
- **GPU intersection (155–190)**: `compute_intersection_gpu` allocates device buffers, zeros the matrix, copies relabeled masks to device, launches `build_intersection`, records kernel time, copies the matrix back, and frees buffers.
- **CSV writers (192–219)**: `save_matrix_csv` dumps the dense intersection matrix; `save_metrics_csv` writes forward/backward/symmetric metrics plus timing (`timing_ms_gpu`, `timing_ms_kernel`, `timing_ms_cpu`) and estimated GFLOPs.
- **Main (221–295)**: Parses args, loads masks, checks dimensions, relabels both masks, times GPU intersection, computes a CPU intersection baseline, derives RCM metrics on CPU, estimates GFLOPs (`≈3 ops/pixel / kernel time`), saves CSVs, and logs a human-readable summary. Exceptions are caught and printed before exiting with code 1.

## Line-by-line notes
- 1–10: Bring in CUDA runtime and STL utilities; `unordered_map` supports relabeling.
- 12–13: Include lodepng (header/impl) to decode PNG masks without external deps.
- 15–23: `CUDA_CHECK` macro wraps CUDA calls; on error throws with message and file:line.
- 25–41: Define `DirectionalRCM`, `SymmetricRCM`, and `RCMResults` structs holding metric values.
- 43–52: CUDA kernel computes the intersection histogram; each thread handles one pixel, flattens `(a,b)` into `matrix` and uses `atomicAdd` to avoid races.
- 54–64: `compute_directional` precomputes row sums and total to normalize matrix entries.
- 68–88: Iterate rows, find max overlap per row, count nonzero parts, accumulate fragmentation numerator/denominator, sum overlap.
- 89–94: Guard against divide-by-zero, clamp to [0,1], and compute composite score.
- 96–113: `compute_rcm_cpu` runs forward and backward (via a transposed matrix) and averages to symmetric scores.
- 115–138: `Mask` struct and `load_mask`; decode PNG as grayscale (requesting 16-bit), convert to `uint32_t` labels, handle both 8- and 16-bit inputs.
- 140–153: `relabel_inplace` builds a dense ID map from arbitrary labels to `[0, K)` and rewrites the vectors; returns number of unique labels.
- 155–190: `compute_intersection_gpu` allocates device buffers sized by pixel count and `n_a * n_b`, copies relabeled labels, launches the kernel, records kernel time, copies matrix back, and frees device memory.
- 192–200: `save_matrix_csv` writes the dense intersection matrix row by row.
- 204–219: `save_metrics_csv` writes metric rows plus timing breakdown and GFLOPs estimate.
- 221–230: `main` parses CLI args: two mask paths, metrics CSV path, matrix CSV path.
- 232–244: Load both masks; ensure dimensions match; flatten into host int32 vectors.
- 246–248: Relabel both masks to dense IDs and capture the label counts.
- 249–253: Time the GPU intersection pipeline (H2D + kernel + D2H).
- 255–268: CPU intersection baseline over the relabeled masks (double loop on host).
- 269–274: Compute RCM metrics on CPU from the GPU-produced matrix; estimate GFLOPs from kernel time.
- 275–290: Save metrics/matrix to CSV; print human-readable metrics and timing to stdout.
- 291–295: Catch exceptions, print error, and return non-zero on failure.***

## Inputs, outputs, constraints, edge cases
- Inputs: two grayscale PNG masks (8- or 16-bit), same width/height; labels can be arbitrary integers.
- Outputs: metrics CSV (forward/backward/symmetric + timing + GFLOPs) and dense intersection matrix CSV (after relabeling labels to dense IDs).
- Constraints: masks must match spatial dimensions; memory/time scale with `pixels` and `n_a * n_b` post-relabel; kernel launch fixed at 256 threads/block.
- Edge cases: empty/zero-total matrix triggers runtime error; single-label masks yield zero fragmentation; non-PNG formats unsupported; huge label vocabularies can make relabeling and `n_a * n_b` large, risking OOM.

## Possible bugs / limitations
- Dense matrix allocation can fail silently if `n_a * n_b` is very large (would surface as `cudaMalloc` failure); no preflight cap.
- Relabeling is single-threaded and could be slow for very large masks with many unique labels.
- GFLOPs estimate is rough (3 ops/pixel) and ignores transfer costs; kernel timing excludes H2D/D2H.
- Fixed launch configuration may not be optimal across GPUs; no occupancy tuning.

## Improvements to consider
- Add sparse histogram (hash/sort-reduce) for high label counts; cap `n_a * n_b` or warn before allocation.
- Use pinned memory/streams to overlap transfers and compute; make block size configurable.
- Add options to skip CPU baseline, exclude background label, or accept alternative formats (e.g., NPY).
- Enhance reporting: break out H2D, kernel, D2H, and bandwidth alongside GFLOPs.***
