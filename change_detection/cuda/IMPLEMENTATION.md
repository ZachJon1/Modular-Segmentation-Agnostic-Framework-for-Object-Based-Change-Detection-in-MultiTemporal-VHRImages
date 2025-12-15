# CUDA RCM Implementation

This doc summarizes the standalone CUDA C++ Region-Correlation Matrix (RCM) implementation and how to build/run it.

## Files
- `rcm_metrics_cuda.cu` – main program; loads PNG masks (8/16-bit labels), relabels to dense IDs, runs GPU intersection histogram, computes RCM metrics on CPU, writes metrics/matrix CSVs.
- `lodepng.cpp/.h` – bundled PNG decoder (no external image deps).
- `compare_rcm.py` – Python helper to compare CUDA outputs against the Python reference and plot metrics/timings.
- `README.md` – quickstart build/run notes.

## Build
Requires CUDA toolkit (`nvcc`, C++17). No external libs beyond CUDA and bundled lodepng.
```bash
cd <repo_root>
nvcc -O3 -std=c++17 change_detection/cuda/rcm_metrics_cuda.cu change_detection/cuda/lodepng.cpp -o change_detection/cuda/rcm_metrics_cuda
```

## Run (single pair)
```bash
./change_detection/cuda/rcm_metrics_cuda \
  <mask_a.png> <mask_b.png> \
  <metrics_out.csv> <matrix_out.csv>
```
Outputs:
- `metrics_out.csv`: forward/backward/symmetric metrics + timings (`timing_ms_gpu`, `timing_ms_kernel`, `timing_ms_cpu`, `gflops_est`).
- `matrix_out.csv`: intersection matrix with relabeled dense IDs.
- Stdout echoes the metrics and timings.

## Algorithm outline
1) **Load masks** (lodepng) as grayscale; supports 8- or 16-bit labels.
2) **Relabel to dense IDs** on CPU (unordered_map) to minimize matrix size and mirror Python `relabel_sequential`.
3) **GPU intersection**: kernel builds a bincount-style matrix (`atomicAdd`) of size `n_a x n_b`; kernel launch is 256 threads/block over all pixels.
4) **Transfer back** and compute directional & symmetric RCM metrics on CPU (same math as Python).
5) **Timing**: kernel time, GPU total (H2D+kernel+D2H), CPU intersection baseline, rough GFLOPs estimate (`3 ops/pixel / kernel_time`).

## Python comparison (optional)
Use the venv from earlier steps:
```bash
. .venv_cuda/bin/activate
PYTHONPATH=change_detection/src python change_detection/cuda/compare_rcm.py \
  --mask-a <mask_a.png> --mask-b <mask_b.png> \
  --cuda-metrics <metrics_out.csv> \
  --output-dir <report_dir>
```
Produces `combined_metrics.csv`, `timing.csv`, and `plots.png` to confirm parity with the Python reference.

## Notes / constraints
- Masks must share spatial dimensions.
- For small masks, kernel time is tiny but transfer/launch dominates; larger masks should show better GPU payoff.
- Intersection matrix scales with `(#labels in A) x (#labels in B)`; dense relabeling keeps it compact.***
