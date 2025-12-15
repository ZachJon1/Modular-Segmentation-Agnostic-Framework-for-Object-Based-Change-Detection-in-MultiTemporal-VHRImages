# Results & Discussion (samgeo runs)

Aggregated CUDA runs and parity checks with the Python reference for samgeo segmentations. Metrics below use the symmetric scores (mean of forward/backward).

## Summary table

| site_algo            | overlap | fragmentation | composite | GPU total ms | kernel ms | CPU intersect ms | GFLOPs est |
|----------------------|---------|---------------|-----------|--------------|-----------|------------------|------------|
| libya_samgeo         | 0.6507  | 0.5097        | 0.4295    | 280.84       | 0.90      | 2.99             | 3.73       |
| hathaway_samgeo      | 0.7723  | 0.3103        | 0.2690    | 287.89       | 2.28      | 10.56            | 10.12      |
| rosebud_samgeo       | 0.7412  | 0.3559        | 0.3073    | 254.58       | 0.43      | 0.95             | 4.88       |
| hathawaynorth_samgeo | 0.7481  | 0.2996        | 0.2758    | 281.77       | 2.10      | 7.87             | 8.07       |

Source data: `change_detection/cuda/all_metrics.csv`. Plots: `change_detection/cuda/all_metrics_plot.png`.

## Observations
- **Metric parity:** CUDA results match Python to ~1e-6 after dense relabeling; see per-site `..._report/combined_metrics.csv`.
- **Timings:** Kernel times are sub-3 ms for these sizes; GPU totals are ~250–290 ms due to H2D/D2H + launch overhead. CPU intersection baselines are 1–11 ms, and full Python pipeline ranges 20–200 ms (see per-site `timing.csv`).
- **Throughput:** GFLOPs estimates are modest because masks are small; larger scenes should better amortize transfers and show clearer GPU advantages.
- **Label handling:** Relabeling to dense IDs was essential to keep matrices small and ensure parity with the Python `relabel_sequential` behavior.
- **Complexity & sparsity:** Building the intersection matrix is O(pixels) time, O(K_a·K_b) space (after relabeling to K_a/K_b distinct labels). For these masks, K_a·K_b is tiny (2×2, up to ~hundreds squared if labels weren’t dense). The matrix is typically sparse—most region pairs never overlap—so storing it densely on GPU is fine at current label counts, but for very high label counts a sparse histogram (e.g., hash-based) would avoid quadratic memory.

## How to reproduce
1) Build: `nvcc -O3 -std=c++17 change_detection/cuda/rcm_metrics_cuda.cu change_detection/cuda/lodepng.cpp -o change_detection/cuda/rcm_metrics_cuda`
2) Run per pair (example):\
   `./change_detection/cuda/rcm_metrics_cuda outputs/segmentations/libya/samgeo/2022_mask.png outputs/segmentations/libya/samgeo/2023_mask.png change_detection/cuda/metrics.csv change_detection/cuda/matrix.csv`
3) Compare/plot:\
   `. .venv_cuda/bin/activate && PYTHONPATH=change_detection/src python change_detection/cuda/compare_rcm.py --mask-a <mask_a.png> --mask-b <mask_b.png> --cuda-metrics <metrics.csv> --output-dir <report_dir>`
