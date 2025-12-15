# Sample run: libya / samgeo

Command run (compiled binary + provided masks):
```
./change_detection/cuda/rcm_metrics_cuda \
  outputs/segmentations/libya/samgeo/2022_mask.png \
  outputs/segmentations/libya/samgeo/2023_mask.png \
  change_detection/cuda/sample_metrics.csv \
  change_detection/cuda/sample_matrix.csv
```

## Metrics
CUDA (after relabeling to compact IDs) matches the Python implementation to 3–6 decimal places:

- Forward: overlap 0.593642, fragmentation 0.594632, composite 0.500495  
- Backward: overlap 0.707811, fragmentation 0.424721, composite 0.358455  
- Symmetric: overlap 0.650727, fragmentation 0.509677, composite 0.429475  

`change_detection/cuda/sample_matrix.csv` (2×2) shows the intersection counts after relabeling.

## Timing / throughput
- GPU total: 302.6 ms (kernel ~0.45 ms)
- CPU intersection (C++ baseline): 2.21 ms
- Python full pipeline (`rcm_metrics.compute_metrics`): ~43.85 ms
- Estimated GFLOPs: ~7.50 (3 ops/pixel over kernel time)

For this small mask (~1056×1064), GPU launch and H2D/D2H copies dominate; the kernel itself is sub-millisecond. Larger masks should tilt the balance toward the GPU.

## Plots / comparison artifacts
Generated via `change_detection/cuda/compare_rcm.py`:
- Combined metrics: `change_detection/cuda/sample_report/combined_metrics.csv`
- Timings: `change_detection/cuda/sample_report/timing.csv`
- Plot: `change_detection/cuda/sample_report/plots.png` (metrics bars + timing bars)

## Observations
- Numeric parity with the Python version is achieved once labels are relabeled to a dense ID space (mirrors `relabel_sequential` in Python).
- On small inputs, CPU and Python are faster end-to-end despite the GPU kernel being very quick; batching larger scenes or keeping data on GPU would reduce overhead.
