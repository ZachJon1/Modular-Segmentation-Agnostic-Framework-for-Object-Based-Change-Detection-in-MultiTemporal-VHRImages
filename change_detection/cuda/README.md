# CUDA RCM Metrics

Standalone CUDA C++ implementation of the Region-Correlation Matrix (RCM) metrics. The tool loads two segmentation masks (PNG, 8- or 16-bit labels), builds the intersection matrix on GPU, and reports forward/backward/symmetric overlap/fragmentation/composite scores. A small Python helper script can compare the CUDA output against the existing Python implementation and plot timings.

## Build

Requirements: CUDA toolkit (nvcc), C++17. No external image dependencies; PNG decoding is handled by vendored `lodepng`.

```bash
cd <repo_root>
nvcc -O3 -std=c++17 change_detection/cuda/rcm_metrics_cuda.cu change_detection/cuda/lodepng.cpp -o change_detection/cuda/rcm_metrics_cuda
```

## Run

```bash
# Example (libya, samgeo)
./change_detection/cuda/rcm_metrics_cuda \
  outputs/segmentations/libya/samgeo/2022_mask.png \
  outputs/segmentations/libya/samgeo/2023_mask.png \
  change_detection/cuda/metrics.csv \
  change_detection/cuda/matrix.csv
```

Outputs
- `metrics.csv`: forward/backward/symmetric metrics + timing/gflops rows
- `matrix.csv`: intersection matrix (labels are relabeled to a compact 0..N-1 space)
- stdout: same metrics plus timing summary

## Python comparison & plots (optional)

Create a venv and install Python deps (numpy, pandas, pillow, matplotlib, scikit-image, pyyaml, opencv-python-headless):
```bash
python3 -m venv .venv_cuda
. .venv_cuda/bin/activate
pip install numpy pandas pillow matplotlib scikit-image pyyaml opencv-python-headless
```

Generate combined metrics, timings, and a plot:
```bash
. .venv_cuda/bin/activate
PYTHONPATH=change_detection/src python change_detection/cuda/compare_rcm.py \
  --mask-a outputs/segmentations/libya/samgeo/2022_mask.png \
  --mask-b outputs/segmentations/libya/samgeo/2023_mask.png \
  --cuda-metrics change_detection/cuda/metrics.csv \
  --output-dir change_detection/cuda/report
```

## Quick reference commands
- Compile: `nvcc -O3 -std=c++17 change_detection/cuda/rcm_metrics_cuda.cu change_detection/cuda/lodepng.cpp -o change_detection/cuda/rcm_metrics_cuda`
- Run (template): `./change_detection/cuda/rcm_metrics_cuda <mask_a.png> <mask_b.png> <metrics_out.csv> <matrix_out.csv>`
- Run (libya example): `./change_detection/cuda/rcm_metrics_cuda outputs/segmentations/libya/samgeo/2022_mask.png outputs/segmentations/libya/samgeo/2023_mask.png change_detection/cuda/sample_metrics.csv change_detection/cuda/sample_matrix.csv`
- Compare/plot: `. .venv_cuda/bin/activate && PYTHONPATH=change_detection/src python change_detection/cuda/compare_rcm.py --mask-a <mask_a.png> --mask-b <mask_b.png> --cuda-metrics <metrics.csv> --output-dir <report_dir>`

This creates `report/combined_metrics.csv`, `report/timing.csv`, and `report/plots.png`.

## Notes
- Masks are relabeled on CPU to dense label IDs before the GPU histogram to keep the matrix compact.
- GFLOPs is a rough estimate (3 ops/pixel divided by measured kernel time).
- For small masks the GPU kernel is fast but transfer/launch overhead dominates; larger masks should show a clearer speedup.
