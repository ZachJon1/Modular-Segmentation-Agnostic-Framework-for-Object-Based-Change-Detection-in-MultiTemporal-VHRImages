Change Detection Experiments
============================

This repository automates the change‐detection workflow described in the AAAI paper. It takes bi‑temporal UAV imagery, produces multi‑algorithm segmentations, computes Region Correlation Matrix (RCM) metrics, extracts overlay features, runs Change Vector Analysis (CVA), and exports qualitative overlays and quantitative summaries.

*RESDA described in paper == REZsd in code*

Directory Layout
----------------

```
├── change_detection/              # Source package and experiment scripts
│   ├── environment.yml            # Conda environment definition
│   ├── experiments/               # Batch configs (YAML) and runner
│   └── src/change_detection/      # Library modules (segmentation, CVA, fusion…)
├── outputs/                       # Generated artefacts (created after running)
│   ├── segmentations/<site>/<algo>/
│   │   ├── {year}_mask.png        # Segmentation labels (uint16 PNG)
│   │   ├── {year}_overlay.png     # Colour overlay for visual QA
│   │   └── {year}_samgeo.tif      # (SAMGeo) GeoTIFF export from SamGeo
│   ├── metrics/<site>_pairwise.csv          # RCM tables (all algorithm pairs)
│   ├── change_maps/<site>/<algo>/           # CVA artefacts (float map, stretched PNG, binary mask, overlay, summary)
│   └── summaries/segmentations_<site>.csv, summary.csv 
└── README.md                       # This file
```

Data Requirements
-----------------

Place imagery and REZsd masks under `change_detection/data` following the instructions:

```
change_detection/data/
├── pairs/<site>/{2022,2023}.png         # RGB images (one per year)
└── rezsd/<site>/<year>labels.csv        # REZsd segmentation labels (CSV)
```

If REZsd masks are supplied as PNG/TIF (or the legacy JPG previews), store them alongside the CSV; the loader auto-detects whichever extension is present and normalises labels to sequential integers.

The repository already includes sample assets in `supplementary/data/images` and `change_detection/data/rezsd/` for Ann Arbor, Hathaway, HathawayNorth, Rosebud, and Libya. Current runs write results to `outputs/`; any older `change_detection/outputs/` folders are legacy artefacts from early smoke tests.

Environment Setup
-----------------

Create the conda environment (includes `samgeo`, `torch`, `scikit-image`, `opencv`, etc.):

```bash
conda env create -f change_detection/environment.yml
conda activate change_detection
```

Ensure the SAM checkpoint (`sam_vit_h_4b8939.pth`) is available at project root or specify `checkpoint` in the config/CLI.

SAM Checkpoint Helper
---------------------

The 2.6 GB SAM ViT-H weights are not stored in git. Fetch them into `change_detection/sam_vit_h_4b8939.pth` with:

```bash
./scripts/download_sam_weights.sh
```

The script downloads the file directly from the official Segment Anything CDN, verifies its SHA-256 hash, and keeps the temporary download out of version control. Use `./scripts/download_sam_weights.sh --force` to re-download if the file becomes corrupted. The path is already listed in `.gitignore`, so the checkpoint will never be committed by accident.

Command-Line Workflow
---------------------

Segmentations (run per site):

```bash
python -m change_detection.cli segment \
  --data-root change_detection/data \
  --site rosebud \
  --algos felzenszwalb meanshift quickshift rezsd samgeo \
  --outdir outputs/segmentations \
  --out-summary outputs/summaries/segmentations_rosebud.csv
```

RCM metrics:

```bash
python -m change_detection.cli compare-segmentations \
  --segs-root outputs/segmentations/rosebud \
  --outcsv outputs/metrics/rosebud_pairwise.csv
```

CVA & overlays:

```bash
python -m change_detection.cli change-map \
  --data-root change_detection/data \
  --site rosebud \
  --algo samgeo \
  --seg-path-t0 outputs/segmentations/rosebud/samgeo/2022_mask.png \
  --seg-path-t1 outputs/segmentations/rosebud/samgeo/2023_mask.png \
  --outdir outputs/change_maps/rosebud/samgeo \
  --method multi_otsu --classes 3 \
  --method-param block_size=51
```

Outputs include the dense CVA map (`CVA_float.tif`), stretched preview (`C_hat_uint8.png`), binary change mask, and a `change_overlay.png` highlighting the mask on top of the t₁ image.

`--method` now accepts a comprehensive suite of global and local thresholding strategies (multi-otsu, otsu, triangle, li, yen, isodata, mean, median, minimum, niblack, sauvola, adaptive, hysteresis, percentile‑*, manual). Extra parameters can be provided with repeated `--method-param key=value` flags, e.g. `--method adaptive --method-param block_size=41 --method-param offset=-5`.

Threshold gallery (builds a comparison mosaic for a site/algo pair):

```bash
python -m change_detection.cli threshold-gallery \
  --data-root change_detection/data \
  --site libya \
  --algo samgeo \
  --seg-path-t0 outputs/segmentations/libya/samgeo/2022_mask.png \
  --seg-path-t1 outputs/segmentations/libya/samgeo/2023_mask.png \
  --outdir outputs/change_maps/libya/samgeo \
  --overlay-method multi_otsu \
  --ncols 4 --dpi 220
```

The command renders `threshold_gallery.png` (tile layout including T0/T1 imagery, stretched CVA, overlay, and per-method binary masks annotated with percentage change) plus a machine-readable `threshold_summary.json` containing per-method statistics and parameter values.

Batch Experiments
-----------------

Edit `change_detection/experiments/configs/default.yaml` to list sites, algorithms, and parameters. Then run:

```bash
python change_detection/experiments/run_experiments.py \
  --config change_detection/experiments/configs/default.yaml
```

This orchestrates segmentations (Felzenszwalb, MeanShift, Quickshift, REZsd, SAMGeo), generates `outputs/segmentations`, pairwise RCM CSVs, CVA maps, overlays, and summary tables. The shipped config already lists all five algorithms and their starter parameters; adjust the `params:` block to tune hyperparameters (e.g., `meanshift.quantization`, `quickshift.max_dist`, `samgeo.sam_kwargs`), or override paths to alternative data roots.

Key Modules
-----------

- `segmentation.py` – wraps `skimage`, OpenCV mean shift, SAMGeo, and REZsd loaders; writes overlays.
- `overlay_features.py` – extracts gradient, histogram, and GLCM features on overlay cells.
- `cva.py` – computes CVA magnitudes and rasterises them.
- `fusion.py` – applies validity-aware fusion, percentile stretch, and a pluggable threshold registry (global, local, hysteresis, percentile, manual).
- `gallery.py` – assembles threshold comparison mosaics (T0, T1, stretched CVA, overlay, per-method masks).
- `rcm_metrics.py` – builds intersection matrices and reports overlap/fragmentation/composite scores.
- `viz.py` – generates binary or label-colour overlays.
- `cli.py` – exposes the pipeline as `segment`, `compare-segmentations`, `change-map`, `threshold-gallery`.
- `experiments/run_experiments.py` – batch driver leveraging the YAML config.

Results Overview
----------------

`outputs/summaries/summary.csv` collects algorithm counts and change percentages per site. Example (Rosebud):

```
algo         n_regions_t0  n_regions_t1  change_pct
felzenszwalb       284            336        6.93
meanshift          326            372        3.30
quickshift         845            908        2.16
rezsd                5              6       58.88
samgeo             193            193        3.84
```

Tips & Notes
------------

- The CLI honours custom parameters via `--param-file` pointing to a JSON/YAML mapping.
- Segmentations are normalized to sequential labels before metric calculations.
- Segmentation overlays (`outputs/segmentations/.../{year}_overlay.png`) provide an easy QA pass for mask quality; CVA overlays (`change_overlay.png`) highlight detected change on t₁ imagery.
- REZsd CSV labels are parsed directly with `load_csv_mask`, so no manual conversion is required.
- Long-running SAMGeo and Quickshift steps can be parallelised across sites if desired.
