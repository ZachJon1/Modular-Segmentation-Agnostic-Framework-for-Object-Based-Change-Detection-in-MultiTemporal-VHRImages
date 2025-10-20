"""Segmentation algorithms leveraged by the change detection pipeline."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import cv2
import numpy as np
from skimage.segmentation import felzenszwalb, quickshift, relabel_sequential

from . import io
from .utils import LOGGER, ensure_dir
from .viz import save_segmentation_overlay


SAM_DEFAULT_WEIGHTS = pathlib.Path(__file__).resolve().parents[2] / "sam_vit_h_4b8939.pth"


ALIAS_MAP = {
    "felz": "felzenszwalb",
    "felzenszwalb": "felzenszwalb",
    "quick": "quickshift",
    "quickshift": "quickshift",
    "mean_shift": "meanshift",
    "meanshift": "meanshift",
    "samgeo": "samgeo",
    "rezsd": "rezsd",
    "rezsd_load": "rezsd",
}


@dataclass
class SegmentationSummary:
    algo: str
    site: str
    year: str
    n_regions: int
    mask_path: pathlib.Path


def resolve_algo(name: str) -> str:
    """Normalize algorithm aliases."""
    key = name.lower()
    if key not in ALIAS_MAP:
        raise KeyError(f"Unsupported segmentation algorithm {name!r}")
    return ALIAS_MAP[key]


def run_single(
    algo: str,
    *,
    image: np.ndarray,
    image_path: pathlib.Path,
    data_root: pathlib.Path,
    site: str,
    year: str,
    outdir: pathlib.Path,
    params: MutableMapping[str, float] | None = None,
) -> SegmentationSummary:
    """Execute a segmentation algorithm and persist the result."""
    algo_name = resolve_algo(algo)
    params = dict(params or {})

    if algo_name == "felzenszwalb":
        mask = _segment_felzenszwalb(image, params)
    elif algo_name == "quickshift":
        mask = _segment_quickshift(image, params)
    elif algo_name == "meanshift":
        mask = _segment_meanshift(image, params)
    elif algo_name == "samgeo":
        mask = _segment_samgeo(image_path, params, outdir, site, year)
    elif algo_name == "rezsd":
        mask = _load_rezsd_mask(data_root, site, year, params)
    else:
        raise KeyError(f"Unknown algorithm {algo}")

    ensure_dir(outdir)
    mask_path = outdir / f"{year}_mask.png"
    io.save_mask(mask, mask_path)
    overlay_alpha = float(params.get("overlay_alpha", 0.5)) if isinstance(params, dict) else 0.5
    overlay_path = outdir / f"{year}_overlay.png"
    save_segmentation_overlay(image, mask, overlay_path, alpha=overlay_alpha)
    n_regions = int(np.unique(mask).size)
    LOGGER.info(
        "%s | %s %s → %d regions",
        algo_name,
        site,
        year,
        n_regions,
    )
    return SegmentationSummary(
        algo=algo_name,
        site=site,
        year=year,
        n_regions=n_regions,
        mask_path=mask_path,
    )


def segment_site(
    *,
    data_root: pathlib.Path | str,
    site: str,
    algos: Sequence[str],
    outdir: pathlib.Path | str,
    algo_params: Mapping[str, Mapping[str, float]] | None = None,
) -> Sequence[SegmentationSummary]:
    """Run the requested algorithms for both years of a site."""
    base = pathlib.Path(data_root)
    out_base = pathlib.Path(outdir) / site
    summaries: list[SegmentationSummary] = []
    t0_path, t1_path = io.segmentation_paths_for_site(base, site)
    images = {
        "2022": io.load_image(t0_path),
        "2023": io.load_image(t1_path),
    }
    for algo in algos:
        params = (algo_params or {}).get(resolve_algo(algo), {})
        for year, image_path in (("2022", t0_path), ("2023", t1_path)):
            summary = run_single(
                algo,
                image=images[year],
                image_path=image_path,
                data_root=base,
                site=site,
                year=year,
                outdir=out_base / resolve_algo(algo),
                params=dict(params),
            )
            summaries.append(summary)
    return summaries


def _segment_felzenszwalb(image: np.ndarray, params: MutableMapping[str, float]) -> np.ndarray:
    defaults = {"scale": 250, "sigma": 0.8, "min_size": 200}
    defaults.update(params)
    segments = felzenszwalb(image, **defaults)
    relabeled, _, _ = relabel_sequential(segments)
    return relabeled.astype(np.int32)


def _segment_quickshift(image: np.ndarray, params: MutableMapping[str, float]) -> np.ndarray:
    defaults = {"kernel_size": 5, "max_dist": 10, "ratio": 0.5}
    defaults.update(params)
    segments = quickshift(image, convert2lab=True, **defaults)
    relabeled, _, _ = relabel_sequential(segments)
    return relabeled.astype(np.int32)


def _segment_meanshift(image: np.ndarray, params: MutableMapping[str, float]) -> np.ndarray:
    defaults = {"sp": 21, "sr": 21, "max_level": 1, "quantization": 16}
    defaults.update(params)
    scaled = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    filtered = cv2.pyrMeanShiftFiltering(
        scaled,
        sp=float(defaults["sp"]),
        sr=float(defaults["sr"]),
        maxLevel=int(defaults["max_level"]),
    )
    quant = max(int(defaults.get("quantization", 16)), 1)
    reduced = (filtered // quant).astype(np.int32)
    levels = int(np.ceil(256 / quant))
    encoded = (
        reduced[..., 0] * (levels**2)
        + reduced[..., 1] * levels
        + reduced[..., 2]
    )
    relabeled, _, _ = relabel_sequential(encoded)
    return relabeled.astype(np.int32)


def _segment_samgeo(
    image_path: pathlib.Path,
    params: MutableMapping[str, float],
    outdir: pathlib.Path,
    site: str,
    year: str,
) -> np.ndarray:
    precomputed = params.pop("mask_path", None)
    if precomputed:
        return io.load_mask(precomputed)

    try:
        from samgeo import SamGeo
    except ImportError as exc:
        raise ImportError(
            "samgeo is required for the SAMGeo segmentation. "
            "Install samgeo or provide `mask_path` within params."
        ) from exc
    try:
        import torch
    except ImportError as exc:
        raise ImportError("SAMGeo requires PyTorch to determine the compute device.") from exc

    model_type = params.pop("model_type", "vit_h")
    checkpoint = params.pop("checkpoint", None)
    if checkpoint is None and SAM_DEFAULT_WEIGHTS.exists():
        checkpoint = SAM_DEFAULT_WEIGHTS
    if checkpoint is not None:
        checkpoint = pathlib.Path(checkpoint)
    sam_kwargs = params.pop("sam_kwargs", None)
    mask_path = outdir / f"{year}_samgeo.tif"
    ensure_dir(mask_path.parent)

    if sam_kwargs is None:
        sam_kwargs = {
            "points_per_side": 32,
            "pred_iou_thresh": 0.86,
            "stability_score_thresh": 0.92,
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 100,
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = SamGeo(
        model_type=model_type,
        sam_kwargs=sam_kwargs,
        checkpoint=str(checkpoint) if checkpoint else None,
        device=device,
    )
    mask_output = mask_path
    sam.generate(str(image_path), output=str(mask_output), foreground=True)
    mask = io.load_mask(mask_output)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.int32, copy=False)
    relabeled, _, _ = relabel_sequential(mask)
    return relabeled.astype(np.int32)


def _load_rezsd_mask(
    data_root: pathlib.Path,
    site: str,
    year: str,
    params: MutableMapping[str, float],
) -> np.ndarray:
    base_dir = data_root / "rezsd" / site
    override = params.get("paths") if isinstance(params, dict) else None
    if override:
        mask_path = pathlib.Path(override.get(year, ""))
        if not mask_path.is_absolute():
            mask_path = base_dir / mask_path
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing REZsd mask override at {mask_path}")
        return _load_mask_any(mask_path)

    candidates = [
        base_dir / f"{year}_mask.png",
        base_dir / f"{year}.png",
        base_dir / f"{year}labels.png",
        base_dir / f"{year}_mask.tif",
        base_dir / f"{year}.tif",
        base_dir / f"{year}labels.tif",
        base_dir / f"{year}labels.csv",
        base_dir / f"{year}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return _load_mask_any(candidate)
    raise FileNotFoundError(f"Missing REZsd mask for {site} {year} under {base_dir}")


def _load_mask_any(path: pathlib.Path) -> np.ndarray:
    if path.suffix.lower() == ".csv":
        mask = io.load_csv_mask(path)
    else:
        mask = io.load_mask(path)
    relabeled, _, _ = relabel_sequential(mask)
    return relabeled.astype(np.int32)
