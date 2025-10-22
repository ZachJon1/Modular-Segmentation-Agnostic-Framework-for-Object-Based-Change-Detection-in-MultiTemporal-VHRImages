"""Command-line interface for the change detection toolkit."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from . import io, segmentation
from .cva import compute_dense_cva
from .fusion import (
    THRESHOLD_METHOD_CHOICES,
    fuse_and_threshold,
    fuse_validity,
    robust_stretch,
    save_outputs,
    threshold_map,
)
from .gallery import ThresholdResult, save_threshold_gallery
from .overlay_features import extract_features
from .rcm_metrics import pairwise_metrics
from .utils import LOGGER, ensure_dir, load_yaml
from .viz import save_overlay


DEFAULT_GALLERY_METHODS = [
    "multi_otsu",
    "otsu",
    "triangle",
    "li",
    "yen",
    "isodata",
    "mean",
    "median",
    "minimum",
    "niblack",
    "sauvola",
    "adaptive",
    "hysteresis",
    "percentile-90",
    "manual",
]


def _load_params(param_file: str | None) -> Mapping[str, Mapping[str, float]]:
    if not param_file:
        return {}
    path = pathlib.Path(param_file)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_yaml(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_method_params(pairs: list[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid method parameter {item!r}; expected key=value format")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Empty key in method parameter {item!r}")
        params[key] = _autocast(value)
    return params


def _autocast(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if value.startswith("0") and value != "0" and not value.startswith("0."):
            # Treat leading-zero strings as strings to avoid octal interpretations
            raise ValueError
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def cmd_segment(args: argparse.Namespace) -> None:
    params = _load_params(args.param_file)
    summaries = segmentation.segment_site(
        data_root=args.data_root,
        site=args.site,
        algos=args.algos,
        outdir=args.outdir,
        algo_params=params.get("params", params),
    )
    summary_rows = {}
    for summary in summaries:
        summary_rows.setdefault(summary.algo, {})[summary.year] = summary.n_regions
    rows = []
    for algo, metrics in summary_rows.items():
        rows.append(
            {
                "site": args.site,
                "algo": algo,
                "n_regions_t0": metrics.get("2022", np.nan),
                "n_regions_t1": metrics.get("2023", np.nan),
            }
        )
    df = pd.DataFrame(rows)
    ensure_dir(args.out_summary.parent)
    with args.out_summary.open("w", encoding="utf-8") as handle:
        df.to_csv(handle, index=False)
    LOGGER.info("Segmentation summary written to %s", args.out_summary)


def cmd_compare(args: argparse.Namespace) -> None:
    segs_root = pathlib.Path(args.segs_root)
    algorithms = [d for d in segs_root.iterdir() if d.is_dir()]
    if not algorithms:
        raise FileNotFoundError(f"No segmentations found under {segs_root}")
    segmentations_map: Dict[str, Dict[str, np.ndarray]] = {}
    for algo_dir in algorithms:
        masks = {
            "2022": io.load_mask(algo_dir / "2022_mask.png"),
            "2023": io.load_mask(algo_dir / "2023_mask.png"),
        }
        segmentations_map[algo_dir.name] = masks
    df = pairwise_metrics(segmentations_map)
    ensure_dir(args.outcsv.parent)
    df.to_csv(args.outcsv, index=False)
    LOGGER.info("Pairwise metrics written to %s", args.outcsv)


def cmd_change_map(args: argparse.Namespace) -> None:
    data_root = pathlib.Path(args.data_root)
    t0_path, t1_path = io.segmentation_paths_for_site(data_root, args.site)
    image_t0 = io.load_image(t0_path)
    image_t1 = io.load_image(t1_path)
    seg_t0 = io.load_mask(args.seg_path_t0)
    seg_t1 = io.load_mask(args.seg_path_t1)

    features = extract_features(image_t0, image_t1, seg_t0, seg_t1)
    change_map = compute_dense_cva(features)
    method_kwargs = _parse_method_params(args.method_param or [])
    fusion_out = fuse_and_threshold(
        change_map,
        gamma=args.gamma,
        method=args.method,
        classes=args.classes,
        method_kwargs=method_kwargs,
    )
    outdir = pathlib.Path(args.outdir)
    save_outputs(change_map, fusion_out.stretched, fusion_out.binary, outdir)
    save_overlay(image_t1, fusion_out.binary, outdir / "change_overlay.png")

    change_pct = float(fusion_out.binary.mean() * 100.0)
    summary_path = outdir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "site": args.site,
                "algo": args.algo,
                "change_pct": change_pct,
            },
            handle,
            indent=2,
        )
    LOGGER.info("CVA outputs saved to %s (≈%.2f%% change)", outdir, change_pct)


def cmd_threshold_gallery(args: argparse.Namespace) -> None:
    data_root = pathlib.Path(args.data_root)
    t0_path, t1_path = io.segmentation_paths_for_site(data_root, args.site)
    image_t0 = io.load_image(t0_path)
    image_t1 = io.load_image(t1_path)
    seg_t0 = io.load_mask(args.seg_path_t0)
    seg_t1 = io.load_mask(args.seg_path_t1)

    features = extract_features(image_t0, image_t1, seg_t0, seg_t1)
    change_map = compute_dense_cva(features)
    fused = fuse_validity(change_map, valid_t0=None, valid_t1=None, gamma=args.gamma)
    stretched = robust_stretch(fused)

    methods = args.methods or DEFAULT_GALLERY_METHODS
    common_kwargs = _parse_method_params(args.method_param or [])

    results: list[ThresholdResult] = []
    summary_rows: list[Dict[str, Any]] = []

    for method in methods:
        method_kwargs = dict(common_kwargs)
        key = method.lower().replace("-", "_")
        if key in {"niblack", "sauvola"}:
            method_kwargs.setdefault("window_size", args.local_window)
        if key == "sauvola":
            method_kwargs.setdefault("k", args.sauvola_k)
        if key == "adaptive":
            method_kwargs.setdefault("block_size", args.local_window)
            method_kwargs.setdefault("offset", args.adaptive_offset)
            method_kwargs.setdefault("method", args.adaptive_method)
        if key == "hysteresis":
            method_kwargs.setdefault("low_ratio", args.hysteresis_low_ratio)
            method_kwargs.setdefault("high_ratio", args.hysteresis_high_ratio)
        if key == "manual" and "value" not in method_kwargs and "threshold" not in method_kwargs:
            percentile = float(args.manual_percentile)
            data_min = float(stretched.min())
            data_max = float(stretched.max())
            below_max = stretched[stretched < data_max]
            if below_max.size:
                manual_value = float(np.percentile(below_max, percentile))
            else:
                manual_value = data_max
            if manual_value >= data_max:
                manual_value = max(data_max - 1.0, data_min)
            if manual_value <= data_min:
                manual_value = min(data_min + 1.0, data_max)
            method_kwargs["value"] = manual_value

        binary = threshold_map(
            stretched,
            method=method,
            classes=args.classes,
            method_kwargs=method_kwargs,
        )
        percent = float(binary.mean() * 100.0)
        results.append(ThresholdResult(method=method, mask=binary, percent=percent))
        summary_rows.append(
            {
                "method": method,
                "percent": percent,
                "params": {k: _json_safe(v) for k, v in method_kwargs.items()},
            }
        )
        LOGGER.info("Method %-12s -> %.2f%% change", method, percent)

    outdir = pathlib.Path(args.outdir)
    ensure_dir(outdir)
    gallery_path = outdir / args.gallery_name
    save_threshold_gallery(
        image_t0,
        image_t1,
        stretched,
        results,
        gallery_path,
        overlay_method=args.overlay_method,
        ncols=args.ncols,
        dpi=args.dpi,
    )

    summary_path = outdir / args.summary_name
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "site": args.site,
            "algo": args.algo,
            "gamma": args.gamma,
            "classes": args.classes,
            "overlay_method": args.overlay_method,
            "methods": summary_rows,
        }, handle, indent=2)
    LOGGER.info("Threshold gallery saved to %s", gallery_path)
    LOGGER.info("Threshold summary saved to %s", summary_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="change_detection")
    sub = parser.add_subparsers(dest="command", required=True)

    seg_parser = sub.add_parser("segment", help="Run segmentation algorithms.")
    seg_parser.add_argument("--data-root", type=pathlib.Path, required=True)
    seg_parser.add_argument("--site", required=True)
    seg_parser.add_argument("--algos", nargs="+", required=True)
    seg_parser.add_argument("--outdir", type=pathlib.Path, required=True)
    seg_parser.add_argument("--param-file", type=str, default=None)
    seg_parser.add_argument(
        "--out-summary",
        type=pathlib.Path,
        default=pathlib.Path("outputs/summaries/segmentations.csv"),
    )
    seg_parser.set_defaults(func=cmd_segment)

    cmp_parser = sub.add_parser("compare-segmentations", help="Compute RCM metrics.")
    cmp_parser.add_argument("--segs-root", type=pathlib.Path, required=True)
    cmp_parser.add_argument("--outcsv", type=pathlib.Path, required=True)
    cmp_parser.set_defaults(func=cmd_compare)

    change_parser = sub.add_parser("change-map", help="Run CVA and fusion.")
    change_parser.add_argument("--data-root", type=pathlib.Path, required=True)
    change_parser.add_argument("--site", required=True)
    change_parser.add_argument("--algo", required=True)
    change_parser.add_argument("--seg-path-t0", type=pathlib.Path, required=True)
    change_parser.add_argument("--seg-path-t1", type=pathlib.Path, required=True)
    change_parser.add_argument("--outdir", type=pathlib.Path, required=True)
    change_parser.add_argument("--gamma", type=float, default=1.0)
    change_parser.add_argument("--method", choices=THRESHOLD_METHOD_CHOICES, default="multi_otsu")
    change_parser.add_argument("--classes", type=int, default=3)
    change_parser.add_argument(
        "--method-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional threshold parameters (repeatable).",
    )
    change_parser.set_defaults(func=cmd_change_map)

    gallery_parser = sub.add_parser("threshold-gallery", help="Create a threshold comparison gallery.")
    gallery_parser.add_argument("--data-root", type=pathlib.Path, required=True)
    gallery_parser.add_argument("--site", required=True)
    gallery_parser.add_argument("--algo", default="unknown")
    gallery_parser.add_argument("--seg-path-t0", type=pathlib.Path, required=True)
    gallery_parser.add_argument("--seg-path-t1", type=pathlib.Path, required=True)
    gallery_parser.add_argument("--outdir", type=pathlib.Path, required=True)
    gallery_parser.add_argument("--gamma", type=float, default=1.0)
    gallery_parser.add_argument("--classes", type=int, default=3)
    gallery_parser.add_argument(
        "--methods",
        nargs="+",
        choices=THRESHOLD_METHOD_CHOICES,
        default=None,
        help="Threshold methods to include in the gallery (default: comprehensive set).",
    )
    gallery_parser.add_argument(
        "--overlay-method",
        choices=THRESHOLD_METHOD_CHOICES,
        default="multi_otsu",
        help="Method used for the overlay tile.",
    )
    gallery_parser.add_argument(
        "--method-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional threshold parameters applied to all methods (repeatable).",
    )
    gallery_parser.add_argument("--manual-percentile", type=float, default=85.0)
    gallery_parser.add_argument("--local-window", type=int, default=31)
    gallery_parser.add_argument("--sauvola-k", type=float, default=0.2)
    gallery_parser.add_argument("--adaptive-offset", type=float, default=0.0)
    gallery_parser.add_argument("--adaptive-method", default="gaussian")
    gallery_parser.add_argument("--hysteresis-low-ratio", type=float, default=0.5)
    gallery_parser.add_argument("--hysteresis-high-ratio", type=float, default=1.0)
    gallery_parser.add_argument("--ncols", type=int, default=4)
    gallery_parser.add_argument("--dpi", type=int, default=220)
    gallery_parser.add_argument("--gallery-name", default="threshold_gallery.png")
    gallery_parser.add_argument("--summary-name", default="threshold_summary.json")
    gallery_parser.set_defaults(func=cmd_threshold_gallery)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
