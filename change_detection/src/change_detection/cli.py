"""Command-line interface for the change detection toolkit."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from . import io, segmentation
from .cva import compute_dense_cva
from .fusion import fuse_and_threshold, save_outputs
from .overlay_features import extract_features
from .rcm_metrics import pairwise_metrics
from .utils import LOGGER, ensure_dir, load_yaml
from .viz import save_overlay


def _load_params(param_file: str | None) -> Mapping[str, Mapping[str, float]]:
    if not param_file:
        return {}
    path = pathlib.Path(param_file)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_yaml(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    fusion_out = fuse_and_threshold(
        change_map,
        gamma=args.gamma,
        method=args.method,
        classes=args.classes,
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
    change_parser.add_argument("--method", choices=["otsu", "multi_otsu"], default="multi_otsu")
    change_parser.add_argument("--classes", type=int, default=3)
    change_parser.set_defaults(func=cmd_change_map)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

