"""Batch experiment runner following the AAAI workflow."""

from __future__ import annotations

import argparse
import pathlib
import shutil
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from change_detection import io, segmentation
from change_detection.cva import compute_dense_cva
from change_detection.fusion import fuse_and_threshold, save_outputs
from change_detection.overlay_features import extract_features
from change_detection.rcm_metrics import pairwise_metrics
from change_detection.utils import LOGGER, ensure_dir, load_yaml
from change_detection.viz import save_overlay


def load_config(path: pathlib.Path) -> Mapping[str, object]:
    config = load_yaml(path)
    if not config:
        raise ValueError(f"Empty configuration file at {path}")
    return config


def _ensure_site_imagery(data_root: pathlib.Path, site_cfg: Mapping[str, object]) -> None:
    site_name = site_cfg["name"]
    site_dir = data_root / "pairs" / site_name
    ensure_dir(site_dir)
    t0_dest = site_dir / "2022.png"
    t1_dest = site_dir / "2023.png"

    def _copy_if_needed(src_field: str, dest: pathlib.Path, *, year: str | None = None) -> None:
        src_value = site_cfg.get(src_field)
        if not src_value:
            return
        src_path = pathlib.Path(src_value)
        if not src_path.is_absolute():
            src_path = data_root / src_path
        if not src_path.exists():
            LOGGER.warning("Source %s (%s) missing for site %s.", src_field, src_path, site_name)
            return
        if year and src_path.suffix.lower() == ".csv":
            dest = dest.with_name(f"{year}labels.csv")
        elif year and dest.suffix != src_path.suffix and src_path.suffix:
            dest = dest.with_suffix(src_path.suffix)
        if dest.exists():
            return
        shutil.copyfile(src_path, dest)

    _copy_if_needed("t0", t0_dest)
    _copy_if_needed("t1", t1_dest)

    rezsd_dir = data_root / "rezsd" / site_name
    ensure_dir(rezsd_dir)
    _copy_if_needed("rezsd_t0", rezsd_dir / "2022_mask.png", year="2022")
    _copy_if_needed("rezsd_t1", rezsd_dir / "2023_mask.png", year="2023")


def run_experiments(config: Mapping[str, object]) -> None:
    data_root = pathlib.Path(config["data_root"])
    algorithms = config.get("algorithms", [])
    if not algorithms:
        raise ValueError("Config must include a non-empty `algorithms` list.")
    algo_params = config.get("params", {})
    fusion_params = config.get("fusion", {"method": "multi_otsu", "classes": 3})

    seg_out_root = pathlib.Path("outputs/segmentations")
    metrics_root = pathlib.Path("outputs/metrics")
    change_root = pathlib.Path("outputs/change_maps")
    summary_rows = []

    sites = config.get("sites", [])
    if not sites:
        raise ValueError("Config missing `sites`.")

    for site_cfg in sites:
        site_name = site_cfg["name"]
        LOGGER.info("=== Site: %s ===", site_name)
        _ensure_site_imagery(data_root, site_cfg)
        # Ensure imagery exists
        t0_path, t1_path = io.segmentation_paths_for_site(data_root, site_name)
        image_t0 = io.load_image(t0_path)
        image_t1 = io.load_image(t1_path)

        seg_summaries = segmentation.segment_site(
            data_root=data_root,
            site=site_name,
            algos=algorithms,
            outdir=seg_out_root,
            algo_params=algo_params,
        )
        region_counts: Dict[str, Dict[str, int]] = {}
        for summary in seg_summaries:
            region_counts.setdefault(summary.algo, {})[summary.year] = summary.n_regions

        seg_path_by_algo = {
            algo: {
                "2022": seg_out_root / site_name / segmentation.resolve_algo(algo) / "2022_mask.png",
                "2023": seg_out_root / site_name / segmentation.resolve_algo(algo) / "2023_mask.png",
            }
            for algo in algorithms
        }
        # Metrics
        try:
            segmentations_map = {
                segmentation.resolve_algo(algo): {
                    "2022": io.load_mask(paths["2022"]),
                    "2023": io.load_mask(paths["2023"]),
                }
                for algo, paths in seg_path_by_algo.items()
            }
            metrics_df = pairwise_metrics(segmentations_map)
            metrics_path = metrics_root / f"{site_name}_pairwise.csv"
            ensure_dir(metrics_path.parent)
            metrics_df.to_csv(metrics_path, index=False)
            LOGGER.info("RCM metrics saved to %s", metrics_path)
        except FileNotFoundError as exc:
            LOGGER.warning("Skipping metrics for %s: %s", site_name, exc)

        # Change maps
        for algo in algorithms:
            algo_key = segmentation.resolve_algo(algo)
            seg_t0_path = seg_path_by_algo[algo]["2022"]
            seg_t1_path = seg_path_by_algo[algo]["2023"]
            if not seg_t0_path.exists() or not seg_t1_path.exists():
                LOGGER.warning("Skipping change map for %s/%s due to missing segmentation.", site_name, algo)
                continue
            seg_t0 = io.load_mask(seg_t0_path)
            seg_t1 = io.load_mask(seg_t1_path)
            features = extract_features(image_t0, image_t1, seg_t0, seg_t1)
            change_map = compute_dense_cva(features)
            fusion_out = fuse_and_threshold(change_map, **fusion_params)

            algo_out = change_root / site_name / algo_key
            save_outputs(change_map, fusion_out.stretched, fusion_out.binary, algo_out)
            save_overlay(image_t1, fusion_out.binary, algo_out / "change_overlay.png")
            change_pct = float(fusion_out.binary.mean() * 100.0)
            summary_rows.append(
                {
                    "site": site_name,
                    "algo": algo_key,
                    "n_regions_t0": region_counts.get(algo_key, {}).get("2022", np.nan),
                    "n_regions_t1": region_counts.get(algo_key, {}).get("2023", np.nan),
                    "change_pct": change_pct,
                }
            )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = pathlib.Path("outputs/summaries/summary.csv")
        ensure_dir(summary_path.parent)
        summary_df.to_csv(summary_path, index=False)
        LOGGER.info("Summaries stored in %s", summary_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run change detection experiments.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    run_experiments(config)


if __name__ == "__main__":
    main()
