"""
Compare CUDA C++ RCM metrics against the existing Python implementation.

Outputs:
- combined_metrics.csv with forward/backward/symmetric metrics from both implementations
- timing.csv with CPU (Python), CPU intersection (C++), and GPU timings
- plots.png showing metric bars and timing bars
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from change_detection import rcm_metrics


def load_cuda_metrics(metrics_csv: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    df = pd.read_csv(metrics_csv, header=0)
    metrics_rows = df[df["direction"].isin({"forward", "backward", "symmetric"})].copy()
    timing_rows = df[~df["direction"].isin({"forward", "backward", "symmetric"})]
    timings: dict[str, float] = {}
    for _, row in timing_rows.iterrows():
        key = str(row["direction"])
        val = row["overlap"]
        if pd.notna(val):
            timings[key] = float(val)
    return metrics_rows, timings


def run_python_metrics(mask_a_path: Path, mask_b_path: Path) -> tuple[rcm_metrics.RCMResults, float]:
    mask_a = np.asarray(Image.open(mask_a_path)).astype(np.int32)
    mask_b = np.asarray(Image.open(mask_b_path)).astype(np.int32)
    t0 = time.time()
    res = rcm_metrics.compute_metrics(mask_a, mask_b)
    ms = (time.time() - t0) * 1000.0
    return res, ms


def build_combined(cuda_metrics: pd.DataFrame, py_res: rcm_metrics.RCMResults) -> pd.DataFrame:
    rows = []
    for direction, vals in [
        ("forward", py_res.forward),
        ("backward", py_res.backward),
        ("symmetric", py_res.symmetric),
    ]:
        rows.append(
            {
                "direction": direction,
                "impl": "python",
                "overlap": vals.overlap,
                "fragmentation": vals.fragmentation,
                "composite": vals.composite,
            }
        )
    for _, row in cuda_metrics.iterrows():
        rows.append(
            {
                "direction": row["direction"],
                "impl": "cuda_cpp",
                "overlap": row["overlap"],
                "fragmentation": row["fragmentation"],
                "composite": row["composite"],
            }
        )
    return pd.DataFrame(rows)


def plot_results(combined: pd.DataFrame, timings: dict[str, float], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    directions = ["forward", "backward", "symmetric"]
    metrics = ["overlap", "fragmentation", "composite"]
    width = 0.35
    x = np.arange(len(metrics))
    colors = {"python": "#4e79a7", "cuda_cpp": "#f28e2b"}
    for idx, direction in enumerate(directions):
        ax = axes[0] if idx == 0 else axes[0]
    # Single stacked plot combining directions for clarity
    axes[0].clear()
    for i, direction in enumerate(directions):
        subset = combined[combined["direction"] == direction]
        for j, impl in enumerate(["python", "cuda_cpp"]):
            data = subset[subset["impl"] == impl][metrics].iloc[0].values
            axes[0].bar(x + (j - 0.5) * width + i * (width * 2 + 0.1), data, width, label=f"{direction}-{impl}" if i == 0 else "", color=colors.get(impl, None))
    axes[0].set_xticks([r + width / 2 for r in range(len(metrics) + 2)])
    axes[0].set_xticklabels(metrics + ["", ""])
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].set_title("RCM metrics")

    # Timing comparison
    timing_plot = {k: v for k, v in timings.items() if not k.startswith("gflops")}
    labels = list(timing_plot.keys())
    tvals = [timing_plot[k] for k in labels]
    axes[1].bar(labels, tvals, color=["#4e79a7", "#59a14f", "#f28e2b"])
    axes[1].set_ylabel("ms")
    axes[1].set_title("Timing comparison")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CUDA C++ RCM metrics to Python implementation.")
    parser.add_argument("--mask-a", required=True, type=Path)
    parser.add_argument("--mask-b", required=True, type=Path)
    parser.add_argument("--cuda-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cuda_metrics, cuda_timings = load_cuda_metrics(args.cuda_metrics)
    py_res, py_ms = run_python_metrics(args.mask_a, args.mask_b)
    combined = build_combined(cuda_metrics, py_res)
    combined.to_csv(args.output_dir / "combined_metrics.csv", index=False)

    # Timing summary: Python full, CUDA GPU total, CUDA CPU intersection (from csv)
    gpu_total = cuda_timings.get("timing_ms_gpu", np.nan)
    cpu_intersection = cuda_timings.get("timing_ms_cpu", np.nan)
    gflops_est = cuda_timings.get("gflops_est", np.nan)
    timings = {
        "python_full": py_ms,
        "cpu_intersection_cpp": cpu_intersection,
        "gpu_total_cpp": gpu_total,
        "gflops_est_cpp": gflops_est,
    }
    pd.DataFrame([timings]).to_csv(args.output_dir / "timing.csv", index=False)
    plot_results(combined, timings, args.output_dir / "plots.png")


if __name__ == "__main__":
    main()
