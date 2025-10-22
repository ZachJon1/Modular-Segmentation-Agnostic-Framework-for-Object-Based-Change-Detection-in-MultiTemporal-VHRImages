"""Utilities for composing threshold galleries."""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir
from .viz import overlay_mask


@dataclass(frozen=True)
class ThresholdResult:
    """Container for threshold outputs used in galleries."""

    method: str
    mask: np.ndarray
    percent: float


def save_threshold_gallery(
    image_t0: np.ndarray,
    image_t1: np.ndarray,
    stretched: np.ndarray,
    results: Sequence[ThresholdResult],
    outpath: pathlib.Path | str,
    *,
    overlay_method: str | None = None,
    ncols: int = 4,
    dpi: int = 200,
) -> None:
    """Persist a tile-based visualization comparing thresholding strategies."""

    if not results:
        raise ValueError("At least one threshold result is required to build a gallery.")

    outpath = pathlib.Path(outpath)
    ensure_dir(outpath.parent)

    overlay_method = overlay_method or results[0].method
    overlay_result = _pick_overlay_result(results, overlay_method)
    overlay_image = overlay_mask(image_t1, overlay_result.mask.astype(np.uint8))

    base_tiles = [
        ("T0 Image", _prepare_display(image_t0), {"cmap": None}),
        ("T1 Image", _prepare_display(image_t1), {"cmap": None}),
        ("Stretched CVA", stretched, {"cmap": "magma", "vmin": 0, "vmax": 255}),
        (
            f"Overlay ({_format_method_label(overlay_result.method)})",
            _prepare_display(overlay_image),
            {"cmap": None},
        ),
    ]

    total_tiles = len(base_tiles) + len(results)
    nrows = math.ceil(total_tiles / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.6), dpi=dpi)
    axes_iter = iter(np.asarray(axes).reshape(-1))

    for title, image, display_kwargs in base_tiles:
        ax = next(axes_iter)
        _plot_image(ax, image, title, **display_kwargs)

    for res in results:
        ax = next(axes_iter)
        label = _format_method_label(res.method)
        ax.imshow(res.mask.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"{label}\n{res.percent:.1f}% change", fontsize=10)
        ax.axis("off")
        ax.text(
            0.02,
            0.94,
            f"{res.percent:.1f}%",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color="yellow",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.6},
        )

    for ax in axes_iter:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _prepare_display(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.dtype.kind == "f":
        return np.clip(image, 0.0, 1.0)
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def _plot_image(ax, image: np.ndarray, title: str, cmap: str | None = None, **kwargs) -> None:
    ax.imshow(image if cmap is None else image, cmap=cmap, **kwargs)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _format_method_label(method: str) -> str:
    pretty = method.replace("_", " ").replace("-", " ")
    return pretty.title()


def _pick_overlay_result(results: Iterable[ThresholdResult], method: str) -> ThresholdResult:
    for res in results:
        if res.method == method:
            return res
    return next(iter(results))
