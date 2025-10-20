"""Fusion and thresholding utilities for change maps."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tifffile
from skimage.filters import threshold_multiotsu, threshold_otsu

from .io import save_image
from .utils import ensure_dir


@dataclass(frozen=True)
class FusionOutputs:
    fused: np.ndarray
    stretched: np.ndarray
    binary: np.ndarray


def fuse_validity(
    change_map: np.ndarray,
    valid_t0: np.ndarray | None,
    valid_t1: np.ndarray | None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Apply validity-aware fusion strategy."""
    fused = change_map.copy()
    if valid_t0 is None or valid_t1 is None:
        return fused
    both = (valid_t0 > 0) & (valid_t1 > 0)
    xor = (valid_t0 > 0) ^ (valid_t1 > 0)
    fused[~both & ~xor] = 0.0
    fused[xor] = gamma * fused[xor]
    return fused


def robust_stretch(
    change_map: np.ndarray,
    *,
    lower: float = 1.0,
    upper: float = 99.0,
    eps: float = 1e-6,
) -> np.ndarray:
    p1, p99 = np.percentile(change_map, [lower, upper])
    clipped = np.clip(change_map, p1, p99)
    stretched = 255.0 * (clipped - p1) / (p99 - p1 + eps)
    return stretched.astype(np.float32)


def threshold_map(
    stretched: np.ndarray,
    *,
    method: str = "multi_otsu",
    classes: int = 3,
) -> np.ndarray:
    if method == "otsu":
        threshold = threshold_otsu(stretched)
        binary = stretched >= threshold
    elif method == "multi_otsu":
        thresholds = threshold_multiotsu(stretched, classes=classes)
        binary = stretched >= thresholds[-1]
    else:
        raise ValueError(f"Unknown thresholding method {method!r}")
    return binary.astype(np.uint8)


def save_outputs(
    change_map: np.ndarray,
    stretched: np.ndarray,
    binary: np.ndarray,
    outdir: pathlib.Path | str,
) -> None:
    out = pathlib.Path(outdir)
    ensure_dir(out)
    tifffile.imwrite(out / "CVA_float.tif", change_map.astype(np.float32))
    save_image(out / "C_hat_uint8.png", (stretched / 255.0).astype(np.float32))
    save_image(out / "change_mask.png", binary.astype(np.uint8))


def fuse_and_threshold(
    change_map: np.ndarray,
    *,
    valid_t0: np.ndarray | None = None,
    valid_t1: np.ndarray | None = None,
    gamma: float = 1.0,
    method: str = "multi_otsu",
    classes: int = 3,
) -> FusionOutputs:
    fused = fuse_validity(change_map, valid_t0, valid_t1, gamma=gamma)
    stretched = robust_stretch(fused)
    binary = threshold_map(stretched, method=method, classes=classes)
    return FusionOutputs(fused=fused, stretched=stretched, binary=binary)
