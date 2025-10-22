"""Fusion and thresholding utilities for change maps."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import tifffile
from skimage.filters import (
    apply_hysteresis_threshold,
    threshold_isodata,
    threshold_li,
    threshold_local,
    threshold_mean,
    threshold_minimum,
    threshold_multiotsu,
    threshold_niblack,
    threshold_otsu,
    threshold_sauvola,
    threshold_triangle,
    threshold_yen,
)

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
    method_kwargs: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Threshold a stretched change map using the selected strategy."""

    method_kwargs = dict(method_kwargs or {})
    canonical = _normalize_threshold_method(method, method_kwargs)
    if canonical == "multi_otsu":
        method_kwargs.setdefault("classes", classes)
    binary = _apply_threshold(canonical, stretched, method_kwargs)
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
    method_kwargs: Mapping[str, Any] | None = None,
) -> FusionOutputs:
    fused = fuse_validity(change_map, valid_t0, valid_t1, gamma=gamma)
    stretched = robust_stretch(fused)
    binary = threshold_map(
        stretched,
        method=method,
        classes=classes,
        method_kwargs=method_kwargs,
    )
    return FusionOutputs(fused=fused, stretched=stretched, binary=binary)


def _normalize_threshold_method(method: str, method_kwargs: dict[str, Any]) -> str:
    method_clean = method.strip().lower()

    if method_clean.startswith("percentile"):
        remainder = method_clean[len("percentile") :]
        remainder = remainder.lstrip("-_")
        if remainder:
            try:
                percentile = float(remainder)
                method_kwargs.setdefault("percentile", percentile)
            except ValueError:
                pass
        return "percentile"

    alias_key = method_clean.replace("-", "_").replace(" ", "")
    canonical = _THRESHOLD_ALIASES.get(alias_key, alias_key)
    if canonical not in _THRESHOLD_FUNCTIONS:
        raise ValueError(f"Unknown thresholding method {method!r}")
    return canonical


def _apply_threshold(method: str, stretched: np.ndarray, kwargs: Mapping[str, Any]) -> np.ndarray:
    handler = _THRESHOLD_FUNCTIONS[method]
    return handler(stretched, **kwargs)


def _threshold_multi_otsu(stretched: np.ndarray, *, classes: int = 3, **_: Any) -> np.ndarray:
    if classes < 2:
        raise ValueError("multi_otsu requires at least two classes")
    thresholds = threshold_multiotsu(stretched, classes=classes)
    return stretched >= thresholds[-1]


def _threshold_otsu(stretched: np.ndarray, **_: Any) -> np.ndarray:
    threshold = threshold_otsu(stretched)
    return stretched >= threshold


def _threshold_triangle(stretched: np.ndarray, **_: Any) -> np.ndarray:
    threshold = threshold_triangle(stretched)
    return stretched >= threshold


def _threshold_li(stretched: np.ndarray, **_: Any) -> np.ndarray:
    threshold = threshold_li(stretched)
    return stretched >= threshold


def _threshold_yen(stretched: np.ndarray, **_: Any) -> np.ndarray:
    threshold = threshold_yen(stretched)
    return stretched >= threshold


def _threshold_isodata(stretched: np.ndarray, **_: Any) -> np.ndarray:
    threshold = threshold_isodata(stretched)
    return stretched >= threshold


def _threshold_mean(stretched: np.ndarray, **_: Any) -> np.ndarray:
    threshold = threshold_mean(stretched)
    return stretched >= threshold


def _threshold_median(stretched: np.ndarray, **_: Any) -> np.ndarray:
    threshold = float(np.median(stretched))
    return stretched >= threshold


def _threshold_minimum(stretched: np.ndarray, **_: Any) -> np.ndarray:
    try:
        threshold = threshold_minimum(stretched)
    except RuntimeError:
        threshold = threshold_otsu(stretched)
    return stretched >= threshold


def _threshold_niblack(
    stretched: np.ndarray,
    *,
    window_size: int = 25,
    k: float = 0.2,
    **_: Any,
) -> np.ndarray:
    threshold = threshold_niblack(stretched, window_size=window_size, k=k)
    return stretched >= threshold


def _threshold_sauvola(
    stretched: np.ndarray,
    *,
    window_size: int = 25,
    k: float = 0.2,
    **_: Any,
) -> np.ndarray:
    threshold = threshold_sauvola(stretched, window_size=window_size, k=k)
    return stretched >= threshold


def _threshold_manual(
    stretched: np.ndarray,
    *,
    value: float | None = None,
    threshold: float | None = None,
    **_: Any,
) -> np.ndarray:
    manual_value = value if value is not None else threshold
    if manual_value is None:
        raise ValueError("manual thresholding requires a 'value' parameter")
    return stretched >= float(manual_value)


def _threshold_percentile(stretched: np.ndarray, *, percentile: float = 90.0, **_: Any) -> np.ndarray:
    percentile = float(percentile)
    percentile = float(np.clip(percentile, 0.0, 100.0))
    threshold = np.percentile(stretched, percentile)
    return stretched >= threshold


def _threshold_hysteresis(
    stretched: np.ndarray,
    *,
    low: float | None = None,
    high: float | None = None,
    low_ratio: float = 0.5,
    high_ratio: float = 1.0,
    reference: float | None = None,
    **_: Any,
) -> np.ndarray:
    if high is None:
        if reference is None:
            reference = threshold_otsu(stretched)
        high = float(reference * high_ratio)
    if low is None:
        low = float(high * low_ratio)
    data_min = float(stretched.min())
    data_max = float(stretched.max())
    low = float(np.clip(low, data_min, data_max))
    high = float(np.clip(high, data_min, data_max))
    if high <= low:
        raise ValueError("hysteresis requires high threshold greater than low threshold")
    return apply_hysteresis_threshold(stretched, low, high)


def _threshold_adaptive(
    stretched: np.ndarray,
    *,
    block_size: int = 35,
    offset: float = 0.0,
    method: str = "gaussian",
    **_: Any,
) -> np.ndarray:
    threshold = threshold_local(stretched, block_size=block_size, method=method, offset=offset)
    return stretched >= threshold


_THRESHOLD_FUNCTIONS: dict[str, Any] = {
    "multi_otsu": _threshold_multi_otsu,
    "otsu": _threshold_otsu,
    "triangle": _threshold_triangle,
    "li": _threshold_li,
    "yen": _threshold_yen,
    "isodata": _threshold_isodata,
    "mean": _threshold_mean,
    "median": _threshold_median,
    "minimum": _threshold_minimum,
    "niblack": _threshold_niblack,
    "sauvola": _threshold_sauvola,
    "manual": _threshold_manual,
    "percentile": _threshold_percentile,
    "hysteresis": _threshold_hysteresis,
    "adaptive": _threshold_adaptive,
}


_THRESHOLD_ALIASES: dict[str, str] = {
    "multi_otsu": "multi_otsu",
    "multiotsu": "multi_otsu",
    "otsu": "otsu",
    "triangle": "triangle",
    "li": "li",
    "yen": "yen",
    "isodata": "isodata",
    "isodata_map": "isodata",
    "isodatamap": "isodata",
    "mean": "mean",
    "median": "median",
    "minimum": "minimum",
    "niblack": "niblack",
    "sauvola": "sauvola",
    "manual": "manual",
    "hysteresis": "hysteresis",
    "adaptive": "adaptive",
    "percentile": "percentile",
    "percentile-90": "percentile",
    "percentile_90": "percentile",
    "percentile90": "percentile",
}


SUPPORTED_THRESHOLD_METHODS: tuple[str, ...] = tuple(sorted(_THRESHOLD_FUNCTIONS.keys()))
THRESHOLD_METHOD_CHOICES: tuple[str, ...] = tuple(
    sorted(
        {
            "adaptive",
            "hysteresis",
            "isodata",
            "isodata_map",
            "li",
            "manual",
            "mean",
            "median",
            "minimum",
            "multi-otsu",
            "multi_otsu",
            "multiotsu",
            "niblack",
            "otsu",
            "percentile",
            "percentile-90",
            "sauvola",
            "triangle",
            "yen",
        }
    )
)
