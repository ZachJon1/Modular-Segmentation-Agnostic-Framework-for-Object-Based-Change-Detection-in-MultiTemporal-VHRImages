"""Change Vector Analysis utilities."""

from __future__ import annotations

import numpy as np

from .overlay_features import OverlayFeatures


def compute_change_vectors(
    features: OverlayFeatures,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute CVA magnitudes per overlay cell."""
    diff = features.features_t1 - features.features_t0
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape[0] != diff.shape[1]:
            raise ValueError("Weights must align with feature dimensionality.")
        diff = diff * weights[np.newaxis, :]
    magnitudes = np.linalg.norm(diff, axis=1)
    return magnitudes.astype(np.float32)


def rasterize_change_map(
    magnitudes: np.ndarray,
    overlay_map: np.ndarray,
) -> np.ndarray:
    """Assign per-cell magnitudes back to an image-sized array."""
    change_map = np.zeros_like(overlay_map, dtype=np.float32)
    for idx, value in enumerate(magnitudes):
        change_map[overlay_map == idx] = value
    return change_map


def compute_dense_cva(features: OverlayFeatures, weights: np.ndarray | None = None) -> np.ndarray:
    magnitudes = compute_change_vectors(features, weights=weights)
    return rasterize_change_map(magnitudes, features.overlay_map)

