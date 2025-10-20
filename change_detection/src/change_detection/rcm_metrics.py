"""Region-Correlation Matrix metrics as described in the AAAI draft."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
from skimage.segmentation import relabel_sequential

from .utils import LOGGER


@dataclass(frozen=True)
class DirectionalRCM:
    overlap: float
    fragmentation: float
    composite: float


@dataclass(frozen=True)
class SymmetricRCM:
    overlap: float
    fragmentation: float
    composite: float


@dataclass(frozen=True)
class RCMResults:
    """Container mixing directional and symmetric scores."""

    forward: DirectionalRCM
    backward: DirectionalRCM
    symmetric: SymmetricRCM
    matrix: np.ndarray
    matrix_norm: np.ndarray


def _ensure_sequential(mask: np.ndarray) -> np.ndarray:
    relabeled, _, _ = relabel_sequential(mask)
    return relabeled.astype(np.int32, copy=False)


def intersection_matrix(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    if mask_a.shape != mask_b.shape:
        raise ValueError("Segmentation masks must share the same spatial dimensions.")
    a = _ensure_sequential(mask_a)
    b = _ensure_sequential(mask_b)
    n_a = int(a.max()) + 1
    n_b = int(b.max()) + 1
    joint = a.astype(np.int64) * n_b + b.astype(np.int64)
    counts = np.bincount(joint.ravel(), minlength=n_a * n_b)
    matrix = counts.reshape((n_a, n_b))
    return matrix


def _normalize(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum()
    if total == 0:
        raise ValueError("Empty intersection matrix encountered.")
    return matrix / float(total)


def _directional_scores(matrix: np.ndarray) -> DirectionalRCM:
    norm = _normalize(matrix)
    row_sums = norm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_i = row_sums  # already normalized by total
        max_per_row = norm.max(axis=1)
        p_i = np.count_nonzero(norm > 0, axis=1)
        m_i = np.divide(max_per_row, row_sums, where=row_sums > 0, out=np.zeros_like(max_per_row))
        numerator = np.sum(w_i * (p_i - 1) * (1.0 - m_i))
        denominator = np.sum(w_i * (p_i - 1))
        fragmentation = numerator / denominator if denominator > 0 else 0.0
        overlap = float(np.sum(max_per_row))

    overlap = float(np.clip(overlap, 0.0, 1.0))
    fragmentation = float(np.clip(fragmentation, 0.0, 1.0))
    composite = _composite(fragmentation, overlap)
    return DirectionalRCM(overlap=overlap, fragmentation=fragmentation, composite=composite)


def _composite(fragmentation: float, overlap: float, alpha: float = 0.5, beta: float = 0.5) -> float:
    return float(
        (alpha * fragmentation + beta * (1.0 - overlap)) / (alpha + beta),
    )


def compute_metrics(mask_a: np.ndarray, mask_b: np.ndarray) -> RCMResults:
    matrix = intersection_matrix(mask_a, mask_b)
    forward = _directional_scores(matrix)
    backward = _directional_scores(matrix.T)
    overlap_sym = np.mean([forward.overlap, backward.overlap])
    fragmentation_sym = np.mean([forward.fragmentation, backward.fragmentation])
    composite_sym = np.mean([forward.composite, backward.composite])
    symmetric = SymmetricRCM(
        overlap=overlap_sym,
        fragmentation=fragmentation_sym,
        composite=composite_sym,
    )
    norm = _normalize(matrix)
    _validate_matrix(matrix, norm)
    return RCMResults(
        forward=forward,
        backward=backward,
        symmetric=symmetric,
        matrix=matrix,
        matrix_norm=norm,
    )


def _validate_matrix(matrix: np.ndarray, matrix_norm: np.ndarray) -> None:
    row_check = np.isclose(matrix_norm.sum(axis=1), matrix.sum(axis=1) / matrix.sum(), atol=1e-6)
    if not np.all(row_check):
        LOGGER.warning("RCM validation failed for one or more rows.")


def pairwise_metrics(
    segmentations: Mapping[str, Mapping[str, np.ndarray]],
) -> pd.DataFrame:
    """Compute pairwise metrics across algorithms for both years."""
    records = []
    algorithms = sorted(segmentations.keys())
    for algo_i, algo_j in itertools.product(algorithms, repeat=2):
        if algo_i == algo_j:
            continue
        metrics_t0 = compute_metrics(segmentations[algo_i]["2022"], segmentations[algo_j]["2022"])
        metrics_t1 = compute_metrics(segmentations[algo_i]["2023"], segmentations[algo_j]["2023"])
        record = {
            "algo_i": algo_i,
            "algo_j": algo_j,
            "overlap_ab_t0": metrics_t0.forward.overlap,
            "overlap_ba_t0": metrics_t0.backward.overlap,
            "overlap_sym_t0": metrics_t0.symmetric.overlap,
            "fragmentation_ab_t0": metrics_t0.forward.fragmentation,
            "fragmentation_ba_t0": metrics_t0.backward.fragmentation,
            "fragmentation_sym_t0": metrics_t0.symmetric.fragmentation,
            "composite_ab_t0": metrics_t0.forward.composite,
            "composite_ba_t0": metrics_t0.backward.composite,
            "composite_sym_t0": metrics_t0.symmetric.composite,
            "overlap_ab_t1": metrics_t1.forward.overlap,
            "overlap_ba_t1": metrics_t1.backward.overlap,
            "overlap_sym_t1": metrics_t1.symmetric.overlap,
            "fragmentation_ab_t1": metrics_t1.forward.fragmentation,
            "fragmentation_ba_t1": metrics_t1.backward.fragmentation,
            "fragmentation_sym_t1": metrics_t1.symmetric.fragmentation,
            "composite_ab_t1": metrics_t1.forward.composite,
            "composite_ba_t1": metrics_t1.backward.composite,
            "composite_sym_t1": metrics_t1.symmetric.composite,
        }
        records.append(record)
    df = pd.DataFrame.from_records(records)
    return df
