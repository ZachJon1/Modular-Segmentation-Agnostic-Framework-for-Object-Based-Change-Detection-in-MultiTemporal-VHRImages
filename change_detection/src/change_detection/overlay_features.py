"""Feature extraction on overlay cells between two segmentations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops


@dataclass(frozen=True)
class OverlayFeatures:
    labels: np.ndarray
    features_t0: np.ndarray
    features_t1: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    overlay_map: np.ndarray


def compute_overlay(seg_t0: np.ndarray, seg_t1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return overlay indices for each pixel."""
    max_t1 = int(seg_t1.max()) + 1
    overlay_ids = seg_t0.astype(np.int64) * max_t1 + seg_t1.astype(np.int64)
    unique_ids, inverse = np.unique(overlay_ids, return_inverse=True)
    labels = inverse.reshape(seg_t0.shape)
    return unique_ids, labels


def extract_features(
    image_t0: np.ndarray,
    image_t1: np.ndarray,
    seg_t0: np.ndarray,
    seg_t1: np.ndarray,
) -> OverlayFeatures:
    """Compute overlay cell features for both times."""
    min_h = min(seg_t0.shape[0], seg_t1.shape[0], image_t0.shape[0], image_t1.shape[0])
    min_w = min(seg_t0.shape[1], seg_t1.shape[1], image_t0.shape[1], image_t1.shape[1])

    def _crop(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            return arr[:min_h, :min_w, ...]
        return arr[:min_h, :min_w]

    seg_t0 = _crop(seg_t0)
    seg_t1 = _crop(seg_t1)
    image_t0 = _crop(image_t0)
    image_t1 = _crop(image_t1)

    unique_ids, overlay_map = compute_overlay(seg_t0, seg_t1)
    n_regions = unique_ids.shape[0]
    feats_t0 = np.zeros((n_regions, 10), dtype=np.float32)
    feats_t1 = np.zeros_like(feats_t0)

    gray0 = _to_gray(np.clip(image_t0, 0.0, 1.0))
    gray1 = _to_gray(np.clip(image_t1, 0.0, 1.0))
    gray0_u8 = (gray0 * 255).astype(np.uint8)
    gray1_u8 = (gray1 * 255).astype(np.uint8)

    sobel0_x = _sobel(gray0)
    sobel0_y = _sobel(gray0, axis="y")
    sobel1_x = _sobel(gray1)
    sobel1_y = _sobel(gray1, axis="y")

    for idx in range(n_regions):
        mask = overlay_map == idx
        feats_t0[idx] = _features_for_region(gray0_u8, sobel0_x, sobel0_y, mask)
        feats_t1[idx] = _features_for_region(gray1_u8, sobel1_x, sobel1_y, mask)

    pooled = np.vstack([feats_t0, feats_t1])
    means = pooled.mean(axis=0)
    stds = pooled.std(axis=0)
    stds[stds == 0] = 1.0

    feats_t0 = (feats_t0 - means) / stds
    feats_t1 = (feats_t1 - means) / stds

    return OverlayFeatures(
        labels=unique_ids,
        features_t0=feats_t0,
        features_t1=feats_t1,
        means=means,
        stds=stds,
        overlay_map=overlay_map,
    )


def _sobel(gray: np.ndarray, axis: str = "x") -> np.ndarray:
    import cv2

    if axis == "x":
        return cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    return cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)


def _features_for_region(
    gray_u8: np.ndarray,
    sobel_x: np.ndarray,
    sobel_y: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros(10, dtype=np.float32)

    gx_vals = sobel_x[mask]
    gy_vals = sobel_y[mask]
    grad_mean_x = float(np.mean(np.abs(gx_vals)))
    grad_mean_y = float(np.mean(np.abs(gy_vals)))
    grad_var = float(np.var(gx_vals) + np.var(gy_vals))

    values = gray_u8[mask]
    hist, _ = np.histogram(values, bins=256, range=(0, 255))
    hist_var = float(np.var(hist))

    glcm_feats = _glcm_features(gray_u8, mask)

    return np.array(
        [
            grad_mean_x,
            grad_mean_y,
            grad_var,
            hist_var,
            *glcm_feats,
        ],
        dtype=np.float32,
    )


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] == 1:
        return image[..., 0]
    return rgb2gray(image)


def _glcm_features(gray_u8: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    coords = np.argwhere(mask)
    if coords.shape[0] < 2:
        return (0.0,) * 6
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    sub_img = gray_u8[y0:y1, x0:x1]
    sub_mask = mask[y0:y1, x0:x1]
    try:
        glcm = graycomatrix(
            sub_img,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True,
            mask=sub_mask,
        )
    except TypeError:
        cleaned = sub_img.copy()
        cleaned[~sub_mask] = 0
        glcm = graycomatrix(
            cleaned,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True,
        )

    contrast = float(graycoprops(glcm, "contrast")[0, 0])
    dissimilarity = float(graycoprops(glcm, "dissimilarity")[0, 0])
    homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
    energy = float(graycoprops(glcm, "energy")[0, 0])
    correlation = float(graycoprops(glcm, "correlation")[0, 0])
    asm = float(graycoprops(glcm, "ASM")[0, 0])
    return contrast, dissimilarity, homogeneity, energy, correlation, asm
