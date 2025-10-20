"""Visualization helpers."""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
from skimage.color import label2rgb

from .io import save_image
from .utils import ensure_dir


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a binary mask on top of an image."""
    if image.shape[:2] != mask.shape[:2]:
        min_h = min(image.shape[0], mask.shape[0])
        min_w = min(image.shape[1], mask.shape[1])
        image = image[:min_h, :min_w, ...] if image.ndim == 3 else image[:min_h, :min_w]
        mask = mask[:min_h, :min_w]
    if image.ndim == 2:
        base = np.stack([image, image, image], axis=-1)
    else:
        base = image.copy()
    if base.dtype != np.uint8:
        base = np.clip(base * 255, 0, 255).astype(np.uint8)
    overlay = base.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)
    return blended


def save_overlay(image: np.ndarray, mask: np.ndarray, outpath: pathlib.Path | str) -> None:
    outpath = pathlib.Path(outpath)
    ensure_dir(outpath.parent)
    over = overlay_mask(image, mask)
    save_image(outpath, over.astype(np.uint8))


def segmentation_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = 0.5,
    bg_label: int = -1,
) -> np.ndarray:
    """Return an image with segmentation regions color-coded atop the base image."""
    if image.ndim == 2:
        base = np.stack([image, image, image], axis=-1)
    else:
        base = image.copy()
    if base.dtype != np.float32 and base.dtype != np.float64:
        base = base.astype(np.float32)
        if base.dtype == np.uint8 or base.max() > 1.0:
            base = base / 255.0
    else:
        base = np.clip(base, 0.0, 1.0)
    overlay = label2rgb(labels, image=base, alpha=alpha, bg_label=bg_label, kind="overlay")
    return (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)


def save_segmentation_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    outpath: pathlib.Path | str,
    *,
    alpha: float = 0.5,
    bg_label: int = -1,
) -> None:
    """Persist a segmentation overlay image."""
    outpath = pathlib.Path(outpath)
    ensure_dir(outpath.parent)
    overlay = segmentation_overlay(image, labels, alpha=alpha, bg_label=bg_label)
    save_image(outpath, overlay)
