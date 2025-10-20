"""I/O helpers for reading imagery, masks, and metadata."""

from __future__ import annotations

import csv
import pathlib
from typing import Iterable, Tuple

import numpy as np
from PIL import Image
from skimage import io as skio
from skimage import img_as_float32, img_as_ubyte
from skimage.color import rgb2gray

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SUPPORTED_MASK_EXTS = {".png", ".tif", ".tiff", ".npy"}


class UnsupportedFileTypeError(RuntimeError):
    """Raised when attempting to load an unsupported file extension."""


def _validate_extension(path: pathlib.Path, allowed: Iterable[str]) -> None:
    if path.suffix.lower() not in allowed:
        raise UnsupportedFileTypeError(f"Unsupported file extension for {path!s}")


def load_image(path: pathlib.Path | str, *, as_gray: bool = False) -> np.ndarray:
    """Load an RGB/gray image as float32 in [0, 1]."""
    filepath = pathlib.Path(path)
    _validate_extension(filepath, SUPPORTED_IMAGE_EXTS)
    image = skio.imread(filepath)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    if as_gray:
        if image.shape[-1] == 1:
            image = image[..., 0]
        else:
            image = rgb2gray(image)
            image = image[..., np.newaxis]
    image = img_as_float32(image)
    return image


def save_image(path: pathlib.Path | str, image: np.ndarray) -> None:
    """Save a floating image (0-1) to disk as uint8."""
    filepath = pathlib.Path(path)
    ensure_parent(filepath)
    if image.dtype.kind == "f":
        data = np.clip(image, 0.0, 1.0)
        data = img_as_ubyte(data)
    elif image.dtype == np.uint8:
        data = image
    else:
        raise ValueError(f"Unsupported image dtype {image.dtype}")
    Image.fromarray(data).save(filepath)


def ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_mask(path: pathlib.Path | str) -> np.ndarray:
    """Load a segmentation mask as int32."""
    filepath = pathlib.Path(path)
    _validate_extension(filepath, SUPPORTED_MASK_EXTS)
    if filepath.suffix.lower() == ".npy":
        mask = np.load(filepath)
    else:
        mask = np.array(Image.open(filepath))
    return mask.astype(np.int32, copy=False)


def save_mask(mask: np.ndarray, path: pathlib.Path | str) -> None:
    """Persist a segmentation mask as uint16 PNG by default."""
    filepath = pathlib.Path(path)
    ensure_parent(filepath)
    mask = mask.astype(np.int32, copy=False)
    if filepath.suffix.lower() == ".npy":
        np.save(filepath, mask)
    else:
        Image.fromarray(mask.astype(np.uint16)).save(filepath)


def load_csv_mask(path: pathlib.Path | str) -> np.ndarray:
    """Load a segmentation from a CSV file (as produced in supplementary data)."""
    filepath = pathlib.Path(path)
    with filepath.open("r", newline="") as handle:
        reader = csv.reader(handle)
        rows = [[int(float(value)) for value in row] for row in reader]
    return np.asarray(rows, dtype=np.int32)


def segmentation_paths_for_site(data_root: pathlib.Path | str, site: str) -> Tuple[pathlib.Path, pathlib.Path]:
    """Return image pair paths (t0, t1) for the requested site."""
    base = pathlib.Path(data_root) / "pairs" / site
    t0 = base / "2022.png"
    t1 = base / "2023.png"
    if not t0.exists() or not t1.exists():
        raise FileNotFoundError(f"Missing imagery for site {site!r} under {base}")
    return t0, t1
