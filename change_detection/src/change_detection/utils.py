"""Shared utility helpers used across the change detection package."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, MutableMapping

import yaml


LOGGER = logging.getLogger("change_detection")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


def ensure_dir(path: pathlib.Path | str) -> pathlib.Path:
    """Create a directory if it does not yet exist and return it."""
    directory = pathlib.Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml(path: pathlib.Path | str) -> Dict[str, Any]:
    """Load a YAML file returning a dictionary."""
    with pathlib.Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def save_json(data: Mapping[str, Any], path: pathlib.Path | str, *, indent: int = 2) -> None:
    """Persist a mapping to disk as JSON."""
    with pathlib.Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent)


@dataclass(frozen=True)
class Timer:
    """Convenience wall-clock timer."""

    label: str
    start_time: float = time.time()

    def __enter__(self) -> "Timer":
        object.__setattr__(self, "start_time", time.time())
        LOGGER.info("⏱️ %s ...", self.label)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.time() - self.start_time
        LOGGER.info("✅ %s in %.2fs", self.label, elapsed)


def slugify(text: str) -> str:
    """Slugify a string for filesystem usage."""
    import re

    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-")


def iter_sites(data_root: pathlib.Path | str) -> Iterator[pathlib.Path]:
    """Yield site directories within ``pairs`` under the provided data root."""
    base = pathlib.Path(data_root) / "pairs"
    if not base.exists():
        return iter(())
    for path in sorted(base.iterdir()):
        if path.is_dir():
            yield path


def merge_dicts(
    base: MutableMapping[str, Any],
    override: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Recursively merge nested dictionaries."""

    def _merge(target: MutableMapping[str, Any], src: Mapping[str, Any]) -> None:
        for key, value in src.items():
            if (
                key in target
                and isinstance(target[key], MutableMapping)
                and isinstance(value, Mapping)
            ):
                _merge(target[key], value)
            else:
                target[key] = value

    _merge(base, override)
    return base


@contextlib.contextmanager
def working_directory(path: pathlib.Path | str) -> Iterator[None]:
    """Temporarily switch the working directory."""
    previous = pathlib.Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)
