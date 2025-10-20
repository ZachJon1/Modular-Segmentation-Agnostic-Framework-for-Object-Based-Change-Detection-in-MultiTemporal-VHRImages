"""Top-level package for the change detection toolkit."""

from importlib.metadata import PackageNotFoundError, version


def _package_version() -> str:
    try:
        return version("change_detection")
    except PackageNotFoundError:
        return "0.0.0"


__all__ = ["_package_version"]
__version__ = _package_version()

