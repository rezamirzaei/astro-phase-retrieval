"""Phase Retrieval for Astronomical Wavefront Sensing."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _local_project_version() -> str | None:
    """Return the version declared in the local pyproject when running from a checkout."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle).get("project", {})
    version_value = project.get("version")
    return version_value if isinstance(version_value, str) else None


_checkout_version = _local_project_version()

if _checkout_version is not None:
    __version__ = _checkout_version
else:
    try:
        __version__ = version("phase-retrieval")
    except PackageNotFoundError:
        __version__ = "0.0.0-dev"

__all__ = [
    "__version__",
    "algorithms",
    "data",
    "metrics",
    "models",
    "optics",
    "pipeline",
    "visualization",
]
