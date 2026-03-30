"""Phase Retrieval for Astronomical Wavefront Sensing."""

from importlib.metadata import version, PackageNotFoundError

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
    "visualization",
]
