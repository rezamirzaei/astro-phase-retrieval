"""Data management service — FITS listing, preset downloading, synthetic generation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from web.config import settings

logger = logging.getLogger(__name__)


def list_fits_files() -> list[dict[str, object]]:
    """Return metadata for every FITS / ``.npy`` file under *data_dir*."""
    data_dir = settings.data_dir
    results: list[dict[str, object]] = []
    if not data_dir.exists():
        return results
    for ext in ("*.fits", "*.npy"):
        results.extend(
            {
                "filename": p.name,
                "filepath": str(p),
                "size_bytes": p.stat().st_size,
            }
            for p in sorted(data_dir.rglob(ext))
        )
    return results


def resolve_fits_path(filename: str) -> Path:
    """Find a FITS / npy file by name under *data_dir*.

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    data_dir = settings.data_dir
    for ext in ("*.fits", "*.npy"):
        for p in data_dir.rglob(ext):
            if p.name == filename:
                return p
    raise FileNotFoundError(f"No data file named '{filename}' in {data_dir}")


def generate_synthetic_psf(
    *,
    name: str = "synthetic",
    grid_size: int = 128,
    aberration_rms: float = 0.5,
    telescope: str = "hst",
    filter_name: str = "F606W",
) -> Path:
    """Create a synthetic PSF ``.npy`` file and return its path."""
    from src.models.config import PupilConfig, TelescopeType
    from src.optics.propagator import forward_model
    from src.optics.pupils import build_pupil
    from src.optics.zernike import zernike_basis

    tele = TelescopeType(telescope)
    pupil_cfg = PupilConfig(telescope=tele, grid_size=grid_size)
    pupil = build_pupil(pupil_cfg)

    # Random Zernike aberration
    n_terms = 15
    basis, _rho, _theta = zernike_basis(n_terms, grid_size, start_j=2)
    rng = np.random.default_rng(42)
    coeffs = rng.normal(0, aberration_rms, n_terms)
    phase: np.ndarray = np.tensordot(coeffs, basis, axes=1)
    phase[pupil.amplitude == 0] = 0.0

    psf = forward_model(pupil.amplitude, phase)

    out_dir = settings.data_dir / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace("/", "_").replace("\\", "_")
    out_path = out_dir / f"{safe_name}_{grid_size}.npy"
    np.save(str(out_path), psf)
    logger.info("Saved synthetic PSF → %s", out_path)
    return out_path
