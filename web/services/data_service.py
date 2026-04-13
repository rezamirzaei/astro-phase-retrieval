"""Data management service — FITS listing, preset downloading, synthetic generation."""

from __future__ import annotations

import json
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
    n_zernike: int = 15,
    telescope: str = "hst",
    filter_name: str = "F606W",
    photon_count: float = 0.0,
    read_noise_std: float = 0.0,
    center_offset_pixels: tuple[float, float] = (0.0, 0.0),
    background_level: float = 0.0,
    bandwidth_fraction: float = 0.0,
    spectral_samples: int = 1,
    spectral_weighting: str = "delta",
    field_defocus_waves: float = 0.0,
    detector_sigma_pixels: float = 0.0,
    jitter_sigma_pixels: float = 0.0,
    pixel_integration_width: float = 1.0,
    random_seed: int = 42,
) -> Path:
    """Create a synthetic PSF ``.npy`` file and return its path."""
    from src.data.synthetic import generate_synthetic_psf as _generate_synthetic_dataset

    dataset = _generate_synthetic_dataset(
        grid_size=grid_size,
        rms_aberration=aberration_rms,
        n_zernike=n_zernike,
        photon_count=photon_count,
        read_noise_std=read_noise_std,
        telescope=telescope,
        random_seed=random_seed,
        center_offset_pixels=center_offset_pixels,
        background_level=background_level,
        bandwidth_fraction=bandwidth_fraction,
        spectral_samples=spectral_samples,
        spectral_weighting=spectral_weighting,
        field_defocus_waves=field_defocus_waves,
        detector_sigma_pixels=detector_sigma_pixels,
        jitter_sigma_pixels=jitter_sigma_pixels,
        pixel_integration_width=pixel_integration_width,
    )

    out_dir = settings.data_dir / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace("/", "_").replace("\\", "_")
    out_path = out_dir / f"{safe_name}_{grid_size}.npy"
    metadata_path = out_dir / f"{safe_name}_{grid_size}.json"
    np.save(str(out_path), dataset.psf_data.image)
    metadata_path.write_text(
        json.dumps(
            {
                "filename": out_path.name,
                "filter_name": filter_name,
                "generation": {
                    "grid_size": grid_size,
                    "aberration_rms": aberration_rms,
                    "n_zernike": n_zernike,
                    "telescope": telescope,
                    "photon_count": photon_count,
                    "read_noise_std": read_noise_std,
                    "center_offset_pixels": [
                        float(center_offset_pixels[0]),
                        float(center_offset_pixels[1]),
                    ],
                    "background_level": background_level,
                    "bandwidth_fraction": bandwidth_fraction,
                    "spectral_samples": spectral_samples,
                    "spectral_weighting": spectral_weighting,
                    "field_defocus_waves": field_defocus_waves,
                    "detector_sigma_pixels": detector_sigma_pixels,
                    "jitter_sigma_pixels": jitter_sigma_pixels,
                    "pixel_integration_width": pixel_integration_width,
                    "random_seed": random_seed,
                },
                "psf_metadata": dataset.psf_data.metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved synthetic PSF → %s", out_path)
    return out_path
