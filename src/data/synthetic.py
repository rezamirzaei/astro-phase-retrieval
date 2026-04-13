"""Synthetic data generator for phase-retrieval testing and benchmarking.

Generates configurable synthetic PSF datasets with:
- Zernike-composed aberrations (arbitrary RMS level)
- Noise models (Poisson photon noise, Gaussian read noise)
- Multiple telescope pupil geometries

Usage
-----
>>> from src.data.synthetic import generate_synthetic_psf
>>> psf_data, true_phase = generate_synthetic_psf(grid_size=128, rms_aberration=0.5)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift as ndimage_shift  # type: ignore[import-untyped]

from src.models.config import PupilConfig, TelescopeType
from src.models.optics import PSFData, PupilModel
from src.optics.propagator import forward_model
from src.optics.pupils import build_pupil


@dataclass
class SyntheticDataset:
    """Container for a synthetic phase-retrieval test case."""

    psf_data: PSFData
    pupil: PupilModel
    true_phase: np.ndarray
    true_psf: np.ndarray  # noiseless PSF for reference
    noise_level: float
    center_offset_pixels: tuple[float, float] = (0.0, 0.0)
    background_level: float = 0.0
    bandwidth_fraction: float = 0.0
    spectral_weighting: str = "delta"
    field_defocus_waves: float = 0.0
    detector_sigma_pixels: float = 0.0
    jitter_sigma_pixels: float = 0.0
    pixel_integration_width: float = 1.0


def generate_synthetic_psf(
    grid_size: int = 128,
    rms_aberration: float = 0.5,
    n_zernike: int = 15,
    photon_count: float = 0.0,
    read_noise_std: float = 0.0,
    telescope: TelescopeType = TelescopeType.GENERIC_CIRCULAR,
    wavelength_m: float = 606e-9,
    random_seed: int = 42,
    center_offset_pixels: tuple[float, float] = (0.0, 0.0),
    background_level: float = 0.0,
    bandwidth_fraction: float = 0.0,
    spectral_samples: int = 1,
    spectral_weighting: str = "delta",
    field_defocus_waves: float = 0.0,
    detector_sigma_pixels: float = 0.0,
    jitter_sigma_pixels: float = 0.0,
    pixel_integration_width: float = 1.0,
) -> SyntheticDataset:
    """Generate a synthetic PSF with configurable aberrations and noise.

    Parameters
    ----------
    grid_size : int
        Square grid side length (must be power of 2).
    rms_aberration : float
        Target RMS wavefront error in radians.
    n_zernike : int
        Number of Zernike terms for the aberration (starting from j=2).
    photon_count : float
        If > 0, apply Poisson noise with this total photon count.
    read_noise_std : float
        If > 0, add Gaussian read noise with this standard deviation.
    telescope : TelescopeType
        Telescope pupil type.
    wavelength_m : float
        Observation wavelength in metres.
    random_seed : int
        RNG seed for reproducibility.

    Returns
    -------
    SyntheticDataset
        Complete synthetic dataset including PSF, pupil, and ground truth.
    """
    rng = np.random.default_rng(random_seed)
    telescope_enum = (
        telescope if isinstance(telescope, TelescopeType) else TelescopeType(str(telescope))
    )

    # Build pupil
    pupil_cfg = PupilConfig(
        telescope=telescope_enum,
        grid_size=grid_size,
        primary_radius=1.0,
        secondary_radius=0.3 if telescope_enum != TelescopeType.JWST else 0.15,
        spider_width=0.02 if telescope_enum == TelescopeType.HST else 0.0,
        n_spiders=4 if telescope_enum == TelescopeType.HST else 0,
        wavelength_m=wavelength_m,
        pixel_scale_arcsec=0.04,
        bandwidth_fraction=bandwidth_fraction,
        spectral_samples=spectral_samples,
        spectral_weighting=spectral_weighting,
        field_defocus_waves=field_defocus_waves,
        detector_sigma_pixels=detector_sigma_pixels,
        jitter_sigma_pixels=jitter_sigma_pixels,
        pixel_integration_width=pixel_integration_width,
    )
    pupil = build_pupil(pupil_cfg)
    support = pupil.amplitude > 0

    # Generate Zernike-composed aberration
    true_phase = _generate_zernike_phase(
        grid_size,
        n_zernike,
        rms_aberration,
        support,
        rng,
    )

    # Generate noiseless PSF
    noiseless_psf = forward_model(
        pupil.amplitude,
        true_phase,
        **pupil.forward_model_kwargs(),
    )

    if center_offset_pixels != (0.0, 0.0):
        noiseless_psf = ndimage_shift(
            noiseless_psf,
            shift=center_offset_pixels,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float64)
        total = noiseless_psf.sum()
        if total > 0:
            noiseless_psf /= total

    # Add noise
    noisy_psf = noiseless_psf.copy()
    noise_level = 0.0

    if photon_count > 0:
        # Poisson photon noise
        counts = rng.poisson(noiseless_psf * photon_count)
        noisy_psf = counts.astype(np.float64) / photon_count
        noise_level += 1.0 / np.sqrt(photon_count)

    if read_noise_std > 0:
        # Gaussian read noise
        read_noise = rng.normal(0, read_noise_std, noisy_psf.shape)
        noisy_psf = np.maximum(noisy_psf + read_noise, 0.0)
        noise_level += read_noise_std

    if background_level > 0:
        noisy_psf = noisy_psf + background_level
        noise_level += background_level

    # Normalise
    total = noisy_psf.sum()
    if total > 0:
        noisy_psf /= total

    psf_data = PSFData(
        image=noisy_psf,
        pixel_scale_arcsec=0.04,
        wavelength_m=wavelength_m,
        filter_name="SYNTH",
        telescope=telescope_enum.value,
        obs_id=f"synth-{random_seed}",
        metadata={
            "source_kind": "synthetic",
            "center_offset_pixels": [
                float(center_offset_pixels[0]),
                float(center_offset_pixels[1]),
            ],
            "background_level": float(background_level),
            "photon_count": float(photon_count),
            "read_noise_std": float(read_noise_std),
            "rms_aberration_rad": float(rms_aberration),
            "bandwidth_fraction": float(bandwidth_fraction),
            "spectral_samples": int(spectral_samples),
            "spectral_weighting": str(spectral_weighting),
            "field_defocus_waves": float(field_defocus_waves),
            "detector_sigma_pixels": float(detector_sigma_pixels),
            "jitter_sigma_pixels": float(jitter_sigma_pixels),
            "pixel_integration_width": float(pixel_integration_width),
        },
    )

    return SyntheticDataset(
        psf_data=psf_data,
        pupil=pupil,
        true_phase=true_phase,
        true_psf=noiseless_psf,
        noise_level=noise_level,
        center_offset_pixels=(float(center_offset_pixels[0]), float(center_offset_pixels[1])),
        background_level=float(background_level),
        bandwidth_fraction=float(bandwidth_fraction),
        spectral_weighting=str(spectral_weighting),
        field_defocus_waves=float(field_defocus_waves),
        detector_sigma_pixels=float(detector_sigma_pixels),
        jitter_sigma_pixels=float(jitter_sigma_pixels),
        pixel_integration_width=float(pixel_integration_width),
    )


def _generate_zernike_phase(
    grid_size: int,
    n_terms: int,
    rms_target: float,
    support: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a random Zernike-composed phase aberration.

    Draws random coefficients for Zernike polynomials j=2..n_terms+1,
    then scales the resulting phase map to achieve the target RMS.
    """
    from src.optics.zernike import zernike_basis

    basis, rho, theta = zernike_basis(n_terms, grid_size, start_j=2)

    # Random Zernike coefficients (normally distributed)
    coefficients = rng.standard_normal(n_terms)

    # Compose phase map
    phase = np.zeros((grid_size, grid_size), dtype=np.float64)
    for i in range(n_terms):
        phase += coefficients[i] * basis[i]

    # Apply support
    phase[~support] = 0.0

    # Remove piston and scale to target RMS
    vals = phase[support]
    if len(vals) > 0:
        phase[support] -= vals.mean()
        current_rms = np.sqrt(np.mean(phase[support] ** 2))
        if current_rms > 0:
            phase *= rms_target / current_rms

    return phase
