"""Shared fixtures for the phase-retrieval test suite.

All tests run on **small synthetic data** (64×64 grids) — no real FITS files
or network access required.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.fft import fft2, fftshift, ifftshift

from src.models.config import AlgorithmConfig, AlgorithmName, PupilConfig, TelescopeType
from src.models.optics import PSFData, PupilModel
from src.optics.pupils import build_pupil

GRID = 64


@pytest.fixture()
def pupil_config() -> PupilConfig:
    """Minimal pupil config for fast tests."""
    return PupilConfig(
        telescope=TelescopeType.GENERIC_CIRCULAR,
        grid_size=GRID,
        primary_radius=1.0,
        secondary_radius=0.3,
        spider_width=0.0,
        n_spiders=0,
        wavelength_m=606e-9,
        pixel_scale_arcsec=0.04,
    )


@pytest.fixture()
def pupil(pupil_config: PupilConfig) -> PupilModel:
    """Build a simple circular pupil on a 64×64 grid."""
    return build_pupil(pupil_config)


@pytest.fixture()
def support(pupil: PupilModel) -> np.ndarray:
    """Boolean support mask."""
    return pupil.amplitude > 0


@pytest.fixture()
def true_phase(pupil: PupilModel) -> np.ndarray:
    """A mild astigmatism-like phase map for testing."""
    n = pupil.grid_size
    y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    phase = 0.5 * (x ** 2 - y ** 2) + 0.3 * x * y
    mask = pupil.amplitude > 0
    phase[~mask] = 0.0
    return phase


@pytest.fixture()
def synthetic_psf(pupil: PupilModel, true_phase: np.ndarray) -> np.ndarray:
    """Noiseless PSF generated from the pupil + true_phase."""
    g = pupil.amplitude * np.exp(1j * true_phase)
    G = fftshift(fft2(ifftshift(g)))
    psf = np.abs(G) ** 2
    psf /= psf.sum()
    return psf


@pytest.fixture()
def psf_data(synthetic_psf: np.ndarray) -> PSFData:
    """Wrap the synthetic PSF into a validated PSFData model."""
    return PSFData(
        image=synthetic_psf,
        pixel_scale_arcsec=0.04,
        wavelength_m=606e-9,
        filter_name="SYNTH",
        telescope="test",
        obs_id="synth-001",
    )


@pytest.fixture()
def hio_config() -> AlgorithmConfig:
    return AlgorithmConfig(
        name=AlgorithmName.HYBRID_INPUT_OUTPUT,
        max_iterations=80,
        beta=0.9,
        random_seed=42,
    )
