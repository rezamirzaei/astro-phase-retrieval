"""Tests for Pydantic data models (validation, rejection of bad inputs)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.config import (
    AlgorithmConfig,
    AlgorithmName,
    DataConfig,
    PipelineConfig,
    PupilConfig,
    TelescopeType,
    default_hst_config,
)
from src.models.optics import PSFData, PupilModel, PhaseRetrievalResult


# ── PSFData ───────────────────────────────────────────────────────────────


class TestPSFData:
    def test_valid_construction(self) -> None:
        img = np.zeros((64, 64), dtype=np.float64)
        psf = PSFData(
            image=img,
            pixel_scale_arcsec=0.04,
            wavelength_m=606e-9,
            filter_name="F606W",
            telescope="hst",
        )
        assert psf.image.shape == (64, 64)

    def test_rejects_1d_image(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            PSFData(
                image=np.zeros(64),
                pixel_scale_arcsec=0.04,
                wavelength_m=606e-9,
                filter_name="F606W",
                telescope="hst",
            )

    def test_rejects_non_square(self) -> None:
        with pytest.raises(ValueError, match="square"):
            PSFData(
                image=np.zeros((64, 32)),
                pixel_scale_arcsec=0.04,
                wavelength_m=606e-9,
                filter_name="F606W",
                telescope="hst",
            )


# ── PupilModel ────────────────────────────────────────────────────────────


class TestPupilModel:
    def test_valid_construction(self) -> None:
        amp = np.ones((128, 128), dtype=np.float64)
        pm = PupilModel(amplitude=amp, grid_size=128)
        assert pm.grid_size == 128

    def test_rejects_non_square_amplitude(self) -> None:
        with pytest.raises(ValueError, match="square"):
            PupilModel(amplitude=np.ones((64, 32)), grid_size=64)


# ── AlgorithmConfig ───────────────────────────────────────────────────────


class TestAlgorithmConfig:
    def test_defaults(self) -> None:
        cfg = AlgorithmConfig()
        assert cfg.name == AlgorithmName.HYBRID_INPUT_OUTPUT
        assert cfg.beta == 0.9

    def test_rejects_beta_out_of_range(self) -> None:
        with pytest.raises(Exception):
            AlgorithmConfig(beta=1.5)

    def test_rejects_negative_iterations(self) -> None:
        with pytest.raises(Exception):
            AlgorithmConfig(max_iterations=-1)


# ── PupilConfig ───────────────────────────────────────────────────────────


class TestPupilConfig:
    def test_power_of_two_enforcement(self) -> None:
        with pytest.raises(ValueError, match="power of 2"):
            PupilConfig(grid_size=100)

    def test_accepts_valid_power_of_two(self) -> None:
        cfg = PupilConfig(grid_size=128)
        assert cfg.grid_size == 128


# ── DataConfig ────────────────────────────────────────────────────────────


class TestDataConfig:
    def test_power_of_two_enforcement(self) -> None:
        with pytest.raises(ValueError, match="power of 2"):
            DataConfig(cutout_size=50)


# ── PipelineConfig ────────────────────────────────────────────────────────


class TestPipelineConfig:
    def test_wavelength_synced_to_filter(self) -> None:
        cfg = PipelineConfig(data=DataConfig(filter_name="F814W"))
        assert cfg.pupil.wavelength_m == pytest.approx(814e-9)

    def test_default_hst_config(self) -> None:
        cfg = default_hst_config()
        assert cfg.pupil.telescope == TelescopeType.HST
        assert cfg.data.filter_name == "F606W"


# ── PhaseRetrievalResult ─────────────────────────────────────────────────


class TestPhaseRetrievalResult:
    def test_rejects_3d_phase(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            PhaseRetrievalResult(
                algorithm=AlgorithmName.ERROR_REDUCTION,
                recovered_phase=np.zeros((4, 4, 4)),
                recovered_amplitude=np.zeros((4, 4)),
                reconstructed_psf=np.zeros((4, 4)),
                n_iterations=1,
            )
