"""Tests for quality metrics (Strehl, RMS, Zernike decomposition)."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.quality import (
    compute_rms_phase,
    compute_rms_wavelength,
    compute_strehl_ratio,
    zernike_decomposition,
)
from src.models.optics import PupilModel
from src.optics.propagator import forward_model


# ── RMS phase ─────────────────────────────────────────────────────────────


class TestRMSPhase:
    def test_flat_phase_is_zero(self, pupil: PupilModel, support: np.ndarray) -> None:
        phase = np.zeros((pupil.grid_size, pupil.grid_size))
        assert compute_rms_phase(phase, support) == pytest.approx(0.0)

    def test_constant_phase_is_zero(self, pupil: PupilModel, support: np.ndarray) -> None:
        """Constant phase = pure piston, which is removed → RMS = 0."""
        phase = np.full((pupil.grid_size, pupil.grid_size), 1.5)
        assert compute_rms_phase(phase, support) == pytest.approx(0.0, abs=1e-12)

    def test_nonzero_phase(self, pupil: PupilModel, support: np.ndarray, true_phase: np.ndarray) -> None:
        rms = compute_rms_phase(true_phase, support)
        assert rms > 0.0


# ── Strehl ratio ──────────────────────────────────────────────────────────


class TestStrehlRatio:
    def test_diffraction_limited_is_one(self, pupil: PupilModel) -> None:
        phase = np.zeros_like(pupil.amplitude)
        psf = forward_model(pupil.amplitude, phase)
        strehl = compute_strehl_ratio(psf, pupil.amplitude)
        assert strehl == pytest.approx(1.0, abs=1e-6)

    def test_aberrated_less_than_one(self, pupil: PupilModel, true_phase: np.ndarray) -> None:
        psf = forward_model(pupil.amplitude, true_phase)
        strehl = compute_strehl_ratio(psf, pupil.amplitude)
        assert 0.0 < strehl < 1.0


# ── RMS in wavelengths ───────────────────────────────────────────────────


class TestRMSWavelength:
    def test_conversion(self) -> None:
        rms_rad = 2 * np.pi  # 1 full wave
        assert compute_rms_wavelength(rms_rad, 500e-9) == pytest.approx(1.0)


# ── Zernike decomposition ────────────────────────────────────────────────


class TestZernikeDecomposition:
    def test_flat_phase_small_coefficients(self, pupil: PupilModel, support: np.ndarray) -> None:
        phase = np.zeros((pupil.grid_size, pupil.grid_size))
        coeffs = zernike_decomposition(phase, support, n_terms=10)
        for j, c in coeffs.items():
            assert abs(c) < 0.01, f"Z{j} should be ~0 for flat phase, got {c}"

    def test_returns_correct_keys(self, pupil: PupilModel, support: np.ndarray) -> None:
        phase = np.zeros((pupil.grid_size, pupil.grid_size))
        coeffs = zernike_decomposition(phase, support, n_terms=10)
        # Should have j = 2 .. 11 (skipping piston j=1)
        assert set(coeffs.keys()) == set(range(2, 12))
