"""Tests for quality metrics (Strehl, RMS, Zernike, MTF, SSIM, Phase Structure Function)."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.quality import (
    compute_mtf,
    compute_phase_structure_function,
    compute_rms_phase,
    compute_rms_wavelength,
    compute_ssim,
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

    def test_nonzero_phase(
        self, pupil: PupilModel, support: np.ndarray, true_phase: np.ndarray
    ) -> None:
        rms = compute_rms_phase(true_phase, support)
        assert rms > 0.0

    def test_empty_support_returns_zero(self) -> None:
        phase = np.ones((64, 64))
        support = np.zeros((64, 64), dtype=bool)
        assert compute_rms_phase(phase, support) == 0.0


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

    def test_zero_pupil_returns_zero(self) -> None:
        pupil_amp = np.zeros((64, 64))
        psf = np.zeros((64, 64))
        assert compute_strehl_ratio(psf, pupil_amp) == 0.0


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

    def test_empty_support(self) -> None:
        phase = np.ones((64, 64))
        support = np.zeros((64, 64), dtype=bool)
        coeffs = zernike_decomposition(phase, support, n_terms=5)
        for c in coeffs.values():
            assert c == 0.0


# ── MTF ───────────────────────────────────────────────────────────────────


class TestMTF:
    def test_mtf_dc_is_one(self, pupil: PupilModel) -> None:
        """MTF at zero frequency should be 1.0 (normalised)."""
        phase = np.zeros_like(pupil.amplitude)
        psf = forward_model(pupil.amplitude, phase)
        freqs, mtf_profile = compute_mtf(psf)
        assert mtf_profile[0] == pytest.approx(1.0, abs=0.05)

    def test_mtf_monotone_decreasing_for_perfect_psf(self, pupil: PupilModel) -> None:
        """For a diffraction-limited PSF, MTF should generally decrease with frequency."""
        phase = np.zeros_like(pupil.amplitude)
        psf = forward_model(pupil.amplitude, phase)
        freqs, mtf_profile = compute_mtf(psf)
        # First few points should be higher than last few
        assert np.mean(mtf_profile[:5]) > np.mean(mtf_profile[-5:])

    def test_mtf_returns_correct_length(self, pupil: PupilModel) -> None:
        psf = forward_model(pupil.amplitude, np.zeros_like(pupil.amplitude))
        freqs, mtf_profile = compute_mtf(psf)
        assert len(freqs) == len(mtf_profile)
        assert len(freqs) == pupil.grid_size // 2


# ── SSIM ──────────────────────────────────────────────────────────────────


class TestSSIM:
    def test_identical_images_ssim_one(self, pupil: PupilModel) -> None:
        psf = forward_model(pupil.amplitude, np.zeros_like(pupil.amplitude))
        ssim = compute_ssim(psf, psf)
        assert ssim == pytest.approx(1.0, abs=0.01)

    def test_different_images_ssim_less_than_one(
        self, pupil: PupilModel, true_phase: np.ndarray
    ) -> None:
        psf_perfect = forward_model(pupil.amplitude, np.zeros_like(pupil.amplitude))
        psf_aberrated = forward_model(pupil.amplitude, true_phase)
        ssim = compute_ssim(psf_perfect, psf_aberrated)
        assert ssim < 1.0

    def test_explicit_data_range(self, pupil: PupilModel) -> None:
        psf = forward_model(pupil.amplitude, np.zeros_like(pupil.amplitude))
        ssim = compute_ssim(psf, psf, data_range=1.0)
        assert ssim == pytest.approx(1.0, abs=0.01)

    def test_zero_images_fallback(self) -> None:
        """Two all-zero images: data_range computed as 0, fallback to 1.0."""
        z = np.zeros((32, 32))
        ssim = compute_ssim(z, z)
        assert isinstance(ssim, float)


# ── Phase Structure Function ─────────────────────────────────────────────


class TestPhaseStructureFunction:
    def test_flat_phase_zero_structure(self, pupil: PupilModel, support: np.ndarray) -> None:
        phase = np.zeros((pupil.grid_size, pupil.grid_size))
        seps, sf = compute_phase_structure_function(phase, support, max_sep=10)
        assert len(seps) == 10
        for val in sf:
            assert val == pytest.approx(0.0, abs=1e-10)

    def test_nonzero_phase_positive_structure(
        self, pupil: PupilModel, support: np.ndarray, true_phase: np.ndarray
    ) -> None:
        seps, sf = compute_phase_structure_function(true_phase, support, max_sep=10)
        # Structure function should be non-negative and generally increasing
        assert all(v >= 0 for v in sf)
        assert sf[-1] > sf[0]  # larger separations → larger phase differences

    def test_default_max_sep(self, pupil: PupilModel, support: np.ndarray) -> None:
        """When max_sep is not given, should default to grid_size // 4."""
        phase = np.zeros((pupil.grid_size, pupil.grid_size))
        seps, sf = compute_phase_structure_function(phase, support)
        assert len(seps) == pupil.grid_size // 4
