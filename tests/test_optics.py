"""Tests for the optics module: pupils, Zernike polynomials, propagation."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.config import PupilConfig, TelescopeType
from src.optics.propagator import forward_model, make_complex_pupil, pupil_to_psf
from src.optics.pupils import build_pupil
from src.optics.zernike import (
    ZERNIKE_NAMES,
    _noll_lookup,
    zernike,
    zernike_basis,
)


# ── Pupil construction ────────────────────────────────────────────────────


class TestBuildPupil:
    @pytest.mark.parametrize("telescope", list(TelescopeType))
    def test_all_telescopes_build(self, telescope: TelescopeType) -> None:
        cfg = PupilConfig(
            telescope=telescope,
            grid_size=64,
            primary_radius=1.0,
            secondary_radius=0.3,
        )
        pupil = build_pupil(cfg)
        assert pupil.amplitude.shape == (64, 64)
        assert pupil.amplitude.min() >= 0.0
        assert pupil.amplitude.max() <= 1.0

    def test_open_fraction_is_reasonable(self, pupil_config: PupilConfig) -> None:
        pupil = build_pupil(pupil_config)
        frac = pupil.amplitude.mean()
        # A circular aperture with central obstruction → ~50-80 % open
        assert 0.2 < frac < 0.95


# ── FFT propagation ──────────────────────────────────────────────────────


class TestPropagation:
    def test_energy_conservation(self, pupil) -> None:
        """PSF should sum to ~1 (normalised)."""
        phase = np.zeros((pupil.grid_size, pupil.grid_size))
        psf = forward_model(pupil.amplitude, phase)
        assert psf.sum() == pytest.approx(1.0, abs=1e-10)

    def test_diffraction_limited_peak_is_at_centre(self, pupil) -> None:
        phase = np.zeros((pupil.grid_size, pupil.grid_size))
        psf = forward_model(pupil.amplitude, phase)
        cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
        n = pupil.grid_size
        assert abs(cy - n // 2) <= 1
        assert abs(cx - n // 2) <= 1

    def test_make_complex_pupil_amplitude(self, pupil) -> None:
        phase = np.zeros_like(pupil.amplitude)
        cpupil = make_complex_pupil(pupil.amplitude, phase)
        np.testing.assert_allclose(np.abs(cpupil), pupil.amplitude)

    def test_pupil_to_psf_non_negative(self, pupil) -> None:
        cpupil = pupil.amplitude.astype(complex)
        psf = pupil_to_psf(cpupil)
        assert np.all(psf >= 0)


# ── Zernike polynomials ──────────────────────────────────────────────────


class TestZernike:
    def test_noll_table_first_entries(self) -> None:
        assert _noll_lookup(1) == (0, 0)   # piston
        assert _noll_lookup(2) == (1, 1)   # tip
        assert _noll_lookup(3) == (1, -1)  # tilt
        assert _noll_lookup(4) == (2, 0)   # defocus
        assert _noll_lookup(11) == (4, 0)  # spherical

    def test_piston_is_constant(self) -> None:
        n = 64
        y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
        rho = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        Z1 = zernike(1, rho, theta)
        # Inside unit circle, piston should be constant
        inside = rho <= 1.0
        vals = Z1[inside]
        assert np.std(vals) < 1e-12

    def test_orthogonality(self) -> None:
        """Low-order Zernike polynomials should be approximately orthogonal."""
        n_terms = 8
        basis, rho, theta = zernike_basis(n_terms, 128, start_j=1)
        mask = rho <= 1.0
        n_pts = mask.sum()

        # Gram matrix
        G = np.zeros((n_terms, n_terms))
        for i in range(n_terms):
            for j in range(n_terms):
                G[i, j] = np.sum(basis[i][mask] * basis[j][mask]) / n_pts

        # Off-diagonal should be near zero
        off_diag = G - np.diag(np.diag(G))
        assert np.max(np.abs(off_diag)) < 0.15  # discrete grid → not perfect

    def test_basis_shape(self) -> None:
        basis, rho, theta = zernike_basis(10, 64)
        assert basis.shape == (10, 64, 64)
        assert rho.shape == (64, 64)
