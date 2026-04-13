"""Tests for the optics module: pupils, Zernike polynomials, propagation."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.config import PupilConfig, TelescopeType
from src.optics.propagator import (
    add_defocus,
    forward_model,
    make_complex_pupil,
    psf_to_pupil,
    pupil_to_psf,
)
from src.optics.pupils import _spider_mask, build_pupil
from src.optics.zernike import (
    _noll_lookup,
    radial_polynomial,
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

    def test_unknown_telescope_raises(self) -> None:
        cfg = PupilConfig(grid_size=64)
        cfg.telescope = "unknown"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown telescope"):
            build_pupil(cfg)

    def test_spider_mask_no_spiders(self) -> None:
        x = np.zeros((64, 64))
        y = np.zeros((64, 64))
        mask = _spider_mask(x, y, n_spiders=0, width_frac=0.01)
        assert np.all(mask == 1.0)

    def test_spider_mask_zero_width(self) -> None:
        x = np.zeros((64, 64))
        y = np.zeros((64, 64))
        mask = _spider_mask(x, y, n_spiders=4, width_frac=0.0)
        assert np.all(mask == 1.0)


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

    def test_psf_to_pupil_roundtrip_shape(self, pupil) -> None:
        cpupil = make_complex_pupil(pupil.amplitude, np.zeros_like(pupil.amplitude))
        from numpy.fft import fft2, fftshift, ifftshift

        G = fftshift(fft2(ifftshift(cpupil)))
        recovered = psf_to_pupil(G)
        assert recovered.shape == cpupil.shape

    def test_add_defocus_preserves_shape(self, pupil) -> None:
        phase = np.zeros_like(pupil.amplitude)
        defocused = add_defocus(phase, pupil.amplitude, defocus_waves=1.0)
        assert defocused.shape == phase.shape

    def test_add_defocus_modifies_phase(self, pupil) -> None:
        phase = np.zeros_like(pupil.amplitude)
        defocused = add_defocus(phase, pupil.amplitude, defocus_waves=1.0)
        support = pupil.amplitude > 0
        assert np.any(defocused[support] != 0.0)

    def test_forward_model_detector_effects_preserve_normalisation(self, pupil) -> None:
        phase = np.zeros_like(pupil.amplitude)
        psf = forward_model(
            pupil.amplitude,
            phase,
            detector_sigma_pixels=0.7,
            jitter_sigma_pixels=0.3,
            pixel_integration_width=1.5,
        )
        assert psf.sum() == pytest.approx(1.0, abs=1e-10)

    def test_forward_model_detector_effects_reduce_peak(self, pupil) -> None:
        phase = np.zeros_like(pupil.amplitude)
        sharp = forward_model(pupil.amplitude, phase)
        blurred = forward_model(
            pupil.amplitude,
            phase,
            detector_sigma_pixels=0.8,
            pixel_integration_width=1.8,
        )
        assert blurred.max() < sharp.max()

    def test_forward_model_polychromatic_effects_reduce_peak(self, pupil) -> None:
        phase = np.zeros_like(pupil.amplitude)
        mono = forward_model(pupil.amplitude, phase, wavelength_m=606e-9)
        poly = forward_model(
            pupil.amplitude,
            phase,
            wavelength_m=606e-9,
            bandwidth_fraction=0.15,
            spectral_samples=5,
            field_defocus_waves=0.1,
        )
        assert poly.sum() == pytest.approx(1.0, abs=1e-10)
        assert poly.max() < mono.max()


# ── Zernike polynomials ──────────────────────────────────────────────────


class TestZernike:
    def test_noll_table_first_entries(self) -> None:
        assert _noll_lookup(1) == (0, 0)  # piston
        assert _noll_lookup(2) == (1, 1)  # tip
        assert _noll_lookup(3) == (1, -1)  # tilt
        assert _noll_lookup(4) == (2, 0)  # defocus
        assert _noll_lookup(11) == (4, 0)  # spherical

    def test_piston_is_constant(self) -> None:
        n = 64
        y, x = np.mgrid[-1 : 1 : complex(0, n), -1 : 1 : complex(0, n)]  # type: ignore[misc]
        rho = np.sqrt(x**2 + y**2)
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

    def test_noll_lookup_beyond_table(self) -> None:
        """Noll indices > 37 should use the general formula."""
        for j in [38, 39, 40, 45, 50]:
            n, m = _noll_lookup(j)
            assert n >= 0
            # (n − |m|) must be even for a valid Zernike
            assert (n - abs(m)) % 2 == 0

    def test_radial_polynomial_odd_nm_returns_zero(self) -> None:
        """R_n^m is zero when (n - |m|) is odd."""
        rho = np.linspace(0, 1, 50)
        result = radial_polynomial(3, 2, rho)
        np.testing.assert_allclose(result, 0.0)

    def test_radial_polynomial_defocus(self) -> None:
        """R_2^0(rho) = 2*rho^2 - 1."""
        rho = np.linspace(0, 1, 100)
        result = radial_polynomial(2, 0, rho)
        expected = 2 * rho**2 - 1
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_zernike_basis_start_j_1(self) -> None:
        """start_j=1 includes piston."""
        basis, rho, theta = zernike_basis(5, 64, start_j=1)
        assert basis.shape == (5, 64, 64)
        # First term (piston) should be constant inside the pupil
        inside = rho <= 1.0
        vals = basis[0][inside]
        assert np.std(vals) < 1e-12
