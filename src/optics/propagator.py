"""FFT-based optical propagation between pupil and focal planes."""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import gaussian_filter, uniform_filter  # type: ignore[import-untyped]


def _normalise_psf(psf: np.ndarray) -> np.ndarray:
    """Normalise a focal-plane intensity image to unit sum when possible."""
    total = psf.sum()
    if total > 0:
        psf = psf / total
    return psf


def _apply_detector_effects(
    psf: np.ndarray,
    *,
    detector_sigma_pixels: float = 0.0,
    jitter_sigma_pixels: float = 0.0,
    pixel_integration_width: float = 1.0,
) -> np.ndarray:
    """Apply simple detector-aware focal-plane effects to an intensity PSF."""
    result = psf.astype(np.float64, copy=True)

    if pixel_integration_width > 1.0:
        result = uniform_filter(result, size=float(pixel_integration_width), mode="nearest")

    blur_sigma = float(detector_sigma_pixels) + float(jitter_sigma_pixels)
    if blur_sigma > 0:
        result = gaussian_filter(result, sigma=blur_sigma, mode="nearest")

    return _normalise_psf(result)


def _spectral_grid(
    *,
    wavelength_m: float,
    bandwidth_fraction: float,
    spectral_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return wavelength samples and normalized weights for simple band averaging."""
    if spectral_samples <= 1 or bandwidth_fraction <= 0:
        return np.array([wavelength_m], dtype=np.float64), np.array([1.0], dtype=np.float64)

    half_span = 0.5 * bandwidth_fraction * wavelength_m
    wavelengths = np.linspace(
        wavelength_m - half_span,
        wavelength_m + half_span,
        spectral_samples,
        dtype=np.float64,
    )
    wavelengths = np.clip(wavelengths, 1e-12, None)
    centre = (spectral_samples - 1) / 2.0
    sigma = max(spectral_samples / 4.0, 1e-12)
    weights = np.exp(-0.5 * ((np.arange(spectral_samples, dtype=np.float64) - centre) / sigma) ** 2)
    weights /= weights.sum()
    return wavelengths, weights


def pupil_to_psf(
    pupil_complex: np.ndarray,
    *,
    detector_sigma_pixels: float = 0.0,
    jitter_sigma_pixels: float = 0.0,
    pixel_integration_width: float = 1.0,
) -> np.ndarray:
    """Propagate a complex pupil field to a focal-plane PSF intensity."""
    ft = fftshift(fft2(ifftshift(pupil_complex)))
    psf = np.abs(ft) ** 2
    psf = _normalise_psf(psf)
    return _apply_detector_effects(
        psf,
        detector_sigma_pixels=detector_sigma_pixels,
        jitter_sigma_pixels=jitter_sigma_pixels,
        pixel_integration_width=pixel_integration_width,
    )


def psf_to_pupil(psf_amplitude: np.ndarray) -> np.ndarray:
    """Inverse-propagate focal-plane amplitude back to pupil plane."""
    return fftshift(ifft2(ifftshift(psf_amplitude)))


def make_complex_pupil(
    amplitude: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    """Combine amplitude and phase into a complex pupil field."""
    return amplitude * np.exp(1j * phase)  # type: ignore[no-any-return]


def forward_model(
    amplitude: np.ndarray,
    phase: np.ndarray,
    *,
    wavelength_m: float = 606e-9,
    bandwidth_fraction: float = 0.0,
    spectral_samples: int = 1,
    field_defocus_waves: float = 0.0,
    detector_sigma_pixels: float = 0.0,
    jitter_sigma_pixels: float = 0.0,
    pixel_integration_width: float = 1.0,
) -> np.ndarray:
    """Full forward model: pupil (A, φ) → PSF intensity."""
    wavelengths, weights = _spectral_grid(
        wavelength_m=wavelength_m,
        bandwidth_fraction=bandwidth_fraction,
        spectral_samples=spectral_samples,
    )

    psf_accum = np.zeros_like(amplitude, dtype=np.float64)
    for wavelength, weight in zip(wavelengths, weights, strict=False):
        phase_sample = phase * (wavelength_m / wavelength)
        if field_defocus_waves != 0:
            phase_sample = add_defocus(
                phase_sample,
                amplitude,
                defocus_waves=field_defocus_waves,
            )
        psf_accum += float(weight) * pupil_to_psf(
            make_complex_pupil(amplitude, phase_sample),
            detector_sigma_pixels=detector_sigma_pixels,
            jitter_sigma_pixels=jitter_sigma_pixels,
            pixel_integration_width=pixel_integration_width,
        )
    return _normalise_psf(psf_accum)


def add_defocus(
    phase: np.ndarray,
    pupil_amplitude: np.ndarray,
    defocus_waves: float,
) -> np.ndarray:
    """Add a known defocus aberration to a phase map."""
    n = phase.shape[0]
    y, x = np.mgrid[-1 : 1 : complex(0, n), -1 : 1 : complex(0, n)]  # type: ignore[misc]
    rho2 = x**2 + y**2
    defocus_rad = 2 * np.pi * defocus_waves * rho2
    mask = pupil_amplitude > 0
    result = phase.copy()
    result[mask] += defocus_rad[mask]
    return result
