"""FFT-based optical propagation between pupil and focal planes."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift  # type: ignore[import-untyped]


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
    """Apply detector transfer effects using a simple OTF/MTF model."""
    result = psf.astype(np.float64, copy=True)
    ny, nx = result.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    fx_grid, fy_grid = np.meshgrid(fx, fy)

    pixel_mtf = np.sinc(fx_grid * pixel_integration_width) * np.sinc(
        fy_grid * pixel_integration_width
    )
    blur_sigma = float(detector_sigma_pixels) + float(jitter_sigma_pixels)
    blur_mtf = np.exp(-2.0 * np.pi**2 * blur_sigma**2 * (fx_grid**2 + fy_grid**2))

    spectrum = fft2(ifftshift(result), workers=-1)
    filtered = np.real(fftshift(ifft2(spectrum * pixel_mtf * blur_mtf, workers=-1)))
    filtered[filtered < 0] = 0.0
    return _normalise_psf(filtered)


def _spectral_grid(
    *,
    wavelength_m: float,
    bandwidth_fraction: float,
    spectral_samples: int,
    spectral_weighting: str = "delta",
) -> tuple[np.ndarray, np.ndarray]:
    """Return wavelength samples and normalized weights for band averaging."""
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
    if spectral_weighting == "uniform":
        weights = np.ones(spectral_samples, dtype=np.float64)
    else:
        centre = wavelength_m
        sigma_m = max((bandwidth_fraction * wavelength_m) / 2.355, 1e-12)
        weights = np.exp(-0.5 * ((wavelengths - centre) / sigma_m) ** 2)
    weights = weights / weights.sum()
    return wavelengths, weights


def pupil_to_psf(
    pupil_complex: np.ndarray,
    *,
    detector_sigma_pixels: float = 0.0,
    jitter_sigma_pixels: float = 0.0,
    pixel_integration_width: float = 1.0,
) -> np.ndarray:
    """Propagate a complex pupil field to a focal-plane PSF intensity."""
    ft = fftshift(fft2(ifftshift(pupil_complex), workers=-1))
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
    return fftshift(ifft2(ifftshift(psf_amplitude), workers=-1))


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
    spectral_weighting: str = "delta",
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
        spectral_weighting=spectral_weighting,
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
    """Add a Zernike-like defocus aberration to a phase map."""
    n = phase.shape[0]
    y, x = np.mgrid[-1 : 1 : complex(0, n), -1 : 1 : complex(0, n)]  # type: ignore[misc]
    rho = np.sqrt(x**2 + y**2)
    defocus_zernike = 2.0 * rho**2 - 1.0
    mask = (pupil_amplitude > 0) & (rho <= 1.0)
    defocus_rad = np.pi * defocus_waves * defocus_zernike
    result = phase.copy()
    result[mask] += defocus_rad[mask]
    return result
