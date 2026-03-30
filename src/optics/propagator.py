"""FFT-based optical propagation between pupil and focal planes."""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def pupil_to_psf(pupil_complex: np.ndarray) -> np.ndarray:
    """Propagate a complex pupil field to a focal-plane PSF intensity.

    Parameters
    ----------
    pupil_complex : ndarray
        Complex pupil field A(x,y) · exp(i·φ(x,y)).

    Returns
    -------
    ndarray
        Normalised focal-plane intensity |F{pupil}|².
    """
    ft = fftshift(fft2(ifftshift(pupil_complex)))
    psf = np.abs(ft) ** 2
    total = psf.sum()
    if total > 0:
        psf /= total
    return psf


def psf_to_pupil(psf_amplitude: np.ndarray) -> np.ndarray:
    """Inverse-propagate focal-plane amplitude back to pupil plane.

    Parameters
    ----------
    psf_amplitude : ndarray
        Complex focal-plane field (amplitude with phase estimate).

    Returns
    -------
    ndarray
        Complex pupil-plane field.
    """
    return fftshift(ifft2(ifftshift(psf_amplitude)))


def make_complex_pupil(
    amplitude: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    """Combine amplitude and phase into a complex pupil field.

    Parameters
    ----------
    amplitude : ndarray
        Pupil amplitude (0/1 mask or soft apodisation).
    phase : ndarray
        Pupil phase in radians.

    Returns
    -------
    ndarray
        Complex field A · exp(iφ).
    """
    return amplitude * np.exp(1j * phase)


def forward_model(
    amplitude: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    """Full forward model: pupil (A, φ) → PSF intensity.

    Parameters
    ----------
    amplitude : ndarray
        Pupil amplitude mask.
    phase : ndarray
        Pupil phase (radians).

    Returns
    -------
    ndarray
        Focal-plane PSF intensity (normalised).
    """
    cpupil = make_complex_pupil(amplitude, phase)
    return pupil_to_psf(cpupil)


def add_defocus(
    phase: np.ndarray,
    pupil_amplitude: np.ndarray,
    defocus_waves: float,
) -> np.ndarray:
    """Add a known defocus aberration to a phase map.

    Parameters
    ----------
    phase : ndarray
        Current pupil phase (radians).
    pupil_amplitude : ndarray
        Pupil amplitude mask (used to determine pupil extent).
    defocus_waves : float
        Peak-to-valley defocus in waves (λ).

    Returns
    -------
    ndarray
        Phase + defocus (radians).
    """
    n = phase.shape[0]
    y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    rho2 = x ** 2 + y ** 2
    defocus_rad = 2 * np.pi * defocus_waves * rho2  # Seidel defocus
    mask = pupil_amplitude > 0
    result = phase.copy()
    result[mask] += defocus_rad[mask]
    return result

