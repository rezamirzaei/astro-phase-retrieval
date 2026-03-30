"""Quality metrics for evaluating phase-retrieval results.

Includes Strehl ratio, RMS wavefront error, and Zernike decomposition.
"""

from __future__ import annotations

import numpy as np

from src.optics.propagator import pupil_to_psf, make_complex_pupil


def compute_rms_phase(phase: np.ndarray, support: np.ndarray) -> float:
    """Compute RMS wavefront error over the pupil support.

    Parameters
    ----------
    phase : ndarray
        Pupil-plane phase (radians).
    support : ndarray
        Boolean mask of the pupil region.

    Returns
    -------
    float
        RMS phase error in radians.
    """
    vals = phase[support]
    if len(vals) == 0:
        return 0.0
    vals = vals - vals.mean()  # Remove piston
    return float(np.sqrt(np.mean(vals ** 2)))


def compute_strehl_ratio(psf: np.ndarray, pupil_amplitude: np.ndarray) -> float:
    """Estimate the Strehl ratio.

    Strehl = peak(aberrated PSF) / peak(diffraction-limited PSF).

    Parameters
    ----------
    psf : ndarray
        Reconstructed (aberrated) PSF.
    pupil_amplitude : ndarray
        Pupil amplitude mask.

    Returns
    -------
    float
        Strehl ratio (0–1).
    """
    # Diffraction-limited PSF (zero phase)
    perfect_psf = pupil_to_psf(make_complex_pupil(pupil_amplitude, np.zeros_like(pupil_amplitude)))
    peak_perfect = perfect_psf.max()
    peak_aberrated = psf.max()
    if peak_perfect == 0:
        return 0.0
    return float(min(peak_aberrated / peak_perfect, 1.0))


def compute_rms_wavelength(rms_rad: float, wavelength_m: float) -> float:
    """Convert RMS phase (radians) to RMS wavefront error in waves.

    Parameters
    ----------
    rms_rad : float
        RMS phase in radians.
    wavelength_m : float
        Wavelength in metres.

    Returns
    -------
    float
        RMS error in units of wavelength (waves).
    """
    return rms_rad / (2 * np.pi)


def zernike_decomposition(
    phase: np.ndarray,
    support: np.ndarray,
    n_terms: int = 22,
) -> dict[int, float]:
    """Decompose recovered phase into Zernike coefficients.

    Parameters
    ----------
    phase : ndarray
        Recovered pupil-plane phase (radians).
    support : ndarray
        Boolean pupil support mask.
    n_terms : int
        Number of Zernike terms (starting from j=2, skipping piston).

    Returns
    -------
    dict[int, float]
        Noll index → coefficient (radians).
    """
    from src.optics.zernike import zernike_basis

    n = phase.shape[0]
    basis, rho, theta = zernike_basis(n_terms, n, start_j=2)

    coefficients: dict[int, float] = {}
    for idx in range(n_terms):
        j = idx + 2  # Noll index
        z = basis[idx]
        mask = support & (rho <= 1.0)
        if mask.sum() == 0:
            coefficients[j] = 0.0
            continue
        # Inner product
        coeff = np.sum(phase[mask] * z[mask]) / mask.sum()
        coefficients[j] = float(coeff)

    return coefficients

