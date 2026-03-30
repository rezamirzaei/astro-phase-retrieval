"""Quality metrics for evaluating phase-retrieval results.

Includes Strehl ratio, RMS wavefront error, Zernike decomposition,
MTF (Modulation Transfer Function), SSIM, and Phase Structure Function.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, fftshift

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


# ---------------------------------------------------------------------------
# MTF — Modulation Transfer Function
# ---------------------------------------------------------------------------

def compute_mtf(psf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radially averaged Modulation Transfer Function (MTF).

    The MTF is the normalised magnitude of the OTF (optical transfer function),
    which is the Fourier transform of the PSF.

    Parameters
    ----------
    psf : ndarray
        2-D PSF (normalised to unit sum or not — we normalise internally).

    Returns
    -------
    freqs : ndarray
        Spatial frequency in cycles/pixel (0 to Nyquist = 0.5).
    mtf_profile : ndarray
        Radially averaged MTF (0–1).
    """
    psf_norm = psf / max(psf.sum(), 1e-30)
    otf = fftshift(fft2(psf_norm))
    mtf_2d = np.abs(otf)
    # Normalise so that DC = 1
    dc = mtf_2d.max()
    if dc > 0:
        mtf_2d /= dc

    # Radial average
    n = psf.shape[0]
    cy, cx = n // 2, n // 2
    y, x = np.ogrid[:n, :n]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = n // 2
    freqs = np.arange(0, max_r) / n  # cycles/pixel
    profile = np.array([mtf_2d[r == rr].mean() for rr in range(max_r)])

    return freqs, profile


# ---------------------------------------------------------------------------
# SSIM — Structural Similarity Index
# ---------------------------------------------------------------------------

def compute_ssim(
    observed: np.ndarray,
    reconstructed: np.ndarray,
    *,
    data_range: float | None = None,
) -> float:
    """Compute structural similarity (SSIM) between observed and reconstructed PSFs.

    Uses scikit-image's implementation.

    Parameters
    ----------
    observed : ndarray
        Observed PSF image.
    reconstructed : ndarray
        Reconstructed PSF image (same shape).
    data_range : float | None
        Dynamic range.  If None, computed from the observed image.

    Returns
    -------
    float
        SSIM value in [-1, 1] (1 = perfect match).
    """
    from skimage.metrics import structural_similarity

    obs = observed / max(observed.sum(), 1e-30)
    rec = reconstructed / max(reconstructed.sum(), 1e-30)

    if data_range is None:
        data_range = float(obs.max() - obs.min())
    if data_range <= 0:
        data_range = 1.0

    return float(structural_similarity(obs, rec, data_range=data_range))


# ---------------------------------------------------------------------------
# Phase Structure Function
# ---------------------------------------------------------------------------

def compute_phase_structure_function(
    phase: np.ndarray,
    support: np.ndarray,
    max_sep: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the phase structure function D_φ(r).

    D_φ(r) = ⟨|φ(x) − φ(x + r)|²⟩  averaged over all angles and positions.

    This is a standard diagnostic for optical wavefront quality and
    atmospheric turbulence characterisation.

    Parameters
    ----------
    phase : ndarray
        2-D pupil-plane phase (radians).
    support : ndarray
        Boolean pupil support mask.
    max_sep : int | None
        Maximum separation in pixels (default: grid_size // 4).

    Returns
    -------
    separations : ndarray
        Separation distances in pixels.
    structure_fn : ndarray
        D_φ(r) values (rad²).
    """
    n = phase.shape[0]
    if max_sep is None:
        max_sep = n // 4

    separations = np.arange(1, max_sep + 1)
    structure_fn = np.zeros(len(separations))

    # Phase only within support
    phi = phase.copy()
    phi[~support] = np.nan

    for i, sep in enumerate(separations):
        diffs = []
        # Sample at 8 angles for efficiency
        for angle in np.linspace(0, np.pi, 8, endpoint=False):
            dx = int(round(sep * np.cos(angle)))
            dy = int(round(sep * np.sin(angle)))
            if dx == 0 and dy == 0:
                continue

            # Shifted phase
            phi_shifted = np.full_like(phi, np.nan)
            if dy >= 0 and dx >= 0:
                phi_shifted[dy:, dx:] = phi[:n - dy if dy > 0 else n, :n - dx if dx > 0 else n]
            elif dy >= 0 and dx < 0:
                phi_shifted[dy:, :n + dx] = phi[:n - dy if dy > 0 else n, -dx:]
            elif dy < 0 and dx >= 0:
                phi_shifted[:n + dy, dx:] = phi[-dy:, :n - dx if dx > 0 else n]
            else:
                phi_shifted[:n + dy, :n + dx] = phi[-dy:, -dx:]

            valid = np.isfinite(phi) & np.isfinite(phi_shifted)
            if valid.sum() > 0:
                d = (phi[valid] - phi_shifted[valid]) ** 2
                diffs.append(np.mean(d))

        structure_fn[i] = np.mean(diffs) if diffs else 0.0

    return separations.astype(float), structure_fn


# ---------------------------------------------------------------------------
# Zernike decomposition
# ---------------------------------------------------------------------------

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
