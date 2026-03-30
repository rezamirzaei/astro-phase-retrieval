"""Gerchberg–Saxton (GS) algorithm.

Classic two-plane amplitude-constraint algorithm. In the wavefront-sensing
context the two planes are: pupil plane (known amplitude = pupil mask) and
focal plane (known amplitude = sqrt(measured PSF)).
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from src.algorithms.base import PhaseRetriever


class GerchbergSaxton(PhaseRetriever):
    """Gerchberg–Saxton phase retrieval."""

    def _iterate(
        self,
        *,
        g: np.ndarray,
        pupil_amplitude: np.ndarray,
        target_amplitude: np.ndarray,
        support: np.ndarray,
        iteration: int,
    ) -> tuple[np.ndarray, float]:
        # 1. Pupil → focal via FFT
        G = fftshift(fft2(ifftshift(g)))

        # 2. Enforce focal-plane amplitude constraint (noise-model aware)
        G_prime = self._project_fourier(G, target_amplitude)

        # 3. Focal → pupil via inverse FFT
        g_prime = fftshift(ifft2(ifftshift(G_prime)))

        # 4. Enforce pupil-plane amplitude (identical to ER)
        g_new = np.zeros_like(g_prime)
        g_new[support] = pupil_amplitude[support] * np.exp(1j * np.angle(g_prime[support]))

        # Cost
        cost = self._focal_cost(target_amplitude, G)

        return g_new, cost


