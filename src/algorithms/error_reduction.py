"""Error Reduction (ER) algorithm — Fienup 1982.

The simplest projection-based phase retrieval: alternately enforce the
measured focal-plane amplitude and the known pupil-plane support.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from src.algorithms.base import PhaseRetriever


class ErrorReduction(PhaseRetriever):
    """Fienup's Error-Reduction algorithm."""

    def _iterate(
        self,
        *,
        g: np.ndarray,
        pupil_amplitude: np.ndarray,
        target_amplitude: np.ndarray,
        support: np.ndarray,
        iteration: int,
    ) -> tuple[np.ndarray, float]:
        # 1. Forward propagate: pupil → focal plane
        G = fftshift(fft2(ifftshift(g)))

        # 2. Replace focal-plane amplitude with measured, keep phase
        G_prime = target_amplitude * np.exp(1j * np.angle(G))

        # 3. Inverse propagate back to pupil plane
        g_prime = fftshift(ifft2(ifftshift(G_prime)))

        # 4. ER: project onto support and enforce known amplitude
        g_new = np.zeros_like(g_prime)
        g_new[support] = pupil_amplitude[support] * np.exp(1j * np.angle(g_prime[support]))

        # Cost
        cost = self._focal_cost(target_amplitude, G)

        return g_new, cost



