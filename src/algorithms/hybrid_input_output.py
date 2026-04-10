"""Hybrid Input-Output (HIO) algorithm — Fienup 1982.

The workhorse of iterative phase retrieval.  Uses a feedback parameter β to
escape local minima that trap the simpler Error-Reduction algorithm.

Crucially, the full complex field g is maintained across iterations — the
HIO feedback rule modifies both amplitude and phase outside the support.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

from src.algorithms.base import PhaseRetriever


class HybridInputOutput(PhaseRetriever):
    """Fienup's Hybrid Input-Output (HIO) algorithm.

    The final ``er_finish_fraction`` of iterations switch to Error Reduction
    for a clean convergent finish (standard practice for oscillatory algorithms).
    """

    def _iterate(
        self,
        *,
        g: np.ndarray,
        pupil_amplitude: np.ndarray,
        target_amplitude: np.ndarray,
        support: np.ndarray,
        iteration: int,
    ) -> tuple[np.ndarray, float]:
        # Switch to ER for the final fraction of iterations
        er_start = int(self.config.max_iterations * (1.0 - self.config.er_finish_fraction))
        if iteration > er_start:
            return self._er_step(g, pupil_amplitude, target_amplitude, support)

        beta = self._get_beta(iteration)

        # 1. Forward propagate
        G = fftshift(fft2(ifftshift(g)))

        # 2. Enforce focal-plane amplitude constraint (noise-model aware)
        G_prime = self._project_fourier(G, target_amplitude)

        # 3. Inverse propagate
        g_prime = fftshift(ifft2(ifftshift(G_prime)))

        # 4. HIO update rule
        #    Inside support:  g_new = g'  (with amplitude enforced)
        #    Outside support: g_new = g - β·g'  (feedback)
        g_new = np.where(
            support,
            pupil_amplitude * np.exp(1j * np.angle(g_prime)),
            g - beta * g_prime,
        )

        # Cost
        cost = self._focal_cost(target_amplitude, G)

        return g_new, cost
