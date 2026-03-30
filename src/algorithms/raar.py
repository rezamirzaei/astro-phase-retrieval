"""Relaxed Averaged Alternating Reflections (RAAR) — Luke 2005.

Correct RAAR update (Luke, Inverse Problems 21, 2005):

    g_{k+1} = β/2 · (R_S · R_F + I) · g_k  +  (1 − β) · P_F · g_k

where R_S = 2·P_S − I  and  R_F = 2·P_F − I  are reflectors.

RAAR is a feasibility-seeking algorithm that oscillates around the solution.
To obtain a clean final estimate, ER iterations are applied in the last 10%
of the budget (a standard practice in the phase-retrieval literature).
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from src.algorithms.base import PhaseRetriever


class RAAR(PhaseRetriever):
    """Relaxed Averaged Alternating Reflections + ER finish."""

    # Fraction of total iterations used for the final ER polish
    _ER_FRACTION = 0.1

    def _iterate(
        self,
        *,
        g: np.ndarray,
        pupil_amplitude: np.ndarray,
        target_amplitude: np.ndarray,
        support: np.ndarray,
        iteration: int,
    ) -> tuple[np.ndarray, float]:
        # Switch to ER for the last _ER_FRACTION of iterations
        er_start = int(self.config.max_iterations * (1.0 - self._ER_FRACTION))
        if iteration > er_start:
            return self._er_step(g, pupil_amplitude, target_amplitude, support)

        return self._raar_step(g, pupil_amplitude, target_amplitude, support)

    # ------------------------------------------------------------------
    def _raar_step(
        self,
        g: np.ndarray,
        pupil_amplitude: np.ndarray,
        target_amplitude: np.ndarray,
        support: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        beta = self.config.beta

        # P_F: project onto focal-plane magnitude constraint
        G = fftshift(fft2(ifftshift(g)))
        G_proj = target_amplitude * np.exp(1j * np.angle(G))
        p_f_g = fftshift(ifft2(ifftshift(G_proj)))

        # R_F = 2·P_F − I
        r_f_g = 2.0 * p_f_g - g

        # P_S(R_F(g))
        p_s_r_f_g = np.zeros_like(r_f_g)
        p_s_r_f_g[support] = pupil_amplitude[support] * np.exp(1j * np.angle(r_f_g[support]))

        # R_S(R_F(g)) = 2·P_S(R_F(g)) − R_F(g)
        r_s_r_f_g = 2.0 * p_s_r_f_g - r_f_g

        # RAAR: g_new = β/2·(R_S·R_F + I)·g  +  (1−β)·P_F·g
        g_new = (beta / 2.0) * (r_s_r_f_g + g) + (1.0 - beta) * p_f_g

        cost = self._focal_cost(target_amplitude, G)
        return g_new, cost

    # ------------------------------------------------------------------
    @staticmethod
    def _er_step(
        g: np.ndarray,
        pupil_amplitude: np.ndarray,
        target_amplitude: np.ndarray,
        support: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """One ER step: P_S ∘ P_F."""
        G = fftshift(fft2(ifftshift(g)))
        G_proj = target_amplitude * np.exp(1j * np.angle(G))
        g_prime = fftshift(ifft2(ifftshift(G_proj)))

        g_new = np.zeros_like(g_prime)
        g_new[support] = pupil_amplitude[support] * np.exp(1j * np.angle(g_prime[support]))

        modelled = np.abs(G)
        scale = target_amplitude.sum() / max(modelled.sum(), 1e-30)
        cost = float(np.sum((target_amplitude - modelled * scale) ** 2))
        return g_new, cost




