"""ADMM (Alternating Direction Method of Multipliers) for phase retrieval.

Splits the phase retrieval problem into two tractable subproblems linked
by a dual (Lagrange multiplier) variable:

    Minimise  L_F(G) + L_S(g)
    subject to  G = F{g}

Using the augmented Lagrangian with penalty ρ:

    g-step:   g ← P_S( F⁻¹{G − u} )          (pupil-support projection)
    G-step:   G ← P_F( F{g} + u )              (Fourier-magnitude projection)
    u-step:   u ← u + F{g} − G                 (dual variable update)

ADMM naturally handles the constraint splitting and is particularly
effective when combined with regularization (TV, Poisson noise model).

References:
    Boyd S., Parikh N., Chu E., Peleato B., Eckstein J. (2011)
    "Distributed Optimization and Statistical Learning via ADMM"
    Foundations and Trends in Machine Learning 3(1):1–122

    Chang H., Marchesini S. (2018)
    "ADMM methods for phase retrieval"
    arXiv:1804.05306
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

from src.algorithms.base import PhaseRetriever
from src.models.optics import PhaseRetrievalResult, PSFData


class ADMM(PhaseRetriever):
    """ADMM-based phase retrieval with Fourier/support splitting."""

    _u: np.ndarray | None

    def run(self, psf_data: PSFData) -> PhaseRetrievalResult:
        """Override to initialise dual variable before delegating to base loop."""
        self._u = None  # dual variable, initialised in first iterate
        return super().run(psf_data)

    def _iterate(
        self,
        *,
        g: np.ndarray,
        pupil_amplitude: np.ndarray,
        target_amplitude: np.ndarray,
        support: np.ndarray,
        iteration: int,
    ) -> tuple[np.ndarray, float]:
        # Initialise dual variable on first call
        if self._u is None:
            self._u = np.zeros_like(g, dtype=complex)

        # ── G-step: enforce Fourier-magnitude constraint ──────────────
        Fg = fftshift(fft2(ifftshift(g)))
        G_tilde = Fg + self._u
        G = self._project_fourier(G_tilde, target_amplitude)

        # ── g-step: enforce pupil-support constraint ──────────────────
        g_tilde = fftshift(ifft2(ifftshift(G - self._u)))
        g_new = np.zeros_like(g_tilde)
        g_new[support] = pupil_amplitude[support] * np.exp(1j * np.angle(g_tilde[support] + 1e-30))

        # ── Dual variable update (scaled form: u ← u + Fg − G) ──────
        Fg_new = fftshift(fft2(ifftshift(g_new)))
        self._u = self._u + (Fg_new - G)

        # Cost
        cost = self._focal_cost(target_amplitude, Fg)

        return g_new, cost
