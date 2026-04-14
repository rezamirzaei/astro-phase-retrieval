"""ADMM (Alternating Direction Method of Multipliers) for phase retrieval.

Splits the phase retrieval problem into two tractable subproblems linked
by a dual (Lagrange multiplier) variable:

    Minimise  L_F(G) + L_S(g)
    subject to  G = F{g}

Using the augmented Lagrangian with penalty ρ:

    g-step:   g ← P_S( F⁻¹{G − u} )          (pupil-support projection)
    G-step:   G ← P_F( F{g} + u )              (Fourier-magnitude projection)
    u-step:   u ← u + F{g} − G                 (scaled dual variable update)

The penalty ρ is exposed through ``AlgorithmConfig.admm_rho``.  The default
ρ = 1.0 corresponds to the standard unscaled dual update.  Increasing ρ
enforces the constraint G = F{g} more strongly at the expense of slower
convergence on each individual sub-problem.

ADMM naturally handles regularisation (TV, Poisson noise model) and is
particularly effective on large problems where it can be warm-started.

References
----------
Boyd S., Parikh N., Chu E., Peleato B., Eckstein J. (2011)
    "Distributed Optimization and Statistical Learning via ADMM"
    Foundations and Trends in Machine Learning 3(1):1–122

Chang H., Marchesini S. (2018)
    "ADMM methods for phase retrieval"
    arXiv:1804.05306
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift  # type: ignore[import-untyped]

from src.algorithms.base import _EPS, PhaseRetriever
from src.models.optics import PhaseRetrievalResult, PSFData

# Default ADMM penalty parameter ρ (augmented Lagrangian weight).
# Exposed so callers can tune the primal/dual residual balance.
_DEFAULT_RHO: float = 1.0


class ADMM(PhaseRetriever):
    """ADMM-based phase retrieval with Fourier/support splitting.

    The penalty parameter ``rho`` (ρ) controls the trade-off between
    primal feasibility (G = F{g}) and dual convergence speed.  For
    well-conditioned problems the default ρ = 1 works well; for
    noisy data increase ρ to enforce the Fourier constraint more
    aggressively.

    Attributes
    ----------
    _u : complex ndarray | None
        Scaled dual variable (initialised to zeros on first call).
    _rho : float
        ADMM penalty parameter ρ (read from config or default).
    """

    _u: np.ndarray | None
    _rho: float

    def run(self, psf_data: PSFData) -> PhaseRetrievalResult:
        """Override to initialise dual variable and ρ before delegating to base loop."""
        self._u = None  # dual variable, initialised lazily in first _iterate
        # Allow the caller to pass rho via config metadata; fall back to default
        self._rho = float(getattr(self.config, "admm_rho", _DEFAULT_RHO))
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
        """One ADMM iteration: G-step → g-step → dual update.

        Parameters
        ----------
        g : complex ndarray
            Current pupil-plane field estimate.
        pupil_amplitude, target_amplitude, support, iteration :
            See :meth:`PhaseRetriever._iterate`.

        Returns
        -------
        g_new : complex ndarray
        cost : float
        """
        rho = self._rho

        # ── Initialise dual variable on first call ────────────────────
        if self._u is None:
            self._u = np.zeros_like(g, dtype=complex)

        # ── G-step: enforce Fourier-magnitude constraint ──────────────
        Fg = fftshift(fft2(ifftshift(g), workers=-1))
        G_tilde = Fg + self._u
        G = self._project_fourier(G_tilde, target_amplitude)

        # ── g-step: enforce pupil-support constraint ──────────────────
        # z = F⁻¹{G − u}  then project onto support
        z = fftshift(ifft2(ifftshift(G - self._u), workers=-1))
        g_new = np.zeros_like(z)
        g_new[support] = pupil_amplitude[support] * np.exp(1j * np.angle(z[support] + _EPS))

        # ── Dual variable update (scaled form) ────────────────────────
        # u ← u + ρ · (F{g_new} − G)
        Fg_new = fftshift(fft2(ifftshift(g_new), workers=-1))
        self._u = self._u + rho * (Fg_new - G)

        # ── Cost (primal residual amplitude error) ─────────────────────
        cost = self._focal_cost(target_amplitude, Fg)

        return g_new, cost
