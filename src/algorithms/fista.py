"""FISTA — Fast Iterative Shrinkage-Thresholding Algorithm for phase retrieval.

Implements the proximal-gradient approach to phase retrieval with
acceleration (Beck & Teboulle, SIAM J. Imaging Sciences 2(1):183–202, 2009).

The algorithm minimises:

    min_g  f(g) + λ · R(g)

where f is the data-fidelity term (Fourier amplitude error) and R is a
regulariser (TV or L1-wavelet sparsity).

The FISTA update is:

    g_k     = prox_{λ/L · R}( y_k − (1/L) · ∇f(y_k) )
    t_{k+1} = (1 + √(1 + 4·t_k²)) / 2
    y_{k+1} = g_k + (t_k − 1) / t_{k+1} · (g_k − g_{k-1})

where L is the Lipschitz constant of ∇f and t_k implements Nesterov
acceleration.

Pluggable regularisers:
    - ``none``: plain projected gradient descent
    - ``tv``: total-variation via Chambolle proximal operator
    - ``l1_wavelet``: L1 soft-thresholding (wavelet-domain sparsity)

References
----------
Beck A., Teboulle M. (2009)
    "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems"
    SIAM J. Imaging Sciences 2(1):183–202

Candes E.J., Li X., Soltanolkotabi M. (2015)
    "Phase Retrieval via Wirtinger Flow"
    IEEE Trans. Info. Theory 61(4):1985–2007
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

from src.algorithms.base import _EPS, PhaseRetriever
from src.models.config import Regulariser
from src.models.optics import PhaseRetrievalResult, PSFData


class FISTA(PhaseRetriever):
    r"""FISTA proximal-gradient phase retrieval with pluggable regularisers.

    Attributes
    ----------
    _t : float
        Nesterov acceleration parameter.
    _g_prev : ndarray | None
        Previous iterate for momentum extrapolation.
    """

    _t: float
    _g_prev: np.ndarray | None

    def run(self, psf_data: PSFData) -> PhaseRetrievalResult:
        """Initialise FISTA state before delegating to the base loop."""
        self._t = 1.0
        self._g_prev = None

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
        """One FISTA iteration: gradient step → proximal step → Nesterov extrapolation."""
        n = g.shape[0]
        L = self.config.fista_lipschitz  # Lipschitz constant estimate
        lam = self.config.proximal_weight

        # ── Wirtinger gradient of the data-fidelity term ──────────────
        G = fftshift(fft2(ifftshift(g)))
        Y = target_amplitude ** 2
        I_model = np.abs(G) ** 2
        residual = I_model - Y
        grad_G = residual * G
        grad_g = fftshift(ifft2(ifftshift(grad_G)))

        # Step-size normalised by Lipschitz constant and grid
        mean_intensity = np.mean(Y) + _EPS
        step = 1.0 / (L * mean_intensity * n ** 2)

        # ── Gradient descent step ─────────────────────────────────────
        z = g - step * grad_g

        # ── Proximal step (regulariser) ───────────────────────────────
        z = self._proximal_operator(z, lam * step, support, pupil_amplitude)

        # ── Project onto pupil support ────────────────────────────────
        g_new = np.zeros_like(z)
        g_new[support] = pupil_amplitude[support] * np.exp(
            1j * np.angle(z[support] + _EPS)
        )

        # ── Nesterov acceleration ─────────────────────────────────────
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * self._t ** 2)) / 2.0
        if self._g_prev is not None:
            momentum = (self._t - 1.0) / t_new
            g_accel = g_new + momentum * (g_new - self._g_prev)
            # Project accelerated point onto support
            g_out = np.zeros_like(g_accel)
            g_out[support] = pupil_amplitude[support] * np.exp(
                1j * np.angle(g_accel[support] + _EPS)
            )
        else:
            g_out = g_new

        self._g_prev = g_new
        self._t = t_new

        # ── Cost ──────────────────────────────────────────────────────
        cost = self._focal_cost(target_amplitude, G)

        return g_out, cost

    def _proximal_operator(
        self,
        g: np.ndarray,
        weight: float,
        support: np.ndarray,
        pupil_amplitude: np.ndarray,
    ) -> np.ndarray:
        """Apply the configured proximal operator to the phase of g."""
        reg = self.config.regulariser

        if reg == Regulariser.NONE:
            return g

        if reg == Regulariser.TV:
            # TV proximal on the phase
            phase = np.angle(g)
            phase = self._tv_prox(phase, weight, support)
            amp = np.abs(g)
            return amp * np.exp(1j * phase)  # type: ignore[return-value,no-any-return]

        if reg == Regulariser.L1_WAVELET:
            # L1 soft-thresholding on the phase (wavelet-domain sparsity)
            phase = np.angle(g)
            phase = self._l1_soft_threshold(phase, weight, support)
            amp = np.abs(g)
            return amp * np.exp(1j * phase)  # type: ignore[return-value,no-any-return]

        return g

    @staticmethod
    def _l1_soft_threshold(
        phase: np.ndarray,
        weight: float,
        support: np.ndarray,
    ) -> np.ndarray:
        """L1 soft-thresholding proximal operator.

        Implements prox_{λ·||·||₁}(x) = sign(x) · max(|x| − λ, 0).
        Applied element-wise to the phase within the support.
        """
        result = phase.copy()
        # Soft threshold
        result[support] = np.sign(phase[support]) * np.maximum(
            np.abs(phase[support]) - weight, 0.0
        )
        result[~support] = 0.0
        return result
