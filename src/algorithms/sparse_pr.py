"""Sparse Phase Retrieval — Thresholded Wirtinger Flow (ThWF).

Extends Wirtinger Flow with hard/soft thresholding for sparsity-promoting
phase retrieval.  The key insight is that many optical aberrations are
sparse in some basis (e.g. Zernike, wavelet), so enforcing sparsity acts
as a powerful regulariser.

The algorithm:

    g_{k+1} = T_λ( g_k − μ · ∇L(g_k) )

where T_λ is the thresholding operator and ∇L is the Wirtinger gradient.

References
----------
Cai T.T., Li X., Ma Z. (2016)
    "Optimal rates of convergence for noisy sparse phase retrieval via
    thresholded Wirtinger flow"
    Annals of Statistics 44(5):2221–2251

Netrapalli P., Jain P., Sanghavi S. (2015)
    "Phase Retrieval Using Alternating Minimization"
    IEEE Trans. Signal Processing 63(18):4814–4826
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

from src.algorithms.base import _EPS, PhaseRetriever
from src.models.config import Regulariser


class SparsePhaseRetrieval(PhaseRetriever):
    """Thresholded Wirtinger Flow for sparse phase retrieval.

    Combines the Wirtinger gradient descent with adaptive thresholding
    to promote sparsity in the recovered phase.  Supports both hard
    and soft thresholding modes.
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
        """One ThWF iteration: Wirtinger gradient → threshold → project."""
        n = g.shape[0]
        mu = self.config.wf_step_size
        sparsity_level = self.config.sparsity_threshold

        # ── Forward propagate ─────────────────────────────────────────
        G = fftshift(fft2(ifftshift(g)))

        # ── Wirtinger gradient ────────────────────────────────────────
        Y = target_amplitude**2
        I_model = np.abs(G) ** 2
        residual = I_model - Y
        grad_G = residual * G
        grad_g = fftshift(ifft2(ifftshift(grad_G)))

        # Normalised step size
        mean_intensity = np.mean(Y) + _EPS
        step = mu / (mean_intensity * n**2)

        # ── Gradient descent ──────────────────────────────────────────
        g_new = g - step * grad_g

        # ── Sparsity-promoting thresholding on the phase ──────────────
        phase = np.angle(g_new)

        # Adaptive threshold: decreases over iterations for refinement
        decay = max(0.1, 1.0 - iteration / max(self.config.max_iterations, 1))
        support_phase = np.abs(phase[support])
        threshold = sparsity_level * decay * np.max(support_phase)

        if self.config.regulariser == Regulariser.L1_WAVELET:
            # Soft thresholding
            phase_thresholded = np.sign(phase) * np.maximum(np.abs(phase) - threshold, 0.0)
        else:
            # Hard thresholding with an explicit keep-fraction on the support.
            phase_thresholded = phase.copy()
            phase_thresholded[np.abs(phase) < threshold] = 0.0
            keep_count = max(
                1,
                int(np.ceil(self.config.sparsity_keep_fraction * np.count_nonzero(support))),
            )
            active_values = np.abs(phase_thresholded[support])
            if active_values.size > keep_count:
                topk_threshold = float(np.partition(active_values, -keep_count)[-keep_count])
                phase_thresholded[np.abs(phase_thresholded) < topk_threshold] = 0.0

        # ── Project onto pupil support ────────────────────────────────
        g_out = np.zeros_like(g_new)
        g_out[support] = pupil_amplitude[support] * np.exp(1j * phase_thresholded[support])

        # ── Cost ──────────────────────────────────────────────────────
        cost = self._focal_cost(target_amplitude, G)

        return g_out, cost
