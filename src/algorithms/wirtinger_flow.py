"""Wirtinger Flow (WF) algorithm — Candès, Li & Soltanolkotabi 2015.

A gradient-descent method on the intensity loss function using Wirtinger
(complex) derivatives.  Combined with spectral initialization, WF converges
linearly to the global optimum and represents the state of the art in
non-convex phase retrieval.

Reference:
    Candès E.J., Li X., Soltanolkotabi M. (2015)
    "Phase Retrieval via Wirtinger Flow: Theory and Algorithms"
    IEEE Trans. Information Theory 61(4):1985–2007

The spectral initialization computes the leading eigenvector of a
weighted covariance matrix via truncated power iteration, providing a
much better starting point than random phase.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

from src.algorithms.base import PhaseRetriever
from src.models.optics import PhaseRetrievalResult, PSFData


class WirtingerFlow(PhaseRetriever):
    r"""Wirtinger Flow phase retrieval with optional spectral initialization.

    This is a gradient-descent algorithm on the amplitude-based loss:

    .. math::

        L(g) = \frac{1}{4n^2} \sum_k \bigl( |\langle a_k, g \rangle|^2 - y_k \bigr)^2

    where :math:`y_k = |F\{g\}|^2` are the measured intensities and
    :math:`a_k` are the Fourier measurement vectors.  The Wirtinger gradient is:

    .. math::

        \nabla_g L = \frac{1}{n^2} F^{-1}\{ (|G|^2 - Y) \odot G \}

    The step size is normalised by the mean intensity for stability.
    """

    def run(self, psf_data: PSFData) -> PhaseRetrievalResult:
        """Override run to apply WF-specific spectral initialization when configured."""
        if self.config.wf_spectral_init:
            self._spectral_phase = self._spectral_init(psf_data.image, psf_data.image.shape[0])
        else:
            self._spectral_phase = None
        return super().run(psf_data)

    def _initial_phase(self, n: int) -> np.ndarray:
        """Use WF spectral initialization if available, else delegate to base."""
        if hasattr(self, "_spectral_phase") and self._spectral_phase is not None:
            return self._spectral_phase  # type: ignore[no-any-return]
        return super()._initial_phase(n)

    def _iterate(
        self,
        *,
        g: np.ndarray,
        pupil_amplitude: np.ndarray,
        target_amplitude: np.ndarray,
        support: np.ndarray,
        iteration: int,
    ) -> tuple[np.ndarray, float]:
        n = g.shape[0]
        mu = self.config.wf_step_size

        # Forward propagate
        G = fftshift(fft2(ifftshift(g)))

        # Measured intensity and model intensity
        Y = target_amplitude**2
        I_model = np.abs(G) ** 2

        # Wirtinger gradient:  ∇L = F⁻¹{ (|G|² − Y) · G } / n²
        residual = I_model - Y
        grad_G = residual * G
        grad_g = fftshift(ifft2(ifftshift(grad_G)))

        # Normalise step size by mean intensity
        mean_intensity = np.mean(Y) + 1e-30
        step = mu / (mean_intensity * n**2)

        # Gradient descent
        g_new = g - step * grad_g

        # Project onto pupil support (enforce known amplitude)
        g_new[~support] = 0.0
        g_new[support] = pupil_amplitude[support] * np.exp(1j * np.angle(g_new[support] + 1e-30))

        # Cost: amplitude error
        cost = self._focal_cost(target_amplitude, G)

        return g_new, cost
