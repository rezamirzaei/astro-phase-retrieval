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
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from src.algorithms.base import PhaseRetriever


class WirtingerFlow(PhaseRetriever):
    """Wirtinger Flow phase retrieval with optional spectral initialization.

    This is a gradient-descent algorithm on the amplitude-based loss:

        L(g) = (1/4n²) Σ_k ( |⟨a_k, g⟩|² − y_k )²

    where y_k = |F{g}|² are the measured intensities and a_k are the
    Fourier measurement vectors.  The Wirtinger gradient is:

        ∇_g L = (1/n²) F⁻¹{ (|G|² − Y) ⊙ G }

    The step size is normalised by the mean intensity for stability.
    """

    def run(self, psf_data):
        """Override run to apply spectral initialization when configured."""
        if self.config.wf_spectral_init:
            self._spectral_phase = self._spectral_init(psf_data)
        else:
            self._spectral_phase = None
        return super().run(psf_data)

    def _initial_phase(self, n: int) -> np.ndarray:
        """Use spectral initialization if available, else small random."""
        if hasattr(self, '_spectral_phase') and self._spectral_phase is not None:
            return self._spectral_phase
        return self._rng.uniform(-0.3, 0.3, size=(n, n)).astype(np.float64)

    def _spectral_init(self, psf_data) -> np.ndarray:
        """Spectral initialization via truncated power iteration.

        Computes the leading eigenvector of the weighted measurement matrix
        T = (1/m) Σ_k y_k · a_k a_k^H, which is equivalent to iterating
        z ← F⁻¹{ Y ⊙ F{z} } and normalizing.
        """
        Y = psf_data.image  # intensity
        n = Y.shape[0]
        support = self.pupil.amplitude > 0

        # Power iteration in Fourier domain: z ← IFFT{ Y · FFT{z} }
        rng = np.random.default_rng(self.config.random_seed)
        z = self.pupil.amplitude * np.exp(1j * rng.uniform(-np.pi, np.pi, (n, n)))

        for _ in range(50):
            Z = fftshift(fft2(ifftshift(z)))
            Z = Y * Z  # weight by measured intensity
            z = fftshift(ifft2(ifftshift(Z)))
            # Project onto support
            z[~support] = 0.0
            # Normalise
            norm = np.sqrt(np.sum(np.abs(z)**2))
            if norm > 0:
                z /= norm

        # Scale to match pupil energy
        z *= np.sqrt(np.sum(self.pupil.amplitude**2))

        phase = np.angle(z)
        phase[~support] = 0.0
        return phase

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
        I_model = np.abs(G)**2

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
