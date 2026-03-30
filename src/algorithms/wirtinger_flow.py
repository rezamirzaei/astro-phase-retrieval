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
        """Override run to pick the most promising initialization."""
        self._spectral_phase = self._select_initial_phase(psf_data)
        return super().run(psf_data)

    def _initial_phase(self, n: int) -> np.ndarray:
        """Use the selected initializer if available, else small random."""
        if hasattr(self, "_spectral_phase") and self._spectral_phase is not None:
            return self._spectral_phase
        return self._rng.uniform(-0.3, 0.3, size=(n, n)).astype(np.float64)

    def _select_initial_phase(self, psf_data) -> np.ndarray | None:
        """Choose the best among spectral, zero, and random initial phases."""
        n = psf_data.image.shape[0]
        candidates: list[np.ndarray] = [
            np.zeros((n, n), dtype=np.float64),
            self._rng.uniform(-0.3, 0.3, size=(n, n)).astype(np.float64),
        ]
        if self.config.wf_spectral_init:
            candidates.append(self._spectral_init(psf_data))

        target_amp = np.sqrt(psf_data.image)
        energy_pupil = np.sum(self.pupil.amplitude**2)
        energy_target = np.sum(target_amp**2)
        if energy_target > 0:
            target_amp = target_amp * np.sqrt((energy_pupil * (n**2)) / energy_target)

        support = self.pupil.amplitude > 0
        best_phase: np.ndarray | None = None
        best_cost = float("inf")
        for phase in candidates:
            phase = phase.copy()
            phase[~support] = 0.0
            g = self.pupil.amplitude * np.exp(1j * phase)
            G = fftshift(fft2(ifftshift(g)))
            cost = self._focal_cost(target_amp, G)
            if cost < best_cost:
                best_cost = cost
                best_phase = phase
        return best_phase

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
        G = fftshift(fft2(ifftshift(g)))

        # Projected amplitude-flow variant: descend on amplitude mismatch and
        # backtrack until the focal-plane error decreases.
        model_amplitude = np.abs(G)
        grad_G = (model_amplitude - target_amplitude) * G / np.maximum(model_amplitude, 1e-12)
        grad_g = fftshift(ifft2(ifftshift(grad_G)))

        current_cost = self._focal_cost(target_amplitude, G)
        grad_norm = np.sqrt(np.mean(np.abs(grad_g[support]) ** 2))
        step = self.config.wf_step_size / max(float(grad_norm), 1e-6)

        best_g = g.copy()
        best_cost = current_cost
        for _ in range(8):
            candidate = g - step * grad_g
            candidate[~support] = 0.0
            candidate[support] = pupil_amplitude[support] * np.exp(1j * np.angle(candidate[support] + 1e-30))

            candidate_G = fftshift(fft2(ifftshift(candidate)))
            candidate_cost = self._focal_cost(target_amplitude, candidate_G)
            if candidate_cost <= best_cost:
                best_g = candidate
                best_cost = candidate_cost
                break
            step *= 0.5

        return best_g, best_cost
