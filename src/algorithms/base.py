"""Abstract base class for all phase-retrieval algorithms."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import uniform_filter

from src.models.config import AlgorithmConfig, BetaSchedule, NoiseModel
from src.models.optics import PhaseRetrievalResult, PupilModel, PSFData
from src.metrics.quality import compute_rms_phase, compute_strehl_ratio


class PhaseRetriever(ABC):
    """Base class that every phase-retrieval algorithm must subclass.

    Subclasses implement ``_iterate`` which receives and returns the full
    complex pupil-plane field **g** (not just phase).  This is essential for
    algorithms like HIO and RAAR whose update rules mix amplitude and phase
    information in the feedback term.

    State-of-the-art enhancements (applied transparently in the base loop):
      • Nesterov / heavy-ball **momentum** acceleration
      • **Adaptive β** scheduling (constant / linear / cosine)
      • **Total-variation (TV)** regularization via Chambolle proximal operator
      • **Poisson noise** maximum-likelihood focal-plane projection
    """

    def __init__(self, config: AlgorithmConfig, pupil: PupilModel) -> None:
        self.config = config
        self.pupil = pupil
        self._rng = np.random.default_rng(config.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, psf_data: PSFData) -> PhaseRetrievalResult:
        """Execute the full retrieval loop."""
        target_amp = np.sqrt(psf_data.image)
        pupil_amp = self.pupil.amplitude
        n = pupil_amp.shape[0]
        support = pupil_amp > 0

        # Normalise target amplitude to match pupil energy (Parseval's theorem)
        energy_pupil = np.sum(pupil_amp**2)
        energy_target = np.sum(target_amp**2)
        if energy_target > 0:
            target_amp *= np.sqrt((energy_pupil * (n**2)) / energy_target)

        # Initialise complex pupil field with known amplitude + random phase
        phase0 = self._initial_phase(n)
        g = pupil_amp * np.exp(1j * phase0)

        cost_history: list[float] = []
        converged = False
        t0 = time.perf_counter()
        window = 20  # sliding-window length for convergence check

        # Momentum state
        g_prev = g.copy()

        for iteration in range(1, self.config.max_iterations + 1):
            # ── Momentum extrapolation (Nesterov / heavy-ball) ────────
            if self.config.momentum > 0 and iteration > 1:
                g = g + self.config.momentum * (g - g_prev)

            g_before = g.copy()

            g, cost = self._iterate(
                g=g,
                pupil_amplitude=pupil_amp,
                target_amplitude=target_amp,
                support=support,
                iteration=iteration,
            )

            # ── TV regularization on the phase ────────────────────────
            if self.config.tv_weight > 0:
                phase = np.angle(g)
                phase = self._tv_prox(phase, self.config.tv_weight, support)
                amp = np.abs(g)
                g = amp * np.exp(1j * phase)

            # Store for momentum
            g_prev = g_before

            cost_history.append(float(cost))

            # Convergence: compare the mean cost over the last two windows.
            # This handles algorithms like RAAR whose cost oscillates.
            if len(cost_history) >= 2 * window:
                recent = np.mean(cost_history[-window:])
                previous = np.mean(cost_history[-2 * window:-window])
                if abs(previous - recent) / max(abs(previous), 1e-30) < self.config.tolerance:
                    converged = True
                    break

        elapsed = time.perf_counter() - t0

        # Extract final phase (enforce support)
        phase = np.angle(g)
        phase[~support] = 0.0

        # Build outputs
        from src.optics.propagator import forward_model
        recon_psf = forward_model(pupil_amp, phase)
        rms = compute_rms_phase(phase, support)
        strehl = compute_strehl_ratio(recon_psf, pupil_amp)

        return PhaseRetrievalResult(
            algorithm=self.config.name,
            recovered_phase=phase,
            recovered_amplitude=pupil_amp,
            reconstructed_psf=recon_psf,
            cost_history=cost_history,
            n_iterations=len(cost_history),
            converged=converged,
            elapsed_seconds=elapsed,
            rms_phase_rad=rms,
            strehl_ratio=strehl,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _iterate(
        self,
        *,
        g: np.ndarray,
        pupil_amplitude: np.ndarray,
        target_amplitude: np.ndarray,
        support: np.ndarray,
        iteration: int,
    ) -> tuple[np.ndarray, float]:
        """Perform one iteration.

        Parameters
        ----------
        g : complex ndarray
            Current pupil-plane complex field estimate.
        pupil_amplitude : ndarray
            Known pupil amplitude mask.
        target_amplitude : ndarray
            sqrt(observed PSF) — measured focal-plane amplitude.
        support : ndarray
            Boolean mask of the pupil region.
        iteration : int
            1-based iteration counter.

        Returns
        -------
        (g_new, cost) — updated complex field and focal-plane cost.
        """
        ...

    # ------------------------------------------------------------------
    # Adaptive β scheduling
    # ------------------------------------------------------------------

    def _get_beta(self, iteration: int) -> float:
        """Return the β value for the current iteration, respecting the schedule."""
        beta_max = self.config.beta
        beta_min = self.config.beta_min
        max_iter = self.config.max_iterations
        schedule = self.config.beta_schedule

        if schedule == BetaSchedule.CONSTANT:
            return beta_max

        t = (iteration - 1) / max(max_iter - 1, 1)  # normalised progress [0, 1]

        if schedule == BetaSchedule.LINEAR:
            return beta_max - (beta_max - beta_min) * t

        if schedule == BetaSchedule.COSINE:
            return beta_min + 0.5 * (beta_max - beta_min) * (1 + np.cos(np.pi * t))

        return beta_max

    # ------------------------------------------------------------------
    # Noise-robust focal-plane projection
    # ------------------------------------------------------------------

    def _project_fourier(
        self,
        G: np.ndarray,
        target_amplitude: np.ndarray,
    ) -> np.ndarray:
        """Replace focal-plane amplitude, respecting the configured noise model.

        Gaussian mode (default):
            G' = target_amp · exp(i·angle(G))

        Poisson mode (maximum-likelihood):
            G' = sqrt( I_obs · |G|² / smooth(|G|²) ) · exp(i·angle(G))
            This reduces noise amplification in low-SNR regions.
        """
        if self.config.noise_model == NoiseModel.GAUSSIAN:
            return target_amplitude * np.exp(1j * np.angle(G + 1e-30))

        # Poisson ML projection (Thibault et al., 2012)
        I_obs = target_amplitude**2
        I_model = np.abs(G)**2
        # Smooth the model intensity to stabilise the ratio
        I_smooth = uniform_filter(I_model, size=3)
        I_smooth = np.maximum(I_smooth, 1e-30)
        ratio = np.sqrt(np.maximum(I_obs / I_smooth, 0))
        return ratio * G

    # ------------------------------------------------------------------
    # Total-variation proximal operator (Chambolle 2004)
    # ------------------------------------------------------------------

    @staticmethod
    def _tv_prox(
        phase: np.ndarray,
        weight: float,
        support: np.ndarray,
        n_iter: int = 10,
    ) -> np.ndarray:
        """Proximal operator for isotropic total variation (Chambolle 2004).

        Denoises the recovered phase map while preserving sharp aberration
        boundaries.  Only applied within the pupil support.
        """
        if weight <= 0:
            return phase
        tau = 0.25
        px = np.zeros_like(phase)
        py = np.zeros_like(phase)
        for _ in range(n_iter):
            # Divergence of (px, py)
            div = np.zeros_like(phase)
            div[:, 1:] += px[:, :-1]
            div[:, 0] += 0
            div[:, 1:] -= px[:, 1:]
            # Actually: div_x
            div_x = np.zeros_like(phase)
            div_x[:, 1:] = px[:, :-1] - px[:, 1:]
            div_x[:, 0] = -px[:, 0]

            div_y = np.zeros_like(phase)
            div_y[1:, :] = py[:-1, :] - py[1:, :]
            div_y[0, :] = -py[0, :]

            div_p = div_x + div_y

            # Gradient of (phase + weight * div_p)
            u = phase + weight * div_p
            gx = np.zeros_like(phase)
            gx[:, :-1] = u[:, 1:] - u[:, :-1]
            gy = np.zeros_like(phase)
            gy[:-1, :] = u[1:, :] - u[:-1, :]

            # Update dual variables
            norm_g = np.sqrt(gx**2 + gy**2)
            denom = 1.0 + tau * norm_g / max(weight, 1e-30)
            px = (px + tau * gx) / denom
            py = (py + tau * gy) / denom

        # Final denoised phase
        div_x = np.zeros_like(phase)
        div_x[:, 1:] = px[:, :-1] - px[:, 1:]
        div_x[:, 0] = -px[:, 0]
        div_y = np.zeros_like(phase)
        div_y[1:, :] = py[:-1, :] - py[1:, :]
        div_y[0, :] = -py[0, :]

        result = phase + weight * (div_x + div_y)
        result[~support] = 0.0
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _initial_phase(self, n: int) -> np.ndarray:
        """Small random perturbations around zero (diffraction-limited start)."""
        return self._rng.uniform(-0.3, 0.3, size=(n, n)).astype(np.float64)

    @staticmethod
    def _focal_cost(target_amp: np.ndarray, G: np.ndarray) -> float:
        """Normalised focal-plane amplitude error."""
        modelled_amp = np.abs(G)
        scale = target_amp.sum() / max(modelled_amp.sum(), 1e-30)
        return float(np.sum((target_amp - modelled_amp * scale) ** 2))
