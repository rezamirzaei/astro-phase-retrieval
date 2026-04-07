"""Abstract base class for all phase-retrieval algorithms."""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import uniform_filter  # type: ignore[import-untyped]

from src.metrics.quality import compute_rms_phase, compute_strehl_ratio
from src.models.config import AlgorithmConfig, BetaSchedule, NoiseModel
from src.models.optics import PhaseRetrievalResult, PSFData, PupilModel

# Numerical stability epsilon — used throughout to prevent division by zero
# and to regularise the phase angle of near-zero complex values.
_EPS: float = 1e-30

# Default sliding-window length for convergence checking.  The algorithm
# converges when the relative change in mean cost between two consecutive
# windows drops below `tolerance`.
_CONVERGENCE_WINDOW: int = 20


class PhaseRetriever(ABC):
    """Base class that every phase-retrieval algorithm must subclass.

    Subclasses implement :meth:`_iterate` which receives and returns the full
    complex pupil-plane field **g** (not just phase).  This is essential for
    algorithms like HIO and RAAR whose update rules mix amplitude and phase
    information in the feedback term.

    State-of-the-art enhancements (applied transparently in the base loop):

    * **Heavy-ball momentum** acceleration
    * **Adaptive β** scheduling (constant / linear / cosine)
    * **Total-variation (TV)** regularisation via Chambolle proximal operator
    * **Poisson noise** maximum-likelihood focal-plane projection
    """

    def __init__(self, config: AlgorithmConfig, pupil: PupilModel) -> None:
        self.config = config
        self.pupil = pupil
        self._rng = np.random.default_rng(config.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, psf_data: PSFData) -> PhaseRetrievalResult:
        """Execute the full retrieval loop.

        Parameters
        ----------
        psf_data : PSFData
            Validated observed PSF container (2-D intensity image).

        Returns
        -------
        PhaseRetrievalResult
            Recovered wavefront, reconstructed PSF, cost history, and
            quality metrics (RMS phase, Strehl ratio).

        Raises
        ------
        ValueError
            If the PSF image dimensions do not match the pupil grid.
        """
        # Guard: clamp negatives before sqrt (background-subtracted images
        # may contain negative pixels after sky subtraction).
        target_amp = np.sqrt(np.maximum(psf_data.image, 0.0))
        pupil_amp = self.pupil.amplitude
        n = pupil_amp.shape[0]
        support = pupil_amp > 0

        # Validate that PSF and pupil grids are compatible
        if psf_data.image.shape != (n, n):
            raise ValueError(
                f"PSF image shape {psf_data.image.shape} does not match pupil grid ({n}, {n})"
            )

        # Normalise target amplitude to match pupil energy (Parseval's theorem)
        energy_pupil = np.sum(pupil_amp**2)
        energy_target = np.sum(target_amp**2)
        if energy_target > 0:
            target_amp *= np.sqrt((energy_pupil * (n**2)) / energy_target)
        else:
            warnings.warn(
                "PSF image has zero total energy — retrieval may not converge.",
                stacklevel=2,
            )

        # Initialise complex pupil field with known amplitude + random phase
        phase0 = self._initial_phase(n)
        g = pupil_amp * np.exp(1j * phase0)

        cost_history: list[float] = []
        converged = False
        t0 = time.perf_counter()

        # Adaptive convergence window: shrink for short runs so that early
        # stopping is still possible when max_iterations < 2 * window.
        window = min(_CONVERGENCE_WINDOW, max(self.config.max_iterations // 4, 2))

        # Momentum state — tracks the *output* of the previous iteration
        # (not the momentum-extrapolated input) to implement correct
        # heavy-ball acceleration:  y_k = x_k + μ·(x_k − x_{k−1}).
        g_prev = g.copy()

        for iteration in range(1, self.config.max_iterations + 1):
            # Save current iterate *before* momentum extrapolation
            g_before_momentum = g

            # ── Heavy-ball momentum extrapolation ─────────────────────
            if self.config.momentum > 0 and iteration > 1:
                g = g + self.config.momentum * (g - g_prev)

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

            # Store iterate output for next momentum step
            g_prev = g_before_momentum

            cost_history.append(float(cost))

            # Convergence: compare the mean cost over the last two windows.
            # This handles algorithms like RAAR whose cost oscillates.
            if len(cost_history) >= 2 * window:
                recent = np.mean(cost_history[-window:])
                previous = np.mean(cost_history[-2 * window : -window])
                rel_change = abs(previous - recent) / max(
                    float(abs(previous)),
                    _EPS,
                )
                if rel_change < self.config.tolerance:
                    converged = True
                    break

        elapsed = time.perf_counter() - t0

        # Extract final phase (enforce support)
        phase = np.angle(g)
        phase[~support] = 0.0

        # Build outputs
        # Lazy import to break circular dependency:
        #   base → propagator → (potentially) base
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
        support : ndarray of bool
            Boolean mask of the pupil region.
        iteration : int
            1-based iteration counter.

        Returns
        -------
        g_new : complex ndarray
            Updated complex pupil-plane field.
        cost : float
            Focal-plane cost (amplitude error) for this iteration.
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
            return float(beta_min + 0.5 * (beta_max - beta_min) * (1 + np.cos(np.pi * t)))

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

        Note
        ----
        The Poisson ML projection uses Parseval-normalised amplitudes
        rather than raw photon counts — an approximation appropriate
        when absolute flux calibration is unavailable.
        """
        if self.config.noise_model == NoiseModel.GAUSSIAN:
            return target_amplitude * np.exp(1j * np.angle(G + _EPS))  # type: ignore[no-any-return]

        # Poisson ML projection (Thibault et al., 2012)
        I_obs = target_amplitude**2
        I_model = np.abs(G) ** 2
        # Smooth the model intensity to stabilise the ratio in low-SNR regions
        I_smooth = uniform_filter(I_model, size=3)
        I_smooth = np.maximum(I_smooth, _EPS)
        ratio = np.sqrt(np.maximum(I_obs / I_smooth, 0))
        return ratio * G  # type: ignore[no-any-return]

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

        Solves  min_u  (1/2)||u − phase||² + weight · TV(u)

        via the dual algorithm of Chambolle (2004, JMIV 20:89–97).
        Denoises the recovered phase map while preserving sharp aberration
        boundaries.  Only applied within the pupil support.

        Parameters
        ----------
        phase : ndarray
            Input phase map (radians).
        weight : float
            TV regularisation strength λ (> 0).
        support : ndarray of bool
            Pupil support mask.
        n_iter : int
            Number of dual iterations (default 10).

        Returns
        -------
        ndarray
            TV-denoised phase (same shape as input).
        """
        if weight <= 0:
            return phase

        # Chambolle's convergence condition: τ ≤ 1/||∇||²
        # For 2-D forward differences ||∇||² ≤ 8, so τ ≤ 1/8.
        tau = 0.125
        sigma = tau / weight  # effective step size in the scaled dual update

        px = np.zeros_like(phase)
        py = np.zeros_like(phase)

        for _ in range(n_iter):
            # Divergence of (px, py) using backward differences
            # (adjoint of the forward-difference gradient with a sign flip)
            div_x = np.zeros_like(phase)
            div_x[:, 1:] = px[:, :-1] - px[:, 1:]
            div_x[:, 0] = -px[:, 0]

            div_y = np.zeros_like(phase)
            div_y[1:, :] = py[:-1, :] - py[1:, :]
            div_y[0, :] = -py[0, :]

            div_p = div_x + div_y

            # Chambolle update: u = phase + weight·div_code(p),  v = −u/λ
            # The dual step is  p ← (p − σ·∇u) / (1 + σ·|∇u|)
            u = phase + weight * div_p

            # Forward-difference gradient of u
            gx = np.zeros_like(phase)
            gx[:, :-1] = u[:, 1:] - u[:, :-1]
            gy = np.zeros_like(phase)
            gy[:-1, :] = u[1:, :] - u[:-1, :]

            # Update dual variables with the correct sign (Chambolle 2004)
            norm_g = np.sqrt(gx**2 + gy**2)
            denom = 1.0 + sigma * norm_g
            px = (px - sigma * gx) / denom
            py = (py - sigma * gy) / denom

        # Final denoised phase:  û = phase + weight · div_code(p*)
        div_x = np.zeros_like(phase)
        div_x[:, 1:] = px[:, :-1] - px[:, 1:]
        div_x[:, 0] = -px[:, 0]
        div_y = np.zeros_like(phase)
        div_y[1:, :] = py[:-1, :] - py[1:, :]
        div_y[0, :] = -py[0, :]

        result = phase + weight * (div_x + div_y)
        result[~support] = 0.0
        return result  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _er_step(
        self,
        g: np.ndarray,
        pupil_amplitude: np.ndarray,
        target_amplitude: np.ndarray,
        support: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """One Error-Reduction step: P_S ∘ P_F (clean convergent finish).

        Used by RAAR and Douglas-Rachford as a final polish stage.
        """
        G = fftshift(fft2(ifftshift(g)))
        G_proj = self._project_fourier(G, target_amplitude)
        g_prime = fftshift(ifft2(ifftshift(G_proj)))

        g_new = np.zeros_like(g_prime)
        g_new[support] = pupil_amplitude[support] * np.exp(1j * np.angle(g_prime[support] + _EPS))

        cost = self._focal_cost(target_amplitude, G)
        return g_new, cost

    def _initial_phase(self, n: int) -> np.ndarray:
        """Small random perturbations around zero (diffraction-limited start).

        The range [-0.3, 0.3] rad is appropriate for near-diffraction-limited
        systems (e.g. HST); for strongly aberrated pupils a wider range may
        improve exploration at the cost of slower initial convergence.
        """
        return self._rng.uniform(-0.3, 0.3, size=(n, n)).astype(np.float64)

    @staticmethod
    def _focal_cost(target_amp: np.ndarray, G: np.ndarray) -> float:
        """Normalised focal-plane amplitude error.

        Returns the mean squared difference between the target and
        modelled focal-plane amplitudes (scale-corrected), so the cost
        is comparable across different grid sizes.
        """
        modelled_amp = np.abs(G)
        scale = target_amp.sum() / max(modelled_amp.sum(), _EPS)
        return float(np.mean((target_amp - modelled_amp * scale) ** 2))
