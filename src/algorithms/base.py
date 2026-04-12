"""Abstract base class for all phase-retrieval algorithms.

Design
------
Every concrete algorithm implements :meth:`_iterate`, which performs one
update step and returns the new complex pupil field **g** plus the scalar
cost for that iteration.  The base :meth:`run` loop handles:

* Energy normalisation of the target amplitude
* Spectral or random phase initialisation
* Heavy-ball (Nesterov) momentum acceleration
* Adaptive β scheduling (constant / linear / cosine)
* Total-variation regularisation via the Chambolle (2004) proximal operator
* **Shrink-Wrap** dynamic support refinement (Marchesini et al. 2003)
* Convergence detection via sliding-window relative-change criterion
* Optional Rich progress bar on interactive terminals
"""

from __future__ import annotations

import sys
import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import gaussian_filter, uniform_filter  # type: ignore[import-untyped]

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

# Shrink-Wrap: update every N iterations (standard: every 10 % of budget)
_SW_UPDATE_PERIOD: int = 20


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
        self._psf_image_for_init = psf_data.image
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

        # Rich progress bar — only shown when stdout is an interactive TTY,
        # so it never pollutes piped/CI output.
        _use_rich = sys.stdout.isatty()
        try:
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )
        except ImportError:  # pragma: no cover
            _use_rich = False

        def _run_loop() -> None:
            nonlocal g, g_prev, converged, support

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
                    phase_tv = np.angle(g)
                    phase_tv = self._tv_prox(phase_tv, self.config.tv_weight, support)
                    amp = np.abs(g)
                    g = amp * np.exp(1j * phase_tv)

                # ── Shrink-Wrap dynamic support refinement ────────────────
                if self.config.use_sw_constraint and (iteration % _SW_UPDATE_PERIOD == 0):
                    support = self._shrink_wrap_step(g, pupil_amp, support, iteration)

                # Store iterate output for next momentum step
                g_prev = g_before_momentum

                cost_history.append(float(cost))

                if _task_id is not None:
                    _progress.update(  # type: ignore[union-attr]
                        _task_id,
                        advance=1,
                        description=f"[cyan]{self.config.name.value.upper()}[/]  cost={cost:.4e}",
                    )

                # Convergence: compare the mean cost over the last two windows.
                if len(cost_history) >= 2 * window:
                    recent = np.mean(cost_history[-window:])
                    previous = np.mean(cost_history[-2 * window : -window])
                    rel_change = abs(previous - recent) / max(float(abs(previous)), _EPS)
                    if rel_change < self.config.tolerance:
                        converged = True
                        break

        _progress = None
        _task_id = None
        if _use_rich:
            _progress = Progress(  # type: ignore[assignment]
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                transient=True,
            )
            with _progress:
                _task_id = _progress.add_task(
                    f"[cyan]{self.config.name.value.upper()}",
                    total=self.config.max_iterations,
                )
                _run_loop()
        else:
            _run_loop()

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
            px_prev, py_prev = px.copy(), py.copy()

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

            # Early-exit: check relative change on dual variables
            norm_px = np.sqrt(np.sum(px**2))
            change = np.sqrt(np.sum((px - px_prev) ** 2 + (py - py_prev) ** 2))
            if norm_px > 0 and change / (norm_px + _EPS) < 1e-4:
                break  # pragma: no cover — requires specific convergence conditions

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
    # Shrink-Wrap support refinement (Marchesini et al. 2003)
    # ------------------------------------------------------------------

    def _shrink_wrap_step(
        self,
        g: np.ndarray,
        pupil_amplitude: np.ndarray,
        current_support: np.ndarray,
        iteration: int | None = None,
    ) -> np.ndarray:
        """Refine the support mask using the current estimate of the object.

        The Shrink-Wrap algorithm (Marchesini et al., PRB 68, 140101, 2003)
        dynamically updates the real-space support constraint by:

        1. Smoothing the current amplitude estimate with a Gaussian of width σ
        2. Thresholding at ``support_threshold × max(smoothed amplitude)``
        3. Intersecting with the original pupil mask
        4. Removing small isolated support islands (connectivity filtering)

        σ is annealed from ``sw_sigma_start`` to ``sw_sigma_end`` over
        the iteration budget using exponential decay, allowing the algorithm
        to first find the rough object shape then progressively tighten.

        Parameters
        ----------
        g : complex ndarray
            Current pupil-plane field estimate.
        pupil_amplitude : ndarray
            Original pupil amplitude mask (defines hard outer boundary).
        current_support : ndarray of bool
            Current support estimate (refined in-place).
        iteration : int | None
            Current iteration (used for sigma annealing).

        Returns
        -------
        ndarray of bool
            Updated support mask.
        """
        from scipy.ndimage import label as ndimage_label

        # Annealed sigma: exponential decay from sigma_start to sigma_end
        sigma_start = self.config.sw_sigma_start
        sigma_end = self.config.sw_sigma_end
        if iteration is not None and self.config.max_iterations > 1:
            t = (iteration - 1) / max(self.config.max_iterations - 1, 1)
            sigma = sigma_start * (sigma_end / max(sigma_start, 1e-6)) ** t
        else:
            sigma = sigma_start

        sigma = max(sigma, sigma_end)

        # Smoothed amplitude of the current estimate
        amp = np.abs(g)
        smoothed = gaussian_filter(amp, sigma=sigma)

        # Threshold
        thresh = self.config.support_threshold * float(smoothed.max())
        new_support = (smoothed >= thresh) & (pupil_amplitude > 0)

        # Connectivity filtering: remove small isolated islands
        # (keeps only the largest connected component)
        n_pupil_px = max(float(np.sum(pupil_amplitude > 0)), 1.0)
        min_island_size = max(int(0.01 * n_pupil_px), 4)

        labeled, n_features = ndimage_label(new_support)
        if n_features > 1:
            component_sizes = np.bincount(labeled.ravel())
            # component_sizes[0] is background; find largest foreground
            component_sizes[0] = 0
            largest = component_sizes.argmax()
            # Remove islands smaller than threshold
            for label_id in range(1, n_features + 1):
                if label_id != largest and component_sizes[label_id] < min_island_size:
                    new_support[labeled == label_id] = False

        # Ensure the support does not completely collapse (fallback to pupil)
        if float(new_support.sum()) < 0.05 * n_pupil_px:
            return current_support  # refuse the update to prevent degeneracy

        return new_support  # type: ignore[return-value,no-any-return]

    # ------------------------------------------------------------------
    # Standard Error-Reduction step (used by several algorithms)
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
        """Compute starting phase via spectral initialization or random.

        When ``config.spectral_init`` is *True* (the default) **and** the
        observed PSF has been stored on ``self`` by :meth:`run`, the leading
        eigenvector of the weighted measurement matrix is used — this is the
        same *spectral initialization* that makes Wirtinger Flow effective.

        Otherwise falls back to uniform random in [-1, 1] rad — wide enough
        to cover real HST-class aberrations.
        """
        if (
            self.config.spectral_init
            and hasattr(self, "_psf_image_for_init")
            and self._psf_image_for_init is not None
        ):
            return self._spectral_init(self._psf_image_for_init, n)
        return self._rng.uniform(-1.0, 1.0, size=(n, n)).astype(np.float64)

    def _spectral_init(self, psf_image: np.ndarray, n: int) -> np.ndarray:
        """Spectral initialization via truncated power iteration.

        Computes the leading eigenvector of the weighted measurement matrix
        ``T = (1/m) Σ_k y_k a_k a_k^H`` by iterating
        ``z ← IFFT{ Y · FFT{z} }`` and normalising.
        """
        Y = psf_image
        support = self.pupil.amplitude > 0
        rng = np.random.default_rng(self.config.random_seed)
        z = self.pupil.amplitude * np.exp(1j * rng.uniform(-np.pi, np.pi, (n, n)))

        for _ in range(50):
            Z = fftshift(fft2(ifftshift(z)))
            Z = Y * Z
            z = fftshift(ifft2(ifftshift(Z)))
            z[~support] = 0.0
            norm = np.sqrt(np.sum(np.abs(z) ** 2))
            if norm > 0:
                z /= norm

        z *= np.sqrt(np.sum(self.pupil.amplitude**2))
        phase = np.angle(z)
        phase[~support] = 0.0
        return phase  # type: ignore[no-any-return]

    @staticmethod
    def _focal_cost(target_amp: np.ndarray, G: np.ndarray) -> float:
        """Normalised focal-plane amplitude error (R-factor).

        Returns the sum of squared amplitude residuals normalised by the
        sum of squared target amplitudes, making the cost scale-invariant
        and comparable across different grid sizes.
        """
        modelled_amp = np.abs(G)
        scale = target_amp.sum() / max(modelled_amp.sum(), _EPS)
        residual = target_amp - modelled_amp * scale
        denom = np.sum(target_amp**2)
        if denom > 0:
            return float(np.sum(residual**2) / denom)
        return float(np.sum(residual**2))
