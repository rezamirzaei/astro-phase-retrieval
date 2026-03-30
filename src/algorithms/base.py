"""Abstract base class for all phase-retrieval algorithms."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from src.models.config import AlgorithmConfig
from src.models.optics import PhaseRetrievalResult, PupilModel, PSFData
from src.metrics.quality import compute_rms_phase, compute_strehl_ratio


class PhaseRetriever(ABC):
    """Base class that every phase-retrieval algorithm must subclass.

    Subclasses implement ``_iterate`` which receives and returns the full
    complex pupil-plane field **g** (not just phase).  This is essential for
    algorithms like HIO and RAAR whose update rules mix amplitude and phase
    information in the feedback term.
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

        # Initialise complex pupil field with known amplitude + random phase
        phase0 = self._initial_phase(n)
        g = pupil_amp * np.exp(1j * phase0)

        cost_history: list[float] = []
        converged = False
        t0 = time.perf_counter()
        window = 20  # sliding-window length for convergence check

        for iteration in range(1, self.config.max_iterations + 1):
            g, cost = self._iterate(
                g=g,
                pupil_amplitude=pupil_amp,
                target_amplitude=target_amp,
                support=support,
                iteration=iteration,
            )
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





