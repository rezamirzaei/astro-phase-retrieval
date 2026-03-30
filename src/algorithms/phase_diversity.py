"""Phase Diversity algorithm for wavefront sensing.

Uses a pair of images — one in-focus and one with a known defocus — to
jointly estimate the pupil-plane phase.  This is the method actually used
by STScI for HST focus monitoring and JWST mirror alignment.

Reference:
    Gonsalves R.A. (1982) "Phase Retrieval and Diversity in Adaptive Optics"
    Paxman, Schulz & Fienup (1992) "Joint estimation of object and aberrations…"
"""

from __future__ import annotations

import time

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from src.algorithms.base import PhaseRetriever
from src.metrics.quality import compute_rms_phase, compute_strehl_ratio
from src.models.optics import PhaseRetrievalResult, PSFPair
from src.optics.propagator import add_defocus, forward_model, make_complex_pupil


class PhaseDiversity(PhaseRetriever):
    """Phase-diversity wavefront retrieval from a focused + defocused image pair.

    This overrides `run` to accept a `PSFPair` instead of a single `PSFData`.
    The single-image `_iterate` is still available as a fallback (acts like ER).
    """

    def run_diversity(self, pair: PSFPair) -> PhaseRetrievalResult:
        """Run phase-diversity retrieval on a focused + defocused pair."""
        target_foc = np.sqrt(pair.focused.image)
        target_defoc = np.sqrt(pair.defocused.image)
        pupil_amp = self.pupil.amplitude
        n = pupil_amp.shape[0]
        support = pupil_amp > 0
        defocus_waves = self.config.defocus_waves

        phase = self._initial_phase(n)
        cost_history: list[float] = []
        converged = False

        t0 = time.perf_counter()

        for iteration in range(1, self.config.max_iterations + 1):
            phase, cost = self._iterate_diversity(
                phase=phase,
                pupil_amplitude=pupil_amp,
                target_foc=target_foc,
                target_defoc=target_defoc,
                support=support,
                defocus_waves=defocus_waves,
                iteration=iteration,
            )
            cost_history.append(float(cost))

            if len(cost_history) >= 2:
                delta = abs(cost_history[-2] - cost_history[-1])
                if delta < self.config.tolerance:
                    converged = True
                    break

        elapsed = time.perf_counter() - t0
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
            metadata={"defocus_waves": defocus_waves},
        )

    def _iterate_diversity(
        self,
        *,
        phase: np.ndarray,
        pupil_amplitude: np.ndarray,
        target_foc: np.ndarray,
        target_defoc: np.ndarray,
        support: np.ndarray,
        defocus_waves: float,
        iteration: int,
    ) -> tuple[np.ndarray, float]:
        """One iteration of the phase-diversity Gerchberg-Saxton variant."""
        # --- Channel 1: focused image ---
        g1 = make_complex_pupil(pupil_amplitude, phase)
        G1 = fftshift(fft2(ifftshift(g1)))
        G1_corrected = target_foc * np.exp(1j * np.angle(G1))
        g1_back = fftshift(ifft2(ifftshift(G1_corrected)))
        phase1 = np.angle(g1_back)

        # --- Channel 2: defocused image ---
        phase_defoc = add_defocus(phase, pupil_amplitude, defocus_waves)
        g2 = make_complex_pupil(pupil_amplitude, phase_defoc)
        G2 = fftshift(fft2(ifftshift(g2)))
        G2_corrected = target_defoc * np.exp(1j * np.angle(G2))
        g2_back = fftshift(ifft2(ifftshift(G2_corrected)))
        phase2_with_defoc = np.angle(g2_back)
        phase2 = phase2_with_defoc - add_defocus(
            np.zeros_like(phase), pupil_amplitude, defocus_waves
        )

        # --- Average the two phase estimates ---
        new_phase = 0.5 * (phase1 + phase2)
        new_phase[~support] = 0.0

        # Cost
        cost1 = self._focal_cost(target_foc, G1)
        cost2 = self._focal_cost(target_defoc, G2)

        return new_phase, cost1 + cost2

    # Single-image fallback (acts like ER)
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
        G_prime = target_amplitude * np.exp(1j * np.angle(G))
        g_prime = fftshift(ifft2(ifftshift(G_prime)))
        g_new = np.zeros_like(g_prime)
        g_new[support] = pupil_amplitude[support] * np.exp(1j * np.angle(g_prime[support]))
        cost = self._focal_cost(target_amplitude, G)
        return g_new, cost


