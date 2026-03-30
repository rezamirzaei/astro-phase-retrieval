"""Douglas-Rachford (DR) splitting algorithm for phase retrieval.

The Douglas-Rachford algorithm is a proximal splitting method that finds a
point in the intersection of two constraint sets using reflectors:

    z_{k+1} = z_k + P_S(R_F(z_k)) − P_F(z_k)

where R_F = 2·P_F − I is the reflector of the Fourier-magnitude set and
P_S is the pupil-support projector.

DR has stronger fixed-point convergence guarantees than HIO and is closely
related to RAAR (which is a relaxed variant).  It was shown by Bauschke,
Combettes & Luke (2002) to be equivalent to the averaged alternating
reflections with specific parameters.

References:
    Bauschke H.H., Combettes P.L., Luke D.R. (2002)
    "Phase retrieval, error reduction algorithm, and Fienup variants:
     a view from convex optimization"
    J. Opt. Soc. Am. A 19(7):1334–1345

    Elser V. (2003) "Phase retrieval by iterated projections"
    J. Opt. Soc. Am. A 20(1):40–55

Like RAAR, DR oscillates around the solution.  The final 10% of iterations
switch to ER for a clean convergent finish.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from src.algorithms.base import PhaseRetriever


class DouglasRachford(PhaseRetriever):
    """Douglas-Rachford splitting + ER finish for phase retrieval."""

    _ER_FRACTION = 0.1

    def _iterate(
        self,
        *,
        g: np.ndarray,
        pupil_amplitude: np.ndarray,
        target_amplitude: np.ndarray,
        support: np.ndarray,
        iteration: int,
    ) -> tuple[np.ndarray, float]:
        er_start = int(self.config.max_iterations * (1.0 - self._ER_FRACTION))
        if iteration > er_start:
            return self._er_step(g, pupil_amplitude, target_amplitude, support)

        return self._dr_step(g, pupil_amplitude, target_amplitude, support, iteration)

    def _dr_step(
        self,
        g: np.ndarray,
        pupil_amplitude: np.ndarray,
        target_amplitude: np.ndarray,
        support: np.ndarray,
        iteration: int,
    ) -> tuple[np.ndarray, float]:
        gamma = self._get_beta(iteration)  # relaxation parameter

        # P_F: project onto Fourier-magnitude constraint
        G = fftshift(fft2(ifftshift(g)))
        G_proj = self._project_fourier(G, target_amplitude)
        p_f_g = fftshift(ifft2(ifftshift(G_proj)))

        # R_F = 2·P_F − I (reflector of Fourier set)
        r_f_g = 2.0 * p_f_g - g

        # P_S(R_F(g)): project the reflected point onto the support set
        p_s_rf = np.zeros_like(r_f_g)
        p_s_rf[support] = pupil_amplitude[support] * np.exp(
            1j * np.angle(r_f_g[support] + 1e-30)
        )

        # DR update: z_{k+1} = z_k + γ · (P_S(R_F(z_k)) − P_F(z_k))
        g_new = g + gamma * (p_s_rf - p_f_g)

        cost = self._focal_cost(target_amplitude, G)
        return g_new, cost

    def _er_step(
        self,
        g: np.ndarray,
        pupil_amplitude: np.ndarray,
        target_amplitude: np.ndarray,
        support: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """One ER step: P_S ∘ P_F (clean convergent finish)."""
        G = fftshift(fft2(ifftshift(g)))
        G_proj = self._project_fourier(G, target_amplitude)
        g_prime = fftshift(ifft2(ifftshift(G_proj)))

        g_new = np.zeros_like(g_prime)
        g_new[support] = pupil_amplitude[support] * np.exp(
            1j * np.angle(g_prime[support] + 1e-30)
        )

        cost = self._focal_cost(target_amplitude, G)
        return g_new, cost
