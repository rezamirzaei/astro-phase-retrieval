"""Physics-informed neural field for phase retrieval.

This optional solver uses a coordinate MLP with **random Fourier feature
encoding** to represent the pupil-plane phase.  The network parameters are
optimised directly against the observed PSF through a differentiable
Fourier-optics forward model.

Key design choices
------------------
* **Fourier feature encoding** — maps low-dimensional (x, y) coordinates
  through sin/cos of random projections, enabling the network to learn
  high-spatial-frequency phase structure (Tancik et al. 2020).
* **Two-phase optimisation** — Adam for global exploration followed by
  L-BFGS for precise refinement near the minimum.
* **Physics-informed composite loss** — MSE on PSF intensity, log1p
  loss for the wings, sqrt loss for photon-limited fidelity, and
  Laplacian smoothness to suppress noise.
* **Warm-start from RAAR** — the neural field learns a *residual*
  correction on top of a classical phase estimate, dramatically improving
  convergence and final quality.

References
----------
Tancik M. et al. (2020) "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains", NeurIPS 2020.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.algorithms.base import PhaseRetriever
from src.algorithms.raar import RAAR
from src.metrics.quality import compute_rms_phase, compute_strehl_ratio
from src.models.config import AlgorithmName
from src.models.optics import PSFData, PhaseRetrievalResult
from src.optics.propagator import forward_model

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _TorchModules:
    torch: Any
    nn: Any


class PINNPhaseRetriever(PhaseRetriever):
    """Per-observation physics-informed neural-field phase retriever.

    The solver warm-starts from a classical RAAR reconstruction and uses a
    Fourier-encoded coordinate MLP to learn a residual phase correction
    that further reduces the focal-plane reconstruction error.
    """

    # Fraction of total iterations devoted to L-BFGS refinement
    _LBFGS_FRACTION = 0.20

    def run(self, psf_data: PSFData) -> PhaseRetrievalResult:
        modules = self._import_torch()
        torch = modules.torch
        nn = modules.nn

        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)

        device = self._resolve_device(torch)
        dtype = torch.float32

        # --- Prepare numpy arrays -------------------------------------------
        pupil_amp_np = self.pupil.amplitude.astype(np.float32)
        support_np = pupil_amp_np > 0
        target_np = psf_data.image.astype(np.float32)
        target_np /= max(float(target_np.sum()), 1e-30)
        n = pupil_amp_np.shape[0]

        # --- Warm start from classical solver --------------------------------
        base_phase_np, warm_result = self._warm_start_phase(psf_data)

        # --- Move to torch ---------------------------------------------------
        pupil_amp = torch.tensor(pupil_amp_np, dtype=dtype, device=device)
        support = torch.tensor(support_np, dtype=dtype, device=device)
        target = torch.tensor(target_np, dtype=dtype, device=device)
        base_phase = torch.tensor(base_phase_np.astype(np.float32), dtype=dtype, device=device)
        coords = self._coordinate_features(torch, n, device=device, dtype=dtype)

        # --- Build Fourier-encoded MLP ---------------------------------------
        n_fourier = self.config.pinn_fourier_features
        sigma = self.config.pinn_fourier_sigma
        rng = torch.Generator(device=device)
        if self.config.random_seed is not None:
            rng.manual_seed(self.config.random_seed + 7)
        B_matrix = torch.randn(
            coords.shape[-1], n_fourier, generator=rng, device=device, dtype=dtype,
        ) * sigma

        model = self._build_phase_field(nn, n_fourier, device, dtype)

        # --- Optimiser setup -------------------------------------------------
        adam_iters = max(1, int(self.config.max_iterations * (1.0 - self._LBFGS_FRACTION)))
        lbfgs_budget = max(0, self.config.max_iterations - adam_iters)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.pinn_learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, adam_iters // 3),
            T_mult=1,
            eta_min=self.config.pinn_learning_rate * 0.01,
        )

        # --- State tracking --------------------------------------------------
        cost_history: list[float] = []
        converged = False
        best_loss = float("inf")
        best_phase = np.zeros_like(pupil_amp_np)
        window = 20
        t0 = time.perf_counter()

        warm_objective = None
        if warm_result is not None:
            warm_objective = self._objective_value(
                target_np=target_np,
                reconstructed_psf=warm_result.reconstructed_psf,
            )
            best_loss = warm_objective
            best_phase = warm_result.recovered_phase.astype(np.float64).copy()

        # =====================================================================
        # Phase 1 — Adam exploration
        # =====================================================================
        for iteration in range(1, adam_iters + 1):
            optimizer.zero_grad(set_to_none=True)

            phase = self._forward_phase(
                model, coords, B_matrix, base_phase, support, n, torch,
            )
            psf_pred = self._forward_psf(phase, pupil_amp, torch)
            loss, objective = self._composite_loss(
                psf_pred, target, phase, support, torch,
            )

            loss.backward()
            if self.config.pinn_grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), self.config.pinn_grad_clip)
            optimizer.step()
            scheduler.step()

            loss_value = float(objective.detach().cpu())
            cost_history.append(loss_value)

            if loss_value < best_loss:
                best_loss = loss_value
                best_phase = phase.detach().cpu().numpy().astype(np.float64)

            # Convergence check
            if len(cost_history) >= 2 * window:
                recent = np.mean(cost_history[-window:])
                previous = np.mean(cost_history[-2 * window:-window])
                if abs(previous - recent) / max(abs(previous), 1e-30) < self.config.tolerance:
                    converged = True
                    break

        # =====================================================================
        # Phase 2 — L-BFGS refinement
        # =====================================================================
        if lbfgs_budget > 0 and not converged:
            lbfgs_optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=self.config.pinn_lbfgs_lr,
                max_iter=lbfgs_budget,
                line_search_fn="strong_wolfe",
                history_size=10,
                tolerance_grad=1e-9,
                tolerance_change=1e-12,
            )

            lbfgs_steps: list[float] = []

            def closure():
                lbfgs_optimizer.zero_grad()
                ph = self._forward_phase(
                    model, coords, B_matrix, base_phase, support, n, torch,
                )
                psf_p = self._forward_psf(ph, pupil_amp, torch)
                ls, obj = self._composite_loss(psf_p, target, ph, support, torch)
                ls.backward()
                lbfgs_steps.append(float(obj.detach().cpu()))
                return ls

            lbfgs_optimizer.step(closure)

            cost_history.extend(lbfgs_steps)

            # Update best from L-BFGS
            with torch.no_grad():
                phase_final = self._forward_phase(
                    model, coords, B_matrix, base_phase, support, n, torch,
                )
                psf_final = self._forward_psf(phase_final, pupil_amp, torch)
                _, obj_final = self._composite_loss(
                    psf_final, target, phase_final, support, torch,
                )
                final_val = float(obj_final.cpu())
                if final_val < best_loss:
                    best_loss = final_val
                    best_phase = phase_final.cpu().numpy().astype(np.float64)

        # =====================================================================
        # Build result
        # =====================================================================
        elapsed = time.perf_counter() - t0
        best_phase[~support_np] = 0.0
        recon_psf = forward_model(self.pupil.amplitude, best_phase)
        rms = compute_rms_phase(best_phase, support_np)
        strehl = compute_strehl_ratio(recon_psf, self.pupil.amplitude)
        neural_improved = warm_objective is None or best_loss + 1e-12 < warm_objective
        fallback_used = warm_result is not None and not neural_improved

        return PhaseRetrievalResult(
            algorithm=self.config.name,
            recovered_phase=best_phase,
            recovered_amplitude=self.pupil.amplitude,
            reconstructed_psf=recon_psf,
            cost_history=cost_history,
            n_iterations=len(cost_history),
            converged=converged,
            elapsed_seconds=elapsed,
            rms_phase_rad=rms,
            strehl_ratio=strehl,
            metadata={
                "solver": "fourier_feature_neural_field",
                "device": device,
                "final_objective": cost_history[-1] if cost_history else None,
                "best_objective": best_loss,
                "hidden_features": self.config.pinn_hidden_features,
                "hidden_layers": self.config.pinn_hidden_layers,
                "fourier_features": self.config.pinn_fourier_features,
                "fourier_sigma": self.config.pinn_fourier_sigma,
                "learning_rate": self.config.pinn_learning_rate,
                "sqrt_weight": self.config.pinn_sqrt_weight,
                "log_weight": self.config.pinn_log_weight,
                "warm_start": self.config.pinn_warm_start,
                "warm_start_iterations": self.config.pinn_warm_start_iterations,
                "residual_scale": self.config.pinn_residual_scale,
                "warm_start_objective": warm_objective,
                "fallback_to_warm_start": fallback_used,
                "adam_iterations": adam_iters,
                "lbfgs_budget": lbfgs_budget,
            },
        )

    def _iterate(self, **kwargs):  # type: ignore[override]
        raise NotImplementedError("PINNPhaseRetriever overrides run() directly.")

    # ------------------------------------------------------------------
    # Forward model helpers
    # ------------------------------------------------------------------

    def _forward_phase(
        self, model, coords, B_matrix, base_phase, support, n, torch,
    ):
        """Compute full phase from the neural field output."""
        proj = coords @ B_matrix  # [N, n_fourier]
        fourier = torch.cat([
            torch.sin(2.0 * math.pi * proj),
            torch.cos(2.0 * math.pi * proj),
        ], dim=-1)  # [N, 2*n_fourier]
        phase_raw = model(fourier).reshape(n, n)
        residual = self.config.pinn_residual_scale * math.pi * torch.tanh(phase_raw)
        phase = base_phase + residual
        return phase * support

    @staticmethod
    def _forward_psf(phase, pupil_amp, torch):
        """Differentiable forward model: phase → normalised PSF."""
        field = pupil_amp.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64))
        focal = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field)))
        psf = torch.abs(focal) ** 2
        return psf / psf.sum().clamp_min(1e-30)

    # ------------------------------------------------------------------
    # Composite loss
    # ------------------------------------------------------------------

    def _composite_loss(self, psf_pred, target, phase, support, torch):
        """Compute the physics-informed composite loss.

        Components
        ----------
        1. MSE on intensity (data fidelity)
        2. sqrt-intensity loss (photon-counting fidelity)
        3. log1p-intensity loss (wing sensitivity)
        4. Laplacian smoothness on phase (regularisation)
        """
        eps = 1e-12

        # Data fidelity
        data_loss = torch.mean((psf_pred - target) ** 2)

        # sqrt-intensity (Poisson-like)
        sqrt_loss = torch.mean(
            (torch.sqrt(psf_pred.clamp_min(eps)) - torch.sqrt(target.clamp_min(eps))) ** 2
        )

        # log1p loss — numerically more stable than log10 and emphasises
        # the PSF wings where faint structure lives
        alpha = 1e4
        log_loss = torch.mean(
            (torch.log1p(alpha * psf_pred) - torch.log1p(alpha * target)) ** 2
        )

        # Smoothness (gradient-based)
        smoothness = self._smoothness_penalty(phase, support)

        objective = (
            data_loss
            + self.config.pinn_sqrt_weight * sqrt_loss
            + self.config.pinn_log_weight * log_loss
        )
        loss = objective + self.config.pinn_smoothness_weight * smoothness
        return loss, objective

    # ------------------------------------------------------------------
    # Coordinate encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _coordinate_features(torch: Any, n: int, *, device: str, dtype: Any) -> Any:
        """Build coordinate features for the neural field.

        Uses (x, y, rho, x²−y², xy) — 5 input features that are then
        passed through the random Fourier feature projection.
        """
        axis = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        rho = torch.sqrt(x.square() + y.square()).clamp(max=1.0)
        feats = torch.stack([
            x,
            y,
            rho,
            x.square() - y.square(),
            x * y,
        ], dim=-1)
        return feats.reshape(-1, feats.shape[-1])

    # ------------------------------------------------------------------
    # Smoothness penalty
    # ------------------------------------------------------------------

    @staticmethod
    def _smoothness_penalty(phase: Any, support: Any) -> Any:
        """Gradient-based smoothness penalty on the phase within the pupil."""
        dx = phase[:, 1:] - phase[:, :-1]
        dy = phase[1:, :] - phase[:-1, :]
        sx = support[:, 1:] * support[:, :-1]
        sy = support[1:, :] * support[:-1, :]
        return (dx.square() * sx).mean() + (dy.square() * sy).mean()

    # ------------------------------------------------------------------
    # Network builder
    # ------------------------------------------------------------------

    def _build_phase_field(self, nn, n_fourier: int, device: str, dtype):
        """Build a Fourier-encoded MLP phase field.

        Architecture: [2·n_fourier] → hidden → GELU → … → hidden → GELU → [1]

        GELU activations provide smooth gradients and are well-suited to
        representing physically smooth wavefront phase maps.
        """
        hidden = self.config.pinn_hidden_features
        n_layers = self.config.pinn_hidden_layers
        in_dim = 2 * n_fourier

        layers: list[object] = []
        current = in_dim
        for i in range(n_layers):
            layers.append(nn.Linear(current, hidden))
            layers.append(nn.GELU())
            current = hidden

        layers.append(nn.Linear(current, 1))
        model = nn.Sequential(*layers)

        # Initialise weights
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        return model.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    def _resolve_device(self, torch: Any) -> str:
        requested = self.config.pinn_device
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if requested == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            return "cpu"
        return requested

    # ------------------------------------------------------------------
    # Warm start
    # ------------------------------------------------------------------

    def _warm_start_phase(self, psf_data: PSFData) -> tuple[np.ndarray, PhaseRetrievalResult | None]:
        """Initialise from a classical RAAR reconstruction."""
        if not self.config.pinn_warm_start:
            return np.zeros_like(self.pupil.amplitude, dtype=np.float64), None

        warm_cfg = self.config.model_copy(
            update={
                "name": AlgorithmName.RAAR,
                "max_iterations": min(
                    self.config.pinn_warm_start_iterations,
                    self.config.max_iterations * 3,
                ),
                "momentum": 0.0,
                "tv_weight": 0.0,
                "n_starts": 1,
            }
        )
        warm_result = RAAR(warm_cfg, self.pupil).run(psf_data)
        logger.info(
            "PINN warm-start complete: %d iter, Strehl=%.4f, RMS=%.4f rad",
            warm_result.n_iterations,
            warm_result.strehl_ratio,
            warm_result.rms_phase_rad,
        )
        return warm_result.recovered_phase.astype(np.float64), warm_result

    # ------------------------------------------------------------------
    # Objective for comparison with warm start
    # ------------------------------------------------------------------

    def _objective_value(self, *, target_np: np.ndarray, reconstructed_psf: np.ndarray) -> float:
        """Compute the composite objective in numpy for comparison."""
        eps = 1e-12
        psf = reconstructed_psf.astype(np.float64)
        psf /= max(float(psf.sum()), 1e-30)
        target = target_np.astype(np.float64)

        data_loss = np.mean((psf - target) ** 2)
        sqrt_loss = np.mean(
            (np.sqrt(np.clip(psf, eps, None)) - np.sqrt(np.clip(target, eps, None))) ** 2,
        )
        alpha = 1e4
        log_loss = np.mean(
            (np.log1p(alpha * psf) - np.log1p(alpha * target)) ** 2,
        )
        return float(
            data_loss
            + self.config.pinn_sqrt_weight * sqrt_loss
            + self.config.pinn_log_weight * log_loss
        )

    # ------------------------------------------------------------------
    # PyTorch import
    # ------------------------------------------------------------------

    @staticmethod
    def _import_torch() -> _TorchModules:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PINN support requires a working PyTorch install. "
                "Install the optional extra with `pip install -e .[pinn]`; on macOS, "
                "if Gatekeeper blocks native libraries, remove the quarantine attribute "
                "from the torch package before retrying."
            ) from exc
        return _TorchModules(torch=torch, nn=nn)

