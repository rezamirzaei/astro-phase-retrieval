"""Physics-informed neural field for phase retrieval.

This optional solver uses a small coordinate MLP to represent the pupil-plane
phase. The network parameters are optimized directly against the observed PSF
through a differentiable Fourier-optics forward model.

It is intentionally lightweight: this is not a dataset-trained deep model, but
rather a per-observation neural-field optimizer with physics-informed loss.
"""

from __future__ import annotations

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


@dataclass(slots=True)
class _TorchModules:
    torch: Any
    nn: Any


class PINNPhaseRetriever(PhaseRetriever):
    """Per-observation physics-informed neural network phase retriever."""

    def run(self, psf_data: PSFData) -> PhaseRetrievalResult:
        modules = self._import_torch()
        torch = modules.torch
        nn = modules.nn

        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)

        device = self._resolve_device(torch)
        dtype = torch.float32
        pupil_amp_np = self.pupil.amplitude.astype(np.float32)
        support_np = pupil_amp_np > 0
        target_np = psf_data.image.astype(np.float32)
        target_np /= max(float(target_np.sum()), 1e-30)
        n = pupil_amp_np.shape[0]
        base_phase_np, warm_result = self._warm_start_phase(psf_data)

        pupil_amp = torch.tensor(pupil_amp_np, dtype=dtype, device=device)
        support = torch.tensor(support_np, dtype=dtype, device=device)
        target = torch.tensor(target_np, dtype=dtype, device=device)
        base_phase = torch.tensor(base_phase_np.astype(np.float32), dtype=dtype, device=device)
        coords = self._coordinate_features(torch, n, device=device, dtype=dtype)

        model = _PhaseField(
            nn=nn,
            in_features=coords.shape[-1],
            hidden_features=self.config.pinn_hidden_features,
            hidden_layers=self.config.pinn_hidden_layers,
        ).to(device=device, dtype=dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.pinn_learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, min(self.config.pinn_lr_step, self.config.max_iterations)),
            gamma=self.config.pinn_lr_gamma,
        )

        cost_history: list[float] = []
        converged = False
        best_loss = float("inf")
        best_phase = np.zeros_like(pupil_amp_np)
        t0 = time.perf_counter()
        window = 20
        warm_objective = None
        if warm_result is not None:
            warm_objective = self._objective_value(
                target_np=target_np,
                reconstructed_psf=warm_result.reconstructed_psf,
            )
            best_loss = warm_objective
            best_phase = warm_result.recovered_phase.astype(np.float64).copy()

        for iteration in range(1, self.config.max_iterations + 1):
            optimizer.zero_grad(set_to_none=True)

            phase_raw = model(coords).reshape(n, n)
            residual_phase = self.config.pinn_residual_scale * np.pi * torch.tanh(phase_raw)
            phase = base_phase + residual_phase
            phase = phase * support
            field = pupil_amp.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64))
            focal = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field)))
            psf = torch.abs(focal) ** 2
            psf = psf / psf.sum().clamp_min(1e-30)

            eps = 1e-12
            data_loss = torch.mean((psf - target) ** 2)
            sqrt_loss = torch.mean(
                (
                    torch.sqrt(psf.clamp_min(eps))
                    - torch.sqrt(target.clamp_min(eps))
                ) ** 2
            )
            log_loss = torch.mean(
                (
                    torch.log10(psf.clamp_min(eps))
                    - torch.log10(target.clamp_min(eps))
                ) ** 2
            )
            smoothness = self._smoothness_penalty(phase, support)
            objective = (
                data_loss
                + self.config.pinn_sqrt_weight * sqrt_loss
                + self.config.pinn_log_weight * log_loss
            )
            loss = objective + self.config.pinn_smoothness_weight * smoothness
            loss.backward()
            if self.config.pinn_grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), self.config.pinn_grad_clip)
            optimizer.step()
            scheduler.step()

            loss_value = float(objective.detach().cpu())
            cost_history.append(loss_value)
            if loss_value < best_loss:
                best_loss = loss_value
                best_phase = np.asarray(
                    phase.detach().cpu().to(torch.float32).tolist(),
                    dtype=np.float64,
                )

            if len(cost_history) >= 2 * window:
                recent = np.mean(cost_history[-window:])
                previous = np.mean(cost_history[-2 * window:-window])
                if abs(previous - recent) / max(abs(previous), 1e-30) < self.config.tolerance:
                    converged = True
                    break

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
                "solver": "physics_informed_neural_field",
                "device": device,
                "final_objective": cost_history[-1] if cost_history else None,
                "best_objective": best_loss,
                "hidden_features": self.config.pinn_hidden_features,
                "hidden_layers": self.config.pinn_hidden_layers,
                "learning_rate": self.config.pinn_learning_rate,
                "sqrt_weight": self.config.pinn_sqrt_weight,
                "log_weight": self.config.pinn_log_weight,
                "warm_start": self.config.pinn_warm_start,
                "warm_start_iterations": self.config.pinn_warm_start_iterations,
                "residual_scale": self.config.pinn_residual_scale,
                "warm_start_objective": warm_objective,
                "fallback_to_warm_start": fallback_used,
            },
        )

    def _iterate(self, **kwargs):  # type: ignore[override]
        raise NotImplementedError("PINNPhaseRetriever overrides run() directly.")

    @staticmethod
    def _coordinate_features(torch: Any, n: int, *, device: str, dtype: Any) -> Any:
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

    @staticmethod
    def _smoothness_penalty(phase: Any, support: Any) -> Any:
        dx = phase[:, 1:] - phase[:, :-1]
        dy = phase[1:, :] - phase[:-1, :]
        sx = support[:, 1:] * support[:, :-1]
        sy = support[1:, :] * support[:-1, :]
        return (dx.square() * sx).mean() + (dy.square() * sy).mean()

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

    def _warm_start_phase(self, psf_data: PSFData) -> tuple[np.ndarray, PhaseRetrievalResult | None]:
        if not self.config.pinn_warm_start:
            return np.zeros_like(self.pupil.amplitude, dtype=np.float64), None

        warm_cfg = self.config.model_copy(
            update={
                "name": AlgorithmName.RAAR,
                "max_iterations": min(self.config.pinn_warm_start_iterations, self.config.max_iterations),
                "momentum": 0.0,
                "tv_weight": 0.0,
                "n_starts": 1,
            }
        )
        warm_result = RAAR(warm_cfg, self.pupil).run(psf_data)
        return warm_result.recovered_phase.astype(np.float64), warm_result

    def _objective_value(self, *, target_np: np.ndarray, reconstructed_psf: np.ndarray) -> float:
        eps = 1e-12
        psf = reconstructed_psf.astype(np.float64)
        psf /= max(float(psf.sum()), 1e-30)
        target = target_np.astype(np.float64)
        data_loss = np.mean((psf - target) ** 2)
        sqrt_loss = np.mean((np.sqrt(np.clip(psf, eps, None)) - np.sqrt(np.clip(target, eps, None))) ** 2)
        log_loss = np.mean((np.log10(np.clip(psf, eps, None)) - np.log10(np.clip(target, eps, None))) ** 2)
        return float(data_loss + self.config.pinn_sqrt_weight * sqrt_loss + self.config.pinn_log_weight * log_loss)

    @staticmethod
    def _import_torch() -> _TorchModules:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "PINN support requires a working PyTorch install. "
                "Install the optional extra with `pip install -e .[pinn]`; on macOS, "
                "if Gatekeeper blocks native libraries, remove the quarantine attribute "
                "from the torch package before retrying."
            ) from exc
        return _TorchModules(torch=torch, nn=nn)


class _PhaseField:
    def __new__(
        cls,
        *,
        nn: Any,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
    ) -> Any:
        layers: list[object] = []
        current = in_features
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(current, hidden_features), nn.Tanh()])
            current = hidden_features
        layers.append(nn.Linear(current, 1))
        model = nn.Sequential(*layers)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        return model










