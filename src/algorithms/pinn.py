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
from src.metrics.quality import compute_rms_phase, compute_strehl_ratio
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

        device = self._resolve_device(torch)
        dtype = torch.float32
        pupil_amp_np = self.pupil.amplitude.astype(np.float32)
        support_np = pupil_amp_np > 0
        target_np = psf_data.image.astype(np.float32)
        target_np /= max(float(target_np.sum()), 1e-30)
        n = pupil_amp_np.shape[0]

        pupil_amp = torch.tensor(pupil_amp_np, dtype=dtype, device=device)
        support = torch.tensor(support_np, dtype=dtype, device=device)
        target = torch.tensor(target_np, dtype=dtype, device=device)
        coords = self._coordinate_features(torch, n, device=device, dtype=dtype)

        model = _PhaseField(
            nn=nn,
            in_features=coords.shape[-1],
            hidden_features=self.config.pinn_hidden_features,
            hidden_layers=self.config.pinn_hidden_layers,
        ).to(device=device, dtype=dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.pinn_learning_rate)

        cost_history: list[float] = []
        converged = False
        best_loss = float("inf")
        best_phase = np.zeros_like(pupil_amp_np)
        t0 = time.perf_counter()
        window = 20

        for iteration in range(1, self.config.max_iterations + 1):
            optimizer.zero_grad(set_to_none=True)

            phase = model(coords).reshape(n, n)
            phase = phase * support
            field = pupil_amp.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64))
            focal = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field)))
            psf = torch.abs(focal) ** 2
            psf = psf / psf.sum().clamp_min(1e-30)

            data_loss = torch.mean((psf - target) ** 2)
            smoothness = self._smoothness_penalty(phase, support)
            loss = data_loss + self.config.pinn_smoothness_weight * smoothness
            loss.backward()
            optimizer.step()

            loss_value = float(data_loss.detach().cpu())
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
                "final_data_loss": cost_history[-1] if cost_history else None,
                "best_data_loss": best_loss,
                "hidden_features": self.config.pinn_hidden_features,
                "hidden_layers": self.config.pinn_hidden_layers,
            },
        )

    def _iterate(self, **kwargs):  # type: ignore[override]
        raise NotImplementedError("PINNPhaseRetriever overrides run() directly.")

    @staticmethod
    def _coordinate_features(torch: "torch", n: int, *, device: str, dtype: "torch.dtype") -> "torch.Tensor":
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
    def _smoothness_penalty(phase: "torch.Tensor", support: "torch.Tensor") -> "torch.Tensor":
        dx = phase[:, 1:] - phase[:, :-1]
        dy = phase[1:, :] - phase[:-1, :]
        sx = support[:, 1:] * support[:, :-1]
        sy = support[1:, :] * support[:-1, :]
        return (dx.square() * sx).mean() + (dy.square() * sy).mean()

    def _resolve_device(self, torch: "torch") -> str:
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
        nn: "nn",
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
    ) -> "nn.Module":
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




