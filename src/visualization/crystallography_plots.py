"""Publication-quality visualisation for crystallographic phase-retrieval results."""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from src.models.crystallography import CrystallographyResult, DiffractionPattern
from src.visualization.plots import (
    _CMAP_PHASE,
    _CMAP_PSF,
    _PALETTE,
    set_style,
)

# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------


def plot_diffraction_pattern(
    pattern: DiffractionPattern,
    *,
    log_scale: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a 2-D diffraction pattern (log-scale heatmap)."""
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = cast(plt.Figure, ax.figure)
    img = pattern.image.copy()
    if log_scale:
        img = np.log10(img + 1e-12)
    im = ax.imshow(img, cmap=_CMAP_PSF)
    ax.set_title(f"Diffraction Pattern — {pattern.space_group}\n(COD: {pattern.source_id})")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log₁₀(I)" if log_scale else "Intensity")
    fig.tight_layout()
    return fig


def plot_electron_density(
    result: CrystallographyResult,
    *,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot the recovered 2-D electron density map."""
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = cast(plt.Figure, ax.figure)
    ed = result.electron_density.copy()
    im = ax.imshow(ed, cmap="viridis")
    ax.set_title(f"Electron Density — {result.algorithm.value.upper()}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8, label="ρ(r)")
    fig.tight_layout()
    return fig


def plot_crystallography_phase(
    result: CrystallographyResult,
    *,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot the recovered phase from crystallographic retrieval."""
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = cast(plt.Figure, ax.figure)
    phase = result.recovered_phase.copy()
    vmax = np.max(np.abs(phase))
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(phase, cmap=_CMAP_PHASE, norm=norm)
    ax.set_title(f"Recovered Phase — {result.algorithm.value.upper()}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Phase (rad)")
    fig.tight_layout()
    return fig


def plot_crystallography_result(
    pattern: DiffractionPattern,
    result: CrystallographyResult,
) -> plt.Figure:
    """Side-by-side: Observed diffraction | Recovered phase | Reconstruction | Electron density."""
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))

    # Panel 1: Observed diffraction
    obs = np.log10(pattern.image + 1e-12)
    im0 = axes[0].imshow(obs, cmap=_CMAP_PSF)
    axes[0].set_title("Observed Diffraction")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="log₁₀(I)")

    # Panel 2: Recovered phase
    phase = result.recovered_phase.copy()
    vmax = np.max(np.abs(phase)) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im1 = axes[1].imshow(phase, cmap=_CMAP_PHASE, norm=norm)
    axes[1].set_title(f"Recovered Phase — {result.algorithm.value.upper()}")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="Phase (rad)")

    # Panel 3: Reconstructed diffraction
    rec = np.log10(result.reconstructed_diffraction + 1e-12)
    im2 = axes[2].imshow(rec, cmap=_CMAP_PSF, vmin=obs.min(), vmax=obs.max())
    axes[2].set_title("Reconstructed Diffraction")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="log₁₀(I)")

    # Panel 4: Electron density
    im3 = axes[3].imshow(result.electron_density, cmap="viridis")
    axes[3].set_title("Electron Density")
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3], shrink=0.8, label="ρ(r)")

    fig.suptitle(
        f"Crystallography Phase Retrieval — {result.algorithm.value.upper()}  |  "
        f"R-factor = {result.r_factor:.4f}  |  {result.n_iterations} iter, "
        f"{result.elapsed_seconds:.2f}s",
        fontsize=13,
        fontweight="bold",
        color="black",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_crystal_summary(
    pattern: DiffractionPattern,
    result: CrystallographyResult,
) -> plt.Figure:
    """2×2 composite summary: observed, recovered phase, electron density, convergence."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0) Observed diffraction
    obs = np.log10(pattern.image + 1e-12)
    im0 = axes[0, 0].imshow(obs, cmap=_CMAP_PSF)
    axes[0, 0].set_title(f"Diffraction — {pattern.space_group}")
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    # (0,1) Recovered phase
    phase = result.recovered_phase.copy()
    vmax = np.max(np.abs(phase)) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im1 = axes[0, 1].imshow(phase, cmap=_CMAP_PHASE, norm=norm)
    axes[0, 1].set_title(f"Recovered Phase — {result.algorithm.value.upper()}")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8, label="rad")

    # (1,0) Electron density
    im2 = axes[1, 0].imshow(result.electron_density, cmap="viridis")
    axes[1, 0].set_title("Electron Density Map")
    axes[1, 0].axis("off")
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.8, label="ρ(r)")

    # (1,1) Convergence curve
    axes[1, 1].semilogy(result.cost_history, linewidth=1.5, color=_PALETTE[0])
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Cost")
    axes[1, 1].set_title(f"Convergence — R={result.r_factor:.4f}")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Crystallography Summary — {result.algorithm.value.upper()}\n"
        f"R-factor = {result.r_factor:.4f}  |  {result.n_iterations} iter  |  "
        f"{result.elapsed_seconds:.2f}s",
        fontsize=14,
        fontweight="bold",
        color="black",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_r_factor_comparison(
    results: dict[str, CrystallographyResult],
) -> plt.Figure:
    """Bar chart comparing R-factors across algorithms."""
    set_style()
    names = list(results.keys())
    r_factors = [r.r_factor for r in results.values()]
    times = [r.elapsed_seconds for r in results.values()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bars1 = ax1.bar(
        x - width / 2,
        r_factors,
        width,
        label="R-factor",
        color="#4575b4",
        edgecolor="k",
        linewidth=0.5,
    )
    ax1.set_ylabel("R-factor", color="#4575b4", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#4575b4")
    ax1.set_ylim(0, max(max(r_factors) * 1.3, 0.1) if r_factors else 1.0)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        times,
        width,
        label="Time (s)",
        color="#d73027",
        edgecolor="k",
        linewidth=0.5,
    )
    ax2.set_ylabel("Time (s)", color="#d73027", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#d73027")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=11, color="black")
    ax1.set_title(
        "Algorithm Comparison — R-factor vs. Time",
        fontsize=13,
        fontweight="bold",
        color="black",
    )

    for bar in bars1:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=9,
            color="#4575b4", fontweight="bold",
        )
    for bar in bars2:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}s",
            ha="center", va="bottom", fontsize=9,
            color="#d73027", fontweight="bold",
        )

    fig.legend(
        [bars1, bars2],  # type: ignore[list-item]
        ["R-factor", "Time (s)"],
        loc="upper right",
        bbox_to_anchor=(0.95, 0.92),
        frameon=True,
        facecolor="white",
        edgecolor="#999999",
        framealpha=0.95,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_crystallography_comparison(
    pattern: DiffractionPattern,
    results: dict[str, CrystallographyResult],
) -> plt.Figure:
    """Multi-algorithm comparison: phase + electron density for each algorithm."""
    set_style()
    n_alg = len(results)
    fig, axes = plt.subplots(3, n_alg, figsize=(5.5 * n_alg, 14))
    if n_alg == 1:
        axes = axes.reshape(3, 1)

    for col, (name, res) in enumerate(results.items()):
        # Row 0: Recovered phase
        phase = res.recovered_phase.copy()
        vmax = np.max(np.abs(phase)) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        axes[0, col].imshow(phase, cmap=_CMAP_PHASE, norm=norm)
        axes[0, col].set_title(
            f"{name}\nR = {res.r_factor:.4f}",
            color="black",
        )
        axes[0, col].axis("off")

        # Row 1: Electron density
        axes[1, col].imshow(res.electron_density, cmap="viridis")
        axes[1, col].set_title("Electron Density", color="black")
        axes[1, col].axis("off")

        # Row 2: Convergence
        axes[2, col].semilogy(
            res.cost_history,
            linewidth=1.2,
            color=_PALETTE[col % len(_PALETTE)],
        )
        axes[2, col].set_xlabel("Iteration")
        axes[2, col].set_ylabel("Cost")
        axes[2, col].set_title(
            f"{res.n_iterations} iter, {res.elapsed_seconds:.1f}s",
            color="black",
        )
        axes[2, col].grid(True, alpha=0.3)

    fig.suptitle(
        "Crystallography Algorithm Comparison",
        fontsize=14, fontweight="bold", color="black",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


