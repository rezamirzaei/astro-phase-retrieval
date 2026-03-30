"""Publication-quality visualisation for phase-retrieval results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from src.metrics.quality import compute_ssim
from src.models.optics import PhaseRetrievalResult, PSFData, PupilModel
from src.optics.zernike import ZERNIKE_NAMES


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

_CMAP_PHASE = "RdBu_r"
_CMAP_PSF = "inferno"
_CMAP_PUPIL = "gray"


def set_style() -> None:
    """Apply a clean matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 120,
        "image.origin": "lower",
    })


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_pupil(pupil: PupilModel, *, ax: plt.Axes | None = None) -> plt.Figure:
    """Plot the telescope pupil amplitude mask."""
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    ax.imshow(pupil.amplitude, cmap=_CMAP_PUPIL)
    ax.set_title("Telescope Pupil")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_observed_psf(
    psf: PSFData,
    *,
    log_scale: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot the observed PSF."""
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    img = psf.image.copy()
    if log_scale:
        img = np.log10(img + 1e-12)
    im = ax.imshow(img, cmap=_CMAP_PSF)
    ax.set_title(f"Observed PSF — {psf.filter_name} ({psf.telescope})")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log₁₀(intensity)" if log_scale else "intensity")
    fig.tight_layout()
    return fig


def plot_recovered_phase(
    result: PhaseRetrievalResult,
    support: np.ndarray,
    *,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot the recovered pupil-plane phase."""
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    phase = result.recovered_phase.copy()
    phase[~support] = np.nan
    vmax = np.nanmax(np.abs(phase))
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(phase, cmap=_CMAP_PHASE, norm=norm)
    ax.set_title(f"Recovered Wavefront — {result.algorithm.value.upper()}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Phase (rad)")
    fig.tight_layout()
    return fig


def plot_reconstructed_psf(
    result: PhaseRetrievalResult,
    *,
    log_scale: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot the reconstructed (forward-modelled) PSF."""
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    img = result.reconstructed_psf.copy()
    if log_scale:
        img = np.log10(img + 1e-12)
    im = ax.imshow(img, cmap=_CMAP_PSF)
    ax.set_title(f"Reconstructed PSF — {result.algorithm.value.upper()}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log₁₀(intensity)" if log_scale else "intensity")
    fig.tight_layout()
    return fig


def plot_psf_residual(
    psf: PSFData,
    result: PhaseRetrievalResult,
    *,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot the residual (observed − reconstructed) PSF with a diverging colormap.

    Both images are normalised to unit sum before differencing so the
    residual is on a comparable intensity scale.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    obs = psf.image / max(psf.image.sum(), 1e-30)
    rec = result.reconstructed_psf / max(result.reconstructed_psf.sum(), 1e-30)
    diff = obs - rec

    vmax = np.max(np.abs(diff))
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(diff, cmap="RdBu_r", norm=norm)
    ax.set_title(f"Residual (Obs − Recon) — {result.algorithm.value.upper()}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Δ intensity")
    fig.tight_layout()
    return fig


def plot_psf_comparison(
    psf: PSFData,
    result: PhaseRetrievalResult,
    *,
    log_scale: bool = True,
) -> plt.Figure:
    """Side-by-side: Observed PSF | Reconstructed PSF | Residual | Log-residual.

    This makes it easy to see exactly where the reconstruction differs
    from the measurement.
    """
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    obs = psf.image.copy()
    rec = result.reconstructed_psf.copy()

    # Normalise both to unit sum for fair comparison
    obs_n = obs / max(obs.sum(), 1e-30)
    rec_n = rec / max(rec.sum(), 1e-30)

    # --- Panel 1: Observed PSF (log) ---
    obs_disp = np.log10(obs_n + 1e-12) if log_scale else obs_n
    im0 = axes[0].imshow(obs_disp, cmap=_CMAP_PSF)
    axes[0].set_title("Observed PSF")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="log₁₀(I)" if log_scale else "I")

    # --- Panel 2: Reconstructed PSF (log, same colour range) ---
    rec_disp = np.log10(rec_n + 1e-12) if log_scale else rec_n
    im1 = axes[1].imshow(rec_disp, cmap=_CMAP_PSF,
                          vmin=im0.get_clim()[0], vmax=im0.get_clim()[1])
    axes[1].set_title(f"Reconstructed PSF — {result.algorithm.value.upper()}")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="log₁₀(I)" if log_scale else "I")

    # --- Panel 3: Linear residual (diverging) ---
    diff = obs_n - rec_n
    vmax = np.max(np.abs(diff))
    if vmax == 0:
        vmax = 1.0
    norm_div = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im2 = axes[2].imshow(diff, cmap="RdBu_r", norm=norm_div)
    axes[2].set_title("Residual (Obs − Recon)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="Δ intensity")

    # --- Panel 4: Absolute residual on log scale ---
    abs_diff = np.abs(diff)
    abs_diff_log = np.log10(abs_diff + 1e-12)
    im3 = axes[3].imshow(abs_diff_log, cmap="magma")
    axes[3].set_title("|Residual| (log scale)")
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3], shrink=0.8, label="log₁₀(|Δ|)")

    # Global metrics in suptitle
    rms_resid = np.sqrt(np.mean(diff ** 2))
    max_resid = np.max(np.abs(diff))
    fig.suptitle(
        f"PSF Comparison — {result.algorithm.value.upper()}  |  "
        f"RMS residual = {rms_resid:.2e}  |  Max |residual| = {max_resid:.2e}  |  "
        f"Strehl = {result.strehl_ratio:.4f}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def plot_convergence(
    result: PhaseRetrievalResult,
    *,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot convergence curve (cost vs. iteration)."""
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    ax.semilogy(result.cost_history, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost (focal-plane error)")
    ax.set_title(f"Convergence — {result.algorithm.value.upper()}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_zernike_bar(
    coefficients: dict[int, float],
    *,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Bar chart of Zernike coefficients."""
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    indices = sorted(coefficients.keys())
    values = [coefficients[j] for j in indices]
    labels = [ZERNIKE_NAMES.get(j, f"Z{j}") for j in indices]

    colors = ["#d73027" if v < 0 else "#4575b4" for v in values]
    ax.bar(range(len(indices)), values, color=colors, edgecolor="k", linewidth=0.3)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Coefficient (rad)")
    ax.set_title("Zernike Decomposition of Recovered Wavefront")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Composite summary figure
# ---------------------------------------------------------------------------

def plot_summary(
    psf: PSFData,
    pupil: PupilModel,
    result: PhaseRetrievalResult,
    zernike_coeffs: dict[int, float] | None = None,
) -> plt.Figure:
    """Create a 2×3 summary figure of the full retrieval pipeline."""
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Pupil, Observed PSF, Residual (Obs − Recon)
    plot_pupil(pupil, ax=axes[0, 0])
    plot_observed_psf(psf, ax=axes[0, 1])
    plot_psf_residual(psf, result, ax=axes[0, 2])

    # Row 2: Recovered phase, Convergence, Zernike coefficients
    support = pupil.amplitude > 0
    plot_recovered_phase(result, support, ax=axes[1, 0])
    plot_convergence(result, ax=axes[1, 1])

    if zernike_coeffs is not None:
        plot_zernike_bar(zernike_coeffs, ax=axes[1, 2])
    else:
        axes[1, 2].text(
            0.5, 0.5, "Zernike\ndecomposition\nnot computed",
            ha="center", va="center", fontsize=12, color="gray",
            transform=axes[1, 2].transAxes,
        )
        axes[1, 2].axis("off")

    fig.suptitle(
        f"Phase Retrieval Summary — {result.algorithm.value.upper()} "
        f"({result.n_iterations} iter, Strehl={result.strehl_ratio:.3f}, "
        f"RMS={result.rms_phase_rad:.3f} rad)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_algorithm_comparison(
    results: dict[str, PhaseRetrievalResult],
    support: np.ndarray,
) -> plt.Figure:
    """Compare multiple algorithms side-by-side."""
    set_style()
    n_alg = len(results)
    fig, axes = plt.subplots(2, n_alg, figsize=(5 * n_alg, 9))
    if n_alg == 1:
        axes = axes.reshape(2, 1)

    for col, (name, res) in enumerate(results.items()):
        # Phase
        phase = res.recovered_phase.copy()
        phase[~support] = np.nan
        vmax = np.nanmax(np.abs(phase)) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        axes[0, col].imshow(phase, cmap=_CMAP_PHASE, norm=norm)
        axes[0, col].set_title(f"{name}\nStrehl={res.strehl_ratio:.3f}")
        axes[0, col].axis("off")

        # Convergence
        axes[1, col].semilogy(res.cost_history, linewidth=1.2)
        axes[1, col].set_xlabel("Iteration")
        axes[1, col].set_ylabel("Cost")
        axes[1, col].set_title(f"{res.n_iterations} iter, {res.elapsed_seconds:.1f}s")
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle("Algorithm Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Additional plot types
# ---------------------------------------------------------------------------

def _azimuthal_average(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the azimuthal (radial) average of a 2-D image centred on its peak."""
    cy, cx = np.unravel_index(np.argmax(image), image.shape)
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cx, cy, image.shape[1] - cx - 1, image.shape[0] - cy - 1)
    radii = np.arange(0, max_r)
    profile = np.array([image[r == rr].mean() for rr in radii])
    return radii.astype(float), profile


def plot_radial_profile(
    psf: PSFData,
    result: PhaseRetrievalResult,
    pupil: PupilModel,
) -> plt.Figure:
    """Azimuthally averaged radial profile: observed vs reconstructed vs diffraction-limited.

    This *line plot* reveals differences in the PSF wings that are invisible
    in 2-D images.
    """
    set_style()
    from src.optics.propagator import forward_model

    obs = psf.image / max(psf.image.sum(), 1e-30)
    rec = result.reconstructed_psf / max(result.reconstructed_psf.sum(), 1e-30)
    perfect_psf = forward_model(pupil.amplitude, np.zeros_like(pupil.amplitude))
    perfect_psf = perfect_psf / max(perfect_psf.sum(), 1e-30)

    fig, ax = plt.subplots(figsize=(9, 5))
    for data, label, ls in [
        (obs, "Observed", "-"),
        (rec, "Reconstructed", "--"),
        (perfect_psf, "Diffraction-limited", ":"),
    ]:
        r, prof = _azimuthal_average(data)
        ax.semilogy(r, prof + 1e-15, label=label, linewidth=1.8, linestyle=ls)

    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel("Azimuthal-average intensity")
    ax.set_title(f"Radial PSF Profile — {result.algorithm.value.upper()}")
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    fig.tight_layout()
    return fig


def plot_psf_cross_sections(
    psf: PSFData,
    result: PhaseRetrievalResult,
) -> plt.Figure:
    """Horizontal and vertical cross-sections through the PSF peak.

    Two *line-plot* panels that show fine-grained agreement / disagreement
    at every pixel along two orthogonal cuts.
    """
    set_style()
    obs = psf.image / max(psf.image.sum(), 1e-30)
    rec = result.reconstructed_psf / max(result.reconstructed_psf.sum(), 1e-30)
    cy, cx = np.unravel_index(np.argmax(obs), obs.shape)

    fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(14, 5))

    # Horizontal cut
    ax_h.semilogy(obs[cy, :], label="Observed", linewidth=1.5)
    ax_h.semilogy(rec[cy, :], label="Reconstructed", linewidth=1.5, linestyle="--")
    ax_h.axvline(cx, color="gray", linestyle=":", linewidth=0.8)
    ax_h.set_xlabel("x (pixels)")
    ax_h.set_ylabel("Intensity")
    ax_h.set_title("Horizontal Cross-Section")
    ax_h.legend(frameon=True, fontsize=9)
    ax_h.grid(True, alpha=0.3)

    # Vertical cut
    ax_v.semilogy(obs[:, cx], label="Observed", linewidth=1.5)
    ax_v.semilogy(rec[:, cx], label="Reconstructed", linewidth=1.5, linestyle="--")
    ax_v.axvline(cy, color="gray", linestyle=":", linewidth=0.8)
    ax_v.set_xlabel("y (pixels)")
    ax_v.set_ylabel("Intensity")
    ax_v.set_title("Vertical Cross-Section")
    ax_v.legend(frameon=True, fontsize=9)
    ax_v.grid(True, alpha=0.3)

    fig.suptitle(
        f"PSF Cross-Sections — {result.algorithm.value.upper()}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def plot_wavefront_3d(
    result: PhaseRetrievalResult,
    support: np.ndarray,
) -> plt.Figure:
    """3-D surface plot of the recovered wavefront phase over the pupil."""
    set_style()
    phase = result.recovered_phase.copy()
    phase[~support] = np.nan
    n = phase.shape[0]

    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Mask outside the pupil for a clean surface
    Z = np.where(support, phase, np.nan)
    ax.plot_surface(
        X, Y, Z,
        cmap=_CMAP_PHASE,
        rstride=2, cstride=2,
        alpha=0.9,
        linewidth=0,
        antialiased=True,
    )
    ax.set_xlabel("x (normalised)")
    ax.set_ylabel("y (normalised)")
    ax.set_zlabel("Phase (rad)")
    ax.set_title(
        f"Recovered Wavefront — {result.algorithm.value.upper()}\n"
        f"RMS = {result.rms_phase_rad:.4f} rad",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def plot_encircled_energy(
    psf: PSFData,
    result: PhaseRetrievalResult,
    pupil: PupilModel,
) -> plt.Figure:
    """Encircled energy vs. radius: observed, reconstructed, diffraction-limited.

    Classic engineering plot that shows what fraction of total energy is
    captured within a given aperture radius.
    """
    set_style()
    from src.optics.propagator import forward_model

    obs = psf.image / max(psf.image.sum(), 1e-30)
    rec = result.reconstructed_psf / max(result.reconstructed_psf.sum(), 1e-30)
    perfect_psf = forward_model(pupil.amplitude, np.zeros_like(pupil.amplitude))
    perfect_psf = perfect_psf / max(perfect_psf.sum(), 1e-30)

    fig, ax = plt.subplots(figsize=(9, 5))

    for data, label, ls in [
        (obs, "Observed", "-"),
        (rec, "Reconstructed", "--"),
        (perfect_psf, "Diffraction-limited", ":"),
    ]:
        cy, cx = np.unravel_index(np.argmax(data), data.shape)
        yy, xx = np.ogrid[:data.shape[0], :data.shape[1]]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_r = int(min(cx, cy, data.shape[1] - cx - 1, data.shape[0] - cy - 1))
        radii = np.arange(1, max_r)
        ee = np.array([data[r <= rr].sum() for rr in radii])
        ax.plot(radii, ee, label=label, linewidth=1.8, linestyle=ls)

    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel("Encircled Energy (fraction)")
    ax.set_title(f"Encircled Energy — {result.algorithm.value.upper()}")
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def plot_zernike_polar(
    coefficients: dict[int, float],
) -> plt.Figure:
    """Polar lollipop chart of Zernike coefficients.

    Each aberration is placed at its azimuthal order angle, with distance
    from centre proportional to coefficient magnitude. Gives a quick
    visual sense of which directions the wavefront is aberrated.
    """
    set_style()
    from src.optics.zernike import _noll_lookup

    indices = sorted(coefficients.keys())
    values = [coefficients[j] for j in indices]
    labels = [ZERNIKE_NAMES.get(j, f"Z{j}") for j in indices]

    # Map each Zernike to an angle based on its azimuthal order m
    angles = []
    for j in indices:
        n, m = _noll_lookup(j)
        # Spread them around the circle: use m to set base angle, offset by n
        angle = np.pi / 2 + m * np.pi / (max(abs(m), 1) + n * 0.2)
        angles.append(angle)

    # De-overlap: add small offsets for terms at similar angles
    angles = np.array(angles, dtype=float)
    for i in range(len(angles)):
        for k in range(i):
            if abs(angles[i] - angles[k]) < 0.15:
                angles[i] += 0.2

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    abs_vals = np.abs(values)
    colors = ["#d73027" if v < 0 else "#4575b4" for v in values]

    for angle, r, color, label in zip(angles, abs_vals, colors, labels):
        ax.plot([angle, angle], [0, r], color=color, linewidth=2, alpha=0.8)
        ax.plot(angle, r, "o", color=color, markersize=8)
        ax.annotate(
            label, (angle, r),
            textcoords="offset points", xytext=(5, 5),
            fontsize=7, ha="left",
        )

    ax.set_title("Zernike Polar Map\n(red = negative, blue = positive)", fontsize=12, pad=20)
    ax.set_ylim(0, max(abs_vals) * 1.3 if max(abs_vals) > 0 else 1.0)
    fig.tight_layout()
    return fig


def plot_algorithm_dashboard(
    psf: PSFData,
    results: dict[str, PhaseRetrievalResult],
    support: np.ndarray,
    pupil: PupilModel,
) -> plt.Figure:
    """Comprehensive 4-row × N-column dashboard comparing all algorithms.

    Rows: recovered phase | reconstructed PSF | residual | radial profile.
    Each column is one algorithm.
    """
    set_style()
    from src.optics.propagator import forward_model

    n_alg = len(results)
    fig, axes = plt.subplots(4, n_alg, figsize=(5.5 * n_alg, 20))
    if n_alg == 1:
        axes = axes.reshape(4, 1)

    obs_n = psf.image / max(psf.image.sum(), 1e-30)
    perfect_psf = forward_model(pupil.amplitude, np.zeros_like(pupil.amplitude))
    perfect_psf = perfect_psf / max(perfect_psf.sum(), 1e-30)
    r_obs, prof_obs = _azimuthal_average(obs_n)
    r_perf, prof_perf = _azimuthal_average(perfect_psf)

    for col, (name, res) in enumerate(results.items()):
        # Row 0 — Recovered phase (heatmap)
        phase = res.recovered_phase.copy()
        phase[~support] = np.nan
        vmax = np.nanmax(np.abs(phase)) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im0 = axes[0, col].imshow(phase, cmap=_CMAP_PHASE, norm=norm)
        axes[0, col].set_title(f"{name}\nStrehl={res.strehl_ratio:.3f}  RMS={res.rms_phase_rad:.3f}")
        axes[0, col].axis("off")
        fig.colorbar(im0, ax=axes[0, col], shrink=0.7, label="rad")

        # Row 1 — Reconstructed PSF (log heatmap)
        rec_n = res.reconstructed_psf / max(res.reconstructed_psf.sum(), 1e-30)
        rec_log = np.log10(rec_n + 1e-12)
        im1 = axes[1, col].imshow(rec_log, cmap=_CMAP_PSF)
        axes[1, col].set_title("Reconstructed PSF")
        axes[1, col].axis("off")
        fig.colorbar(im1, ax=axes[1, col], shrink=0.7, label="log₁₀(I)")

        # Row 2 — Residual (diverging heatmap)
        diff = obs_n - rec_n
        vmax_d = np.max(np.abs(diff)) or 1.0
        norm_d = TwoSlopeNorm(vmin=-vmax_d, vcenter=0, vmax=vmax_d)
        im2 = axes[2, col].imshow(diff, cmap="RdBu_r", norm=norm_d)
        axes[2, col].set_title(f"Residual  RMS={np.sqrt(np.mean(diff**2)):.2e}")
        axes[2, col].axis("off")
        fig.colorbar(im2, ax=axes[2, col], shrink=0.7, label="Δ I")

        # Row 3 — Radial profile (line plot)
        r_rec, prof_rec = _azimuthal_average(rec_n)
        axes[3, col].semilogy(r_obs, prof_obs + 1e-15, label="Observed", linewidth=1.3)
        axes[3, col].semilogy(r_rec, prof_rec + 1e-15, label="Reconstructed", linewidth=1.3, linestyle="--")
        axes[3, col].semilogy(r_perf, prof_perf + 1e-15, label="Diffraction-lim.", linewidth=1.0, linestyle=":")
        axes[3, col].set_xlabel("Radius (px)")
        axes[3, col].set_ylabel("Intensity")
        axes[3, col].set_title("Radial Profile")
        axes[3, col].legend(fontsize=7, frameon=True)
        axes[3, col].grid(True, alpha=0.3)

    # Row labels
    for row, label in enumerate(["Recovered Phase", "Reconstructed PSF", "Residual", "Radial Profile"]):
        axes[row, 0].annotate(
            label, xy=(-0.3, 0.5), xycoords="axes fraction",
            fontsize=12, fontweight="bold", rotation=90,
            ha="center", va="center",
        )

    fig.suptitle(
        "Algorithm Dashboard — Phase · PSF · Residual · Radial",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0.03, 0, 1, 0.96])
    return fig


def plot_strehl_rms_bar(
    results: dict[str, PhaseRetrievalResult],
) -> plt.Figure:
    """Grouped bar chart comparing Strehl ratio and RMS phase across algorithms."""
    set_style()
    names = list(results.keys())
    strehls = [r.strehl_ratio for r in results.values()]
    rms_vals = [r.rms_phase_rad for r in results.values()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars1 = ax1.bar(x - width / 2, strehls, width, label="Strehl Ratio",
                     color="#4575b4", edgecolor="k", linewidth=0.5)
    ax1.set_ylabel("Strehl Ratio", color="#4575b4")
    ax1.tick_params(axis="y", labelcolor="#4575b4")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, rms_vals, width, label="RMS Phase (rad)",
                     color="#d73027", edgecolor="k", linewidth=0.5)
    ax2.set_ylabel("RMS Phase (rad)", color="#d73027")
    ax2.tick_params(axis="y", labelcolor="#d73027")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=11)
    ax1.set_title("Algorithm Performance — Strehl Ratio vs. RMS Wavefront Error",
                   fontsize=13, fontweight="bold")

    # Add value labels on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, color="#4575b4")
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, color="#d73027")

    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.92), frameon=True)
    fig.tight_layout()
    return fig


def plot_pinn_benchmark(
    psf: PSFData,
    results: dict[str, PhaseRetrievalResult],
) -> plt.Figure:
    """Focused benchmark view for PINN vs. classical baselines.

    Layout:
      - Top-left: observed PSF (reference)
      - Top row remaining columns: reconstructed PSFs
      - Bottom-left: compact metric legend/notes
      - Bottom row remaining columns: absolute residuals on log scale
    """
    set_style()
    names = list(results.keys())
    n_alg = len(names)
    fig, axes = plt.subplots(2, n_alg + 1, figsize=(4.8 * (n_alg + 1), 9))

    obs = psf.image / max(psf.image.sum(), 1e-30)
    obs_log = np.log10(obs + 1e-12)
    im_ref = axes[0, 0].imshow(obs_log, cmap=_CMAP_PSF)
    axes[0, 0].set_title("Observed PSF")
    axes[0, 0].axis("off")
    fig.colorbar(im_ref, ax=axes[0, 0], shrink=0.75, label="log₁₀(I)")

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.03,
        0.97,
        "Metrics shown in titles:\nStrehl / RMS / SSIM\nResidual panels show log₁₀(|Obs − Recon|)",
        transform=axes[1, 0].transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )

    for col, name in enumerate(names, start=1):
        res = results[name]
        rec = res.reconstructed_psf / max(res.reconstructed_psf.sum(), 1e-30)
        ssim = compute_ssim(obs, rec)

        rec_log = np.log10(rec + 1e-12)
        im_top = axes[0, col].imshow(rec_log, cmap=_CMAP_PSF)
        axes[0, col].set_title(
            f"{name}\nStrehl={res.strehl_ratio:.3f}  RMS={res.rms_phase_rad:.3f}\nSSIM={ssim:.5f}",
            fontsize=11,
        )
        axes[0, col].axis("off")
        fig.colorbar(im_top, ax=axes[0, col], shrink=0.75, label="log₁₀(I)")

        abs_resid = np.abs(obs - rec)
        im_bot = axes[1, col].imshow(np.log10(abs_resid + 1e-12), cmap="magma")
        axes[1, col].set_title(
            f"|Residual|  max={np.max(abs_resid):.2e}\nTime={res.elapsed_seconds:.2f}s  Final cost={res.cost_history[-1]:.3g}",
            fontsize=10,
        )
        axes[1, col].axis("off")
        fig.colorbar(im_bot, ax=axes[1, col], shrink=0.75, label="log₁₀(|Δ|)")

    fig.suptitle("PINN Benchmark — Real-data comparison against classical baselines", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 150) -> None:
    """Save a figure to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_multi_observation_grid(
    observations: list[dict],
) -> plt.Figure:
    """Compare phase retrieval across multiple real observations.

    Parameters
    ----------
    observations : list[dict]
        Each dict has keys: ``"label"``, ``"psf"`` (PSFData),
        ``"result"`` (PhaseRetrievalResult), ``"support"`` (ndarray).

    Shows 4 rows per observation: observed PSF, recovered phase,
    residual, radial profile.
    """
    set_style()
    n_obs = len(observations)
    fig, axes = plt.subplots(4, n_obs, figsize=(5.5 * n_obs, 20))
    if n_obs == 1:
        axes = axes.reshape(4, 1)

    for col, obs in enumerate(observations):
        psf = obs["psf"]
        res = obs["result"]
        support = obs["support"]
        label = obs["label"]

        # Row 0 — Observed PSF (log heatmap)
        obs_n = psf.image / max(psf.image.sum(), 1e-30)
        obs_log = np.log10(obs_n + 1e-12)
        im0 = axes[0, col].imshow(obs_log, cmap=_CMAP_PSF)
        axes[0, col].set_title(f"{label}\n{psf.filter_name} ({psf.telescope})")
        axes[0, col].axis("off")
        fig.colorbar(im0, ax=axes[0, col], shrink=0.7, label="log₁₀(I)")

        # Row 1 — Recovered phase (diverging heatmap)
        phase = res.recovered_phase.copy()
        phase[~support] = np.nan
        vmax = np.nanmax(np.abs(phase)) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im1 = axes[1, col].imshow(phase, cmap=_CMAP_PHASE, norm=norm)
        axes[1, col].set_title(f"Strehl={res.strehl_ratio:.3f}  RMS={res.rms_phase_rad:.3f}")
        axes[1, col].axis("off")
        fig.colorbar(im1, ax=axes[1, col], shrink=0.7, label="rad")

        # Row 2 — Residual (diverging heatmap)
        rec_n = res.reconstructed_psf / max(res.reconstructed_psf.sum(), 1e-30)
        diff = obs_n - rec_n
        vmax_d = np.max(np.abs(diff)) or 1.0
        norm_d = TwoSlopeNorm(vmin=-vmax_d, vcenter=0, vmax=vmax_d)
        im2 = axes[2, col].imshow(diff, cmap="RdBu_r", norm=norm_d)
        axes[2, col].set_title(f"Residual  RMS={np.sqrt(np.mean(diff**2)):.2e}")
        axes[2, col].axis("off")
        fig.colorbar(im2, ax=axes[2, col], shrink=0.7, label="Δ I")

        # Row 3 — Convergence (line plot)
        axes[3, col].semilogy(res.cost_history, linewidth=1.3, color="#4575b4")
        axes[3, col].set_xlabel("Iteration")
        axes[3, col].set_ylabel("Cost")
        axes[3, col].set_title(f"{res.n_iterations} iter, {res.elapsed_seconds:.1f}s")
        axes[3, col].grid(True, alpha=0.3)

    # Row labels
    for row, label in enumerate(["Observed PSF", "Recovered Phase", "Residual", "Convergence"]):
        axes[row, 0].annotate(
            label, xy=(-0.3, 0.5), xycoords="axes fraction",
            fontsize=12, fontweight="bold", rotation=90,
            ha="center", va="center",
        )

    fig.suptitle(
        "Multi-Observation Comparison — Real HST Data",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0.03, 0, 1, 0.96])
    return fig


def plot_multi_observation_radial(
    observations: list[dict],
) -> plt.Figure:
    """Overlay radial PSF profiles from multiple real observations on one plot.

    Parameters
    ----------
    observations : list[dict]
        Each dict has keys: ``"label"``, ``"psf"`` (PSFData),
        ``"result"`` (PhaseRetrievalResult).
    """
    set_style()
    fig, (ax_obs, ax_rec) = plt.subplots(1, 2, figsize=(16, 6))

    for obs in observations:
        psf = obs["psf"]
        res = obs["result"]
        label = obs["label"]

        obs_n = psf.image / max(psf.image.sum(), 1e-30)
        rec_n = res.reconstructed_psf / max(res.reconstructed_psf.sum(), 1e-30)

        r_o, p_o = _azimuthal_average(obs_n)
        r_r, p_r = _azimuthal_average(rec_n)

        ax_obs.semilogy(r_o, p_o + 1e-15, label=label, linewidth=1.5)
        ax_rec.semilogy(r_r, p_r + 1e-15, label=label, linewidth=1.5)

    ax_obs.set_xlabel("Radius (px)")
    ax_obs.set_ylabel("Azimuthal-average intensity")
    ax_obs.set_title("Observed PSF — Radial Profiles")
    ax_obs.legend(fontsize=9, frameon=True)
    ax_obs.grid(True, alpha=0.3)
    ax_obs.set_xlim(left=0)

    ax_rec.set_xlabel("Radius (px)")
    ax_rec.set_ylabel("Azimuthal-average intensity")
    ax_rec.set_title("Reconstructed PSF — Radial Profiles")
    ax_rec.legend(fontsize=9, frameon=True)
    ax_rec.grid(True, alpha=0.3)
    ax_rec.set_xlim(left=0)

    fig.suptitle("Radial Profiles Across Observations", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig






