"""Deep diagnostic: run all algorithms, check convergence, generate comparison figures."""
from __future__ import annotations
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.config import AlgorithmConfig, AlgorithmName, default_hst_config
from src.models.optics import PSFData
from src.data.loader import load_psf_from_fits, prepare_psf_for_retrieval
from src.optics.pupils import build_pupil
from src.algorithms.registry import AlgorithmRegistry
from src.metrics.quality import compute_strehl_ratio, compute_rms_phase
from src.optics.propagator import forward_model

# ---------- Setup ----------
config = default_hst_config()
fits_path = list(Path("data").rglob("*.fits"))[0]
psf_data = load_psf_from_fits(fits_path, config.data, config.pupil)
psf_image = prepare_psf_for_retrieval(psf_data, config.pupil.grid_size)
psf_data_resized = PSFData(
    image=psf_image,
    pixel_scale_arcsec=psf_data.pixel_scale_arcsec,
    wavelength_m=psf_data.wavelength_m,
    filter_name=psf_data.filter_name,
    telescope=psf_data.telescope,
    obs_id=psf_data.obs_id,
)
pupil = build_pupil(config.pupil)
support = pupil.amplitude > 0

print("=== Input Data ===")
print(f"PSF shape: {psf_image.shape}, sum={psf_image.sum():.6f}, max={psf_image.max():.6f}, min={psf_image.min():.6f}")
print(f"Pupil support: {support.sum()}/{support.size} pixels ({support.mean():.1%})")

# ---------- Run all algorithms ----------
algorithms = [
    AlgorithmName.ERROR_REDUCTION,
    AlgorithmName.GERCHBERG_SAXTON,
    AlgorithmName.HYBRID_INPUT_OUTPUT,
    AlgorithmName.RAAR,
]

results = {}
print("\n=== Running Algorithms ===")
for alg in algorithms:
    n_iter = 2000 if alg == AlgorithmName.RAAR else 500
    beta = 0.75 if alg == AlgorithmName.RAAR else 0.9
    ac = AlgorithmConfig(name=alg, max_iterations=n_iter, beta=beta, random_seed=42)
    retriever = AlgorithmRegistry.create(ac, pupil)
    res = retriever.run(psf_data_resized)
    results[alg.value.upper()] = res

    phase = res.recovered_phase
    recon = res.reconstructed_psf
    corr = np.corrcoef(psf_image.ravel(), recon.ravel())[0, 1]
    print(f"  {alg.value.upper():5s}: Strehl={res.strehl_ratio:.4f}  RMS={res.rms_phase_rad:.4f}  "
          f"phase_std={phase[support].std():.4f}  recon_corr={corr:.4f}  "
          f"cost[0]={res.cost_history[0]:.4f}  cost[-1]={res.cost_history[-1]:.6f}")

# ---------- Generate diagnostic figure ----------
print("\n=== Generating diagnostic figures ===")

fig, axes = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle(f"Phase Retrieval — Real HST Data ({psf_data.obs_id}, {psf_data.filter_name})", fontsize=16, y=0.98)

for row_idx, (name, res) in enumerate(results.items()):
    phase = res.recovered_phase.copy()
    phase[~support] = np.nan
    recon = res.reconstructed_psf

    # Col 0: Recovered phase
    ax = axes[row_idx, 0]
    im = ax.imshow(phase, cmap="RdBu_r", origin="lower")
    ax.set_title(f"{name}: Recovered Phase")
    plt.colorbar(im, ax=ax, label="rad")

    # Col 1: Reconstructed PSF (log)
    ax = axes[row_idx, 1]
    im = ax.imshow(np.log10(recon + 1e-12), cmap="inferno", origin="lower",
                   vmin=np.log10(psf_image.max()) - 4)
    ax.set_title(f"{name}: Reconstructed PSF (log)")
    plt.colorbar(im, ax=ax)

    # Col 2: Observed PSF (log)
    ax = axes[row_idx, 2]
    im = ax.imshow(np.log10(psf_image + 1e-12), cmap="inferno", origin="lower",
                   vmin=np.log10(psf_image.max()) - 4)
    ax.set_title(f"{name}: Observed PSF (log)")
    plt.colorbar(im, ax=ax)

    # Col 3: Residual (reconstructed - observed)
    ax = axes[row_idx, 3]
    residual = recon - psf_image
    vmax = max(abs(residual.min()), abs(residual.max()))
    im = ax.imshow(residual, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
    ax.set_title(f"{name}: Residual (Strehl={res.strehl_ratio:.3f})")
    plt.colorbar(im, ax=ax)

fig.tight_layout()
fig.savefig("outputs/diagnostic_all_algorithms.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: outputs/diagnostic_all_algorithms.png")

# ---------- Convergence plot ----------
fig, ax = plt.subplots(figsize=(10, 6))
for name, res in results.items():
    ax.semilogy(res.cost_history, label=f"{name} (Strehl={res.strehl_ratio:.3f})", linewidth=2)
ax.set_xlabel("Iteration", fontsize=13)
ax.set_ylabel("Cost (focal-plane error)", fontsize=13)
ax.set_title("Convergence — All Algorithms on Real HST Data", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/convergence_all.png", dpi=150)
plt.close(fig)
print("  Saved: outputs/convergence_all.png")

# ---------- RAAR detailed ----------
raar = results["RAAR"]
print(f"\n=== RAAR Detailed ===")
print(f"Cost history length: {len(raar.cost_history)}")
print(f"Cost decreasing overall: {raar.cost_history[0]} -> {raar.cost_history[-1]}")
# Check if cost oscillates
diffs = np.diff(raar.cost_history)
increases = np.sum(diffs > 0)
decreases = np.sum(diffs <= 0)
print(f"Cost increases: {increases}, decreases: {decreases}")
print(f"Cost monotonic: {increases == 0}")

# ---------- RAAR recovered image figure ----------
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("RAAR Phase Retrieval — Recovered Wavefront & PSF", fontsize=15, y=1.02)

# Observed PSF
ax = axes[0]
im = ax.imshow(np.log10(psf_image + 1e-12), cmap="inferno", origin="lower",
               vmin=np.log10(psf_image.max()) - 4)
ax.set_title("Observed PSF (log scale)")
plt.colorbar(im, ax=ax, shrink=0.8)

# Recovered phase
phase_raar = raar.recovered_phase.copy()
phase_raar[~support] = np.nan
ax = axes[1]
im = ax.imshow(phase_raar, cmap="RdBu_r", origin="lower")
ax.set_title(f"Recovered Phase\n(RMS = {raar.rms_phase_rad:.3f} rad)")
plt.colorbar(im, ax=ax, shrink=0.8, label="rad")

# Reconstructed PSF
ax = axes[2]
recon_raar = raar.reconstructed_psf
im = ax.imshow(np.log10(recon_raar + 1e-12), cmap="inferno", origin="lower",
               vmin=np.log10(psf_image.max()) - 4)
ax.set_title(f"Reconstructed PSF (log)\nStrehl = {raar.strehl_ratio:.3f}")
plt.colorbar(im, ax=ax, shrink=0.8)

# Convergence
ax = axes[3]
ax.semilogy(raar.cost_history, "b-", alpha=0.7, linewidth=0.8)
ax.set_xlabel("Iteration")
ax.set_ylabel("Cost")
ax.set_title(f"Convergence ({raar.n_iterations} iter)")
ax.grid(True, alpha=0.3)
# Mark the RAAR→ER transition
er_start = int(len(raar.cost_history) * 0.9)
ax.axvline(er_start, color="red", linestyle="--", alpha=0.7, label="RAAR→ER switch")
ax.legend()

fig.tight_layout()
fig.savefig("outputs/summary_raar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: outputs/summary_raar.png")

print("\nDone!")



