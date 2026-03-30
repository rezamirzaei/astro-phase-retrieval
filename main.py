# %% [markdown]
# # 🔭 Phase Retrieval for HST Wavefront Sensing
#
# **Real-world problem**: Recover the Hubble Space Telescope's wavefront aberrations
# from actual stellar PSF observations downloaded from the MAST archive.
#
# Every optical telescope introduces wavefront errors (defocus, coma, astigmatism,
# spherical aberration …). A star is a point source — its image is the telescope's
# Point Spread Function (PSF). The detector records only **intensity** (|E|²),
# losing all phase information. Phase retrieval recovers this lost wavefront phase
# from the intensity measurement, given the known pupil geometry.
#
# This is exactly the technique NASA used to diagnose HST's famous primary-mirror
# spherical aberration in 1990, and how JWST aligns its mirror segments today.

# %% [markdown]
# ## 1 · Configuration
#
# Everything is validated by **Pydantic** models — typos, out-of-range values,
# and shape mismatches are caught before any computation begins.

# %%
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Project imports
from src.models.config import (
    AlgorithmConfig,
    AlgorithmName,
    DataConfig,
    PipelineConfig,
    PupilConfig,
    TelescopeType,
    default_hst_config,
)
from src.models.optics import PSFData, PhaseRetrievalResult, PupilModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
)

# %%
# Build a validated pipeline configuration for HST / WFC3-UVIS / F606W
config = default_hst_config()
print(config.model_dump_json(indent=2))

# %% [markdown]
# ## 2 · Download Real HST Data from MAST
#
# We fetch calibrated flat-fielded exposures (_flt.fits_) of the white-dwarf
# standard star **GRW+70°5824** — a bright, isolated point source observed
# regularly for HST calibration. This is **real data** from the telescope.

# %%
from src.data.downloader import search_and_download, list_cached_fits

# Check if we already have data cached
cached = list_cached_fits(config.data.data_dir)
if cached:
    print(f"✅ Found {len(cached)} cached FITS file(s) — skipping download.")
    fits_paths = cached
else:
    print("⬇️  Downloading real HST data from MAST archive …")
    fits_paths = search_and_download(config.data)
    print(f"✅ Downloaded {len(fits_paths)} file(s).")

print("Files:", [p.name for p in fits_paths])

# %% [markdown]
# ## 3 · Load & Extract the PSF
#
# We open the FITS file, locate the brightest star, extract a square cutout,
# subtract the sky background, and normalise. The result is a Pydantic-validated
# `PSFData` object.

# %%
from src.data.loader import load_psf_from_fits, prepare_psf_for_retrieval

# Load the first file
psf_data = load_psf_from_fits(fits_paths[0], config.data, config.pupil)
print(f"PSF shape:       {psf_data.image.shape}")
print(f"Filter:          {psf_data.filter_name}")
print(f"Telescope:       {psf_data.telescope}")
print(f"Pixel scale:     {psf_data.pixel_scale_arcsec}″/px")
print(f"Observation ID:  {psf_data.obs_id}")

# %%
# Resize to the algorithm grid
psf_image = prepare_psf_for_retrieval(psf_data, config.pupil.grid_size)
print(f"Working grid size: {psf_image.shape}")

# Update the PSFData object with the resized image
psf_data_resized = PSFData(
    image=psf_image,
    pixel_scale_arcsec=psf_data.pixel_scale_arcsec,
    wavelength_m=psf_data.wavelength_m,
    filter_name=psf_data.filter_name,
    telescope=psf_data.telescope,
    obs_id=psf_data.obs_id,
)

# %% [markdown]
# ## 4 · Build the HST Pupil Model
#
# HST has a 2.4 m primary mirror, a 0.792 m secondary mirror obstruction,
# and 4 spider vanes. We model this analytically.

# %%
from src.optics.pupils import build_pupil
from src.visualization.plots import plot_pupil, plot_observed_psf, set_style

set_style()

pupil = build_pupil(config.pupil)
print(f"Pupil grid:    {pupil.grid_size}×{pupil.grid_size}")
print(f"Open fraction: {pupil.amplitude.mean():.2%}")

fig = plot_pupil(pupil)
plt.show()

# %%
fig = plot_observed_psf(psf_data_resized, log_scale=True)
plt.show()

# %% [markdown]
# ## 5 · Run Phase Retrieval — Single Algorithm
#
# We start with the **Hybrid Input-Output (HIO)** algorithm — the workhorse of
# iterative phase retrieval, invented by Fienup (1982).

# %%
from src.algorithms.registry import AlgorithmRegistry

# HIO retrieval
hio_config = AlgorithmConfig(
    name=AlgorithmName.HYBRID_INPUT_OUTPUT,
    max_iterations=300,
    beta=0.9,
    random_seed=42,
)
retriever = AlgorithmRegistry.create(hio_config, pupil)
print(f"Algorithm: {hio_config.name.value}")
print(f"Available algorithms: {AlgorithmRegistry.available()}")

# %%
result_hio = retriever.run(psf_data_resized)
print(f"✅ HIO complete in {result_hio.elapsed_seconds:.2f}s")
print(f"   Iterations: {result_hio.n_iterations}")
print(f"   Converged:  {result_hio.converged}")
print(f"   Strehl:     {result_hio.strehl_ratio:.4f}")
print(f"   RMS phase:  {result_hio.rms_phase_rad:.4f} rad")

# %% [markdown]
# ## 6 · Visualise the Results

# %%
from src.visualization.plots import (
    plot_recovered_phase,
    plot_reconstructed_psf,
    plot_psf_residual,
    plot_psf_comparison,
    plot_radial_profile,
    plot_psf_cross_sections,
    plot_wavefront_3d,
    plot_encircled_energy,
    plot_convergence,
    plot_summary,
    save_figure,
)
from src.metrics.quality import zernike_decomposition

support = pupil.amplitude > 0

fig = plot_recovered_phase(result_hio, support)
plt.show()

# %%
fig = plot_reconstructed_psf(result_hio, log_scale=True)
plt.show()

# %%
# Direct comparison: Observed vs Reconstructed with residuals
fig = plot_psf_comparison(psf_data_resized, result_hio, log_scale=True)
save_figure(fig, config.output_dir / "psf_comparison_hio.png")
plt.show()
print(f"📁 Saved to {config.output_dir / 'psf_comparison_hio.png'}")

# %%
# Radial profile — observed vs reconstructed vs diffraction-limited (line plot)
fig = plot_radial_profile(psf_data_resized, result_hio, pupil)
save_figure(fig, config.output_dir / "radial_profile_hio.png")
plt.show()

# %%
# Horizontal & vertical cross-sections through PSF peak (line plots)
fig = plot_psf_cross_sections(psf_data_resized, result_hio)
save_figure(fig, config.output_dir / "cross_sections_hio.png")
plt.show()

# %%
# 3-D surface of recovered wavefront phase
fig = plot_wavefront_3d(result_hio, support)
save_figure(fig, config.output_dir / "wavefront_3d_hio.png")
plt.show()

# %%
# Encircled energy curve — observed vs reconstructed vs diffraction-limited
fig = plot_encircled_energy(psf_data_resized, result_hio, pupil)
save_figure(fig, config.output_dir / "encircled_energy_hio.png")
plt.show()

# %%
fig = plot_convergence(result_hio)
plt.show()

# %% [markdown]
# ## 7 · Zernike Decomposition
#
# Decompose the recovered wavefront into Zernike polynomials to identify
# which optical aberrations are present (defocus, coma, astigmatism, spherical …).

# %%
from src.visualization.plots import plot_zernike_bar, plot_zernike_polar
from src.optics.zernike import ZERNIKE_NAMES

zernike_coeffs = zernike_decomposition(result_hio.recovered_phase, support, n_terms=15)

print("Zernike coefficients (rad):")
for j, coeff in zernike_coeffs.items():
    name = ZERNIKE_NAMES.get(j, f"Z{j}")
    print(f"  j={j:2d}  {name:30s}  {coeff:+.5f}")

fig = plot_zernike_bar(zernike_coeffs)
plt.show()

# %%
# Polar map of Zernike coefficients
fig = plot_zernike_polar(zernike_coeffs)
save_figure(fig, config.output_dir / "zernike_polar_hio.png")
plt.show()

# %% [markdown]
# ## 8 · Full Summary Figure

# %%
fig = plot_summary(psf_data_resized, pupil, result_hio, zernike_coeffs)
save_figure(fig, config.output_dir / "summary_hio.png")
plt.show()
print(f"📁 Saved to {config.output_dir / 'summary_hio.png'}")

# %% [markdown]
# ## 9 · Multi-Observation Analysis
#
# Run phase retrieval on **all real HST images** — different filters (F438W,
# F606W, F814W) and different detectors (WFC3/UVIS, ACS/WFC).

# %%
from src.data.downloader import list_cached_fits, available_presets, download_all_presets
from src.visualization.plots import plot_multi_observation_grid, plot_multi_observation_radial

all_fits = list_cached_fits(config.data.data_dir)
print(f"📂 Cached FITS files: {len(all_fits)}")

if len(all_fits) < 2:
    print("⬇️  Downloading additional real HST observations …")
    download_all_presets(
        config.data.data_dir,
        keys=["hst-wfc3-uvis-f814w", "hst-wfc3-uvis-f438w", "hst-acs-wfc-f606w"],
    )
    all_fits = list_cached_fits(config.data.data_dir)

for fp in all_fits:
    print(f"  • {fp.name}")

# %%
observations = []
for fp in all_fits:
    try:
        psf_i = load_psf_from_fits(fp, config.data, config.pupil)
        img_i = prepare_psf_for_retrieval(psf_i, config.pupil.grid_size)
        psf_i_resized = PSFData(
            image=img_i,
            pixel_scale_arcsec=psf_i.pixel_scale_arcsec,
            wavelength_m=psf_i.wavelength_m,
            filter_name=psf_i.filter_name,
            telescope=psf_i.telescope,
            obs_id=psf_i.obs_id,
        )
        alg_cfg = AlgorithmConfig(name=AlgorithmName.RAAR, max_iterations=500, beta=0.9, random_seed=42)
        res_i = AlgorithmRegistry.create(alg_cfg, pupil).run(psf_i_resized)
        observations.append({
            "label": f"{psf_i.obs_id}\n{psf_i.filter_name}",
            "psf": psf_i_resized,
            "result": res_i,
            "support": support,
        })
        print(f"  ✅ {fp.name:30s}  {psf_i.filter_name:6s}  Strehl={res_i.strehl_ratio:.4f}")
    except Exception as exc:
        print(f"  ⚠️  {fp.name}: {exc}")

# %%
if len(observations) >= 2:
    fig = plot_multi_observation_grid(observations)
    save_figure(fig, config.output_dir / "multi_observation_grid.png")
    plt.show()

    fig = plot_multi_observation_radial(observations)
    save_figure(fig, config.output_dir / "multi_observation_radial.png")
    plt.show()

    # Per-observation PSF comparison
    for obs in observations:
        lbl = obs["label"].replace("\n", "_")
        fig = plot_psf_comparison(obs["psf"], obs["result"], log_scale=True)
        save_figure(fig, config.output_dir / f"psf_comparison_{lbl}.png")
        plt.show()

# %% [markdown]
# ## 10 · Compare All Algorithms
#
# Run every implemented algorithm on the same data and compare convergence,
# recovered wavefronts, and Strehl ratios.

# %%
from src.visualization.plots import plot_algorithm_comparison, plot_algorithm_dashboard, plot_strehl_rms_bar

algorithms_to_compare = [
    AlgorithmName.ERROR_REDUCTION,
    AlgorithmName.GERCHBERG_SAXTON,
    AlgorithmName.HYBRID_INPUT_OUTPUT,
    AlgorithmName.RAAR,
]

results: dict[str, PhaseRetrievalResult] = {}

for alg_name in algorithms_to_compare:
    # RAAR needs more iterations (oscillates, then ER-finish cleans up)
    n_iter = 1000 if alg_name == AlgorithmName.RAAR else 300
    alg_config = AlgorithmConfig(
        name=alg_name,
        max_iterations=n_iter,
        beta=0.9,
        random_seed=42,
    )
    retriever = AlgorithmRegistry.create(alg_config, pupil)
    res = retriever.run(psf_data_resized)
    results[alg_name.value.upper()] = res
    print(
        f"  {alg_name.value.upper():5s} — "
        f"{res.n_iterations:4d} iter, "
        f"Strehl={res.strehl_ratio:.4f}, "
        f"RMS={res.rms_phase_rad:.4f} rad, "
        f"{res.elapsed_seconds:.2f}s"
    )

# %%
fig = plot_algorithm_comparison(results, support)
save_figure(fig, config.output_dir / "algorithm_comparison.png")
plt.show()
print(f"📁 Saved to {config.output_dir / 'algorithm_comparison.png'}")

# %%
# Full dashboard: phase · PSF · residual · radial profile for every algorithm
fig = plot_algorithm_dashboard(psf_data_resized, results, support, pupil)
save_figure(fig, config.output_dir / "algorithm_dashboard.png")
plt.show()
print(f"📁 Saved to {config.output_dir / 'algorithm_dashboard.png'}")

# %%
# Strehl ratio vs RMS phase grouped bar chart
fig = plot_strehl_rms_bar(results)
save_figure(fig, config.output_dir / "strehl_rms_comparison.png")
plt.show()
print(f"📁 Saved to {config.output_dir / 'strehl_rms_comparison.png'}")

# %% [markdown]
# ## 10 · Convergence Comparison

# %%
fig, ax = plt.subplots(figsize=(9, 5))
for name, res in results.items():
    ax.semilogy(res.cost_history, label=f"{name} (Strehl={res.strehl_ratio:.3f})", linewidth=1.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("Cost (focal-plane error)")
ax.set_title("Convergence Comparison — All Algorithms")
ax.legend(frameon=True, fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_figure(fig, config.output_dir / "convergence_comparison.png")
plt.show()

# %% [markdown]
# ## 11 · Results Summary Table

# %%
from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="Phase Retrieval Results — Real HST Data", show_lines=True)
table.add_column("Algorithm", style="bold cyan")
table.add_column("Iterations", justify="right")
table.add_column("Converged", justify="center")
table.add_column("Strehl Ratio", justify="right")
table.add_column("RMS Phase (rad)", justify="right")
table.add_column("Time (s)", justify="right")

for name, res in results.items():
    table.add_row(
        name,
        str(res.n_iterations),
        "✅" if res.converged else "❌",
        f"{res.strehl_ratio:.4f}",
        f"{res.rms_phase_rad:.4f}",
        f"{res.elapsed_seconds:.2f}",
    )
console.print(table)

# %% [markdown]
# ## 12 · Export Results
#
# Every result is a Pydantic model — we can serialise the non-array metadata
# to JSON for reproducibility and record-keeping.

# %%
import json

for name, res in results.items():
    summary = {
        "algorithm": res.algorithm.value,
        "n_iterations": res.n_iterations,
        "converged": res.converged,
        "strehl_ratio": res.strehl_ratio,
        "rms_phase_rad": res.rms_phase_rad,
        "elapsed_seconds": res.elapsed_seconds,
        "timestamp": res.timestamp.isoformat(),
    }
    out_path = config.output_dir / f"result_{name.lower()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"📁 {out_path}")

print("\n🎉 Pipeline complete — all outputs saved to", config.output_dir)
