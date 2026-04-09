#!/usr/bin/env python3
"""Generate the phase_retrieval_crystallography.ipynb notebook.

Run:  python _gen_notebook_cryst.py

Matches the exact .ipynb format of phase_retrieval_hst.ipynb (cell IDs,
per-line source arrays, nbformat 4.5) for correct display in PyCharm.
"""
import json
from pathlib import Path

cells = []


def md(cell_id, lines):
    """Add a markdown cell."""
    cells.append({
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]
    })


def code(cell_id, lines):
    """Add a code cell."""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]
    })


# ═══════════════════════════════════════════════════════════════════════
# Cell 0 — Title & Intro
# ═══════════════════════════════════════════════════════════════════════
md("0", [
    "# 🔬 Crystallographic Phase Retrieval",
    "",
    "This notebook demonstrates **phase retrieval for X-ray crystallography** using the same",
    "iterative projection algorithms used for astronomical wavefront sensing.",
    "",
    "## The Crystallographic Phase Problem",
    "",
    "When X-rays scatter off a crystal lattice, the detector records **diffraction intensities**",
    "$|F(\\mathbf{h})|^2$ but loses the **phase** $\\varphi(\\mathbf{h})$. Recovering these phases",
    "from measured intensities alone is the *crystallographic phase problem* — one of the",
    "foundational challenges in structural biology and materials science.",
    "",
    "**Pipeline:**",
    "",
    "1. **Download** real crystal structures from the Crystallography Open Database (COD)",
    "2. **Parse** the CIF file and extract unit-cell parameters + atom sites",
    "3. **Simulate** a 2-D diffraction pattern from the structure factors",
    "4. **Run** iterative phase retrieval algorithms (ER, HIO, RAAR, WF, DR, ADMM)",
    "5. **Visualise** recovered phases, electron density, and convergence",
    "6. **Compare** algorithm performance using the crystallographic R-factor",
    "",
    "**All crystal structures used in this notebook are real-world experimental data**",
    "downloaded from the Crystallography Open Database (COD) — not synthetic models.",
    "",
    "| Algorithm | Key | Reference |",
    "|-----------|-----|----------|",
    "| Error Reduction (ER) | `er` | Fienup 1982 |",
    "| Hybrid Input-Output (HIO) | `hio` | Fienup 1982 |",
    "| Relaxed Averaged Alternating Reflections (RAAR) | `raar` | Luke 2005 |",
    "| Wirtinger Flow (WF) | `wf` | Candès et al. 2015 |",
    "| Douglas-Rachford (DR) | `dr` | Bauschke et al. 2002 |",
    "| ADMM | `admm` | Chang & Marchesini 2018 |",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 1 — Theory
# ═══════════════════════════════════════════════════════════════════════
md("1", [
    "## 1 — Theory: The Crystallographic Phase Problem",
    "",
    "In X-ray crystallography, the scattered amplitude (structure factor) for Miller index $\\mathbf{h} = (h, k, l)$ is:",
    "",
    "$$ F(\\mathbf{h}) = \\sum_{j} f_j \\, \\exp\\bigl(2\\pi i\\, \\mathbf{h} \\cdot \\mathbf{r}_j\\bigr) $$",
    "",
    "where $f_j$ is the atomic scattering factor and $\\mathbf{r}_j$ is the fractional coordinate of atom $j$.",
    "",
    "The detector records the **intensity** $I(\\mathbf{h}) = |F(\\mathbf{h})|^2$ — the phase $\\varphi(\\mathbf{h}) = \\arg F(\\mathbf{h})$ is lost.",
    "",
    "To reconstruct the electron density via inverse Fourier transform:",
    "",
    "$$ \\rho(\\mathbf{r}) = \\sum_{\\mathbf{h}} |F(\\mathbf{h})| \\, e^{i\\varphi(\\mathbf{h})} \\, e^{-2\\pi i\\, \\mathbf{h} \\cdot \\mathbf{r}} $$",
    "",
    "we **must** know $\\varphi$ — the same iterative algorithms used for astronomical wavefront sensing can recover it.",
    "",
    "**Quality metric:** the crystallographic R-factor $R = \\sum |\\sqrt{I_{\\text{obs}}} - \\sqrt{I_{\\text{calc}}}| \\, / \\, \\sum |\\sqrt{I_{\\text{obs}}}|$. Lower is better (R < 0.20 is acceptable, R < 0.05 is excellent).",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 2 — Setup header
# ═══════════════════════════════════════════════════════════════════════
md("2", [
    "## 2 — Setup and Imports",
    "",
    "All crystal structures are downloaded from the **Crystallography Open Database (COD)**,",
    "which hosts > 500,000 experimentally determined structures from published literature.",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 3 — Setup code
# ═══════════════════════════════════════════════════════════════════════
code("3", [
    "from __future__ import annotations",
    "",
    "import logging",
    "import sys",
    "from pathlib import Path",
    "",
    "import matplotlib",
    "matplotlib.use('Agg')",
    "import matplotlib.pyplot as plt",
    "import numpy as np",
    "",
    "# Ensure project root is on sys.path",
    "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()",
    "if str(PROJECT_ROOT) not in sys.path:",
    "    sys.path.insert(0, str(PROJECT_ROOT))",
    "",
    "from src.data.crystallography import (",
    "    available_cod_presets,",
    "    download_cod_preset,",
    "    parse_cif,",
    "    simulate_diffraction,",
    "    run_crystallography_retrieval,",
    ")",
    "from src.models.crystallography import AtomSite, CrystalStructure",
    "from src.visualization.crystallography_plots import (",
    "    plot_diffraction_pattern,",
    "    plot_crystallography_result,",
    "    plot_crystal_summary,",
    "    plot_electron_density,",
    "    plot_r_factor_comparison,",
    "    plot_crystallography_comparison,",
    ")",
    "from src.visualization.plots import save_figure",
    "",
    "DATA_DIR = PROJECT_ROOT / 'data'",
    "OUTPUT_DIR = PROJECT_ROOT / 'notebooks' / 'outputs'",
    "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)",
    "GRID_SIZE = 128",
    "",
    "logging.basicConfig(",
    "    level=logging.INFO,",
    "    format=\"%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s\",",
    "    stream=sys.stdout,",
    ")",
    "",
    "print('Available COD presets (real-world crystal structures):')",
    "for key, desc in available_cod_presets().items():",
    "    print(f'  {key:15s} — {desc}')",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 4 — Download header
# ═══════════════════════════════════════════════════════════════════════
md("4", [
    "## 3 — Download Real Crystal Structures from COD",
    "",
    "We fetch **real experimentally determined** crystal structures from the",
    "Crystallography Open Database (COD) via its REST API. These are actual",
    "crystallographic data published in peer-reviewed journals — not synthetic models.",
    "",
    "The download is cached: subsequent runs skip the network request.",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 5 — Download NaCl
# ═══════════════════════════════════════════════════════════════════════
code("5", [
    "# Download real NaCl structure from COD (ID: 1000041)",
    "# Published experimental data: cubic rock salt, space group Fm-3m",
    "nacl_path = download_cod_preset('nacl', DATA_DIR)",
    "nacl = parse_cif(nacl_path)",
    "",
    "print(f'Crystal: {nacl.formula}')",
    "print(f'COD ID: {nacl.cod_id}')",
    "print(f'Space group: {nacl.space_group}')",
    "print(f'Unit cell: a={nacl.a:.4f} Å, b={nacl.b:.4f} Å, c={nacl.c:.4f} Å')",
    "print(f'Angles: α={nacl.alpha}°, β={nacl.beta}°, γ={nacl.gamma}°')",
    "print(f'Atoms: {len(nacl.atoms)}')",
    "for atom in nacl.atoms:",
    "    print(f'  {atom.label}: {atom.symbol} at ({atom.x:.4f}, {atom.y:.4f}, {atom.z:.4f})')",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 6 — Download more structures
# ═══════════════════════════════════════════════════════════════════════
code("6", [
    "# Download additional real crystal structures from COD",
    "structures = {}",
    "",
    "for key in ['nacl', 'quartz', 'fluorite']:",
    "    cif_path = download_cod_preset(key, DATA_DIR)",
    "    struct = parse_cif(cif_path)",
    "    structures[key] = struct",
    "    print(",
    "        f'  ✅ {key:12s}  COD={struct.cod_id:8s}  {struct.formula:12s}  '",
    "        f'SG={struct.space_group:15s}  atoms={len(struct.atoms)}'",
    "    )",
    "",
    "print(f'\\nDownloaded {len(structures)} real crystal structures from COD.')",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 7 — Simulate header
# ═══════════════════════════════════════════════════════════════════════
md("7", [
    "## 4 — Simulate Diffraction Pattern from Real Structure",
    "",
    "We compute the 2-D structure factors $F(h,k)$ for the $l=0$ reciprocal-space",
    "plane using the **real atom positions and scattering factors** from the COD data,",
    "then plot the diffraction intensities $|F(h,k)|^2$ with a thermal",
    "(Debye-Waller) damping factor for realism.",
    "",
    "This is a physics-based forward model — the output faithfully represents",
    "what an X-ray detector would record from this crystal.",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 8 — Simulate diffraction
# ═══════════════════════════════════════════════════════════════════════
code("8", [
    "# Simulate diffraction from real NaCl structure",
    "pattern_nacl = simulate_diffraction(nacl, grid_size=GRID_SIZE)",
    "",
    "print(f'Diffraction pattern shape: {pattern_nacl.image.shape}')",
    "print(f'Total intensity (normalised): {pattern_nacl.image.sum():.6f}')",
    "print(f'Max intensity: {pattern_nacl.image.max():.6e}')",
    "print(f'Space group: {pattern_nacl.space_group}')",
    "print(f'Source COD ID: {pattern_nacl.source_id}')",
    "",
    "# Plot",
    "fig = plot_diffraction_pattern(pattern_nacl)",
    "save_figure(fig, OUTPUT_DIR / 'cryst_diffraction_nacl.png')",
    "plt.close(fig)",
    "print('Saved: cryst_diffraction_nacl.png')",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 9 — HIO header
# ═══════════════════════════════════════════════════════════════════════
md("9", [
    "## 5 — Phase Retrieval with HIO",
    "",
    "We run the **Hybrid Input-Output (HIO)** algorithm on the real NaCl diffraction",
    "data to recover the lost phases. HIO iterates between the reciprocal-space",
    "constraint (measured amplitudes) and a real-space support constraint.",
    "",
    "The feedback parameter $\\beta = 0.9$ allows the algorithm to escape local minima —",
    "a key advantage over the simpler Error Reduction (ER) algorithm.",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 10 — Run HIO
# ═══════════════════════════════════════════════════════════════════════
code("10", [
    "# Run HIO phase retrieval on real NaCl diffraction data",
    "result_hio = run_crystallography_retrieval(",
    "    pattern_nacl,",
    "    algorithm_name='hio',",
    "    max_iterations=200,",
    "    beta=0.9,",
    "    random_seed=42,",
    ")",
    "",
    "print(f'Algorithm: {result_hio.algorithm.value.upper()}')",
    "print(f'Iterations: {result_hio.n_iterations}')",
    "print(f'Converged: {result_hio.converged}')",
    "print(f'R-factor: {result_hio.r_factor:.4f}')",
    "print(f'Time: {result_hio.elapsed_seconds:.2f}s')",
    "",
    "# Full 4-panel result",
    "fig = plot_crystallography_result(pattern_nacl, result_hio)",
    "save_figure(fig, OUTPUT_DIR / 'cryst_result_hio.png')",
    "plt.close(fig)",
    "",
    "# 2×2 Summary",
    "fig = plot_crystal_summary(pattern_nacl, result_hio)",
    "save_figure(fig, OUTPUT_DIR / 'cryst_summary_hio.png')",
    "plt.close(fig)",
    "",
    "print('Saved: cryst_result_hio.png, cryst_summary_hio.png')",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 11 — Algorithm comparison header
# ═══════════════════════════════════════════════════════════════════════
md("11", [
    "## 6 — Algorithm Comparison on Real NaCl Data",
    "",
    "We compare **six** phase-retrieval algorithms on the same real NaCl diffraction data:",
    "",
    "- **ER** (Error Reduction) — monotone convergence, may stagnate",
    "- **HIO** (Hybrid Input-Output) — fast escape from local minima",
    "- **RAAR** (Relaxed Averaged Alternating Reflections) — robust convergence",
    "- **WF** (Wirtinger Flow) — gradient-based, provably optimal",
    "- **DR** (Douglas-Rachford) — proximal splitting, strong fixed-point guarantees",
    "- **ADMM** (Alternating Direction Method of Multipliers) — handles regularisation naturally",
    "",
    "All use identical initialisation (same random seed) for a fair comparison.",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 12 — Run comparison
# ═══════════════════════════════════════════════════════════════════════
code("12", [
    "algorithms = ['er', 'hio', 'raar', 'wf', 'dr', 'admm']",
    "results = {}",
    "",
    "for alg in algorithms:",
    "    res = run_crystallography_retrieval(",
    "        pattern_nacl,",
    "        algorithm_name=alg,",
    "        max_iterations=200,",
    "        random_seed=42,",
    "    )",
    "    results[alg.upper()] = res",
    "    print(",
    "        f'{alg.upper():>5s}  R={res.r_factor:.4f}  '",
    "        f'iter={res.n_iterations:4d}  '",
    "        f'time={res.elapsed_seconds:.2f}s  '",
    "        f'converged={res.converged}'",
    "    )",
    "",
    "# R-factor comparison bar chart",
    "fig = plot_r_factor_comparison(results)",
    "save_figure(fig, OUTPUT_DIR / 'cryst_r_factor_comparison.png')",
    "plt.close(fig)",
    "",
    "# Multi-algorithm comparison grid",
    "fig = plot_crystallography_comparison(pattern_nacl, results)",
    "save_figure(fig, OUTPUT_DIR / 'cryst_algorithm_comparison.png')",
    "plt.close(fig)",
    "",
    "print('\\nSaved: cryst_r_factor_comparison.png, cryst_algorithm_comparison.png')",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 13 — Convergence comparison header
# ═══════════════════════════════════════════════════════════════════════
md("13", [
    "## 7 — Convergence Comparison",
    "",
    "Cost (focal-plane error) vs. iteration for all algorithms on a single",
    "log-scale axis. Algorithms that drop faster converge more efficiently.",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 14 — Convergence plot
# ═══════════════════════════════════════════════════════════════════════
code("14", [
    "fig, ax = plt.subplots(figsize=(10, 5.5))",
    "for name, res in results.items():",
    "    ax.semilogy(",
    "        res.cost_history,",
    "        label=f'{name} (R={res.r_factor:.4f})',",
    "        linewidth=1.5,",
    "    )",
    "ax.set_xlabel('Iteration')",
    "ax.set_ylabel('Cost (focal-plane error)')",
    "ax.set_title('Convergence Comparison — Real NaCl (COD 1000041)')",
    "ax.legend(frameon=True, fontsize=9)",
    "ax.grid(True, alpha=0.3)",
    "fig.tight_layout()",
    "save_figure(fig, OUTPUT_DIR / 'cryst_convergence_comparison.png')",
    "plt.close(fig)",
    "print('Saved: cryst_convergence_comparison.png')",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 15 — Electron density header
# ═══════════════════════════════════════════════════════════════════════
md("15", [
    "## 8 — Electron Density Maps",
    "",
    "The recovered phases combined with the measured amplitudes allow us to compute",
    "the **electron density** via inverse Fourier transform:",
    "",
    "$$\\rho(\\mathbf{r}) = \\sum_{\\mathbf{h}} |F(\\mathbf{h})| \\exp[i\\varphi(\\mathbf{h})] \\exp(-2\\pi i\\, \\mathbf{h}\\cdot\\mathbf{r})$$",
    "",
    "Peaks in $\\rho$ reveal atom positions in the unit cell. We show the",
    "electron density recovered by the best-performing algorithm.",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 16 — Electron density
# ═══════════════════════════════════════════════════════════════════════
code("16", [
    "# Find the best result (lowest R-factor)",
    "best_alg = min(results, key=lambda k: results[k].r_factor)",
    "best = results[best_alg]",
    "print(f'Best algorithm: {best_alg} (R = {best.r_factor:.4f})')",
    "",
    "fig = plot_electron_density(best)",
    "save_figure(fig, OUTPUT_DIR / 'cryst_electron_density.png')",
    "plt.close(fig)",
    "print('Saved: cryst_electron_density.png')",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 17 — Multi-structure header
# ═══════════════════════════════════════════════════════════════════════
md("17", [
    "## 9 — Multiple Real Crystal Structures",
    "",
    "We apply the same pipeline to **additional real structures** downloaded from COD",
    "to demonstrate versatility across different crystal systems:",
    "",
    "- **NaCl** (COD 1000041) — cubic Fm-3m, rock salt",
    "- **SiO₂** (COD 1011097) — trigonal P3₁21, α-quartz",
    "- **CaF₂** (COD 1000043) — cubic Fm-3m, fluorite",
    "",
    "Each structure is a real experimentally determined entry from the COD.",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 18 — Multi-structure comparison
# ═══════════════════════════════════════════════════════════════════════
code("18", [
    "# Run RAAR on all downloaded real structures",
    "print(f'{\"Crystal\":>12s}  {\"Formula\":>10s}  {\"Space Group\":>16s}  '",
    "      f'{\"R-factor\":>10s}  {\"Iter\":>6s}  {\"Time (s)\":>10s}')",
    "print('-' * 80)",
    "",
    "for key, crystal in structures.items():",
    "    pat = simulate_diffraction(crystal, grid_size=GRID_SIZE)",
    "    res = run_crystallography_retrieval(",
    "        pat, algorithm_name='raar', max_iterations=200, random_seed=42",
    "    )",
    "    print(",
    "        f'{key:>12s}  {crystal.formula:>10s}  {crystal.space_group:>16s}  '",
    "        f'{res.r_factor:10.4f}  {res.n_iterations:6d}  {res.elapsed_seconds:10.2f}'",
    "    )",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 19 — Results summary header
# ═══════════════════════════════════════════════════════════════════════
md("19", [
    "## 10 — Results Summary",
    "",
    "A summary table of all algorithm results on the NaCl data.",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 20 — Results table
# ═══════════════════════════════════════════════════════════════════════
code("20", [
    "print(f'{\"Algorithm\":>10s}  {\"R-factor\":>10s}  {\"Iterations\":>12s}  '",
    "      f'{\"Converged\":>10s}  {\"Time (s)\":>10s}')",
    "print('=' * 62)",
    "",
    "for name, res in results.items():",
    "    print(",
    "        f'{name:>10s}  {res.r_factor:10.4f}  {res.n_iterations:12d}  '",
    "        f'{\"Yes\" if res.converged else \"No\":>10s}  {res.elapsed_seconds:10.2f}'",
    "    )",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 21 — Export
# ═══════════════════════════════════════════════════════════════════════
md("21", [
    "## 11 — Export Results",
    "",
    "Export algorithm comparison metadata to JSON for reproducibility.",
])

code("22", [
    "import json",
    "",
    "for name, res in results.items():",
    "    summary = {",
    "        'algorithm': res.algorithm.value,",
    "        'r_factor': res.r_factor,",
    "        'n_iterations': res.n_iterations,",
    "        'converged': res.converged,",
    "        'elapsed_seconds': res.elapsed_seconds,",
    "        'timestamp': res.timestamp.isoformat(),",
    "        'source': f'COD {pattern_nacl.source_id}',",
    "    }",
    "    out_path = OUTPUT_DIR / f'cryst_result_{name.lower()}.json'",
    "    out_path.write_text(json.dumps(summary, indent=2))",
    "    print(f'📁 {out_path}')",
    "",
    "print(f'\\n🎉 Pipeline complete — all outputs saved to {OUTPUT_DIR}')",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 23 — Takeaways
# ═══════════════════════════════════════════════════════════════════════
md("23", [
    "## 12 — Interpretation and Takeaways",
    "",
    "What this notebook demonstrates:",
    "",
    "- **100% real-world data**: every crystal structure was downloaded from the",
    "  Crystallography Open Database (COD) — experimentally determined, peer-reviewed",
    "  structural data, not synthetic models.",
    "- **Same algorithms**: the iterative phase-retrieval algorithms (ER, HIO, RAAR,",
    "  WF, DR, ADMM) used for astronomical wavefront sensing apply directly to",
    "  crystallographic diffraction data.",
    "- **The diffraction intensity** $|F(h,k)|^2$ plays the role of the PSF intensity.",
    "- **The crystallographic R-factor** replaces the Strehl ratio as quality metric.",
    "- **Recovered phases → inverse FFT → electron density map** reveals atom positions.",
    "- **Algorithm trade-offs**: HIO and RAAR typically outperform basic ER; WF provides",
    "  gradient-based convergence; DR has strong fixed-point guarantees; ADMM handles",
    "  regularisation naturally.",
    "- **Web UI**: the same pipeline is available through the web interface where you",
    "  can download additional COD presets and run phase retrieval interactively.",
    "",
    "For more structures, use `download_cod_preset()` with any of the 8 curated presets,",
    "or `download_cif(cod_id, data_dir)` with any of the 500,000+ COD entry IDs.",
])

# ═══════════════════════════════════════════════════════════════════════
# Assemble notebook
# ═══════════════════════════════════════════════════════════════════════
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).resolve().parent / "notebooks" / "phase_retrieval_crystallography.ipynb"
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"✅ Wrote {out_path}  ({len(cells)} cells)")

