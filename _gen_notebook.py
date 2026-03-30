#!/usr/bin/env python3
"""Generate the updated phase_retrieval_hst.ipynb notebook.

Run:  python _gen_notebook.py
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
# Cell 0 — Title & intro (UPDATED: added SOTA algorithms + enhancements)
# ═══════════════════════════════════════════════════════════════════════
md("0", [
    "# \U0001f52d Phase Retrieval for Astronomical Wavefront Sensing",
    "",
    "This notebook demonstrates a production-ready iterative phase-retrieval pipeline",
    "applied to **real Hubble Space Telescope (HST)** point-spread function (PSF)",
    "observations downloaded from the Mikulski Archive for Space Telescopes (MAST).",
    "",
    "When HST (or any telescope) images a point source (star), the resulting PSF is",
    "",
    "$$",
    "I(\\mathbf{u}) = \\bigl| \\mathcal{F}\\{ A(\\mathbf{x})\\, e^{i\\,\\varphi(\\mathbf{x})} \\} \\bigr|^2",
    "$$",
    "",
    "where $A$ is the known pupil amplitude (aperture shape) and $\\varphi$ is the",
    "unknown wavefront phase we wish to recover. The detector records only the",
    "**intensity** $I$, so the phase is lost \u2014 this is the *phase problem*.",
    "",
    "**Phase retrieval** iteratively recovers $\\varphi$ from $I$ alone, given $A$.",
    "",
    "This is precisely how NASA diagnosed HST's famous 2.2 mm spherical aberration",
    "in 1990 and how JWST's 18 mirror segments are co-phased today.",
    "",
    "**Implemented algorithms:**",
    "",
    "| Algorithm | Key | Reference |",
    "|-----------|-----|----------|",
    "| Error Reduction (ER) | `er` | Fienup 1982 |",
    "| Gerchberg\u2013Saxton (GS) | `gs` | Gerchberg & Saxton 1972 |",
    "| Hybrid Input-Output (HIO) | `hio` | Fienup 1982 |",
    "| Relaxed Averaged Alternating Reflections (RAAR) | `raar` | Luke 2005 |",
    "| Phase Diversity (PD) | `phase_diversity` | Gonsalves 1982 |",
    "| **Wirtinger Flow (WF)** | `wf` | Cand\u00e8s, Li & Soltanolkotabi 2015 |",
    "| **Douglas-Rachford (DR)** | `dr` | Bauschke, Combettes & Luke 2002 |",
    "| **ADMM** | `admm` | Chang & Marchesini 2018 |",
    "| **Physics-Informed Neural Field** *(optional)* | `pinn` | Differentiable Fourier-optics neural field |",
    "",
    "**State-of-the-art enhancements** (built into every algorithm):",
    "",
    "| Enhancement | CLI flag | Description |",
    "|-------------|----------|-------------|",
    "| Nesterov Momentum | `--momentum 0.5` | Heavy-ball acceleration |",
    "| Adaptive \u03b2 Scheduling | `--beta-schedule cosine` | Cosine / linear annealing |",
    "| TV Regularization | `--tv-weight 0.01` | Chambolle proximal for noise-robust phase |",
    "| Poisson Noise Model | `--noise-model poisson` | ML projection for photon-limited data |",
    "| Multi-Start | `--n-starts 5` | Multiple random restarts |",
    "",
    "**Quality metrics:** Strehl ratio \u00b7 RMS wavefront error \u00b7 Zernike decomposition \u00b7 **MTF** \u00b7 **SSIM** \u00b7 **Phase Structure Function**",
    "",
    "**Notebook scope note:** the default notebook benchmarks the seven FFT-based iterative solvers. The optional `pinn` method is available through the package and CLI when PyTorch is installed, but it is intentionally excluded from the default notebook run because it is slower and environment-dependent.",
    "",
    "Primary references:",
    "",
    "- Fienup J.R. (1982), *Phase retrieval algorithms: a comparison*, Applied Optics 21(15)",
    "- Gerchberg R.W. & Saxton W.O. (1972), *A practical algorithm for the determination of phase*, Optik 35",
    "- Luke D.R. (2005), *Relaxed averaged alternating reflections for diffraction imaging*, Inverse Problems 21",
    "- Gonsalves R.A. (1982), *Phase retrieval and diversity in adaptive optics*, Optical Engineering 21",
    "- Cand\u00e8s E.J., Li X., Soltanolkotabi M. (2015), *Phase Retrieval via Wirtinger Flow*, IEEE Trans. IT 61(4)",
    "- Bauschke H.H., Combettes P.L., Luke D.R. (2002), *Phase retrieval, error reduction algorithm, and Fienup variants*, JOSA A 19(7)",
    "- Chang H., Marchesini S. (2018), *ADMM methods for phase retrieval*, arXiv:1804.05306",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 1 — Theory: Phase Problem (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════
md("1", [
    "## 1 \u2014 Theory: The Phase Problem in Optical Astronomy",
    "",
    "An optical system images a distant point source (star) by collecting the electromagnetic field across its aperture and focusing it onto a detector. The complex field at the pupil plane is:",
    "",
    "$$ E_{\\text{pupil}}(\\mathbf{x}) = A(\\mathbf{x}) \\, e^{i\\,\\varphi(\\mathbf{x})} $$",
    "",
    "where:",
    "- $A(\\mathbf{x})$ is the **pupil amplitude** (binary mask of the aperture geometry \u2014 primary mirror, secondary obstruction, spider vanes)",
    "- $\\varphi(\\mathbf{x})$ is the **wavefront phase** encoding all optical aberrations (defocus, coma, astigmatism, spherical, etc.)",
    "",
    "The focal-plane image (PSF) is the squared modulus of the Fourier transform:",
    "",
    "$$ I(\\mathbf{u}) = \\bigl| \\mathcal{F}\\{E_{\\text{pupil}}\\}(\\mathbf{u}) \\bigr|^2 $$",
    "",
    "The detector records only $I$ \u2014 all phase information in the Fourier domain is **lost**. This is the *phase problem*: given $I$ and $A$, recover $\\varphi$.",
    "",
    "The problem is **non-convex** and has ambiguities (trivial: global phase shift, conjugate inversion). However, in telescope wavefront sensing the known pupil support $A$ provides a powerful constraint that makes the problem well-posed in practice.",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 2 — Theory: Algorithms (UPDATED: added WF, DR, ADMM)
# ═══════════════════════════════════════════════════════════════════════
md("2", [
    "## 2 \u2014 Theory: Iterative Phase Retrieval Algorithms",
    "",
    "All implemented algorithms share the same skeleton:",
    "",
    "1. Start with a random phase estimate $\\varphi_0$",
    "2. **Forward-propagate**: form $E = A \\cdot e^{i\\varphi_k}$, compute $G = \\mathcal{F}\\{E\\}$",
    "3. **Focal-plane constraint**: replace $|G|$ with $\\sqrt{I_{\\text{measured}}}$, keep phase of $G$",
    "4. **Back-propagate**: $E' = \\mathcal{F}^{-1}\\{G'\\}$",
    "5. **Pupil-plane constraint**: enforce known amplitude $A$ and support, extract updated $\\varphi_{k+1}$",
    "6. Repeat until convergence",
    "",
    "The algorithms differ in **step 5** \u2014 how they update the pupil-plane estimate:",
    "",
    "- **Error Reduction (ER)**: simply project onto the support \u2014 $g_{k+1} = P_S(g')$",
    "- **Gerchberg\u2013Saxton (GS)**: same as ER in this two-plane formulation",
    "- **Hybrid Input-Output (HIO)**: outside support, uses feedback: $g_{k+1} = g_k - \\beta \\, g'$ (escapes local minima)",
    "- **RAAR**: convex relaxation: $g_{k+1} = \\beta \\, P_S(R_F(g_k)) + (1-\\beta)\\, P_F(g_k)$ (better convergence guarantees)",
    "- **Phase Diversity**: uses two images (focused + defocused) for a jointly constrained problem",
    "",
    "### State-of-the-Art Algorithms",
    "",
    "**Wirtinger Flow (WF)** \u2014 Gradient descent on the intensity loss using Wirtinger (complex) derivatives: $g_{k+1} = g_k - \\frac{\\mu}{n^2} \\mathcal{F}^{-1}\\{(|G_k|^2 - I) \\cdot G_k\\}$. Combined with *spectral initialization* (leading eigenvector of the weighted measurement matrix), WF converges linearly to the global optimum (Cand\u00e8s et al. 2015).",
    "",
    "**Douglas-Rachford (DR)** \u2014 Proximal splitting with reflectors: $z_{k+1} = z_k + \\gamma(P_S(R_F(z_k)) - P_F(z_k))$. DR has stronger fixed-point convergence guarantees than HIO (Bauschke et al. 2002).",
    "",
    "**ADMM** \u2014 Alternating Direction Method of Multipliers splits the problem with a dual variable $u$: $G \\leftarrow P_F(\\mathcal{F}\\{g\\} + u)$, $g \\leftarrow P_S(\\mathcal{F}^{-1}\\{G - u\\})$, $u \\leftarrow u + \\mathcal{F}\\{g\\} - G$. Naturally handles regularization (Chang & Marchesini 2018).",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 3 — Theory: Zernike (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════
md("3", [
    "## 3 \u2014 Theory: Zernike Polynomial Wavefront Decomposition",
    "",
    "Once the wavefront phase $\\varphi(\\mathbf{x})$ is recovered, we decompose it into the **Zernike polynomial basis** \u2014 the standard orthogonal basis on the unit disk, using the Noll (1976) single-index ordering:",
    "",
    "$$ \\varphi(\\rho, \\theta) = \\sum_{j=2}^{N} a_j \\, Z_j(\\rho, \\theta) $$",
    "",
    "Each coefficient $a_j$ maps to a named optical aberration:",
    "",
    "| Noll j | Name | Effect |",
    "|--------|------|--------|",
    "| 2, 3 | Tip / Tilt | Image shift |",
    "| 4 | Defocus | Blurring |",
    "| 5, 6 | Astigmatism | Elliptical PSF |",
    "| 7, 8 | Coma | Comet-shaped PSF |",
    "| 11 | Primary Spherical | Halo around core |",
    "",
    "The Strehl ratio (peak intensity relative to a perfect telescope) is related to the RMS wavefront error by the Mar\u00e9chal approximation:",
    "",
    "$$ S \\approx e^{-(2\\pi \\, \\sigma_{\\text{WFE}} / \\lambda)^2} $$",
    "",
    "A well-corrected telescope has $\\sigma_{\\text{WFE}} < \\lambda/14$ (Strehl > 0.8).",
])

# Cell 4 — Setup header (UNCHANGED)
md("4", [
    "## 4 \u2014 Setup and Pydantic Configuration",
    "",
    "All experiment settings are Pydantic-validated, catching configuration errors before any computation. The `PipelineConfig` aggregates data, optics, and algorithm sub-configurations.",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 5 — Setup code (UPDATED: added BetaSchedule, NoiseModel imports)
# ═══════════════════════════════════════════════════════════════════════
code("5", [
    "from __future__ import annotations",
    "",
    "import logging",
    "import sys",
    "from pathlib import Path",
    "",
    "import matplotlib.pyplot as plt",
    "import numpy as np",
    "import json",
    "",
    "# Auto-reload source modules when they change on disk",
    "%load_ext autoreload",
    "%autoreload 2",
    "",
    "# Ensure project root is on sys.path for imports",
    "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()",
    "if str(PROJECT_ROOT) not in sys.path:",
    "    sys.path.insert(0, str(PROJECT_ROOT))",
    "",
    "from src.models.config import (",
    "    AlgorithmConfig, AlgorithmName, BetaSchedule, NoiseModel, DataConfig,",
    "    PipelineConfig, PupilConfig, TelescopeType, default_hst_config,",
    ")",
    "from src.models.optics import PSFData, PhaseRetrievalResult, PupilModel",
    "",
    "logging.basicConfig(level=logging.INFO, format=\"%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s\", stream=sys.stdout)",
    "",
    "%matplotlib inline",
])

# Cell 6 — Config (UNCHANGED)
code("6", [
    "config = default_hst_config()",
    "",
    "# Ensure data dir resolves to the project-root data/ regardless of CWD",
    "config.data.data_dir = PROJECT_ROOT / \"data\"",
    "config.output_dir = PROJECT_ROOT / \"notebooks\" / \"outputs\"",
    "",
    "print(config.model_dump_json(indent=2))",
])

# Cell 7 — Download header (UNCHANGED)
md("7", [
    "## 5 \u2014 Download Real HST Data from MAST",
    "",
    "We fetch calibrated flat-fielded exposures (`_flt.fits`) of the white-dwarf standard star **GRW+70\u00b05824** \u2014 a bright, isolated point source observed regularly by STScI for HST calibration. This is **real photon data** from the telescope, not a simulation.",
    "",
    "The download uses `astroquery.mast` to query the Mikulski Archive for Space Telescopes. Files are cached locally so subsequent runs skip the download.",
])

# Cell 8 — Download code (UNCHANGED)
code("8", [
    "from src.data.downloader import search_and_download, list_cached_fits",
    "",
    "cached = list_cached_fits(config.data.data_dir)",
    "if cached:",
    "    print(f\"\\u2705 Found {len(cached)} cached FITS file(s) \\u2014 skipping download.\")",
    "    fits_paths = cached",
    "else:",
    "    print(\"\\u2b07\\ufe0f  Downloading real HST data from MAST archive \\u2026\")",
    "    fits_paths = search_and_download(config.data)",
    "    print(f\"\\u2705 Downloaded {len(fits_paths)} file(s).\")",
    "",
    "print(\"Files:\", [p.name for p in fits_paths])",
])

# Cell 9 — Load header (UNCHANGED)
md("9", [
    "## 6 \u2014 Load FITS Image and Extract the Stellar PSF",
    "",
    "The pipeline:",
    "1. Opens the calibrated FITS file with `astropy.io.fits`",
    "2. Locates the brightest point source (star) in the image",
    "3. Extracts a square cutout centred on the star",
    "4. Subtracts the sky background (low-percentile estimate)",
    "5. Normalises so the PSF sums to unity",
    "",
    "The result is a Pydantic-validated `PSFData` model that rejects non-square, non-2-D, or otherwise invalid inputs at construction time.",
])

# Cell 10 (UNCHANGED)
code("10", [
    "from src.data.loader import load_psf_from_fits, prepare_psf_for_retrieval",
    "",
    "psf_data = load_psf_from_fits(fits_paths[0], config.data, config.pupil)",
    "print(f\"PSF shape:       {psf_data.image.shape}\")",
    "print(f\"Filter:          {psf_data.filter_name}\")",
    "print(f\"Telescope:       {psf_data.telescope}\")",
    "print(f\"Pixel scale:     {psf_data.pixel_scale_arcsec}\\u2033/px\")",
    "print(f\"Observation ID:  {psf_data.obs_id}\")",
])

# Cell 11 (UNCHANGED)
code("11", [
    "psf_image = prepare_psf_for_retrieval(psf_data, config.pupil.grid_size)",
    "print(f\"Working grid size: {psf_image.shape}\")",
    "",
    "psf_data_resized = PSFData(",
    "    image=psf_image,",
    "    pixel_scale_arcsec=psf_data.pixel_scale_arcsec,",
    "    wavelength_m=psf_data.wavelength_m,",
    "    filter_name=psf_data.filter_name,",
    "    telescope=psf_data.telescope,",
    "    obs_id=psf_data.obs_id,",
    ")",
])

# Cell 12 — Pupil header (UNCHANGED)
md("12", [
    "## 7 \u2014 Telescope Pupil Model",
    "",
    "HST has a 2.4 m primary mirror with a 0.792 m secondary-mirror central obstruction and four spider vanes supporting it. This geometry defines the pupil amplitude mask $A(\\mathbf{x})$ \u2014 the known constraint for phase retrieval.",
    "",
    "The pupil is modelled analytically as an annular aperture with radial vane obscurations. We also show the observed PSF in log scale to reveal the diffraction features (Airy rings, spider-vane diffraction spikes).",
])

# Cell 13 — Pupil code (UNCHANGED)
code("13", [
    "from src.optics.pupils import build_pupil",
    "from src.visualization.plots import plot_pupil, plot_observed_psf, set_style",
    "",
    "set_style()",
    "",
    "pupil = build_pupil(config.pupil)",
    "print(f\"Pupil grid:    {pupil.grid_size}\\u00d7{pupil.grid_size}\")",
    "print(f\"Open fraction: {pupil.amplitude.mean():.2%}\")",
    "",
    "fig, axes = plt.subplots(1, 2, figsize=(11, 5))",
    "plot_pupil(pupil, ax=axes[0])",
    "plot_observed_psf(psf_data_resized, log_scale=True, ax=axes[1])",
    "fig.tight_layout()",
    "fig",
])

# Cell 14 — RAAR header (UNCHANGED)
md("14", [
    "## 8 \u2014 Run Phase Retrieval: RAAR (Best for Real Data)",
    "",
    "We start with the **Relaxed Averaged Alternating Reflections** algorithm (Luke 2005) \u2014 a modern convex-relaxation method that outperforms classical algorithms on real noisy telescope data. The relaxation parameter $\\beta = 0.9$ balances exploration and convergence.",
    "",
    "The algorithm factory (`AlgorithmRegistry`) maps the `AlgorithmName` enum to the concrete implementation, making it trivial to swap algorithms.",
])

# Cell 15 (UNCHANGED)
code("15", [
    "from src.algorithms.registry import AlgorithmRegistry",
    "",
    "raar_config = AlgorithmConfig(name=AlgorithmName.RAAR, max_iterations=500, beta=0.9, random_seed=42)",
    "retriever = AlgorithmRegistry.create(raar_config, pupil)",
    "print(f\"Algorithm: {raar_config.name.value}\")",
    "print(f\"Available: {AlgorithmRegistry.available()}\")",
])

# Cell 16 (UNCHANGED)
code("16", [
    "result_raar = retriever.run(psf_data_resized)",
    "",
    "print(f\"\\u2705 RAAR complete in {result_raar.elapsed_seconds:.2f}s\")",
    "print(f\"   Iterations: {result_raar.n_iterations}\")",
    "print(f\"   Converged:  {result_raar.converged}\")",
    "print(f\"   Strehl:     {result_raar.strehl_ratio:.4f}\")",
    "print(f\"   RMS phase:  {result_raar.rms_phase_rad:.4f} rad\")",
])

# Cell 17 — Results header (UNCHANGED)
md("17", [
    "## 9 \u2014 Results: Recovered Wavefront and Reconstructed PSF",
    "",
    "The key outputs of phase retrieval:",
    "- **Recovered wavefront phase** $\\varphi(\\mathbf{x})$ over the pupil (radians)",
    "- **Reconstructed PSF** from forward-modelling the recovered wavefront",
    "- **Convergence curve** showing the focal-plane cost vs. iteration",
])

# Cell 18 (UNCHANGED)
code("18", [
    "from src.visualization.plots import (",
    "    plot_recovered_phase, plot_reconstructed_psf, plot_convergence,",
    "    plot_psf_comparison, plot_psf_residual, plot_radial_profile,",
    "    plot_psf_cross_sections, plot_wavefront_3d, plot_encircled_energy,",
    "    save_figure,",
    ")",
    "",
    "support = pupil.amplitude > 0",
    "",
    "fig, axes = plt.subplots(1, 3, figsize=(16, 5))",
    "plot_recovered_phase(result_raar, support, ax=axes[0])",
    "plot_observed_psf(psf_data_resized, log_scale=True, ax=axes[1])",
    "axes[1].set_title(\"Observed PSF (data)\")",
    "plot_reconstructed_psf(result_raar, log_scale=True, ax=axes[2])",
    "fig.tight_layout()",
    "fig",
])

# Cell 19 (UNCHANGED)
code("19", ["fig = plot_convergence(result_raar)", "fig"])

# Cell 20-29 (UNCHANGED — PSF comparison through encircled energy)
md("20", [
    "## 10 \u2014 PSF Comparison: Observed vs Reconstructed vs Residual",
    "",
    "The observed and reconstructed PSFs may look nearly identical \u2014 that means the algorithm is working. To see **where** they differ, we show them side-by-side with a residual map (diverging colourmap, red/blue = over/under-estimated) and a log-scale absolute residual that reveals faint structure differences in the wings.",
])
code("21", [
    "fig = plot_psf_comparison(psf_data_resized, result_raar, log_scale=True)",
    "save_figure(fig, config.output_dir / \"psf_comparison_raar.png\")",
    "fig",
])
md("22", [
    "## 11 \u2014 Radial PSF Profile",
    "",
    "An azimuthally-averaged radial profile (line plot on a log scale) comparing the observed PSF, the reconstruction, and the theoretical diffraction-limited PSF. Differences in the **wings** (beyond the Airy core) are far more visible in this 1-D representation than in any 2-D image.",
])
code("23", [
    "fig = plot_radial_profile(psf_data_resized, result_raar, pupil)",
    "save_figure(fig, config.output_dir / \"radial_profile_raar.png\")",
    "fig",
])
md("24", [
    "## 12 \u2014 PSF Cross-Sections",
    "",
    "Horizontal and vertical cuts through the PSF peak. These **line plots** give a pixel-by-pixel comparison along two orthogonal axes \u2014 you can immediately see where the reconstruction deviates from the data.",
])
code("25", [
    "fig = plot_psf_cross_sections(psf_data_resized, result_raar)",
    "save_figure(fig, config.output_dir / \"cross_sections_raar.png\")",
    "fig",
])
md("26", [
    "## 13 \u2014 3-D Wavefront Surface",
    "",
    "A three-dimensional surface rendering of the recovered wavefront phase $\\varphi(\\mathbf{x})$ across the pupil. Peaks and valleys correspond to optical path differences (in radians). This view makes it intuitive to see which parts of the mirror are \"ahead\" or \"behind\" the ideal wavefront.",
])
code("27", [
    "fig = plot_wavefront_3d(result_raar, support)",
    "save_figure(fig, config.output_dir / \"wavefront_3d_raar.png\")",
    "fig",
])
md("28", [
    "## 14 \u2014 Encircled Energy",
    "",
    "The encircled energy curve shows what fraction of the total PSF flux falls within a circular aperture of a given radius. Comparing observed, reconstructed, and diffraction-limited curves reveals how much energy is scattered into the wings by aberrations \u2014 this is a standard optical-engineering metric used for specifying telescope performance.",
])
code("29", [
    "fig = plot_encircled_energy(psf_data_resized, result_raar, pupil)",
    "save_figure(fig, config.output_dir / \"encircled_energy_raar.png\")",
    "fig",
])

# Cell 30-34 — Zernike (UNCHANGED)
md("30", [
    "## 15 \u2014 Zernike Decomposition of the Recovered Wavefront",
    "",
    "We project the recovered phase onto the Zernike basis to identify which optical aberrations dominate. The bar chart below shows each coefficient $a_j$ in radians. Large defocus or spherical terms indicate focus errors; large coma or astigmatism terms indicate misalignment.",
])
code("31", [
    "from src.metrics.quality import zernike_decomposition",
    "from src.visualization.plots import plot_zernike_bar, plot_zernike_polar",
    "from src.optics.zernike import ZERNIKE_NAMES",
    "",
    "zernike_coeffs = zernike_decomposition(result_raar.recovered_phase, support, n_terms=15)",
    "",
    "print(\"Zernike coefficients (rad):\")",
    "for j, coeff in zernike_coeffs.items():",
    "    name = ZERNIKE_NAMES.get(j, f\"Z{j}\")",
    "    print(f\"  j={j:2d}  {name:30s}  {coeff:+.5f}\")",
])
code("32", ["fig = plot_zernike_bar(zernike_coeffs)", "fig"])
md("33", [
    "### Zernike Polar Map",
    "",
    "A polar lollipop chart: each Zernike aberration is placed at an angle based on its azimuthal order, with distance from centre proportional to the coefficient magnitude. Blue = positive, red = negative. This gives an intuitive visual \"fingerprint\" of which directions the wavefront is aberrated.",
])
code("34", [
    "fig = plot_zernike_polar(zernike_coeffs)",
    "save_figure(fig, config.output_dir / \"zernike_polar_raar.png\")",
    "fig",
])

# ═══════════════════════════════════════════════════════════════════════
# NEW Cell 34a — MTF, SSIM, Phase Structure Function
# ═══════════════════════════════════════════════════════════════════════
md("34a", [
    "## 15b \u2014 Advanced Metrics: MTF, SSIM, Phase Structure Function",
    "",
    "Three new state-of-the-art quality metrics:",
    "",
    "- **MTF (Modulation Transfer Function)**: the spatial-frequency response of the optical system \u2014 how well the telescope preserves contrast at different spatial frequencies.",
    "- **SSIM (Structural Similarity Index)**: a perceptual metric comparing observed and reconstructed PSFs, sensitive to luminance, contrast, and structure.",
    "- **Phase Structure Function** $D_\\varphi(r) = \\langle |\\varphi(x) - \\varphi(x+r)|^2 \\rangle$: a standard wavefront/turbulence diagnostic that reveals spatial correlations in the recovered phase.",
])
code("34b", [
    "from src.metrics.quality import compute_mtf, compute_ssim, compute_phase_structure_function",
    "from src.optics.propagator import forward_model",
    "",
    "# --- MTF ---",
    "freqs_obs, mtf_obs = compute_mtf(psf_data_resized.image)",
    "freqs_rec, mtf_rec = compute_mtf(result_raar.reconstructed_psf)",
    "perfect_psf = forward_model(pupil.amplitude, np.zeros_like(pupil.amplitude))",
    "freqs_perf, mtf_perf = compute_mtf(perfect_psf)",
    "",
    "fig, ax = plt.subplots(figsize=(9, 5))",
    "ax.plot(freqs_obs, mtf_obs, label=\"Observed\", linewidth=1.8)",
    "ax.plot(freqs_rec, mtf_rec, label=\"Reconstructed\", linewidth=1.8, linestyle=\"--\")",
    "ax.plot(freqs_perf, mtf_perf, label=\"Diffraction-limited\", linewidth=1.5, linestyle=\":\")",
    "ax.set_xlabel(\"Spatial frequency (cycles/pixel)\")",
    "ax.set_ylabel(\"MTF\")",
    "ax.set_title(\"Modulation Transfer Function\")",
    "ax.legend(frameon=True)",
    "ax.grid(True, alpha=0.3)",
    "ax.set_xlim(0, 0.5)",
    "ax.set_ylim(0, 1.05)",
    "fig.tight_layout()",
    "save_figure(fig, config.output_dir / \"mtf_comparison.png\")",
    "fig",
])
code("34c", [
    "# --- SSIM ---",
    "ssim_val = compute_ssim(psf_data_resized.image, result_raar.reconstructed_psf)",
    "print(f\"SSIM (observed vs reconstructed): {ssim_val:.4f}  (1.0 = perfect match)\")",
])
code("34d", [
    "# --- Phase Structure Function ---",
    "seps, sf = compute_phase_structure_function(result_raar.recovered_phase, support, max_sep=50)",
    "",
    "fig, ax = plt.subplots(figsize=(9, 5))",
    "ax.plot(seps, sf, linewidth=2, color=\"#4575b4\")",
    "ax.set_xlabel(\"Separation r (pixels)\")",
    "ax.set_ylabel(r\"$D_\\varphi(r)$ (rad$^2$)\")",
    "ax.set_title(\"Phase Structure Function\")",
    "ax.grid(True, alpha=0.3)",
    "fig.tight_layout()",
    "save_figure(fig, config.output_dir / \"phase_structure_function.png\")",
    "fig",
])

# Cell 35-36 — Summary figure (UNCHANGED)
md("35", [
    "## 16 \u2014 Full Summary Figure",
    "",
    "A single 2\u00d73 composite showing the entire pipeline: pupil model, observed PSF, residual map, recovered wavefront, convergence, and Zernike bar chart.",
])
code("36", [
    "from src.visualization.plots import plot_summary, save_figure",
    "",
    "fig = plot_summary(psf_data_resized, pupil, result_raar, zernike_coeffs)",
    "save_figure(fig, config.output_dir / \"summary_raar.png\")",
    "print(f\"\\U0001f4c1 Saved to {config.output_dir / 'summary_raar.png'}\")",
    "fig",
])

# ═══════════════════════════════════════════════════════════════════════
# NEW Cell 36a — SOTA Enhancements Demo
# ═══════════════════════════════════════════════════════════════════════
md("36a", [
    "## 16b \u2014 State-of-the-Art Enhancements Demo",
    "",
    "Demonstrating the advanced features built into the base class:",
    "",
    "1. **Wirtinger Flow** with spectral initialization",
    "2. **Cosine \u03b2 annealing** on RAAR",
    "3. **Momentum acceleration** on HIO",
    "4. **Multi-start optimization** to escape local minima",
])
code("36b", [
    "# --- Wirtinger Flow with spectral initialization ---",
    "wf_cfg = AlgorithmConfig(",
    "    name=AlgorithmName.WIRTINGER_FLOW,",
    "    max_iterations=300,",
    "    random_seed=42,",
    "    wf_spectral_init=True,",
    "    wf_step_size=0.5,",
    ")",
    "result_wf = AlgorithmRegistry.create(wf_cfg, pupil).run(psf_data_resized)",
    "print(f\"\\u2705 Wirtinger Flow: Strehl={result_wf.strehl_ratio:.4f}, RMS={result_wf.rms_phase_rad:.4f} rad, {result_wf.elapsed_seconds:.2f}s\")",
])
code("36c", [
    "# --- RAAR with cosine \u03b2 annealing ---",
    "raar_cosine_cfg = AlgorithmConfig(",
    "    name=AlgorithmName.RAAR,",
    "    max_iterations=500,",
    "    beta=0.95,",
    "    beta_min=0.5,",
    "    beta_schedule=BetaSchedule.COSINE,",
    "    random_seed=42,",
    ")",
    "result_raar_cosine = AlgorithmRegistry.create(raar_cosine_cfg, pupil).run(psf_data_resized)",
    "print(f\"\\u2705 RAAR (cosine \\u03b2): Strehl={result_raar_cosine.strehl_ratio:.4f}, RMS={result_raar_cosine.rms_phase_rad:.4f} rad\")",
])
code("36d", [
    "# --- HIO with Nesterov momentum ---",
    "hio_mom_cfg = AlgorithmConfig(",
    "    name=AlgorithmName.HYBRID_INPUT_OUTPUT,",
    "    max_iterations=300,",
    "    beta=0.9,",
    "    momentum=0.3,",
    "    random_seed=42,",
    ")",
    "result_hio_mom = AlgorithmRegistry.create(hio_mom_cfg, pupil).run(psf_data_resized)",
    "print(f\"\\u2705 HIO + momentum: Strehl={result_hio_mom.strehl_ratio:.4f}, RMS={result_hio_mom.rms_phase_rad:.4f} rad\")",
])
code("36e", [
    "# --- Multi-start optimization ---",
    "from src.algorithms.multi_start import multi_start_run",
    "",
    "ms_cfg = AlgorithmConfig(",
    "    name=AlgorithmName.HYBRID_INPUT_OUTPUT,",
    "    max_iterations=300,",
    "    beta=0.9,",
    "    random_seed=42,",
    "    n_starts=5,",
    ")",
    "result_ms = multi_start_run(ms_cfg, pupil, psf_data_resized)",
    "print(f\"\\u2705 Multi-start HIO (5 starts): Strehl={result_ms.strehl_ratio:.4f}, RMS={result_ms.rms_phase_rad:.4f} rad\")",
])

# Cell 37-46 — Multi-observation (UNCHANGED)
md("37", [
    "## 17 \u2014 Multi-Observation Analysis: Different Filters & Detectors",
    "",
    "So far we've analysed a single observation. Now we run phase retrieval on **all real HST images** available \u2014 different filters (F438W blue, F606W visible, F814W near-IR) and different detectors (WFC3/UVIS, ACS/WFC). Every file is a **real calibrated exposure** downloaded from the MAST archive \u2014 no synthetic data.",
    "",
    "The PSF shape is wavelength-dependent: shorter wavelengths produce sharper diffraction-limited cores but are more sensitive to high-spatial-frequency aberrations; longer wavelengths produce broader cores but are more tolerant. Comparing phase retrieval across wavelengths reveals whether the recovered aberrations are consistent (they should be \u2014 the optics don't change with wavelength).",
])
code("38", [
    "from src.data.downloader import download_all_presets, list_cached_fits, available_presets, FILTER_WAVELENGTH_M",
    "from src.visualization.plots import plot_multi_observation_grid, plot_multi_observation_radial",
    "",
    "# Show what presets are available",
    "print(\"Available observation presets:\")",
    "for key, desc in available_presets().items():",
    "    print(f\"  {key:30s}  {desc}\")",
])
code("39", [
    "# Download any missing observations (skips already-cached files)",
    "all_fits = list_cached_fits(config.data.data_dir)",
    "print(f\"\\n\U0001f4c2 Currently cached: {len(all_fits)} FITS file(s)\")",
    "",
    "if len(all_fits) < 2:",
    "    print(\"\u2b07\ufe0f  Downloading additional real HST observations from MAST \u2026\")",
    "    download_all_presets(",
    "        config.data.data_dir,",
    "        keys=[\"hst-wfc3-uvis-f814w\", \"hst-wfc3-uvis-f438w\", \"hst-acs-wfc-f606w\"],",
    "    )",
    "    all_fits = list_cached_fits(config.data.data_dir)",
    "    print(f\"\u2705 Now have {len(all_fits)} FITS file(s)\")",
    "else:",
    "    print(\"\u2705 Already have multiple observations \u2014 skipping download.\")",
    "",
    "for fp in all_fits:",
    "    print(f\"  \u2022 {fp.name}\")",
])
code("40", [
    "# Process each real observation: load \u2192 extract PSF \u2192 run RAAR",
    "from src.data.loader import load_psf_from_fits, prepare_psf_for_retrieval",
    "",
    "observations = []",
    "for fp in all_fits:",
    "    try:",
    "        psf_i = load_psf_from_fits(fp, config.data, config.pupil)",
    "        img_i = prepare_psf_for_retrieval(psf_i, config.pupil.grid_size)",
    "        psf_i_resized = PSFData(",
    "            image=img_i,",
    "            pixel_scale_arcsec=psf_i.pixel_scale_arcsec,",
    "            wavelength_m=psf_i.wavelength_m,",
    "            filter_name=psf_i.filter_name,",
    "            telescope=psf_i.telescope,",
    "            obs_id=psf_i.obs_id,",
    "        )",
    "",
    "        alg_cfg = AlgorithmConfig(name=AlgorithmName.RAAR, max_iterations=500, beta=0.9, random_seed=42)",
    "        res_i = AlgorithmRegistry.create(alg_cfg, pupil).run(psf_i_resized)",
    "",
    "        observations.append({",
    "            \"label\": f\"{psf_i.obs_id}\\n{psf_i.filter_name}\",",
    "            \"psf\": psf_i_resized,",
    "            \"result\": res_i,",
    "            \"support\": support,",
    "        })",
    "        print(",
    "            f\"  \u2705 {fp.name:30s}  filter={psf_i.filter_name:6s}  \"",
    "            f\"Strehl={res_i.strehl_ratio:.4f}  RMS={res_i.rms_phase_rad:.4f} rad\"",
    "        )",
    "    except Exception as exc:",
    "        print(f\"  \u26a0\ufe0f  {fp.name}: {exc}\")",
    "",
    "print(f\"\\nProcessed {len(observations)} real observations.\")",
])
md("41", [
    "### Multi-Observation Grid",
    "",
    "4 rows \u00d7 N columns: observed PSF (log), recovered wavefront phase, residual (obs\u2212recon), and convergence curve for each real observation. You can immediately compare how the PSF changes with wavelength and how the recovered wavefront is consistent across filters.",
])
code("42", [
    "fig = plot_multi_observation_grid(observations)",
    "save_figure(fig, config.output_dir / \"multi_observation_grid.png\")",
    "fig",
])
md("43", [
    "### Radial Profiles Across Observations",
    "",
    "Overlaid azimuthally-averaged radial profiles for all observations on one plot. Shorter wavelengths (F438W) produce sharper cores; longer wavelengths (F814W) have broader profiles. The reconstruction quality can also be compared.",
])
code("44", [
    "fig = plot_multi_observation_radial(observations)",
    "save_figure(fig, config.output_dir / \"multi_observation_radial.png\")",
    "fig",
])
md("45", [
    "### Per-Observation PSF Comparison",
    "",
    "Detailed 4-panel comparison (observed, reconstructed, residual, log-residual) for each observation.",
])
code("46", [
    "for obs in observations:",
    "    lbl = obs[\"label\"].replace(\"\\n\", \"_\")",
    "    fig = plot_psf_comparison(obs[\"psf\"], obs[\"result\"], log_scale=True)",
    "    save_figure(fig, config.output_dir / f\"psf_comparison_{lbl}.png\")",
    "    plt.show()",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 47 — Algorithm Comparison (UPDATED: 7 default iterative algorithms)
# ═══════════════════════════════════════════════════════════════════════
md("47", [
    "## 18 \u2014 Algorithm Comparison",
    "",
    "We now run the **seven default single-image iterative algorithms** (classic + state-of-the-art) on the same observed PSF and compare their convergence speed, final Strehl ratio, and RMS wavefront error. Each algorithm uses identical initialisation (same random seed) for a fair comparison.",
    "",
    "On real noisy data, RAAR and the SOTA algorithms (WF, DR, ADMM) typically outperform the classic ER/GS \u2014 their advanced update rules avoid the local minima and stagnation that trap simpler projection methods.",
    "",
    "The optional `pinn` solver is available through the package and CLI when PyTorch is installed, but it is excluded from the default notebook comparison because it is slower and has an optional dependency.",
])
code("48", [
    "from src.visualization.plots import plot_algorithm_comparison, plot_algorithm_dashboard, plot_strehl_rms_bar",
    "",
    "algorithms_to_compare = [",
    "    AlgorithmName.ERROR_REDUCTION,",
    "    AlgorithmName.GERCHBERG_SAXTON,",
    "    AlgorithmName.HYBRID_INPUT_OUTPUT,",
    "    AlgorithmName.RAAR,",
    "    AlgorithmName.WIRTINGER_FLOW,",
    "    AlgorithmName.DOUGLAS_RACHFORD,",
    "    AlgorithmName.ADMM,",
    "]",
    "",
    "results: dict[str, PhaseRetrievalResult] = {}",
    "",
    "for alg_name in algorithms_to_compare:",
    "    alg_config = AlgorithmConfig(name=alg_name, max_iterations=500, beta=0.9, random_seed=42)",
    "    retriever = AlgorithmRegistry.create(alg_config, pupil)",
    "    res = retriever.run(psf_data_resized)",
    "    results[alg_name.value.upper()] = res",
    "    print(f\"  {alg_name.value.upper():5s} \\u2014 {res.n_iterations:4d} iter, Strehl={res.strehl_ratio:.4f}, RMS={res.rms_phase_rad:.4f} rad, {res.elapsed_seconds:.2f}s\")",
])
code("49", [
    "fig = plot_algorithm_comparison(results, support)",
    "save_figure(fig, config.output_dir / \"algorithm_comparison.png\")",
    "fig",
])

# Cell 50 — Dashboard (UPDATED description)
md("50", [
    "## 19 \u2014 Algorithm Dashboard",
    "",
    "A comprehensive N-row \u00d7 4-column dashboard showing, for every algorithm: the recovered wavefront phase (heatmap), reconstructed PSF (log heatmap), residual map (diverging heatmap), and azimuthally-averaged radial profile (line plot). This single figure makes it easy to compare **how** each algorithm reconstructs the wavefront differently.",
])
code("51", [
    "fig = plot_algorithm_dashboard(psf_data_resized, results, support, pupil)",
    "save_figure(fig, config.output_dir / \"algorithm_dashboard.png\")",
    "fig",
])

# Cell 52 (UNCHANGED)
md("52", [
    "## 20 \u2014 Algorithm Performance: Strehl vs RMS",
    "",
    "A grouped bar chart with dual y-axes: Strehl ratio (blue, left axis) and RMS wavefront phase error in radians (red, right axis). Value labels on each bar make quantitative comparison immediate.",
])
code("53", [
    "fig = plot_strehl_rms_bar(results)",
    "save_figure(fig, config.output_dir / \"strehl_rms_comparison.png\")",
    "fig",
])

# Cell 54 — Convergence (UPDATED description)
md("54", [
    "## 21 \u2014 Convergence Comparison",
    "",
    "Plotting the focal-plane cost function vs. iteration for all seven algorithms on a single log-scale axis. HIO, RAAR, and the SOTA algorithms (WF, DR, ADMM) typically converge faster than ER/GS due to their advanced update rules.",
])
code("55", [
    "fig, ax = plt.subplots(figsize=(10, 5))",
    "for name, res in results.items():",
    "    ax.semilogy(res.cost_history, label=f\"{name} (Strehl={res.strehl_ratio:.3f})\", linewidth=1.5)",
    "ax.set_xlabel(\"Iteration\")",
    "ax.set_ylabel(\"Cost (focal-plane error)\")",
    "ax.set_title(\"Convergence Comparison \\u2014 All 7 Algorithms\")",
    "ax.legend(frameon=True, fontsize=8, ncol=2)",
    "ax.grid(True, alpha=0.3)",
    "fig.tight_layout()",
    "save_figure(fig, config.output_dir / \"convergence_comparison.png\")",
    "fig",
])

# Cell 56 (UNCHANGED header)
md("56", [
    "## 22 \u2014 Results Summary Table",
    "",
    "Quantitative comparison across all algorithms. The table shows iteration count, convergence status, Strehl ratio, RMS phase error, and wall-clock time.",
])
code("57", [
    "from rich.console import Console",
    "from rich.table import Table",
    "",
    "console = Console()",
    "table = Table(title=\"Phase Retrieval Results \\u2014 Real HST Data (All 7 Algorithms)\", show_lines=True)",
    "table.add_column(\"Algorithm\", style=\"bold cyan\")",
    "table.add_column(\"Iterations\", justify=\"right\")",
    "table.add_column(\"Converged\", justify=\"center\")",
    "table.add_column(\"Strehl Ratio\", justify=\"right\")",
    "table.add_column(\"RMS Phase (rad)\", justify=\"right\")",
    "table.add_column(\"Time (s)\", justify=\"right\")",
    "",
    "for name, res in results.items():",
    "    table.add_row(",
    "        name,",
    "        str(res.n_iterations),",
    "        \"\\u2705\" if res.converged else \"\\u274c\",",
    "        f\"{res.strehl_ratio:.4f}\",",
    "        f\"{res.rms_phase_rad:.4f}\",",
    "        f\"{res.elapsed_seconds:.2f}\",",
    "    )",
    "console.print(table)",
])

# Cell 58 (UNCHANGED)
md("58", [
    "## 23 \u2014 Export Results",
    "",
    "Every result is a Pydantic model. We serialise the scalar metrics to JSON for reproducibility and downstream analysis.",
])
code("59", [
    "for name, res in results.items():",
    "    summary = {",
    "        \"algorithm\": res.algorithm.value,",
    "        \"n_iterations\": res.n_iterations,",
    "        \"converged\": res.converged,",
    "        \"strehl_ratio\": res.strehl_ratio,",
    "        \"rms_phase_rad\": res.rms_phase_rad,",
    "        \"elapsed_seconds\": res.elapsed_seconds,",
    "        \"timestamp\": res.timestamp.isoformat(),",
    "    }",
    "    out_path = config.output_dir / f\"result_{name.lower()}.json\"",
    "    out_path.parent.mkdir(parents=True, exist_ok=True)",
    "    out_path.write_text(json.dumps(summary, indent=2))",
    "    print(f\"\\U0001f4c1 {out_path}\")",
    "",
    "print(f\"\\n\\U0001f389 Pipeline complete \\u2014 all outputs saved to {config.output_dir}\")",
])

# ═══════════════════════════════════════════════════════════════════════
# Cell 60 — Takeaways (UPDATED)
# ═══════════════════════════════════════════════════════════════════════
md("60", [
    "## 24 \u2014 Interpretation and Takeaways",
    "",
    "What this notebook demonstrates:",
    "",
    "- **Real data, not synthetic**: the PSF was recorded by HST's WFC3/UVIS detector from actual stellar photons collected in orbit.",
    "- **7 iterative algorithms compared by default**: from the classic ER/GS (1972\u20131982) through RAAR (2005) to state-of-the-art Wirtinger Flow (2015), Douglas-Rachford, and ADMM (2018). The optional `pinn` solver is available separately when PyTorch is installed.",
    "- **State-of-the-art enhancements**: Nesterov momentum acceleration, adaptive \u03b2 cosine annealing, TV regularization, Poisson noise model, and multi-start optimization are all built into the base class and work with every algorithm.",
    "- **Comprehensive metrics**: Strehl ratio, RMS wavefront error, Zernike decomposition, MTF, SSIM, and Phase Structure Function provide a complete picture of optical quality.",
    "- **Phase retrieval works**: all algorithms successfully recover a wavefront that, when forward-modelled, reproduces the observed PSF structure.",
    "- **Algorithm trade-offs**: HIO/RAAR/DR converge faster than ER/GS. Wirtinger Flow with spectral init provides a strong starting point. ADMM handles regularization naturally.",
    "- **Zernike decomposition** reveals the dominant aberrations \u2014 this is directly actionable for telescope alignment and optical design.",
    "- **Pydantic validation** catches misconfiguration early \u2014 wrong image shapes, out-of-range parameters, and type errors are all caught at model construction.",
    "- **Modular design**: swapping telescopes (HST \u2192 JWST), filters, or algorithms requires changing only the configuration \u2014 not the pipeline code.",
    "",
    "This is the same fundamental technique used operationally by NASA/STScI for HST focus monitoring and JWST mirror alignment \u2014 scaled to a clean, reproducible, state-of-the-art implementation.",
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
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = Path(__file__).resolve().parent / "notebooks" / "phase_retrieval_hst.ipynb"
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"\u2705 Wrote {out_path}  ({len(cells)} cells)")
