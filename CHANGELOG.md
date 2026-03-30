# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.0.0] — 2026-03-30

### Added — State-of-the-Art Algorithms

- **Wirtinger Flow (WF)**: Gradient-descent phase retrieval with spectral
  initialization (Candès, Li & Soltanolkotabi 2015). Converges linearly to the
  global optimum. Key `wf`.
- **Douglas-Rachford (DR)**: Proximal splitting algorithm with provably better
  fixed-point convergence than HIO (Bauschke, Combettes & Luke 2002). Key `dr`.
- **ADMM**: Alternating Direction Method of Multipliers with Fourier/support
  splitting and dual variable (Chang & Marchesini 2018). Key `admm`.

### Added — Algorithm Enhancements

- **Nesterov/heavy-ball momentum** acceleration — applies to all algorithms
  transparently via `--momentum` flag.
- **Adaptive β scheduling**: constant, linear ramp-down, or cosine annealing
  via `--beta-schedule` flag.
- **Total-variation (TV) regularization** via Chambolle proximal operator for
  noise-robust phase recovery. `--tv-weight` flag.
- **Poisson noise model**: maximum-likelihood focal-plane projection for
  photon-limited observations. `--noise-model poisson` flag.
- **Multi-start optimization**: run from N random seeds, return the best result.
  `--n-starts` flag.

### Added — Quality Metrics

- **MTF** (`compute_mtf`): radially averaged Modulation Transfer Function.
- **SSIM** (`compute_ssim`): Structural Similarity Index between observed and
  reconstructed PSFs (uses scikit-image).
- **Phase Structure Function** (`compute_phase_structure_function`): standard
  wavefront diagnostic D_φ(r).

### Changed

- All existing algorithms (ER, GS, HIO, RAAR) now use noise-model-aware
  Fourier projection and adaptive β from the enhanced base class.
- CLI `compare` command now runs all 7 algorithms (was 4).
- CLI gained `--beta-schedule`, `--momentum`, `--tv-weight`, `--noise-model`,
  and `--n-starts` flags for run and compare subcommands.

## [1.0.0] — 2026-03-30

### Added

- **Algorithms**: Error Reduction (ER), Gerchberg–Saxton (GS), Hybrid Input-Output
  (HIO), Relaxed Averaged Alternating Reflections (RAAR), and Phase Diversity.
- **Real data pipeline**: automated download of calibrated HST/WFC3 and ACS
  observations from the MAST archive via `astroquery`.
- **Pupil models**: HST (annular + 4 spider vanes), JWST (hexagonal segments +
  3 struts), and generic circular aperture.
- **Zernike decomposition**: Noll-ordered polynomials up to j = 37 for wavefront
  analysis.
- **Quality metrics**: Strehl ratio, RMS wavefront error, convergence tracking.
- **Visualization**: 15+ publication-quality plot types (summary dashboards,
  radial profiles, 3-D wavefront surfaces, encircled energy, polar Zernike maps).
- **CLI**: `phase-retrieval run`, `compare`, and `download` subcommands.
- **Pydantic models**: fully validated configuration and result containers.
- **Test suite**: 40+ pytest tests on synthetic data (no network required).
- **CI**: GitHub Actions workflow with linting (Ruff) and testing (Python 3.11/3.12).
- **PEP 561**: `py.typed` marker for downstream type checking.
