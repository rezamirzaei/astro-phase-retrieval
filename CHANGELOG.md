# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
