# Phase Retrieval for Astronomical Wavefront Sensing

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/<owner>/phase-retrieval/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/phase-retrieval/actions/workflows/ci.yml)
[![Typed](https://img.shields.io/badge/typing-PEP561-brightgreen.svg)](https://peps.python.org/pep-0561/)

Production-ready implementation of iterative phase retrieval algorithms applied to
**real Hubble Space Telescope (HST)** point-spread function (PSF) observations.

---

## Real-World Problem

Every optical telescope introduces wavefront aberrations (defocus, astigmatism,
coma, spherical aberration, etc.). When a telescope images a point source (star),
the resulting PSF encodes these aberrations — but the detector only records
**intensity** (the squared modulus), losing all phase information.

**Phase retrieval** recovers the lost wavefront phase from intensity-only PSF
measurements, given knowledge of the telescope pupil geometry. This is exactly how
NASA diagnosed HST's famous spherical aberration in 1990 and how JWST's mirrors
are aligned today.

## How It Works

```
    ┌──────────┐      FFT       ┌──────────┐
    │  Pupil   │  ──────────►  │  Focal   │
    │  Plane   │  A·exp(iφ)    │  Plane   │
    │ (phase φ)│  ◄──────────  │  (|E|²)  │
    └──────────┘    IFFT       └──────────┘
         │                          │
     Enforce known             Enforce measured
     pupil support             PSF amplitude
         │                          │
         └──── iterate until ───────┘
               convergence
```

The algorithms alternate between two planes connected by the Fourier transform,
enforcing known constraints in each plane until the phase estimate converges.

## Data

This project downloads **real calibrated HST/WFC3 observations** of standard
calibration stars from the Mikulski Archive for Space Telescopes (MAST). These are
not synthetic — they are actual photons collected by HST.

## Algorithms

| Algorithm | Key | Description |
|-----------|-----|-------------|
| Error Reduction (ER) | `er` | Fienup's basic projection algorithm |
| Gerchberg–Saxton (GS) | `gs` | Classic two-plane amplitude constraint |
| Hybrid Input-Output (HIO) | `hio` | Fienup's workhorse with feedback parameter β |
| Relaxed Averaged Alternating Reflections (RAAR) | `raar` | Luke's modern convex-relaxation method |
| Phase Diversity (PD) | `phase_diversity` | Joint estimation from focused + defocused pair |

## Installation

```bash
# Clone the repository
git clone https://github.com/<owner>/phase-retrieval.git
cd phase-retrieval

# Install in editable mode (recommended for development)
pip install -e ".[dev]"
```

## Quick Start

### As a script (PyCharm / Jupyter-style)

```bash
python main.py
```

### Via the CLI

```bash
# Download real HST data from MAST
phase-retrieval download --preset hst-wfc3-uvis-f606w

# Run a single algorithm
phase-retrieval run --algorithm hio --iterations 500

# Compare all algorithms on the same observation
phase-retrieval compare --iterations 500

# List available observation presets
phase-retrieval download --list
```

### As a Python module

```bash
python -m src run --algorithm raar
```

## Testing

```bash
# Run the full test suite
pytest

# With coverage
pytest --cov=src --cov-report=term-missing
```

All tests use small synthetic data (64×64 grids) — no network or real FITS files
required.

## Project Structure

```
src/
├── algorithms/      Phase retrieval algorithm implementations
│   ├── base.py          Abstract base class with iterative loop
│   ├── error_reduction.py
│   ├── gerchberg_saxton.py
│   ├── hybrid_input_output.py
│   ├── raar.py
│   ├── phase_diversity.py
│   └── registry.py      Factory pattern for algorithm selection
├── cli.py           Command-line interface (argparse)
├── data/            MAST downloader & FITS loader
│   ├── downloader.py    Curated observations from the MAST archive
│   └── loader.py        FITS I/O, source detection, cutout extraction
├── metrics/         Strehl ratio, RMS wavefront error, Zernike decomposition
├── models/          Pydantic-validated data models
│   ├── config.py        Pipeline, algorithm, pupil, data configs
│   └── optics.py        PSFData, PupilModel, PhaseRetrievalResult
├── optics/          Pupil models, Zernike basis, Fourier propagation
│   ├── propagator.py    FFT forward/inverse, defocus injection
│   ├── pupils.py        HST, JWST, generic circular pupil builders
│   └── zernike.py       Noll-ordered Zernike polynomials
└── visualization/   Publication-quality plotting (matplotlib)
tests/               Pytest test suite (synthetic data only)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding conventions,
and the pull-request workflow.

## License

This project is licensed under the [MIT License](LICENSE).
