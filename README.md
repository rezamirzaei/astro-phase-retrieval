# Phase Retrieval for Astronomical Wavefront Sensing

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/<owner>/phase-retrieval/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/phase-retrieval/actions/workflows/ci.yml)
[![Typed](https://img.shields.io/badge/typing-PEP561-brightgreen.svg)](https://peps.python.org/pep-0561/)

Production-ready implementation of **classic and state-of-the-art** phase retrieval
algorithms applied to **real Hubble Space Telescope (HST)** point-spread function
(PSF) observations.

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

### Classic Algorithms

| Algorithm | Key | Description |
|-----------|-----|-------------|
| Error Reduction (ER) | `er` | Fienup's basic projection algorithm (1982) |
| Gerchberg–Saxton (GS) | `gs` | Classic two-plane amplitude constraint |
| Hybrid Input-Output (HIO) | `hio` | Fienup's workhorse with feedback parameter β |
| Phase Diversity (PD) | `phase_diversity` | Joint estimation from focused + defocused pair |

### State-of-the-Art Algorithms

| Algorithm | Key | Description |
|-----------|-----|-------------|
| Relaxed Averaged Alternating Reflections (RAAR) | `raar` | Luke's convex-relaxation method (2005) |
| **Wirtinger Flow (WF)** | `wf` | Gradient descent with spectral initialization (Candès et al. 2015) |
| **Douglas-Rachford (DR)** | `dr` | Proximal splitting with provable convergence (Bauschke et al. 2002) |
| **ADMM** | `admm` | Alternating Direction Method of Multipliers (Chang & Marchesini 2018) |

### Advanced Enhancements

All algorithms benefit from these state-of-the-art enhancements built into the base class:

| Feature | Flag | Description |
|---------|------|-------------|
| **Nesterov Momentum** | `--momentum 0.5` | Heavy-ball acceleration for faster convergence |
| **Adaptive β Scheduling** | `--beta-schedule cosine` | Cosine / linear annealing of the feedback parameter |
| **TV Regularization** | `--tv-weight 0.01` | Chambolle proximal operator for noise-robust phase |
| **Poisson Noise Model** | `--noise-model poisson` | Maximum-likelihood projection for photon-limited data |
| **Multi-Start** | `--n-starts 5` | Multiple random restarts, returns best result |

## Quality Metrics

| Metric | Description |
|--------|-------------|
| **Strehl Ratio** | Peak intensity vs. diffraction limit |
| **RMS Wavefront Error** | Root-mean-square phase error (radians / waves) |
| **Zernike Decomposition** | Noll-ordered polynomial coefficients |
| **MTF** | Modulation Transfer Function (spatial frequency response) |
| **SSIM** | Structural Similarity Index between observed & reconstructed PSF |
| **Phase Structure Function** | D_φ(r) — standard wavefront/turbulence diagnostic |

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

# Run state-of-the-art Wirtinger Flow with spectral init
phase-retrieval run --algorithm wf --iterations 300

# Use advanced features: cosine β schedule + momentum + TV regularization
phase-retrieval run --algorithm raar --iterations 1000 \
    --beta-schedule cosine --momentum 0.3 --tv-weight 0.01

# Multi-start optimization (5 random restarts, keeps best)
phase-retrieval run --algorithm hio --n-starts 5

# Compare all 7 algorithms on the same observation
phase-retrieval compare --iterations 500

# Poisson noise model for low-SNR data
phase-retrieval run --algorithm admm --noise-model poisson

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
│   ├── base.py          Abstract base class (momentum, TV, adaptive β, noise model)
│   ├── error_reduction.py
│   ├── gerchberg_saxton.py
│   ├── hybrid_input_output.py
│   ├── raar.py
│   ├── wirtinger_flow.py    ★ State-of-the-art: gradient + spectral init
│   ├── douglas_rachford.py  ★ Proximal splitting
│   ├── admm.py              ★ ADMM with Fourier/support splitting
│   ├── phase_diversity.py
│   ├── multi_start.py       Multi-start optimization runner
│   └── registry.py          Factory pattern for algorithm selection
├── cli.py           Command-line interface (argparse)
├── data/            MAST downloader & FITS loader
│   ├── downloader.py    Curated observations from the MAST archive
│   └── loader.py        FITS I/O, source detection, cutout extraction
├── metrics/         Strehl, RMS, Zernike, MTF, SSIM, Phase Structure Function
│   └── quality.py
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

## References

- Fienup J.R. (1982) "Phase retrieval algorithms: a comparison" *Applied Optics*
- Gerchberg R.W., Saxton W.O. (1972) "A practical algorithm for the determination of phase" *Optik*
- Luke D.R. (2005) "Relaxed averaged alternating reflections" *Inverse Problems*
- Candès E.J., Li X., Soltanolkotabi M. (2015) "Phase Retrieval via Wirtinger Flow" *IEEE Trans. IT*
- Bauschke H.H., Combettes P.L., Luke D.R. (2002) "Phase retrieval, error reduction algorithm, and Fienup variants" *JOSA A*
- Chang H., Marchesini S. (2018) "ADMM methods for phase retrieval" *arXiv:1804.05306*
- Gonsalves R.A. (1982) "Phase Retrieval and Diversity in Adaptive Optics" *Optical Engineering*
- Chambolle A. (2004) "An algorithm for total variation minimization" *JMIV*

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding conventions,
and the pull-request workflow.

## License

This project is licensed under the [MIT License](LICENSE).
