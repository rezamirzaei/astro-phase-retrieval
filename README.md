# Phase Retrieval for Astronomical Wavefront Sensing

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

Production-ready implementation of iterative phase retrieval algorithms applied to
**real Hubble Space Telescope (HST)** point-spread function (PSF) observations.

## Real-World Problem

Every optical telescope introduces wavefront aberrations (defocus, astigmatism,
coma, spherical aberration, etc.). When a telescope images a point source (star),
the resulting PSF encodes these aberrations — but the detector only records
**intensity** (the squared modulus), losing all phase information.

**Phase retrieval** recovers the lost wavefront phase from intensity-only PSF
measurements, given knowledge of the telescope pupil geometry. This is exactly how
NASA diagnosed HST's famous spherical aberration in 1990 and how JWST's mirrors
are aligned today.

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

## Quick Start

```bash
# Install
pip install -e .

# Run the full notebook in PyCharm (open main.py → Run as Scientific)
python main.py
```

## Project Structure

```
src/
├── models/          Pydantic-validated data models
├── algorithms/      Phase retrieval algorithm implementations
├── data/            MAST downloader & FITS loader
├── optics/          Pupil models, Zernike basis, Fourier propagation
├── visualization/   Publication-quality plotting
└── metrics/         Strehl ratio, RMS wavefront error, convergence
```

## License

This project is licensed under the [MIT License](LICENSE).



