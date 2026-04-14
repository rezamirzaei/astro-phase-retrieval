# 🔭 Phase Retrieval for Astronomical Wavefront Sensing

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/rezamirzaeifard/phase-retrieval/actions/workflows/ci.yml/badge.svg)](https://github.com/rezamirzaeifard/phase-retrieval/actions/workflows/ci.yml)
[![Typed](https://img.shields.io/badge/typing-PEP561-brightgreen.svg)](https://peps.python.org/pep-0561/)

Well-tested **research and engineering toolkit** for classic and modern phase
retrieval algorithms applied to **real Hubble Space Telescope (HST)** point-spread
function (PSF) observations and deterministic synthetic benchmarks.

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

## Architecture

```
phase-retrieval/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py               Command-line interface (argparse)
│   ├── py.typed              PEP 561 marker
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base.py           Abstract base class (momentum, TV, adaptive β, noise model)
│   │   ├── error_reduction.py
│   │   ├── gerchberg_saxton.py
│   │   ├── hybrid_input_output.py
│   │   ├── raar.py
│   │   ├── wirtinger_flow.py    ★ State-of-the-art: gradient + spectral init
│   │   ├── douglas_rachford.py  ★ Proximal splitting
│   │   ├── admm.py              ★ ADMM with Fourier/support splitting
│   │   ├── pinn.py              ★ Physics-informed neural field (optional)
│   │   ├── phase_diversity.py
│   │   ├── multi_start.py       Multi-start optimization runner
│   │   └── registry.py          Factory pattern for algorithm selection
│   ├── data/
│   │   ├── downloader.py        Curated observations from the MAST archive
│   │   └── loader.py            FITS I/O, source detection, cutout extraction
│   ├── metrics/
│   │   └── quality.py           Strehl, RMS, Zernike, MTF, SSIM, Phase Structure Function
│   ├── models/
│   │   ├── config.py            Pipeline, algorithm, pupil, data configs (Pydantic)
│   │   └── optics.py            PSFData, PupilModel, PhaseRetrievalResult
│   ├── optics/
│   │   ├── propagator.py        FFT forward/inverse, defocus injection
│   │   ├── pupils.py            HST, JWST, generic circular pupil builders
│   │   └── zernike.py           Noll-ordered Zernike polynomials
│   └── visualization/
│       └── plots.py             Publication-quality plotting (matplotlib)
├── tests/                       Pytest test suite (synthetic data only — no network)
├── notebooks/
│   └── phase_retrieval_hst.ipynb  Full interactive tutorial
├── pyproject.toml
├── docs/
│   ├── changelog.md
│   ├── contributing.md
│   └── security.md
└── LICENSE
```

## Data

This project downloads **real calibrated HST/WFC3 observations** of standard
calibration stars from the Mikulski Archive for Space Telescopes (MAST). These are
not synthetic — they are actual photons collected by HST.

The preprocessing path records background estimation, centroid refinement,
quality-mask repair, optional DQ-mask usage, and recentering provenance for
each loaded cutout. The reporting layer can also include perturbation-based
uncertainty summaries for supported workflows.

## Algorithms

### Classic Algorithms

| Algorithm | Key | Description |
|-----------|-----|-------------|
| Error Reduction (ER) | `er` | Fienup's basic projection algorithm (1982) |
| Gerchberg–Saxton (GS) | `gs` | Classic two-plane amplitude constraint |
| Hybrid Input-Output (HIO) | `hio` | Fienup's workhorse with feedback parameter β |
| Phase Diversity (PD) | `phase_diversity` | Joint estimation from focused + defocused pair |

### Modern Algorithms

| Algorithm | Key | Description |
|-----------|-----|-------------|
| Relaxed Averaged Alternating Reflections (RAAR) | `raar` | Luke's convex-relaxation method (2005) |
| **Wirtinger Flow (WF)** | `wf` | Gradient descent with spectral initialization (Candès et al. 2015) |
| **Douglas-Rachford (DR)** | `dr` | Proximal splitting with provable convergence (Bauschke et al. 2002) |
| **ADMM** | `admm` | Alternating Direction Method of Multipliers (Chang & Marchesini 2018) |
| **FISTA** | `fista` | Proximal-gradient with adaptive √t Lipschitz scheduling (Beck & Teboulle 2009) |
| **Sparse Phase Retrieval (ThWF)** | `sparse_pr` | Thresholded Wirtinger Flow for sparsity-promoting recovery (Cai et al. 2016) |
| **Physics-Informed Neural Field** | `pinn` | Coordinate MLP optimized through differentiable Fourier optics |

### Practical Enhancements

Many algorithms can use these built-in enhancements:

| Feature | Flag | Description |
|---------|------|-------------|
| **Nesterov Momentum** | `--momentum 0.5` | Heavy-ball acceleration for faster convergence |
| **Adaptive β Scheduling** | `--beta-schedule cosine` | Cosine / linear annealing of the feedback parameter |
| **TV Regularization** | `--tv-weight 0.01` | Chambolle proximal operator for noise-robust phase |
| **Poisson Noise Model** | `--noise-model poisson` | Maximum-likelihood projection for photon-limited data |
| **Multi-Start** | `--n-starts 5` | Multiple random restarts, returns best result |
| **Uncertainty Ensemble** | `--uncertainty-samples 8` | Perturbation-based confidence intervals for saved metrics |
| **Reference Baseline Check** | automatic on supported filters | Compare observed/reconstructed PSFs against curated STScI HST/JWST references |

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
git clone https://github.com/rezamirzaeifard/phase-retrieval.git
cd phase-retrieval

# Install in editable mode (recommended for development)
pip install -e ".[dev]"

# Optional: enable the PINN/neural-field solver
pip install -e ".[pinn]"
```

After installation, the canonical Python package name is `phase_retrieval`.
For backwards compatibility, the in-repository `src` package still works when running from a checkout.

## Quick Start

### As a script (PyCharm / Jupyter-style)

```bash
python scripts/main.py
```

### Via the CLI

```bash
# Run a single algorithm
phase-retrieval run --algorithm hio --iterations 500

# Run state-of-the-art Wirtinger Flow with spectral init
phase-retrieval run --algorithm wf --iterations 300

# Use advanced features: cosine β schedule + momentum + TV regularization
phase-retrieval run --algorithm raar --iterations 1000 \
    --beta-schedule cosine --momentum 0.3 --tv-weight 0.01

# Multi-start optimization (5 random restarts, keeps best)
phase-retrieval run --algorithm hio --n-starts 5

# Compare all algorithms on the same observation
phase-retrieval compare --iterations 500

# Poisson noise model for low-SNR data
phase-retrieval run --algorithm raar --noise-model poisson

# List available observation presets
phase-retrieval download --list

# Optional PINN / neural-field solver (requires pip install -e ".[pinn]")
phase-retrieval run --algorithm pinn --iterations 300

# Deterministic synthetic benchmark with JSON / CSV / Markdown reports
phase-retrieval benchmark --algorithms er,hio,raar,wf --cases clean-low,poisson-hst
```

### As a Python module

```bash
python -m phase_retrieval run --algorithm raar
```

If you are running directly from a source checkout, `python -m src ...` remains supported.

## Testing

```bash
# Run the full test suite (471 tests)
pytest

# With coverage
pytest --cov=src --cov=web --cov-report=term-missing

# Web API tests only
pytest tests/test_web.py tests/test_crystallography_web.py -v
```

## Web Application

A full-stack **REST + WebSocket API** built with FastAPI, with JWT authentication,
background job queue, real-time progress streaming, and an Angular 18 frontend.

```bash
# Quick start with Docker
docker compose up --build

# Or local dev
pip install -e ".[web,dev]"
uvicorn web.main:app --reload --port 8000
```

**Key web features:**
- 10 phase-retrieval algorithms with background job execution
- Real-time WebSocket progress streaming
- JWT auth with refresh tokens, bcrypt, rate limiting, security headers
- File upload (FITS/NPY/CIF), batch ZIP export, paginated results
- X-ray crystallography workflow (COD → diffraction → phase retrieval)
- Health/readiness probes, structured logging, request-ID tracing
- Interactive API docs at `/docs` (Swagger UI) and `/redoc` (ReDoc)

See [`web/README.md`](web/README.md) for full endpoint documentation.

## Validation Scope

The repository includes deterministic synthetic benchmarks, perturbation cases
(including mis-centering and residual background), and rich regression tests.
Those artifacts provide **internal validation evidence** for software quality
and comparative behavior.

For supported real-data configurations, runs also record **external
reference-baseline checks** against curated STScI HST/JWST PSF documentation
values. These serve as transparent sanity checks rather than end-to-end mission
validation.

They do **not** by themselves prove instrument-grade scientific validity on real
HST/JWST data. Real-data conclusions should still be treated as model-dependent
and preprocessing-sensitive unless validated against trusted external baselines.

## Supported vs Experimental

**Supported by current repository evidence**
- deterministic synthetic benchmarking with known-truth phase
- preprocessing provenance and quality-mask diagnostics
- perturbation-based confidence intervals for reported metrics
- curated external reference-baseline checks for selected HST/JWST detector + filter pairs

**Still experimental / approximate**
- simple broadband spectral weighting and detector-transfer modeling
- field-dependent defocus hooks
- scientific interpretation of real-data reconstructions without external baselines

## Benchmarking & Reproducibility

The CLI includes a deterministic synthetic benchmark harness for regression
testing and algorithm comparison:

```bash
phase-retrieval benchmark \
    --algorithms er,gs,hio,raar,wf,dr,admm,fista,sparse_pr \
    --cases clean-low,clean-hst,poisson-hst,noisy-high \
    --iterations 40 \
    --output-dir outputs/benchmark
```

This writes three reports:

- `benchmark_results.json` — full per-case metrics and rankings
- `benchmark_summary.csv` — aggregate per-algorithm table for spreadsheets/CI
- `benchmark_study.json` / `benchmark_study.csv` — convergence, robustness-drop, and failure-mode summaries
- `benchmark_report.md` — human-readable experiment summary

and two visual comparison artifacts:

- `benchmark_leaderboard.png` — ranked aggregate-score comparison plot
- `benchmark_case_heatmap.png` — per-case performance heatmap across algorithms

Regular pipeline runs now also persist richer output manifests:

- `config.json` — algorithm configuration snapshot
- `result.json` — top-level run summary
- `metrics.json` — SSIM, radial-profile error, encircled-energy error, convergence summary, Zernike terms
- `reference_validation.json` — curated STScI baseline comparison when supported
- `provenance.json` — source file metadata, preprocessing steps, and pupil/algorithm provenance
- `evaluation_report.json` / `evaluation_report.md` — paper-style single-run evaluation summary

For CLI `run`, the same evaluation artifacts are written as:

- `evaluation_<algorithm>.json`
- `evaluation_<algorithm>.md`

For CLI `compare`, aggregate comparison reports are also written:

- `comparison_report.json`
- `comparison_report.md`

and real-data comparison plots are exported when saving is enabled:

- `algorithm_comparison.png`
- `algorithm_dashboard.png`
- `strehl_rms_comparison.png`

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

See [docs/contributing.md](docs/contributing.md) for development setup, coding conventions,
and the pull-request workflow.

## License

This project is licensed under the [MIT License](LICENSE).
