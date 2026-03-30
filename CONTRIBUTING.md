# Contributing to Phase Retrieval

Thanks for considering a contribution! This guide will help you get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/<you>/phase-retrieval.git
cd phase-retrieval

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Full test suite
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_algorithms.py -v
```

## Code Quality

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for lint issues
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Check formatting
ruff format --check src/ tests/

# Auto-format
ruff format src/ tests/
```

## Pull Request Workflow

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** for any new functionality — aim for the tests in `tests/` to
   stay fully synthetic (no network, no real FITS files).
3. **Run the checks** locally: `ruff check`, `ruff format --check`, `pytest`.
4. **Open a PR** with a clear description of *what* and *why*.

## Project Layout

```
src/
├── algorithms/    Phase retrieval algorithm implementations
├── data/          MAST downloader & FITS loader
├── metrics/       Strehl ratio, RMS, Zernike decomposition
├── models/        Pydantic-validated data models
├── optics/        Pupil models, Zernike basis, FFT propagation
└── visualization/ Publication-quality plotting
tests/             Pytest test suite (synthetic data only)
```

## Conventions

- **Type hints** on all public functions.
- **Docstrings** in NumPy style for public API.
- **No real data in tests** — generate synthetic pupils and PSFs.
- **Pydantic models** for any structured data crossing module boundaries.
