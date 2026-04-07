# Contributing to Phase Retrieval

Thanks for considering a contribution! This guide will help you get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/rezamirzaeifard/phase-retrieval.git
cd phase-retrieval

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Optional: install the PINN neural-field solver
pip install -e ".[pinn]"
```

## Running Tests

```bash
# Full test suite
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_algorithms.py -v

# Run only tests matching a keyword
pytest -k "raar" -v
```

## Code Quality

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting,
and [mypy](https://mypy.readthedocs.io/) for type checking:

```bash
# Check for lint issues
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Check formatting
ruff format --check src/ tests/

# Auto-format
ruff format src/ tests/

# Type check
mypy src/ --ignore-missing-imports
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
│   ├── base.py        Abstract base class with shared enhancements
│   ├── registry.py    Factory pattern for algorithm selection
│   ├── multi_start.py Multi-start optimization runner
│   └── *.py           Individual algorithm implementations
├── data/          MAST downloader & FITS loader
├── metrics/       Strehl ratio, RMS, Zernike, MTF, SSIM, Phase Structure Function
├── models/        Pydantic-validated data models (config + optics)
├── optics/        Pupil models, Zernike basis, FFT propagation
└── visualization/ Publication-quality plotting
tests/             Pytest test suite (synthetic data only)
notebooks/         Jupyter notebook tutorials
```

## Adding a New Algorithm

1. Create `src/algorithms/my_algorithm.py` with a class that inherits from `PhaseRetriever`.
2. Implement the `_iterate()` method — it receives and returns the complex pupil field.
3. Add an entry to `AlgorithmName` enum in `src/models/config.py`.
4. Register it in `src/algorithms/registry.py`.
5. Add it to `src/algorithms/__init__.py`.
6. Write tests in `tests/test_algorithms.py`.

## Conventions

- **Type hints** on all public functions.
- **Docstrings** in NumPy style for public API.
- **No real data in tests** — generate synthetic pupils and PSFs.
- **Pydantic models** for any structured data crossing module boundaries.
- **Keep dependencies minimal** — optional features (e.g., PINN/PyTorch) use extras.
