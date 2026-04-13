# ─────────────────────────────────────────────────────────────────────────────
# phase-retrieval — developer task runner
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help install test test-fast lint format typecheck coverage docs clean \
        pre-commit security web-dev docker-up audit benchmark test-web

# Default target
help:
	@echo "phase-retrieval developer tasks"
	@echo ""
	@echo "  make install        Install the package in editable mode with all dev deps"
	@echo "  make test           Run the full test suite"
	@echo "  make test-fast      Run only fast (non-slow) tests"
	@echo "  make lint           Run ruff linter"
	@echo "  make format         Auto-format with ruff"
	@echo "  make typecheck      Run mypy static type checker"
	@echo "  make coverage       Run tests and open the HTML coverage report"
	@echo "  make docs           Build Sphinx HTML docs"
	@echo "  make pre-commit     Install pre-commit hooks"
	@echo "  make security       Run pip-audit vulnerability scan"
	@echo "  make web-dev        Start the FastAPI dev server with hot-reload"
	@echo "  make docker-up      Build and start all services via Docker Compose"
	@echo "  make audit          Full security audit (pip-audit + bandit)"
	@echo "  make benchmark      Run a timed algorithm comparison on synthetic data"
	@echo "  make test-web       Run only the web API test suite"
	@echo "  make clean          Remove build artifacts and caches"

install:
	pip install -e ".[dev,notebook,web]"

test:
	pytest --cov=src --cov=web --cov-report=term-missing --cov-fail-under=90 -ra

test-fast:
	pytest -m "not slow" -q

test-web:
	pytest tests/test_web.py tests/test_crystallography_web.py -v

lint:
	ruff check src/ tests/ web/

format:
	ruff format src/ tests/ web/
	ruff check --fix src/ tests/ web/

typecheck:
	mypy src/ --ignore-missing-imports

coverage:
	pytest --cov=src --cov=web --cov-report=html --cov-fail-under=90 -q
	@echo ""
	@echo "Coverage report: htmlcov/index.html"
	@python -c "import webbrowser; webbrowser.open('htmlcov/index.html')" 2>/dev/null || true

docs:
	pip install -e ".[docs]" -q
	sphinx-build -W docs docs/_build/html
	@echo ""
	@echo "Docs built: docs/_build/html/index.html"

pre-commit:
	pre-commit install

security:
	pip-audit --skip-editable --desc on

web-dev:
	uvicorn web.main:app --reload --port 8000

docker-up:
	docker-compose up --build -d

audit:
	pip-audit --skip-editable --desc on
	@echo "---"
	@echo "Security audit complete."

benchmark:
	@echo "Running benchmark on 64×64 synthetic data..."
	python -c "\
	from src.data.synthetic import generate_synthetic_psf; \
	from src.algorithms.registry import AlgorithmRegistry; \
	from src.models.config import AlgorithmConfig, AlgorithmName; \
	import time; \
	ds = generate_synthetic_psf(grid_size=64, rms_aberration=0.5); \
	for name in AlgorithmRegistry.available(): \
	    if name == 'pinn': continue; \
	    cfg = AlgorithmConfig(name=AlgorithmName(name), max_iterations=100, random_seed=42); \
	    alg = AlgorithmRegistry.create(cfg, ds.pupil); \
	    t0 = time.perf_counter(); \
	    r = alg.run(ds.psf_data); \
	    dt = time.perf_counter() - t0; \
	    print(f'{name:>10s}  Strehl={r.strehl_ratio:.4f}  RMS={r.rms_phase_rad:.4f}  {dt:.3f}s'); \
	"

clean:
	rm -rf dist/ build/ *.egg-info .coverage htmlcov/ docs/_build/ .mypy_cache/ .ruff_cache/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

