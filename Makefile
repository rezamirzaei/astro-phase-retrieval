# ─────────────────────────────────────────────────────────────────────────────
# phase-retrieval — developer task runner
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help install test test-fast lint format typecheck coverage docs clean \
        pre-commit security

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
	@echo "  make clean          Remove build artifacts and caches"

install:
	pip install -e ".[dev,notebook]"

test:
	pytest --cov=src --cov-report=term-missing --cov-fail-under=95 -ra

test-fast:
	pytest -m "not slow" -q

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

coverage:
	pytest --cov=src --cov-report=html --cov-fail-under=95 -q
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

clean:
	rm -rf dist/ build/ *.egg-info .coverage htmlcov/ docs/_build/ .mypy_cache/ .ruff_cache/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

