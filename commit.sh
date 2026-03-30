#!/usr/bin/env bash
# One-shot commit script — run with: bash commit.sh
set -e
cd "$(dirname "$0")"

git add -A

git commit -m "Add CI, CLI, test suite, and project tooling for v1.0.0 release

Establish production-ready project infrastructure around the existing
phase-retrieval algorithms and optics pipeline.

New files:
- .gitignore for Python, IDE, data, and output artifacts
- GitHub Actions CI workflow (Ruff lint + pytest on Python 3.11/3.12)
- CONTRIBUTING.md with dev setup, testing, and PR workflow
- CHANGELOG.md with 1.0.0 release notes
- src/py.typed PEP 561 marker for downstream type checking
- src/__main__.py to enable python -m src invocation
- src/cli.py with argparse CLI: run, compare, download subcommands
- tests/ with conftest and 7 test modules covering models, optics,
  algorithms, metrics, CLI, data helpers, and visualization

Updated files:
- pyproject.toml: Production/Stable classifier, Python 3.12 support,
  pytest-cov + mypy dev deps, CLI entry point, Ruff/pytest/mypy config
- README.md: comprehensive rewrite with architecture diagram, CLI docs,
  install instructions, and contributing link
- main.py: fix duplicate section numbering (two §10 -> 10, 11, 12, 13)
- All 7 subpackage __init__.py: add __all__ exports and public API
  re-exports
- src/__init__.py: expose __version__ via importlib.metadata"

echo "✅ Committed successfully."

# Clean up this script
rm -- "$0"
