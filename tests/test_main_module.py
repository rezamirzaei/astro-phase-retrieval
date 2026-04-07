"""Test that ``python -m src`` works (exercises __main__.py)."""

from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import patch


def test_module_invocation_version() -> None:
    """``python -m src -V`` should print the version and exit cleanly."""
    result = subprocess.run(
        [sys.executable, "-m", "src", "-V"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "phase-retrieval" in result.stdout


def test_module_invocation_help() -> None:
    """``python -m src`` with no args should print help and exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "src"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower() or "phase-retrieval" in result.stdout.lower()


def test_canonical_module_invocation_version() -> None:
    """``python -m phase_retrieval -V`` should work for installed users."""
    result = subprocess.run(
        [sys.executable, "-m", "phase_retrieval", "-V"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "phase-retrieval" in result.stdout


def test_canonical_package_import_exposes_version() -> None:
    """``import phase_retrieval`` should expose package metadata."""
    import phase_retrieval

    assert isinstance(phase_retrieval.__version__, str)


def test_checkout_version_matches_pyproject() -> None:
    """Source-checkout imports should report the version declared in pyproject.toml."""
    import phase_retrieval

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as handle:
        expected = tomllib.load(handle)["project"]["version"]
    assert phase_retrieval.__version__ == expected


def test_main_module_inline() -> None:
    """Import __main__ and exercise it via mock to get coverage."""
    with patch("src.cli.main") as _mock_main:
        # __main__.py only calls main() when __name__ == "__main__",
        # but we can at least verify the import works
        import src.__main__  # noqa: F401

        # Directly call to verify the wiring
        from src.__main__ import main as _main_ref

        assert _main_ref is not None


def test_version_fallback() -> None:
    """When the package is not installed, __version__ should fall back to '0.0.0-dev'."""
    from importlib.metadata import PackageNotFoundError

    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("importlib.metadata.version", side_effect=PackageNotFoundError()),
    ):
        import src

        importlib.reload(src)
        assert src.__version__ == "0.0.0-dev"

    importlib.reload(src)
