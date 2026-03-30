"""Smoke tests for the CLI (no network, no real FITS)."""

from __future__ import annotations

from unittest.mock import patch
import pytest

from src.cli import main


class TestCLI:
    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    def test_download_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--list should print preset names without hitting the network."""
        main(["download", "--list"])
        captured = capsys.readouterr()
        assert "hst-wfc3-uvis-f606w" in captured.out

    def test_run_with_mock_fits(
        self, tmp_path, pupil, psf_data, capsys
    ) -> None:
        """Mock the FITS loading chain and verify `run` completes."""
        fake_fits = tmp_path / "fake.fits"
        fake_fits.touch()

        with (
            patch("src.data.loader.load_psf_from_fits", return_value=psf_data),
            patch("src.data.loader.prepare_psf_for_retrieval", return_value=psf_data.image),
        ):
            main([
                "run",
                "--algorithm", "er",
                "--iterations", "10",
                "--fits", str(fake_fits),
                "-o", str(tmp_path / "out"),
            ])

        captured = capsys.readouterr()
        assert "ER" in captured.out
        assert "Strehl" in captured.out

    def test_compare_with_mock_fits(self, tmp_path, psf_data, capsys) -> None:
        """Mock the FITS loading chain and verify `compare` completes."""
        fake_fits = tmp_path / "fake.fits"
        fake_fits.touch()

        with (
            patch("src.data.loader.load_psf_from_fits", return_value=psf_data),
            patch("src.data.loader.prepare_psf_for_retrieval", return_value=psf_data.image),
        ):
            main([
                "compare",
                "--iterations", "3",
                "--fits", str(fake_fits),
                "-o", str(tmp_path / "out"),
            ])

        captured = capsys.readouterr()
        assert "Algorithm" in captured.out
        assert "RAAR" in captured.out

