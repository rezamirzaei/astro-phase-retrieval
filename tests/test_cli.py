"""Smoke tests for the CLI (no network, no real FITS)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.cli import _sync_pupil_to_image, main
from src.models.config import PipelineConfig


class TestCLI:
    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["-V"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "phase-retrieval" in captured.out

    def test_verbose_download_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["-v", "download", "--list"])
        captured = capsys.readouterr()
        assert "hst-wfc3-uvis-f606w" in captured.out

    def test_download_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--list should print preset names without hitting the network."""
        main(["download", "--list"])
        captured = capsys.readouterr()
        assert "hst-wfc3-uvis-f606w" in captured.out

    def test_run_with_mock_fits(self, tmp_path, pupil, psf_data, capsys) -> None:
        """Mock the FITS loading chain and verify `run` completes."""
        fake_fits = tmp_path / "fake.fits"
        fake_fits.touch()

        with (
            patch("src.data.loader.load_psf_from_fits", return_value=psf_data),
            patch("src.data.loader.prepare_psf_for_retrieval", return_value=psf_data.image),
        ):
            main(
                [
                    "run",
                    "--algorithm",
                    "er",
                    "--iterations",
                    "10",
                    "--fits",
                    str(fake_fits),
                    "-o",
                    str(tmp_path / "out"),
                ]
            )

        capsys.readouterr()

    def test_run_auto_discovered_fits(self, tmp_path, psf_data, capsys) -> None:
        """When no --fits is given, should auto-discover cached FITS."""
        fake_fits = tmp_path / "auto.fits"
        fake_fits.touch()

        with (
            patch("src.data.downloader.list_cached_fits", return_value=[fake_fits]),
            patch("src.data.loader.load_psf_from_fits", return_value=psf_data),
            patch("src.data.loader.prepare_psf_for_retrieval", return_value=psf_data.image),
        ):
            main(
                [
                    "run",
                    "--algorithm",
                    "er",
                    "--iterations",
                    "5",
                    "-o",
                    str(tmp_path / "out"),
                ]
            )
        captured = capsys.readouterr()
        assert "ER" in captured.out

    def test_run_no_cached_fits_exits(self, tmp_path, capsys) -> None:
        """When no FITS available and none specified, should exit with error."""
        with (
            patch("src.data.downloader.list_cached_fits", return_value=[]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main(
                [
                    "run",
                    "--algorithm",
                    "er",
                    "--iterations",
                    "5",
                    "-o",
                    str(tmp_path / "out"),
                ]
            )
        assert exc_info.value.code == 1

    def test_run_multi_start(self, tmp_path, psf_data, capsys) -> None:
        fake_fits = tmp_path / "fake.fits"
        fake_fits.touch()

        with (
            patch("src.data.loader.load_psf_from_fits", return_value=psf_data),
            patch("src.data.loader.prepare_psf_for_retrieval", return_value=psf_data.image),
        ):
            main(
                [
                    "run",
                    "--algorithm",
                    "er",
                    "--iterations",
                    "5",
                    "--n-starts",
                    "2",
                    "--fits",
                    str(fake_fits),
                    "-o",
                    str(tmp_path / "out"),
                ]
            )
        captured = capsys.readouterr()
        assert "ER" in captured.out

    def test_run_with_enhancements(self, tmp_path, psf_data, capsys) -> None:
        fake_fits = tmp_path / "fake.fits"
        fake_fits.touch()

        with (
            patch("src.data.loader.load_psf_from_fits", return_value=psf_data),
            patch("src.data.loader.prepare_psf_for_retrieval", return_value=psf_data.image),
        ):
            main(
                [
                    "run",
                    "--algorithm",
                    "raar",
                    "--iterations",
                    "5",
                    "--beta-schedule",
                    "cosine",
                    "--momentum",
                    "0.3",
                    "--tv-weight",
                    "0.01",
                    "--noise-model",
                    "poisson",
                    "--fits",
                    str(fake_fits),
                    "-o",
                    str(tmp_path / "out"),
                ]
            )
        capsys.readouterr()

    def test_compare_with_mock_fits(self, tmp_path, psf_data, capsys) -> None:
        """Mock the FITS loading chain and verify `compare` completes."""
        fake_fits = tmp_path / "fake.fits"
        fake_fits.touch()

        with (
            patch("src.data.loader.load_psf_from_fits", return_value=psf_data),
            patch("src.data.loader.prepare_psf_for_retrieval", return_value=psf_data.image),
        ):
            main(
                [
                    "compare",
                    "--iterations",
                    "3",
                    "--fits",
                    str(fake_fits),
                    "-o",
                    str(tmp_path / "out"),
                ]
            )

        captured = capsys.readouterr()
        assert "Algorithm" in captured.out
        assert "RAAR" in captured.out
        assert "ER" in captured.out
        assert "Strehl" in captured.out

    def test_compare_auto_discovered_fits(self, tmp_path, psf_data, capsys) -> None:
        """Compare should auto-discover cached FITS when none specified."""
        fake_fits = tmp_path / "auto.fits"
        fake_fits.touch()

        with (
            patch("src.data.downloader.list_cached_fits", return_value=[fake_fits]),
            patch("src.data.loader.load_psf_from_fits", return_value=psf_data),
            patch("src.data.loader.prepare_psf_for_retrieval", return_value=psf_data.image),
        ):
            main(
                [
                    "compare",
                    "--iterations",
                    "3",
                    "-o",
                    str(tmp_path / "out"),
                ]
            )
        captured = capsys.readouterr()
        assert "ER" in captured.out

    def test_compare_no_cached_fits_exits(self, tmp_path) -> None:
        with (
            patch("src.data.downloader.list_cached_fits", return_value=[]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main(["compare", "--iterations", "3", "-o", str(tmp_path / "out")])
        assert exc_info.value.code == 1

    def test_compare_multi_start(self, tmp_path, psf_data, capsys) -> None:
        fake_fits = tmp_path / "fake.fits"
        fake_fits.touch()

        with (
            patch("src.data.loader.load_psf_from_fits", return_value=psf_data),
            patch("src.data.loader.prepare_psf_for_retrieval", return_value=psf_data.image),
        ):
            main(
                [
                    "compare",
                    "--iterations",
                    "3",
                    "--n-starts",
                    "2",
                    "--fits",
                    str(fake_fits),
                    "-o",
                    str(tmp_path / "out"),
                ]
            )
        capsys.readouterr()

    def test_download_preset_mock(self, tmp_path, capsys) -> None:
        with patch("src.data.downloader.download_preset", return_value=[tmp_path / "f.fits"]):
            main(["download", "-p", "hst-wfc3-uvis-f606w", "-d", str(tmp_path)])
        captured = capsys.readouterr()
        assert "Downloading" in captured.out


class TestSyncPupilToImage:
    def test_non_square_raises(self) -> None:
        config = PipelineConfig()
        with pytest.raises(ValueError, match="square"):
            _sync_pupil_to_image(config, (64, 128))

    def test_matching_size_no_change(self) -> None:
        config = PipelineConfig()
        config.pupil.grid_size = 64
        result = _sync_pupil_to_image(config, (64, 64))
        assert result.pupil.grid_size == 64

    def test_mismatched_size_rebuilds(self) -> None:
        config = PipelineConfig()
        config.pupil.grid_size = 256
        result = _sync_pupil_to_image(config, (128, 128))
        assert result.pupil.grid_size == 128


class TestHasTorch:
    def test_has_torch_returns_bool(self) -> None:
        from src.cli import _has_torch

        result = _has_torch()
        assert isinstance(result, bool)

    def test_has_torch_false_when_import_fails(self) -> None:
        import sys

        from src.cli import _has_torch

        # Temporarily remove torch from sys.modules so the import is re-attempted
        saved = sys.modules.pop("torch", None)
        try:
            import builtins

            _real_import = builtins.__import__

            def _mock_import(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("mocked")
                return _real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_mock_import):
                result = _has_torch()
            assert result is False
        finally:
            if saved is not None:
                sys.modules["torch"] = saved
