"""Smoke tests for the CLI (no network, no real FITS)."""

from __future__ import annotations

import json
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

    def test_invalid_algorithm_exits_with_argparse_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--algorithm", "not-a-real-algorithm"])
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "invalid choice" in captured.err

    def test_invalid_download_preset_exits_with_argparse_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["download", "--preset", "not-a-real-preset"])
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "invalid choice" in captured.err
        assert "Downloading preset" not in captured.out

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
        """Mock the FITS loading chain and verify `run` completes and writes valid JSON."""
        fake_fits = tmp_path / "fake.fits"
        fake_fits.touch()
        out_dir = tmp_path / "out"

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
                    str(out_dir),
                ]
            )

        capsys.readouterr()

        # Verify that the JSON result file exists and has the expected keys
        result_file = out_dir / "result_er.json"
        assert result_file.exists(), "result_er.json was not written"
        assert (out_dir / "evaluation_er.json").exists()
        assert (out_dir / "evaluation_er.md").exists()
        result_data = json.loads(result_file.read_text())
        for key in ("algorithm", "strehl_ratio", "rms_phase_rad", "n_iterations", "converged"):
            assert key in result_data, f"Missing key '{key}' in result JSON"
        assert result_data["algorithm"] == "er"
        assert 0.0 <= result_data["strehl_ratio"] <= 1.0
        assert result_data["rms_phase_rad"] >= 0.0

    def test_run_quiet_flag(self, tmp_path, psf_data, capsys) -> None:
        """--quiet suppresses progress output but still writes the JSON file."""
        fake_fits = tmp_path / "fake.fits"
        fake_fits.touch()
        out_dir = tmp_path / "quiet_out"

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
                    "--fits",
                    str(fake_fits),
                    "-o",
                    str(out_dir),
                    "--quiet",
                ]
            )

        captured = capsys.readouterr()
        # No emoji / progress lines in stdout
        assert "✅" not in captured.out
        # JSON file still written
        assert (out_dir / "result_er.json").exists()
        assert (out_dir / "evaluation_er.json").exists()

    def test_run_output_format_json(self, tmp_path, psf_data, capsys) -> None:
        """--output-format json prints a JSON object to stdout."""
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
                    "--fits",
                    str(fake_fits),
                    "-o",
                    str(tmp_path / "json_out"),
                    "--output-format",
                    "json",
                ]
            )

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["algorithm"] == "er"
        assert "strehl_ratio" in parsed

    def test_run_with_uncertainty_outputs_json(self, tmp_path, psf_data, capsys) -> None:
        fake_fits = tmp_path / "fake.fits"
        fake_fits.touch()
        out_dir = tmp_path / "unc_out"

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
                    "4",
                    "--uncertainty-samples",
                    "2",
                    "--fits",
                    str(fake_fits),
                    "-o",
                    str(out_dir),
                    "--quiet",
                ]
            )

        capsys.readouterr()
        assert (out_dir / "uncertainty_er.json").exists()

    def test_log_format_json(self, capsys) -> None:
        """--log-format json should not crash."""
        main(["--log-format", "json", "download", "--list"])
        captured = capsys.readouterr()
        assert "hst-wfc3-uvis-f606w" in captured.out

    def test_compare_saves_json_files(self, tmp_path, psf_data, capsys) -> None:
        """compare --save writes JSON files for each algorithm."""
        fake_fits = tmp_path / "fake.fits"
        fake_fits.touch()
        out_dir = tmp_path / "cmp_out"

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
                    str(out_dir),
                    "--save",
                ]
            )

        capsys.readouterr()
        # At least one JSON file should have been written
        json_files = list(out_dir.glob("result_*.json"))
        assert len(json_files) >= 1
        for jf in json_files:
            data = json.loads(jf.read_text())
            assert "algorithm" in data
            assert "strehl_ratio" in data
        assert (out_dir / "comparison_report.json").exists()
        assert (out_dir / "comparison_report.md").exists()
        assert (out_dir / "algorithm_comparison.png").exists()
        assert (out_dir / "algorithm_dashboard.png").exists()
        assert (out_dir / "strehl_rms_comparison.png").exists()
        comparison_payload = json.loads((out_dir / "comparison_report.json").read_text())
        assert "artifacts" in comparison_payload
        assert "algorithm_comparison_plot" in comparison_payload["artifacts"]

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

    def test_benchmark_command_writes_reports(self, tmp_path, capsys) -> None:
        main(
            [
                "benchmark",
                "--algorithms",
                "er,hio",
                "--cases",
                "clean-low",
                "--iterations",
                "2",
                "-o",
                str(tmp_path / "bench"),
            ]
        )
        captured = capsys.readouterr()
        assert "Algorithm" in captured.out
        assert (tmp_path / "bench" / "benchmark_results.json").exists()
        assert (tmp_path / "bench" / "benchmark_summary.csv").exists()
        assert (tmp_path / "bench" / "benchmark_report.md").exists()
        assert (tmp_path / "bench" / "benchmark_leaderboard.png").exists()
        assert (tmp_path / "bench" / "benchmark_case_heatmap.png").exists()


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
        from src.cli import _has_torch

        with patch("importlib.util.find_spec", return_value=None):
            result = _has_torch()
        assert result is False


class TestJsonFormatter:
    """Exercise the _JsonFormatter used for --log-format json."""

    def test_format_produces_valid_json(self) -> None:
        import logging

        from src.cli import _JsonFormatter

        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["msg"] == "hello world"
        assert "ts" in parsed
