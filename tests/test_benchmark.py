"""Tests for the deterministic synthetic benchmark harness."""

from __future__ import annotations

import json

from src.benchmark import available_benchmark_cases, run_benchmark
from src.models.config import AlgorithmName


class TestBenchmark:
    def test_run_benchmark_exports_reports(self, tmp_path) -> None:
        case = available_benchmark_cases()["clean-low"]
        summary = run_benchmark(
            algorithms=[AlgorithmName.ERROR_REDUCTION, AlgorithmName.HYBRID_INPUT_OUTPUT],
            cases=[case],
            max_iterations=3,
            output_dir=tmp_path,
        )

        assert len(summary.records) == 2
        assert len(summary.aggregate) == 2
        assert summary.aggregate[0]["mean_score"] >= 0.0

        json_path = tmp_path / "benchmark_results.json"
        csv_path = tmp_path / "benchmark_summary.csv"
        md_path = tmp_path / "benchmark_report.md"
        study_json = tmp_path / "benchmark_study.json"
        study_csv = tmp_path / "benchmark_study.csv"
        leaderboard_png = tmp_path / "benchmark_leaderboard.png"
        heatmap_png = tmp_path / "benchmark_case_heatmap.png"
        assert json_path.exists()
        assert csv_path.exists()
        assert md_path.exists()
        assert study_json.exists()
        assert study_csv.exists()
        assert leaderboard_png.exists()
        assert heatmap_png.exists()

        payload = json.loads(json_path.read_text())
        assert payload["cases"][0]["key"] == "clean-low"
        assert {row["algorithm"] for row in payload["aggregate"]} == {"er", "hio"}
        assert payload["study"][0]["algorithm"] in {"er", "hio"}
        assert "leaderboard_plot" in payload["artifacts"]
        assert "heatmap_plot" in payload["artifacts"]
        assert "study_json" in payload["artifacts"]
        assert "center_offset_pixels" in payload["cases"][0]

    def test_markdown_contains_ranking(self, tmp_path) -> None:
        case = available_benchmark_cases()["clean-low"]
        summary = run_benchmark(
            algorithms=[AlgorithmName.ERROR_REDUCTION],
            cases=[case],
            max_iterations=2,
            output_dir=tmp_path,
        )
        markdown = summary.to_markdown()
        assert "# Phase Retrieval Benchmark Report" in markdown
        assert "Aggregate ranking" in markdown
        assert "Convergence and Failure-Mode Study" in markdown
        assert "Limits" in markdown
        assert "er" in markdown

    def test_available_cases_include_robustness_scenarios(self) -> None:
        cases = available_benchmark_cases()
        assert "miscentered-hst" in cases
        assert "background-hst" in cases
        assert "broadband-hst" in cases

    def test_records_capture_offset_and_background(self, tmp_path) -> None:
        case = available_benchmark_cases()["miscentered-hst"]
        summary = run_benchmark(
            algorithms=[AlgorithmName.ERROR_REDUCTION],
            cases=[case],
            max_iterations=2,
            output_dir=tmp_path,
        )
        record = summary.records[0]
        assert record["center_offset_pixels"] == [0.75, -0.45]
        assert record["background_level"] == 0.0

    def test_broadband_case_records_spectral_fields(self, tmp_path) -> None:
        case = available_benchmark_cases()["broadband-hst"]
        summary = run_benchmark(
            algorithms=[AlgorithmName.ERROR_REDUCTION],
            cases=[case],
            max_iterations=2,
            output_dir=tmp_path,
        )
        record = summary.records[0]
        assert record["bandwidth_fraction"] > 0.0
        assert record["spectral_samples"] == 5
        assert record["field_defocus_waves"] == 0.15
