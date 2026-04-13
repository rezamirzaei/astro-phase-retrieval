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
        leaderboard_png = tmp_path / "benchmark_leaderboard.png"
        heatmap_png = tmp_path / "benchmark_case_heatmap.png"
        assert json_path.exists()
        assert csv_path.exists()
        assert md_path.exists()
        assert leaderboard_png.exists()
        assert heatmap_png.exists()

        payload = json.loads(json_path.read_text())
        assert payload["cases"][0]["key"] == "clean-low"
        assert {row["algorithm"] for row in payload["aggregate"]} == {"er", "hio"}
        assert "leaderboard_plot" in payload["artifacts"]
        assert "heatmap_plot" in payload["artifacts"]

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
        assert "er" in markdown


