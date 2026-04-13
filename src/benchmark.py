"""Deterministic synthetic benchmark harness for phase-retrieval algorithms.

This module closes the gap between implementation quality and validation by
running a repeatable suite of synthetic test cases, aggregating metrics, and
exporting machine-readable and human-readable reports.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.algorithms.multi_start import multi_start_run
from src.algorithms.registry import AlgorithmRegistry
from src.data.synthetic import SyntheticDataset, generate_synthetic_psf
from src.metrics.quality import (
    compute_encircled_energy_error,
    compute_radial_profile_error,
    compute_rms_phase,
    compute_ssim,
    summarise_convergence,
)
from src.models.config import AlgorithmConfig, AlgorithmName, TelescopeType
from src.visualization.plots import (
    plot_benchmark_case_heatmap,
    plot_benchmark_leaderboard,
    save_figure,
)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One deterministic synthetic benchmark scenario."""

    key: str
    description: str
    grid_size: int = 64
    rms_aberration: float = 0.5
    photon_count: float = 0.0
    read_noise_std: float = 0.0
    center_offset_pixels: tuple[float, float] = (0.0, 0.0)
    background_level: float = 0.0
    bandwidth_fraction: float = 0.0
    spectral_samples: int = 1
    field_defocus_waves: float = 0.0
    telescope: TelescopeType = TelescopeType.GENERIC_CIRCULAR
    seed: int = 42


@dataclass(slots=True)
class BenchmarkSummary:
    """Full benchmark output: raw per-case results plus aggregate ranking."""

    cases: list[BenchmarkCase]
    records: list[dict[str, Any]]
    aggregate: list[dict[str, Any]]
    study: list[dict[str, Any]] = field(default_factory=list)
    output_dir: Path | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the benchmark."""
        return {
            "cases": [
                {
                    "key": case.key,
                    "description": case.description,
                    "grid_size": case.grid_size,
                    "rms_aberration": case.rms_aberration,
                    "photon_count": case.photon_count,
                    "read_noise_std": case.read_noise_std,
                    "center_offset_pixels": [
                        float(case.center_offset_pixels[0]),
                        float(case.center_offset_pixels[1]),
                    ],
                    "background_level": case.background_level,
                    "bandwidth_fraction": case.bandwidth_fraction,
                    "spectral_samples": case.spectral_samples,
                    "field_defocus_waves": case.field_defocus_waves,
                    "telescope": case.telescope.value,
                    "seed": case.seed,
                }
                for case in self.cases
            ],
            "records": self.records,
            "aggregate": self.aggregate,
            "study": self.study,
            "artifacts": self.artifacts,
        }

    def to_markdown(self) -> str:
        """Render a compact Markdown benchmark report."""
        lines = [
            "# Phase Retrieval Benchmark Report",
            "",
            "## Cases",
            "",
            (
                "| Key | Description | Grid | RMS (rad) | Photons | Read noise | "
                "Offset (px) | Background | Telescope |"
            ),
            (
                "|-----|-------------|-----:|----------:|--------:|-----------:|"
                "------------:|-----------:|-----------|"
            ),
        ]
        lines.extend(
            [
                "| "
                f"{case.key} | {case.description} | {case.grid_size} | {case.rms_aberration:.2f} | "
                f"{case.photon_count:.0f} | {case.read_noise_std:.2e} | "
                f"{np.hypot(*case.center_offset_pixels):.2f} | {case.background_level:.2e} | "
                f"{case.telescope.value} |"
                for case in self.cases
            ]
        )

        lines.extend(
            [
                "",
                "## Aggregate ranking",
                "",
                (
                    "| Rank | Algorithm | Score | SSIM | Phase RMS err | "
                    "Radial err | EE err | Converged | Time (s) |"
                ),
                (
                    "|-----:|-----------|------:|-----:|--------------:|"
                    "-----------:|-------:|----------:|---------:|"
                ),
            ]
        )
        for idx, row in enumerate(self.aggregate, start=1):
            lines.append(
                "| "
                f"{idx} | {row['algorithm']} | {row['mean_score']:.4f} | {row['mean_ssim']:.4f} | "
                f"{row['mean_phase_rms_error_rad']:.4f} | {row['mean_radial_profile_error']:.4f} | "
                f"{row['mean_encircled_energy_error']:.4f} | {row['converged_fraction']:.2f} | "
                f"{row['mean_elapsed_seconds']:.3f} |"
            )
        if self.study:
            lines.extend(
                [
                    "",
                    "## Convergence and Failure-Mode Study",
                    "",
                    (
                        "| Algorithm | Clean score | Stress score | Robustness drop | "
                        "Failure rate | Worst case |"
                    ),
                    "|-----------|------------:|-------------:|----------------:|-------------:|-----------|",
                ]
            )
            lines.extend(
                [
                    (
                        f"| {row['algorithm']} | {row['clean_mean_score']:.4f} | "
                        f"{row['stress_mean_score']:.4f} | {row['robustness_drop']:.4f} | "
                        f"{row['failure_rate']:.2f} | {row['worst_case']} |"
                    )
                    for row in self.study
                ]
            )
        lines.extend(
            [
                "",
                "## Limits",
                "",
                "These benchmarks are deterministic synthetic stress tests. They improve internal "
                "validation, but they are not a substitute for external "
                "instrument-grade validation against trusted datasets or "
                "literature baselines.",
            ]
        )
        return "\n".join(lines) + "\n"


_DEFAULT_CASES: dict[str, BenchmarkCase] = {
    "clean-low": BenchmarkCase(
        key="clean-low",
        description="Low-aberration, noiseless circular pupil",
        grid_size=64,
        rms_aberration=0.25,
        telescope=TelescopeType.GENERIC_CIRCULAR,
        seed=11,
    ),
    "clean-hst": BenchmarkCase(
        key="clean-hst",
        description="Moderate aberration, noiseless HST-like pupil",
        grid_size=64,
        rms_aberration=0.50,
        telescope=TelescopeType.HST,
        seed=22,
    ),
    "poisson-hst": BenchmarkCase(
        key="poisson-hst",
        description="Photon-limited HST-like PSF",
        grid_size=64,
        rms_aberration=0.50,
        photon_count=50_000,
        telescope=TelescopeType.HST,
        seed=33,
    ),
    "noisy-high": BenchmarkCase(
        key="noisy-high",
        description="High-aberration HST-like PSF with photon + read noise",
        grid_size=64,
        rms_aberration=1.00,
        photon_count=20_000,
        read_noise_std=1e-5,
        telescope=TelescopeType.HST,
        seed=44,
    ),
    "miscentered-hst": BenchmarkCase(
        key="miscentered-hst",
        description="Moderate-aberration HST-like PSF with subpixel centroid offset",
        grid_size=64,
        rms_aberration=0.50,
        center_offset_pixels=(0.75, -0.45),
        telescope=TelescopeType.HST,
        seed=55,
    ),
    "background-hst": BenchmarkCase(
        key="background-hst",
        description="Moderate-aberration HST-like PSF with residual background pedestal",
        grid_size=64,
        rms_aberration=0.50,
        photon_count=40_000,
        background_level=2e-6,
        telescope=TelescopeType.HST,
        seed=66,
    ),
    "broadband-hst": BenchmarkCase(
        key="broadband-hst",
        description="Moderate-aberration HST-like PSF with simple polychromatic blur",
        grid_size=64,
        rms_aberration=0.50,
        bandwidth_fraction=0.12,
        spectral_samples=5,
        field_defocus_waves=0.15,
        telescope=TelescopeType.HST,
        seed=77,
    ),
}


def available_benchmark_cases() -> dict[str, BenchmarkCase]:
    """Return the built-in synthetic benchmark cases."""
    return dict(_DEFAULT_CASES)


def default_benchmark_algorithms(*, include_pinn: bool = False) -> list[AlgorithmName]:
    """Return a practical default algorithm set for benchmarking."""
    algorithms = [
        AlgorithmName.ERROR_REDUCTION,
        AlgorithmName.GERCHBERG_SAXTON,
        AlgorithmName.HYBRID_INPUT_OUTPUT,
        AlgorithmName.RAAR,
        AlgorithmName.WIRTINGER_FLOW,
        AlgorithmName.DOUGLAS_RACHFORD,
        AlgorithmName.ADMM,
        AlgorithmName.FISTA,
        AlgorithmName.SPARSE_PR,
    ]
    if include_pinn:
        algorithms.append(AlgorithmName.PINN)
    return algorithms


def _phase_rms_error(dataset: SyntheticDataset, recovered_phase: np.ndarray) -> float:
    """Compute RMS phase error after piston removal over the true support.

    Phase retrieval has an inherent global sign ambiguity: φ and −φ produce
    the same PSF intensity.  We evaluate the error for both signs and return
    the minimum, avoiding artificially inflated scores.
    """
    support = dataset.pupil.amplitude > 0
    truth = dataset.true_phase.copy()
    recon = recovered_phase.copy()
    truth[~support] = 0.0
    recon[~support] = 0.0

    truth_vals = truth[support]
    recon_vals = recon[support]
    if truth_vals.size == 0:
        return 0.0

    truth_vals = truth_vals - truth_vals.mean()

    # Try +φ
    recon_plus = recon_vals - recon_vals.mean()
    error_plus = float(np.sqrt(np.mean((recon_plus - truth_vals) ** 2)))

    # Try −φ (global sign flip)
    recon_minus = -recon_vals
    recon_minus = recon_minus - recon_minus.mean()
    error_minus = float(np.sqrt(np.mean((recon_minus - truth_vals) ** 2)))

    return min(error_plus, error_minus)


def _score_record(record: dict[str, Any]) -> float:
    """Combine multiple metrics into a bounded benchmark score in [0, 1]."""
    ssim_score = float(record["ssim"])
    radial_score = 1.0 / (1.0 + float(record["radial_profile_error"]))
    ee_score = 1.0 / (1.0 + 10.0 * float(record["encircled_energy_error"]))
    phase_score = 1.0 / (1.0 + float(record["phase_rms_error_rad"]))
    convergence_score = min(max(float(record["convergence_improvement_ratio"]), 0.0), 1.0)
    score = (
        0.40 * ssim_score
        + 0.20 * radial_score
        + 0.15 * ee_score
        + 0.15 * phase_score
        + 0.10 * convergence_score
    )
    return float(min(max(score, 0.0), 1.0))


def _aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-case records into a ranked per-algorithm summary."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["algorithm"]), []).append(record)

    summary: list[dict[str, Any]] = []
    for algorithm, rows in grouped.items():
        summary.append(
            {
                "algorithm": algorithm,
                "n_cases": len(rows),
                "mean_score": float(np.mean([float(r["score"]) for r in rows])),
                "mean_ssim": float(np.mean([float(r["ssim"]) for r in rows])),
                "mean_phase_rms_error_rad": float(
                    np.mean([float(r["phase_rms_error_rad"]) for r in rows])
                ),
                "mean_radial_profile_error": float(
                    np.mean([float(r["radial_profile_error"]) for r in rows])
                ),
                "mean_encircled_energy_error": float(
                    np.mean([float(r["encircled_energy_error"]) for r in rows])
                ),
                "mean_elapsed_seconds": float(
                    np.mean([float(r["elapsed_seconds"]) for r in rows])
                ),
                "converged_fraction": float(
                    np.mean([1.0 if bool(r["converged"]) else 0.0 for r in rows])
                ),
            }
        )

    return sorted(summary, key=lambda row: (-float(row["mean_score"]), str(row["algorithm"])))


def _case_family(case_key: str) -> str:
    if case_key.startswith("clean"):
        return "clean"
    if "miscentered" in case_key or "background" in case_key:
        return "perturbation"
    if "broadband" in case_key:
        return "model_mismatch"
    return "noise"


def _build_benchmark_study(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["algorithm"]), []).append(record)

    study_rows: list[dict[str, Any]] = []
    for algorithm, rows in grouped.items():
        clean_scores = [
            float(row["score"]) for row in rows if _case_family(str(row["case"])) == "clean"
        ]
        stress_scores = [
            float(row["score"]) for row in rows if _case_family(str(row["case"])) != "clean"
        ]
        failure_rate = float(
            np.mean(
                [
                    1.0
                    if (
                        not bool(row["converged"])
                        or float(row["convergence_improvement_ratio"]) < 0.05
                    )
                    else 0.0
                    for row in rows
                ]
            )
        )
        worst_case_row = min(rows, key=lambda row: float(row["score"]))
        clean_mean = (
            float(np.mean(clean_scores))
            if clean_scores
            else float(np.mean([float(row["score"]) for row in rows]))
        )
        stress_mean = float(np.mean(stress_scores)) if stress_scores else clean_mean
        study_rows.append(
            {
                "algorithm": algorithm,
                "clean_mean_score": clean_mean,
                "stress_mean_score": stress_mean,
                "robustness_drop": max(0.0, clean_mean - stress_mean),
                "failure_rate": failure_rate,
                "convergence_stability": float(
                    np.mean([float(row["convergence_monotonic_fraction"]) for row in rows])
                ),
                "worst_case": str(worst_case_row["case"]),
            }
        )

    return sorted(
        study_rows,
        key=lambda row: (float(row["failure_rate"]), -float(row["stress_mean_score"])),
    )


def _write_reports(summary: BenchmarkSummary, output_dir: Path) -> None:
    """Persist JSON, CSV, and Markdown benchmark reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_json = output_dir / "benchmark_results.json"
    summary_csv = output_dir / "benchmark_summary.csv"
    report_md = output_dir / "benchmark_report.md"
    leaderboard_png = output_dir / "benchmark_leaderboard.png"
    heatmap_png = output_dir / "benchmark_case_heatmap.png"
    study_json = output_dir / "benchmark_study.json"
    study_csv = output_dir / "benchmark_study.csv"

    summary.artifacts.update(
        {
            "results_json": str(results_json),
            "summary_csv": str(summary_csv),
            "report_markdown": str(report_md),
            "study_json": str(study_json),
            "study_csv": str(study_csv),
            "leaderboard_plot": str(leaderboard_png),
            "heatmap_plot": str(heatmap_png),
        }
    )

    results_json.write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm",
                "n_cases",
                "mean_score",
                "mean_ssim",
                "mean_phase_rms_error_rad",
                "mean_radial_profile_error",
                "mean_encircled_energy_error",
                "mean_elapsed_seconds",
                "converged_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(summary.aggregate)

    study_json.write_text(json.dumps(summary.study, indent=2), encoding="utf-8")
    with study_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm",
                "clean_mean_score",
                "stress_mean_score",
                "robustness_drop",
                "failure_rate",
                "convergence_stability",
                "worst_case",
            ],
        )
        writer.writeheader()
        writer.writerows(summary.study)

    report_md.write_text(summary.to_markdown(), encoding="utf-8")

    fig = plot_benchmark_leaderboard(summary.aggregate)
    save_figure(fig, leaderboard_png)
    import matplotlib.pyplot as plt

    plt.close(fig)

    fig = plot_benchmark_case_heatmap(summary.records, metric="score")
    save_figure(fig, heatmap_png)
    plt.close(fig)


def run_benchmark(
    *,
    algorithms: list[AlgorithmName] | None = None,
    cases: list[BenchmarkCase] | None = None,
    max_iterations: int = 80,
    beta: float = 0.9,
    random_seed: int = 42,
    output_dir: Path | None = None,
) -> BenchmarkSummary:
    """Run a deterministic synthetic benchmark across algorithms and cases."""
    selected_cases = cases or list(_DEFAULT_CASES.values())
    selected_algorithms = algorithms or default_benchmark_algorithms()

    records: list[dict[str, Any]] = []
    for case in selected_cases:
        dataset = generate_synthetic_psf(
            grid_size=case.grid_size,
            rms_aberration=case.rms_aberration,
            photon_count=case.photon_count,
            read_noise_std=case.read_noise_std,
            center_offset_pixels=case.center_offset_pixels,
            background_level=case.background_level,
            bandwidth_fraction=case.bandwidth_fraction,
            spectral_samples=case.spectral_samples,
            field_defocus_waves=case.field_defocus_waves,
            telescope=case.telescope,
            random_seed=case.seed,
        )
        support = dataset.pupil.amplitude > 0
        true_rms = compute_rms_phase(dataset.true_phase, support)

        for algorithm in selected_algorithms:
            cfg = AlgorithmConfig(
                name=algorithm,
                max_iterations=max_iterations,
                beta=beta,
                random_seed=random_seed,
            )
            if algorithm == AlgorithmName.PINN:
                cfg = cfg.model_copy(
                    update={
                        "pinn_hidden_features": 32,
                        "pinn_hidden_layers": 2,
                        "pinn_warm_start_iterations": min(20, max_iterations),
                    }
                )

            if cfg.n_starts > 1:
                result = multi_start_run(cfg, dataset.pupil, dataset.psf_data)
            else:
                result = AlgorithmRegistry.create(cfg, dataset.pupil).run(dataset.psf_data)

            convergence = summarise_convergence(result.cost_history)
            record = {
                "case": case.key,
                "description": case.description,
                "algorithm": algorithm.value,
                "grid_size": case.grid_size,
                "telescope": case.telescope.value,
                "rms_aberration_rad": case.rms_aberration,
                "true_phase_rms_rad": true_rms,
                "photon_count": case.photon_count,
                "read_noise_std": case.read_noise_std,
                "center_offset_pixels": [
                    float(case.center_offset_pixels[0]),
                    float(case.center_offset_pixels[1]),
                ],
                "background_level": case.background_level,
                "bandwidth_fraction": case.bandwidth_fraction,
                "spectral_samples": case.spectral_samples,
                "field_defocus_waves": case.field_defocus_waves,
                "ssim": compute_ssim(dataset.psf_data.image, result.reconstructed_psf),
                "phase_rms_error_rad": _phase_rms_error(dataset, result.recovered_phase),
                "phase_active_fraction": float(
                    np.mean(np.abs(result.recovered_phase[support]) > 1e-3)
                ),
                "radial_profile_error": compute_radial_profile_error(
                    dataset.psf_data.image,
                    result.reconstructed_psf,
                ),
                "encircled_energy_error": compute_encircled_energy_error(
                    dataset.psf_data.image,
                    result.reconstructed_psf,
                ),
                "strehl_ratio": result.strehl_ratio,
                "elapsed_seconds": result.elapsed_seconds,
                "n_iterations": result.n_iterations,
                "converged": result.converged,
                "convergence_improvement_ratio": convergence["improvement_ratio"],
                "convergence_monotonic_fraction": convergence["monotonic_fraction"],
            }
            record["score"] = _score_record(record)
            records.append(record)

    aggregate = _aggregate_records(records)
    study = _build_benchmark_study(records)
    summary = BenchmarkSummary(
        cases=selected_cases,
        records=records,
        aggregate=aggregate,
        study=study,
        output_dir=output_dir,
    )
    if output_dir is not None:
        _write_reports(summary, output_dir)
    return summary
