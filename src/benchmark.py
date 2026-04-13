"""Deterministic synthetic benchmark harness for phase-retrieval algorithms.

This module closes the gap between implementation quality and validation by
running a repeatable suite of synthetic test cases, aggregating metrics, and
exporting machine-readable and human-readable reports.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One deterministic synthetic benchmark scenario."""

    key: str
    description: str
    grid_size: int = 64
    rms_aberration: float = 0.5
    photon_count: float = 0.0
    read_noise_std: float = 0.0
    telescope: TelescopeType = TelescopeType.GENERIC_CIRCULAR
    seed: int = 42


@dataclass(slots=True)
class BenchmarkSummary:
    """Full benchmark output: raw per-case results plus aggregate ranking."""

    cases: list[BenchmarkCase]
    records: list[dict[str, Any]]
    aggregate: list[dict[str, Any]]
    output_dir: Path | None = None

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
                    "telescope": case.telescope.value,
                    "seed": case.seed,
                }
                for case in self.cases
            ],
            "records": self.records,
            "aggregate": self.aggregate,
        }

    def to_markdown(self) -> str:
        """Render a compact Markdown benchmark report."""
        lines = [
            "# Phase Retrieval Benchmark Report",
            "",
            "## Cases",
            "",
            "| Key | Description | Grid | RMS (rad) | Photons | Read noise | Telescope |",
            "|-----|-------------|-----:|----------:|--------:|-----------:|-----------|",
        ]
        for case in self.cases:
            lines.append(
                "| "
                f"{case.key} | {case.description} | {case.grid_size} | {case.rms_aberration:.2f} | "
                f"{case.photon_count:.0f} | {case.read_noise_std:.2e} | {case.telescope.value} |"
            )

        lines.extend(
            [
                "",
                "## Aggregate ranking",
                "",
                "| Rank | Algorithm | Score | SSIM | Phase RMS err | Radial err | EE err | Converged | Time (s) |",
                "|-----:|-----------|------:|-----:|--------------:|-----------:|-------:|----------:|---------:|",
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
    """Compute RMS phase error after piston removal over the true support."""
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
    recon_vals = recon_vals - recon_vals.mean()
    return float(np.sqrt(np.mean((recon_vals - truth_vals) ** 2)))


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


def _write_reports(summary: BenchmarkSummary, output_dir: Path) -> None:
    """Persist JSON, CSV, and Markdown benchmark reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "benchmark_results.json").write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )

    with (output_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
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

    (output_dir / "benchmark_report.md").write_text(summary.to_markdown(), encoding="utf-8")


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
                "ssim": compute_ssim(dataset.psf_data.image, result.reconstructed_psf),
                "phase_rms_error_rad": _phase_rms_error(dataset, result.recovered_phase),
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
    summary = BenchmarkSummary(cases=selected_cases, records=records, aggregate=aggregate, output_dir=output_dir)
    if output_dir is not None:
        _write_reports(summary, output_dir)
    return summary

