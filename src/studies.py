"""Reusable validation, benchmark, and sensitivity studies for notebooks and web flows."""

from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from src.models.config import AlgorithmConfig, PipelineConfig
from src.models.optics import PSFData, PupilModel
from src.pipeline import PipelineResult, RetrievalPipeline
from src.validation import compare_against_reference


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fieldnames})


def _metric_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "cv": float(std / mean) if abs(mean) > 1e-12 else 0.0,
    }


def _reference_summary(pipeline_result: PipelineResult) -> dict[str, Any]:
    return compare_against_reference(
        observed_psf=pipeline_result.psf_data.image,
        reconstructed_psf=pipeline_result.result.reconstructed_psf,
        pixel_scale_arcsec=pipeline_result.psf_data.pixel_scale_arcsec,
        telescope=pipeline_result.psf_data.telescope,
        detector=str(pipeline_result.psf_data.metadata.get("detector", "")),
        filter_name=pipeline_result.psf_data.filter_name,
    )


def _reference_pass(reference: dict[str, Any]) -> bool:
    """Return whether all available reference checks are strong."""
    if not reference:
        return False
    summary = reference.get("summary", {})
    if not isinstance(summary, dict) or not summary:
        return False
    return all(str(value) == "strong" for value in summary.values())


def _pipeline_record(
    pipeline_result: PipelineResult,
    *,
    source_name: str,
    group_label: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference = _reference_summary(pipeline_result)
    record = {
        "source_name": source_name,
        "group_label": group_label or "",
        "obs_id": pipeline_result.psf_data.obs_id,
        "telescope": pipeline_result.psf_data.telescope,
        "filter_name": pipeline_result.psf_data.filter_name,
        "detector": str(pipeline_result.psf_data.metadata.get("detector", "")),
        "algorithm": pipeline_result.algorithm_config.name.value,
        "random_seed": pipeline_result.algorithm_config.random_seed,
        "converged": pipeline_result.result.converged,
        "n_iterations": pipeline_result.result.n_iterations,
        "elapsed_seconds": pipeline_result.result.elapsed_seconds,
        "strehl_ratio": pipeline_result.result.strehl_ratio,
        "rms_phase_rad": pipeline_result.result.rms_phase_rad,
        "ssim": pipeline_result.ssim,
        "radial_profile_error": pipeline_result.radial_profile_error,
        "encircled_energy_error": pipeline_result.encircled_energy_error,
        "convergence_final_cost": pipeline_result.convergence_summary.get("final_cost", 0.0),
        "convergence_relative_drop": pipeline_result.convergence_summary.get("relative_drop", 0.0),
        "reference_available": bool(reference),
        "reference_pass": _reference_pass(reference),
        "reference_fwhm_observed": (
            float(reference["observed"]["fwhm_arcsec"]) if reference else None
        ),
        "reference_fwhm_reconstructed": (
            float(reference["reconstructed"]["fwhm_arcsec"]) if reference else None
        ),
        "reference_fwhm_relative_error": (
            float(reference["deviations"]["reconstructed_fwhm_error_arcsec"])
            if reference and "reconstructed_fwhm_error_arcsec" in reference.get("deviations", {})
            else None
        ),
        "reference_ee_relative_error": (
            float(reference["deviations"]["reconstructed_encircled_energy_error"])
            if reference
            and "reconstructed_encircled_energy_error" in reference.get("deviations", {})
            else None
        ),
    }
    if extra:
        record.update(extra)
    return record


def _consistency_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = (
        "strehl_ratio",
        "rms_phase_rad",
        "ssim",
        "radial_profile_error",
        "encircled_energy_error",
    )
    consistency: dict[str, Any] = {
        "n_records": len(records),
        "global": {},
        "by_filter": {},
    }
    for metric in metric_keys:
        values = [float(record[metric]) for record in records if record.get(metric) is not None]
        if values:
            consistency["global"][metric] = _metric_summary(values)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("filter_name", "unknown")), []).append(record)
    for filter_name, group in grouped.items():
        filter_summary: dict[str, Any] = {"n_records": len(group)}
        for metric in metric_keys:
            values = [float(record[metric]) for record in group if record.get(metric) is not None]
            if values:
                filter_summary[metric] = _metric_summary(values)
        consistency["by_filter"][filter_name] = filter_summary
    return consistency


def run_validation_campaign(
    file_paths: list[Path],
    *,
    pipeline_config: PipelineConfig,
    algorithm_config: AlgorithmConfig | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run a reproducible multi-observation validation campaign and save benchmark tables."""
    pipeline = RetrievalPipeline(pipeline_config)
    records: list[dict[str, Any]] = []

    for file_path in file_paths:
        run_output_dir = output_dir / file_path.stem if output_dir is not None else None
        pipeline_result = pipeline.run_from_file(
            file_path,
            algorithm_config=algorithm_config,
            output_dir=run_output_dir,
        )
        records.append(
            _pipeline_record(
                pipeline_result,
                source_name=file_path.name,
                group_label=file_path.parent.name,
                extra={"source_path": str(file_path)},
            )
        )

    completed = [record for record in records if bool(record["converged"])]
    consistency = _consistency_payload(records)
    summary = {
        "n_observations": len(records),
        "n_completed": len(completed),
        "success_rate": float(len(completed) / len(records)) if records else 0.0,
        "reference_coverage": int(sum(1 for record in records if record["reference_available"])),
        "reference_pass_rate": (
            float(sum(1 for record in records if record["reference_pass"]) / len(records))
            if records
            else 0.0
        ),
    }
    payload = {
        "summary": summary,
        "records": records,
        "consistency": consistency,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "validation_campaign.json", payload)
        _write_csv(output_dir / "validation_campaign.csv", records)
        _write_csv(output_dir / "benchmark_table.csv", records)
        _write_json(output_dir / "cross_observation_consistency.json", consistency)

    return payload


def run_seed_sensitivity_study(
    psf_data: PSFData,
    pupil: PupilModel,
    *,
    pipeline_config: PipelineConfig,
    algorithm_config: AlgorithmConfig,
    seeds: list[int],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Measure variability across random seeds for a fixed observation."""
    pipeline = RetrievalPipeline(pipeline_config)
    records: list[dict[str, Any]] = []

    for seed in seeds:
        run_cfg = AlgorithmConfig.model_validate(
            {**algorithm_config.model_dump(), "random_seed": seed}
        )
        run_output_dir = output_dir / f"seed_{seed}" if output_dir is not None else None
        pipeline_result = pipeline.run_from_psf(
            psf_data,
            pupil,
            algorithm_config=run_cfg,
            output_dir=run_output_dir,
        )
        records.append(
            _pipeline_record(
                pipeline_result,
                source_name=psf_data.obs_id,
                group_label="seed_sensitivity",
                extra={"seed": seed},
            )
        )

    summary = {
        "n_seeds": len(seeds),
        "strehl_ratio": _metric_summary([float(record["strehl_ratio"]) for record in records]),
        "rms_phase_rad": _metric_summary([float(record["rms_phase_rad"]) for record in records]),
        "ssim": _metric_summary([float(record["ssim"]) for record in records]),
        "convergence_final_cost": _metric_summary(
            [float(record["convergence_final_cost"]) for record in records]
        ),
    }
    payload = {"summary": summary, "records": records}

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "seed_sensitivity.json", payload)
        _write_csv(output_dir / "seed_sensitivity.csv", records)

    return payload


def run_noise_robustness_study(
    psf_data: PSFData,
    pupil: PupilModel,
    *,
    pipeline_config: PipelineConfig,
    algorithm_config: AlgorithmConfig,
    noise_sigma_fractions: list[float],
    repeats_per_level: int = 3,
    seed: int = 123,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Estimate noise robustness intervals by perturbing one observation across noise levels."""
    pipeline = RetrievalPipeline(pipeline_config)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []

    peak = float(np.max(psf_data.image))
    for noise_fraction in noise_sigma_fractions:
        for repeat_idx in range(repeats_per_level):
            noisy = np.clip(
                psf_data.image + rng.normal(scale=noise_fraction * peak, size=psf_data.image.shape),
                0.0,
                None,
            )
            total = float(np.sum(noisy))
            if total > 0:
                noisy /= total
            noisy_psf = PSFData(
                image=noisy,
                pixel_scale_arcsec=psf_data.pixel_scale_arcsec,
                wavelength_m=psf_data.wavelength_m,
                filter_name=psf_data.filter_name,
                telescope=psf_data.telescope,
                obs_id=psf_data.obs_id,
                metadata={
                    **psf_data.metadata,
                    "noise_sigma_fraction": noise_fraction,
                    "noise_repeat_index": repeat_idx,
                },
            )
            run_output_dir = (
                output_dir / f"noise_{noise_fraction:.4f}" / f"repeat_{repeat_idx}"
                if output_dir is not None
                else None
            )
            pipeline_result = pipeline.run_from_psf(
                noisy_psf,
                pupil,
                algorithm_config=algorithm_config,
                output_dir=run_output_dir,
            )
            records.append(
                _pipeline_record(
                    pipeline_result,
                    source_name=psf_data.obs_id,
                    group_label=f"noise_{noise_fraction:.4f}",
                    extra={
                        "noise_sigma_fraction": noise_fraction,
                        "repeat_index": repeat_idx,
                    },
                )
            )

    intervals: dict[str, dict[str, Any]] = {}
    for noise_fraction in noise_sigma_fractions:
        group = [
            record
            for record in records
            if abs(float(record["noise_sigma_fraction"]) - noise_fraction) < 1e-12
        ]
        intervals[f"{noise_fraction:.4f}"] = {
            "n_runs": len(group),
            "strehl_ratio": _metric_summary([float(record["strehl_ratio"]) for record in group]),
            "rms_phase_rad": _metric_summary([float(record["rms_phase_rad"]) for record in group]),
            "ssim": _metric_summary([float(record["ssim"]) for record in group]),
        }

    payload = {
        "summary": {
            "n_noise_levels": len(noise_sigma_fractions),
            "repeats_per_level": repeats_per_level,
            "noise_intervals": intervals,
        },
        "records": records,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "noise_robustness.json", payload)
        _write_csv(output_dir / "noise_robustness.csv", records)

    return payload


def run_parameter_sweep(
    psf_data: PSFData,
    pupil: PupilModel,
    *,
    pipeline_config: PipelineConfig,
    algorithm_config: AlgorithmConfig,
    sweep_parameters: dict[str, list[Any]],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run a grid search over algorithm parameters and save a reproducible table."""
    pipeline = RetrievalPipeline(pipeline_config)
    keys = sorted(sweep_parameters)
    combinations = [
        dict(zip(keys, values, strict=True))
        for values in product(*(sweep_parameters[key] for key in keys))
    ]
    records: list[dict[str, Any]] = []

    for idx, updates in enumerate(combinations):
        candidate = AlgorithmConfig.model_validate({**algorithm_config.model_dump(), **updates})
        run_output_dir = output_dir / f"combo_{idx:03d}" if output_dir is not None else None
        pipeline_result = pipeline.run_from_psf(
            psf_data,
            pupil,
            algorithm_config=candidate,
            output_dir=run_output_dir,
        )
        records.append(
            _pipeline_record(
                pipeline_result,
                source_name=psf_data.obs_id,
                group_label="parameter_sweep",
                extra=updates,
            )
        )

    best_record = (
        max(records, key=lambda record: float(record["strehl_ratio"])) if records else None
    )
    payload = {
        "summary": {
            "n_combinations": len(combinations),
            "parameters": {key: list(values) for key, values in sweep_parameters.items()},
            "best_by_strehl": best_record,
        },
        "records": records,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "parameter_sweep.json", payload)
        _write_csv(output_dir / "parameter_sweep.csv", records)

    return payload
