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


def _write_markdown(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


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
        "baseline_key": str(reference.get("baseline", {}).get("key", "")) if reference else "",
        "reference_fwhm_agreement": (
            str(reference.get("summary", {}).get("fwhm_agreement", "n/a")) if reference else "n/a"
        ),
        "reference_encircled_energy_agreement": (
            str(reference.get("summary", {}).get("encircled_energy_agreement", "n/a"))
            if reference
            else "n/a"
        ),
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


def _agreement_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = {"strong": 0, "weak": 0, "n/a": 0}
    for record in records:
        value = str(record.get(key, "n/a"))
        counts[value if value in counts else "n/a"] += 1
    return counts


def _reference_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_baseline: dict[str, dict[str, Any]] = {}
    weak_cases: list[dict[str, Any]] = []
    filters_without_reference = sorted(
        {
            str(record.get("filter_name", "unknown"))
            for record in records
            if not bool(record.get("reference_available"))
        }
    )

    for record in records:
        if not bool(record.get("reference_available")):
            continue
        baseline_key = str(record.get("baseline_key") or "unknown")
        entry = by_baseline.setdefault(
            baseline_key,
            {
                "n_records": 0,
                "n_pass": 0,
                "filters": set(),
                "fwhm_agreement": {"strong": 0, "weak": 0, "n/a": 0},
                "encircled_energy_agreement": {"strong": 0, "weak": 0, "n/a": 0},
            },
        )
        entry["n_records"] += 1
        entry["n_pass"] += int(bool(record.get("reference_pass")))
        entry["filters"].add(str(record.get("filter_name", "unknown")))
        entry["fwhm_agreement"][str(record.get("reference_fwhm_agreement", "n/a"))] += 1
        entry["encircled_energy_agreement"][
            str(record.get("reference_encircled_energy_agreement", "n/a"))
        ] += 1
        if not bool(record.get("reference_pass")):
            weak_cases.append(
                {
                    "source_name": record.get("source_name"),
                    "filter_name": record.get("filter_name"),
                    "baseline_key": baseline_key,
                    "fwhm_agreement": record.get("reference_fwhm_agreement"),
                    "encircled_energy_agreement": record.get(
                        "reference_encircled_energy_agreement"
                    ),
                }
            )

    final_by_baseline: dict[str, dict[str, Any]] = {}
    for baseline_key, entry in by_baseline.items():
        n_records = int(entry["n_records"])
        final_by_baseline[baseline_key] = {
            "n_records": n_records,
            "pass_rate": float(entry["n_pass"] / n_records) if n_records else 0.0,
            "filters": sorted(entry["filters"]),
            "fwhm_agreement": entry["fwhm_agreement"],
            "encircled_energy_agreement": entry["encircled_energy_agreement"],
        }

    return {
        "baseline_keys": sorted(final_by_baseline),
        "filters_without_reference": filters_without_reference,
        "fwhm_agreement": _agreement_counts(records, "reference_fwhm_agreement"),
        "encircled_energy_agreement": _agreement_counts(
            records, "reference_encircled_energy_agreement"
        ),
        "by_baseline": final_by_baseline,
        "weak_cases": weak_cases,
    }


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
        "by_baseline": {},
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

    baseline_groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        baseline_key = str(record.get("baseline_key", ""))
        if baseline_key:
            baseline_groups.setdefault(baseline_key, []).append(record)
    for baseline_key, group in baseline_groups.items():
        baseline_summary: dict[str, Any] = {"n_records": len(group)}
        for metric in metric_keys:
            values = [float(record[metric]) for record in group if record.get(metric) is not None]
            if values:
                baseline_summary[metric] = _metric_summary(values)
        consistency["by_baseline"][baseline_key] = baseline_summary
    return consistency


def render_validation_campaign_markdown(payload: dict[str, Any]) -> str:
    """Render a concise markdown summary for a validation campaign."""
    summary = payload["summary"]
    reference = payload["reference_summary"]
    filters_covered = ", ".join(summary["filters_covered"]) or "none"
    filters_without_reference = ", ".join(summary["filters_without_reference"]) or "none"
    lines = [
        "# Validation Campaign Report",
        "",
        "## Summary",
        "",
        f"- Observations: **{summary['n_observations']}**",
        f"- Completed: **{summary['n_completed']}**",
        f"- Success rate: **{summary['success_rate']:.1%}**",
        f"- Reference coverage: **{summary['reference_coverage']}**",
        f"- Reference pass rate: **{summary['reference_pass_rate']:.1%}**",
        f"- Filters covered: {filters_covered}",
        f"- Filters without curated baseline: {filters_without_reference}",
        "",
        "## Agreement Breakdown",
        "",
        f"- FWHM agreement: `{reference['fwhm_agreement']}`",
        f"- Encircled-energy agreement: `{reference['encircled_energy_agreement']}`",
        "",
        "## Baseline Coverage",
        "",
    ]
    if reference["by_baseline"]:
        for baseline_key, entry in reference["by_baseline"].items():
            filters = ", ".join(entry["filters"])
            lines.append(
                f"- `{baseline_key}` — {entry['n_records']} record(s), "
                f"pass rate {entry['pass_rate']:.1%}, filters: {filters}"
            )
    else:
        lines.append("- No curated external baselines were matched in this campaign.")
    lines.extend(["", "## Cases Requiring Review", ""])
    if reference["weak_cases"]:
        lines.extend(
            [
                (
                    f"- `{case['source_name']}` ({case['filter_name']}) against "
                    f"`{case['baseline_key']}` — FWHM: `{case['fwhm_agreement']}`, "
                    f"EE: `{case['encircled_energy_agreement']}`"
                )
                for case in reference["weak_cases"]
            ]
        )
    else:
        lines.append("- No weak reference-agreement cases were recorded.")
    return "\n".join(lines)


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
    reference_summary = _reference_payload(records)
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
        "filters_covered": sorted(
            {str(record.get("filter_name", "unknown")) for record in records}
        ),
        "filters_without_reference": reference_summary["filters_without_reference"],
        "baseline_keys": reference_summary["baseline_keys"],
    }
    payload = {
        "summary": summary,
        "records": records,
        "consistency": consistency,
        "reference_summary": reference_summary,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "validation_campaign.json", payload)
        _write_csv(output_dir / "validation_campaign.csv", records)
        _write_csv(output_dir / "benchmark_table.csv", records)
        _write_json(output_dir / "cross_observation_consistency.json", consistency)
        _write_json(output_dir / "reference_summary.json", reference_summary)
        _write_markdown(
            output_dir / "validation_campaign.md",
            render_validation_campaign_markdown(payload),
        )

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
