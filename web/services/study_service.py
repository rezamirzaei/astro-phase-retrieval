"""Validation-study helpers for the web application."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.models.config import (
    AlgorithmConfig,
    AlgorithmName,
    BetaSchedule,
    NoiseModel,
    Regulariser,
    default_hst_config,
)
from src.studies import run_validation_campaign
from web.config import settings
from web.services.data_service import list_fits_files, resolve_fits_path


def _default_validation_files() -> list[Path]:
    return [
        resolve_fits_path(str(file_info["filename"]))
        for file_info in list_fits_files()
        if str(file_info["filename"]).endswith(".fits")
    ]


def run_web_validation_campaign(
    *,
    fits_filenames: list[str] | None,
    algorithm: AlgorithmName,
    max_iterations: int,
    tolerance: float,
    beta: float,
    beta_schedule: BetaSchedule,
    momentum: float,
    tv_weight: float,
    noise_model: NoiseModel,
    n_starts: int,
    uncertainty_samples: int,
    admm_rho: float,
    wf_step_size: float,
    wf_spectral_init: bool,
    spectral_init: bool,
    regulariser: Regulariser,
    proximal_weight: float,
    sparsity_threshold: float,
    sparsity_keep_fraction: float,
    grid_size: int,
) -> dict[str, Any]:
    """Run a multi-observation validation campaign and persist its artifacts."""
    selected_paths = (
        [resolve_fits_path(filename) for filename in fits_filenames]
        if fits_filenames
        else _default_validation_files()
    )
    if not selected_paths:
        raise ValueError("No FITS observations are available for validation")

    campaign_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = settings.output_dir / "validation_campaigns" / campaign_id

    config = default_hst_config()
    pipeline_config = config.model_copy(
        update={
            "data": config.data.model_copy(update={"cutout_size": grid_size}),
            "pupil": config.pupil.model_copy(update={"grid_size": grid_size}),
            "output_dir": output_dir,
            "uncertainty_samples": uncertainty_samples,
        }
    )
    algorithm_config = AlgorithmConfig(
        name=algorithm,
        max_iterations=max_iterations,
        tolerance=tolerance,
        beta=beta,
        beta_schedule=beta_schedule,
        momentum=momentum,
        tv_weight=tv_weight,
        noise_model=noise_model,
        n_starts=n_starts,
        admm_rho=admm_rho,
        wf_step_size=wf_step_size,
        wf_spectral_init=wf_spectral_init,
        spectral_init=spectral_init,
        regulariser=regulariser,
        proximal_weight=proximal_weight,
        sparsity_threshold=sparsity_threshold,
        sparsity_keep_fraction=sparsity_keep_fraction,
        random_seed=42,
    )
    payload = run_validation_campaign(
        selected_paths,
        pipeline_config=pipeline_config,
        algorithm_config=algorithm_config,
        output_dir=output_dir,
    )
    return {
        "campaign_id": campaign_id,
        "selected_files": [path.name for path in selected_paths],
        "summary": payload["summary"],
        "records": payload["records"],
        "consistency": payload["consistency"],
        "artifacts": sorted(path.name for path in output_dir.glob("*") if path.is_file()),
    }
