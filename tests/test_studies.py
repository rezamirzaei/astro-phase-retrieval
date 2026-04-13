"""Tests for reusable validation and sensitivity studies."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models.config import AlgorithmConfig, AlgorithmName, default_hst_config
from src.pipeline import RetrievalPipeline
from src.studies import (
    run_noise_robustness_study,
    run_parameter_sweep,
    run_seed_sensitivity_study,
    run_validation_campaign,
)


def _gaussian_psf(grid_size: int, offset: float = 0.0) -> np.ndarray:
    coords = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(coords, coords)
    image = np.exp(-((xx - offset) ** 2 + (yy + offset) ** 2) / 0.05)
    return image / np.sum(image)


def test_validation_campaign_writes_reproducible_tables(tmp_path: Path) -> None:
    file_paths: list[Path] = []
    for idx, offset in enumerate((0.0, 0.06)):
        path = tmp_path / f"campaign_{idx}.npy"
        np.save(path, _gaussian_psf(64, offset=offset))
        file_paths.append(path)

    pipeline_config = default_hst_config().model_copy(
        update={
            "pupil": default_hst_config().pupil.model_copy(update={"grid_size": 64}),
            "output_dir": tmp_path / "campaign_outputs",
        }
    )
    algorithm_config = AlgorithmConfig(
        name=AlgorithmName.ERROR_REDUCTION,
        max_iterations=4,
        random_seed=7,
    )

    payload = run_validation_campaign(
        file_paths,
        pipeline_config=pipeline_config,
        algorithm_config=algorithm_config,
        output_dir=tmp_path / "campaign_outputs",
    )

    assert payload["summary"]["n_observations"] == 2
    assert len(payload["records"]) == 2
    assert (tmp_path / "campaign_outputs" / "validation_campaign.json").exists()
    assert (tmp_path / "campaign_outputs" / "benchmark_table.csv").exists()
    assert (tmp_path / "campaign_outputs" / "cross_observation_consistency.json").exists()


def test_sensitivity_studies_produce_json_and_csv(tmp_path: Path) -> None:
    npy_path = tmp_path / "study.npy"
    np.save(npy_path, _gaussian_psf(64))

    pipeline_config = default_hst_config().model_copy(
        update={"pupil": default_hst_config().pupil.model_copy(update={"grid_size": 64})}
    )
    pipeline = RetrievalPipeline(pipeline_config)
    psf_data, pupil = pipeline.load_inputs_from_file(npy_path)
    algorithm_config = AlgorithmConfig(
        name=AlgorithmName.ERROR_REDUCTION,
        max_iterations=4,
        random_seed=11,
    )

    seed_payload = run_seed_sensitivity_study(
        psf_data,
        pupil,
        pipeline_config=pipeline_config,
        algorithm_config=algorithm_config,
        seeds=[11, 13],
        output_dir=tmp_path / "seed",
    )
    noise_payload = run_noise_robustness_study(
        psf_data,
        pupil,
        pipeline_config=pipeline_config,
        algorithm_config=algorithm_config,
        noise_sigma_fractions=[0.0, 0.01],
        repeats_per_level=2,
        output_dir=tmp_path / "noise",
    )
    sweep_payload = run_parameter_sweep(
        psf_data,
        pupil,
        pipeline_config=pipeline_config,
        algorithm_config=algorithm_config,
        sweep_parameters={"beta": [0.8, 0.9], "max_iterations": [3, 4]},
        output_dir=tmp_path / "sweep",
    )

    assert seed_payload["summary"]["n_seeds"] == 2
    assert noise_payload["summary"]["n_noise_levels"] == 2
    assert sweep_payload["summary"]["n_combinations"] == 4
    assert (tmp_path / "seed" / "seed_sensitivity.json").exists()
    assert (tmp_path / "noise" / "noise_robustness.csv").exists()
    assert (tmp_path / "sweep" / "parameter_sweep.csv").exists()
