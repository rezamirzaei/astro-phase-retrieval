"""Uncertainty estimation helpers for phase-retrieval runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift as ndimage_shift  # type: ignore[import-untyped]

from src.algorithms.multi_start import multi_start_run
from src.algorithms.registry import AlgorithmRegistry
from src.metrics.quality import (
    compute_encircled_energy_error,
    compute_radial_profile_error,
    compute_ssim,
)
from src.models.config import AlgorithmConfig
from src.models.optics import PSFData, PupilModel


@dataclass(slots=True)
class UncertaintyResult:
    """Summary of perturbation-based uncertainty estimation."""

    n_samples: int
    sample_records: list[dict[str, float]]
    summary: dict[str, dict[str, float | list[float]]]


def _normalise_image(image: np.ndarray) -> np.ndarray:
    total = float(image.sum())
    if total > 0:
        image = image / total
    return image


def _perturb_psf_image(
    image: np.ndarray,
    *,
    rng: np.random.Generator,
    shift_sigma_pixels: float,
    background_sigma_fraction: float,
    noise_sigma_fraction: float,
) -> np.ndarray:
    """Apply small physically motivated perturbations to an observed PSF."""
    peak = max(float(np.max(image)), 1e-12)
    perturbed = image.astype(np.float64, copy=True)

    if shift_sigma_pixels > 0:
        shift = rng.normal(0.0, shift_sigma_pixels, size=2)
        perturbed = ndimage_shift(
            perturbed,
            shift=tuple(float(v) for v in shift),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

    if background_sigma_fraction > 0:
        perturbed = perturbed + rng.normal(0.0, background_sigma_fraction * peak)

    if noise_sigma_fraction > 0:
        perturbed = perturbed + rng.normal(0.0, noise_sigma_fraction * peak, size=perturbed.shape)

    perturbed = np.maximum(perturbed, 0.0)
    return _normalise_image(perturbed)


def _summarise_metric(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": [0.0, 0.0]}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
    }


def run_uncertainty_analysis(
    *,
    psf_data: PSFData,
    pupil: PupilModel,
    algorithm_config: AlgorithmConfig,
    n_samples: int,
    shift_sigma_pixels: float,
    background_sigma_fraction: float,
    noise_sigma_fraction: float,
    seed: int,
) -> UncertaintyResult:
    """Run perturbation-based uncertainty analysis around one observed PSF."""
    if n_samples <= 0:
        return UncertaintyResult(n_samples=0, sample_records=[], summary={})

    rng = np.random.default_rng(seed)
    records: list[dict[str, float]] = []
    for sample_idx in range(n_samples):
        perturbed_image = _perturb_psf_image(
            psf_data.image,
            rng=rng,
            shift_sigma_pixels=shift_sigma_pixels,
            background_sigma_fraction=background_sigma_fraction,
            noise_sigma_fraction=noise_sigma_fraction,
        )
        perturbed_psf = PSFData(
            image=perturbed_image,
            pixel_scale_arcsec=psf_data.pixel_scale_arcsec,
            wavelength_m=psf_data.wavelength_m,
            filter_name=psf_data.filter_name,
            telescope=psf_data.telescope,
            obs_id=psf_data.obs_id,
            metadata={**psf_data.metadata, "uncertainty_sample": sample_idx},
        )
        sample_cfg = algorithm_config.model_copy(
            update={
                "random_seed": None
                if algorithm_config.random_seed is None
                else int(algorithm_config.random_seed) + sample_idx + 1
            }
        )
        if sample_cfg.n_starts > 1:
            result = multi_start_run(sample_cfg, pupil, perturbed_psf)
        else:
            result = AlgorithmRegistry.create(sample_cfg, pupil).run(perturbed_psf)

        records.append(
            {
                "strehl_ratio": float(result.strehl_ratio),
                "rms_phase_rad": float(result.rms_phase_rad),
                "ssim": float(compute_ssim(perturbed_psf.image, result.reconstructed_psf)),
                "radial_profile_error": float(
                    compute_radial_profile_error(perturbed_psf.image, result.reconstructed_psf)
                ),
                "encircled_energy_error": float(
                    compute_encircled_energy_error(perturbed_psf.image, result.reconstructed_psf)
                ),
                "elapsed_seconds": float(result.elapsed_seconds),
            }
        )

    summary = {
        metric: _summarise_metric([row[metric] for row in records])
        for metric in (
            "strehl_ratio",
            "rms_phase_rad",
            "ssim",
            "radial_profile_error",
            "encircled_energy_error",
            "elapsed_seconds",
        )
    }
    summary["config"] = {
        "shift_sigma_pixels": float(shift_sigma_pixels),
        "background_sigma_fraction": float(background_sigma_fraction),
        "noise_sigma_fraction": float(noise_sigma_fraction),
        "seed": float(seed),
    }
    return UncertaintyResult(n_samples=n_samples, sample_records=records, summary=summary)
