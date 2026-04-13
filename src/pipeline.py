"""Pipeline orchestrator — composable, reusable phase-retrieval workflow.

Provides a single entry-point that composes:

    data loading → algorithm execution → metric computation → visualization → logging

Both the CLI (``src.cli``) and the web service (``web.services.algorithm_service``)
delegate to this module, eliminating duplicated orchestration logic.

Usage
-----
    from src.models.config import default_hst_config
    from src.pipeline import RetrievalPipeline

    config = default_hst_config()
    pipeline = RetrievalPipeline(config)
    result = pipeline.run_from_file(Path("data/psf.fits"))
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.algorithms.multi_start import multi_start_run
from src.algorithms.registry import AlgorithmRegistry
from src.metrics.quality import (
    compute_encircled_energy_error,
    compute_radial_profile_error,
    compute_ssim,
    summarise_convergence,
    zernike_decomposition,
)
from src.models.config import AlgorithmConfig, PipelineConfig
from src.models.optics import PhaseRetrievalResult, PSFData, PupilModel
from src.optics.pupils import build_pupil
from src.reporting import build_evaluation_payload, write_evaluation_report
from src.uncertainty import run_uncertainty_analysis

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Aggregated output from a pipeline run."""

    result: PhaseRetrievalResult
    psf_data: PSFData
    pupil: PupilModel
    config: PipelineConfig
    algorithm_config: AlgorithmConfig
    zernike_coefficients: dict[int, float] = field(default_factory=dict)
    ssim: float = 0.0
    radial_profile_error: float = 0.0
    encircled_energy_error: float = 0.0
    convergence_summary: dict[str, float] = field(default_factory=dict)
    uncertainty_summary: dict[str, Any] = field(default_factory=dict)
    plots_generated: list[str] = field(default_factory=list)
    output_dir: Path | None = None


class RetrievalPipeline:
    """Composable phase-retrieval pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Full pipeline configuration (data, pupil, algorithm, output).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run_from_psf(
        self,
        psf_data: PSFData,
        pupil: PupilModel,
        algorithm_config: AlgorithmConfig | None = None,
        output_dir: Path | None = None,
    ) -> PipelineResult:
        """Run the full pipeline given pre-loaded PSF data and pupil.

        Parameters
        ----------
        psf_data : PSFData
            Validated PSF container.
        pupil : PupilModel
            Telescope pupil model.
        algorithm_config : AlgorithmConfig | None
            Override the pipeline's default algorithm config.
        output_dir : Path | None
            Override the pipeline's default output directory.

        Returns
        -------
        PipelineResult
            Complete pipeline output including metrics and plots.
        """
        alg_cfg = algorithm_config or self.config.algorithm
        out_dir = output_dir or self.config.output_dir

        # ── Execute algorithm ─────────────────────────────────────────
        t0 = time.perf_counter()

        if alg_cfg.n_starts > 1:
            result = multi_start_run(alg_cfg, pupil, psf_data)
        else:
            retriever = AlgorithmRegistry.create(alg_cfg, pupil)
            result = retriever.run(psf_data)

        elapsed_total = time.perf_counter() - t0

        # ── Compute additional metrics ────────────────────────────────
        support = pupil.amplitude > 0
        zc = zernike_decomposition(result.recovered_phase, support)
        ssim = compute_ssim(psf_data.image, result.reconstructed_psf)
        radial_error = compute_radial_profile_error(psf_data.image, result.reconstructed_psf)
        encircled_energy_error = compute_encircled_energy_error(
            psf_data.image,
            result.reconstructed_psf,
        )
        convergence_summary = summarise_convergence(result.cost_history)

        logger.info(
            "Pipeline completed: alg=%s strehl=%.4f rms=%.4f ssim=%.4f iter=%d time=%.2fs",
            alg_cfg.name.value,
            result.strehl_ratio,
            result.rms_phase_rad,
            ssim,
            result.n_iterations,
            elapsed_total,
        )

        pipeline_result = PipelineResult(
            result=result,
            psf_data=psf_data,
            pupil=pupil,
            config=self.config,
            algorithm_config=alg_cfg,
            zernike_coefficients=zc,
            ssim=ssim,
            radial_profile_error=radial_error,
            encircled_energy_error=encircled_energy_error,
            convergence_summary=convergence_summary,
            output_dir=out_dir,
        )

        if self.config.uncertainty_samples > 0:
            uncertainty = run_uncertainty_analysis(
                psf_data=psf_data,
                pupil=pupil,
                algorithm_config=alg_cfg,
                n_samples=self.config.uncertainty_samples,
                shift_sigma_pixels=self.config.uncertainty_shift_sigma_pixels,
                background_sigma_fraction=self.config.uncertainty_background_sigma_fraction,
                noise_sigma_fraction=self.config.uncertainty_noise_sigma_fraction,
                seed=self.config.uncertainty_seed,
            )
            pipeline_result.uncertainty_summary = {
                "n_samples": uncertainty.n_samples,
                "summary": uncertainty.summary,
                "sample_records": uncertainty.sample_records,
            }

        # ── Persist config + results ──────────────────────────────────
        if out_dir is not None:
            self._save_outputs(pipeline_result, out_dir)

        return pipeline_result

    def run_from_file(
        self,
        fits_path: Path,
        algorithm_config: AlgorithmConfig | None = None,
        output_dir: Path | None = None,
    ) -> PipelineResult:
        """Run the full pipeline from a FITS or .npy file path.

        Handles PSF loading, pupil construction, grid synchronisation,
        algorithm execution, and output generation.
        """
        from src.data.loader import load_psf_from_fits, prepare_psf_for_retrieval

        alg_cfg = algorithm_config or self.config.algorithm
        out_dir = output_dir or self.config.output_dir
        grid_size = self.config.pupil.grid_size

        # ── Load PSF ──────────────────────────────────────────────────
        if fits_path.suffix == ".npy":
            psf_image = np.load(str(fits_path)).astype(np.float64)
            original_shape = list(psf_image.shape)
            total = float(psf_image.sum())
            if total > 0:
                psf_image /= total
            if psf_image.shape[0] != grid_size:
                tmp = PSFData(
                    image=psf_image,
                    pixel_scale_arcsec=self.config.pupil.pixel_scale_arcsec,
                    wavelength_m=self.config.pupil.wavelength_m,
                    filter_name="SYNTH",
                    telescope="synthetic",
                    obs_id=fits_path.stem,
                    metadata={
                        "source_kind": "npy",
                        "source_path": str(fits_path),
                        "source_filename": fits_path.name,
                        "original_shape": original_shape,
                        "preprocessing": ["unit_sum_normalisation", "prepare_psf_for_retrieval"],
                    },
                )
                psf_image = prepare_psf_for_retrieval(tmp, grid_size)
            psf_data = PSFData(
                image=psf_image,
                pixel_scale_arcsec=self.config.pupil.pixel_scale_arcsec,
                wavelength_m=self.config.pupil.wavelength_m,
                filter_name="SYNTH",
                telescope="synthetic",
                obs_id=fits_path.stem,
                metadata={
                    "source_kind": "npy",
                    "source_path": str(fits_path),
                    "source_filename": fits_path.name,
                    "original_shape": original_shape,
                    "prepared_grid_size": grid_size,
                    "preprocessing": ["unit_sum_normalisation", "prepare_psf_for_retrieval"],
                },
            )
        else:
            psf_data = load_psf_from_fits(fits_path, self.config.data, self.config.pupil)
            psf_image = prepare_psf_for_retrieval(psf_data, grid_size)
            psf_data = PSFData(
                image=psf_image,
                pixel_scale_arcsec=psf_data.pixel_scale_arcsec,
                wavelength_m=psf_data.wavelength_m,
                filter_name=psf_data.filter_name,
                telescope=psf_data.telescope,
                obs_id=psf_data.obs_id,
                metadata={
                    **psf_data.metadata,
                    "prepared_grid_size": grid_size,
                    "prepared_shape": [int(psf_image.shape[0]), int(psf_image.shape[1])],
                },
            )

        # ── Sync pupil grid to actual image size ──────────────────────
        actual_grid = psf_data.image.shape[0]
        pupil_cfg = self.config.pupil
        if actual_grid != pupil_cfg.grid_size:
            logger.warning(
                "PSF grid %dx%d != pupil grid %d; rebuilding pupil.",
                actual_grid,
                actual_grid,
                pupil_cfg.grid_size,
            )
            pupil_cfg = pupil_cfg.model_copy(update={"grid_size": actual_grid})
        pupil = build_pupil(pupil_cfg)

        return self.run_from_psf(psf_data, pupil, alg_cfg, out_dir)

    @staticmethod
    def _save_outputs(pipeline_result: PipelineResult, out_dir: Path) -> None:
        """Persist config snapshot and result summary to disk."""
        out_dir.mkdir(parents=True, exist_ok=True)

        r = pipeline_result.result
        cfg = pipeline_result.algorithm_config

        # Config snapshot
        config_data = {
            "algorithm": cfg.name.value,
            "max_iterations": cfg.max_iterations,
            "beta": cfg.beta,
            "beta_schedule": cfg.beta_schedule.value,
            "momentum": cfg.momentum,
            "tv_weight": cfg.tv_weight,
            "noise_model": cfg.noise_model.value,
            "n_starts": cfg.n_starts,
            "random_seed": cfg.random_seed,
            "uncertainty_samples": pipeline_result.config.uncertainty_samples,
        }
        (out_dir / "config.json").write_text(json.dumps(config_data, indent=2))

        # Result summary
        summary = {
            "algorithm": r.algorithm.value,
            "n_iterations": r.n_iterations,
            "converged": r.converged,
            "strehl_ratio": r.strehl_ratio,
            "rms_phase_rad": r.rms_phase_rad,
            "ssim": pipeline_result.ssim,
            "radial_profile_error": pipeline_result.radial_profile_error,
            "encircled_energy_error": pipeline_result.encircled_energy_error,
            "elapsed_seconds": r.elapsed_seconds,
            "timestamp": r.timestamp.isoformat(),
        }
        (out_dir / "result.json").write_text(json.dumps(summary, indent=2))

        metrics = {
            "ssim": pipeline_result.ssim,
            "radial_profile_error": pipeline_result.radial_profile_error,
            "encircled_energy_error": pipeline_result.encircled_energy_error,
            "convergence": pipeline_result.convergence_summary,
            "zernike_coefficients": pipeline_result.zernike_coefficients,
        }
        if pipeline_result.uncertainty_summary:
            metrics["uncertainty"] = pipeline_result.uncertainty_summary
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        if pipeline_result.uncertainty_summary:
            (out_dir / "uncertainty.json").write_text(
                json.dumps(pipeline_result.uncertainty_summary, indent=2)
            )

        provenance = {
            "psf": pipeline_result.psf_data.metadata,
            "algorithm": {
                "name": cfg.name.value,
                "max_iterations": cfg.max_iterations,
                "beta": cfg.beta,
                "beta_schedule": cfg.beta_schedule.value,
                "momentum": cfg.momentum,
                "tv_weight": cfg.tv_weight,
                "noise_model": cfg.noise_model.value,
                "n_starts": cfg.n_starts,
                "random_seed": cfg.random_seed,
            },
            "pupil": {
                "grid_size": pipeline_result.pupil.grid_size,
                "support_pixels": int(np.sum(pipeline_result.pupil.amplitude > 0)),
                "approximate_model": True,
                "wavelength_m": pipeline_result.pupil.wavelength_m,
                "bandwidth_fraction": pipeline_result.pupil.bandwidth_fraction,
                "spectral_samples": pipeline_result.pupil.spectral_samples,
                "field_defocus_waves": pipeline_result.pupil.field_defocus_waves,
                "detector_sigma_pixels": pipeline_result.pupil.detector_sigma_pixels,
                "jitter_sigma_pixels": pipeline_result.pupil.jitter_sigma_pixels,
                "pixel_integration_width": pipeline_result.pupil.pixel_integration_width,
            },
        }
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

        evaluation_payload = build_evaluation_payload(
            psf_metadata={
                **pipeline_result.psf_data.metadata,
                "obs_id": pipeline_result.psf_data.obs_id,
            },
            algorithm_name=cfg.name.value,
            algorithm_config=config_data,
            pupil_summary=provenance["pupil"],
            metrics={
                **summary,
                "convergence": pipeline_result.convergence_summary,
                "uncertainty": pipeline_result.uncertainty_summary,
            },
            zernike_coefficients=pipeline_result.zernike_coefficients,
        )
        write_evaluation_report(evaluation_payload, out_dir)

        logger.info("Saved outputs to %s", out_dir)
