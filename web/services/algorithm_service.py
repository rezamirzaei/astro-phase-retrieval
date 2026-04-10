"""Algorithm execution service — run, compare, and persist results."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sqlalchemy.orm import Session

from src.algorithms.registry import AlgorithmRegistry
from src.data.loader import load_psf_from_fits, prepare_psf_for_retrieval
from src.metrics.quality import zernike_decomposition
from src.models.config import (
    AlgorithmConfig,
    AlgorithmName,
    BetaSchedule,
    DataConfig,
    NoiseModel,
    PupilConfig,
    TelescopeType,
)
from src.models.optics import PSFData, PupilModel
from src.optics.pupils import build_pupil
from src.visualization.plots import (
    plot_algorithm_comparison,
    plot_convergence,
    plot_psf_comparison,
    plot_radial_profile,
    plot_recovered_phase,
    plot_strehl_rms_bar,
    plot_summary,
    save_figure,
)
from web.config import settings
from web.models import Job
from web.schemas import AlgorithmDefaults, AlgorithmInfo, PlotReference

matplotlib.use("Agg")  # headless backend for the web server

logger = logging.getLogger(__name__)


ALGORITHM_DEFAULTS: dict[str, AlgorithmDefaults] = {
    "er": AlgorithmDefaults(
        max_iterations=250,
        beta=0.85,
        beta_schedule="constant",
        momentum=0.0,
        tv_weight=0.0,
        noise_model="gaussian",
        grid_size=128,
    ),
    "gs": AlgorithmDefaults(
        max_iterations=300,
        beta=0.9,
        beta_schedule="constant",
        momentum=0.0,
        tv_weight=0.0,
        noise_model="gaussian",
        grid_size=128,
    ),
    "hio": AlgorithmDefaults(
        max_iterations=400,
        beta=0.9,
        beta_schedule="constant",
        momentum=0.25,
        tv_weight=0.0,
        noise_model="gaussian",
        grid_size=128,
    ),
    "raar": AlgorithmDefaults(
        max_iterations=600,
        beta=0.9,
        beta_schedule="cosine",
        momentum=0.1,
        tv_weight=0.0,
        noise_model="gaussian",
        grid_size=128,
    ),
    "wf": AlgorithmDefaults(
        max_iterations=350,
        beta=0.75,
        beta_schedule="constant",
        momentum=0.0,
        tv_weight=0.0,
        noise_model="poisson",
        grid_size=128,
    ),
    "dr": AlgorithmDefaults(
        max_iterations=450,
        beta=0.9,
        beta_schedule="linear",
        momentum=0.0,
        tv_weight=0.0,
        noise_model="gaussian",
        grid_size=128,
    ),
    "admm": AlgorithmDefaults(
        max_iterations=300,
        beta=0.8,
        beta_schedule="constant",
        momentum=0.0,
        tv_weight=1e-4,
        noise_model="gaussian",
        grid_size=128,
    ),
    "pinn": AlgorithmDefaults(
        max_iterations=250,
        beta=0.9,
        beta_schedule="constant",
        momentum=0.0,
        tv_weight=5e-5,
        noise_model="gaussian",
        grid_size=128,
    ),
    "fista": AlgorithmDefaults(
        max_iterations=400,
        beta=0.9,
        beta_schedule="constant",
        momentum=0.0,
        tv_weight=0.0,
        noise_model="gaussian",
        grid_size=128,
    ),
    "sparse_pr": AlgorithmDefaults(
        max_iterations=350,
        beta=0.75,
        beta_schedule="constant",
        momentum=0.0,
        tv_weight=0.0,
        noise_model="gaussian",
        grid_size=128,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_psf(fits_path: Path, grid_size: int) -> tuple[PSFData, PupilModel]:
    """Load a FITS / ``.npy`` file and build a matching pupil."""
    pupil_cfg = PupilConfig(telescope=TelescopeType.HST, grid_size=grid_size)

    if fits_path.suffix == ".npy":
        psf_image: np.ndarray = np.load(str(fits_path)).astype(np.float64)
        total = float(psf_image.sum())
        if total > 0:
            psf_image = psf_image / total
        # Resize to grid_size if needed
        if psf_image.shape[0] != grid_size:
            from src.data.loader import prepare_psf_for_retrieval as _resize

            tmp = PSFData(
                image=psf_image,
                pixel_scale_arcsec=pupil_cfg.pixel_scale_arcsec,
                wavelength_m=pupil_cfg.wavelength_m,
                filter_name="SYNTH",
                telescope="hst",
                obs_id=fits_path.stem,
            )
            psf_image = _resize(tmp, grid_size)
        psf_data = PSFData(
            image=psf_image,
            pixel_scale_arcsec=pupil_cfg.pixel_scale_arcsec,
            wavelength_m=pupil_cfg.wavelength_m,
            filter_name="SYNTH",
            telescope="hst",
            obs_id=fits_path.stem,
        )
    else:
        data_cfg = DataConfig(filter_name="F606W")
        psf_data = load_psf_from_fits(fits_path, data_cfg, pupil_cfg)
        psf_image = prepare_psf_for_retrieval(psf_data, grid_size)
        psf_data = PSFData(
            image=psf_image,
            pixel_scale_arcsec=psf_data.pixel_scale_arcsec,
            wavelength_m=psf_data.wavelength_m,
            filter_name=psf_data.filter_name,
            telescope=psf_data.telescope,
            obs_id=psf_data.obs_id,
        )

    actual_grid = psf_data.image.shape[0]
    if actual_grid != pupil_cfg.grid_size:
        pupil_cfg = pupil_cfg.model_copy(update={"grid_size": actual_grid})
    pupil = build_pupil(pupil_cfg)
    return psf_data, pupil


def _generate_plots(
    psf_data: PSFData,
    pupil: PupilModel,
    result: object,
    output_dir: Path,
) -> list[str]:
    """Generate standard result plots and return their filenames."""
    from src.models.optics import PhaseRetrievalResult

    res: PhaseRetrievalResult = result  # type: ignore[assignment]
    support = pupil.amplitude > 0
    output_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []

    try:
        fig = plot_recovered_phase(res, support)
        save_figure(fig, output_dir / "phase.png")
        plt.close(fig)
        names.append("phase.png")
    except Exception:
        logger.exception("Failed to generate phase plot")

    try:
        fig = plot_psf_comparison(psf_data, res)
        save_figure(fig, output_dir / "psf_comparison.png")
        plt.close(fig)
        names.append("psf_comparison.png")
    except Exception:
        logger.exception("Failed to generate PSF comparison plot")

    try:
        fig = plot_convergence(res)
        save_figure(fig, output_dir / "convergence.png")
        plt.close(fig)
        names.append("convergence.png")
    except Exception:
        logger.exception("Failed to generate convergence plot")

    try:
        fig = plot_radial_profile(psf_data, res, pupil)
        save_figure(fig, output_dir / "radial.png")
        plt.close(fig)
        names.append("radial.png")
    except Exception:
        logger.exception("Failed to generate radial profile plot")

    try:
        zc = zernike_decomposition(res.recovered_phase, support)
        fig = plot_summary(psf_data, pupil, res, zc)
        save_figure(fig, output_dir / "summary.png")
        plt.close(fig)
        names.append("summary.png")
    except Exception:
        logger.exception("Failed to generate summary plot")

    return names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_algorithm(
    db: Session,
    job: Job,
    fits_path: Path,
    grid_size: int = 128,
) -> Job:
    """Execute one algorithm and update *job* in-place."""
    try:
        job.status = "running"
        db.commit()

        psf_data, pupil = _load_psf(fits_path, grid_size)

        cfg_raw: dict[str, object] = json.loads(job.config_json)
        _g = cfg_raw.get
        alg_cfg = AlgorithmConfig(
            name=AlgorithmName(job.algorithm),
            max_iterations=int(str(_g("max_iterations", 300))),
            beta=float(str(_g("beta", 0.9))),
            beta_schedule=BetaSchedule(str(_g("beta_schedule", "constant"))),
            momentum=float(str(_g("momentum", 0.0))),
            tv_weight=float(str(_g("tv_weight", 0.0))),
            noise_model=NoiseModel(str(_g("noise_model", "gaussian"))),
            random_seed=42,
        )

        retriever = AlgorithmRegistry.create(alg_cfg, pupil)
        result = retriever.run(psf_data)

        out_dir = settings.output_dir / str(job.id)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Persist config snapshot for full reproducibility
        (out_dir / "config.json").write_text(
            json.dumps(
                {
                    "algorithm": job.algorithm,
                    "fits_filename": job.fits_filename,
                    "grid_size": grid_size,
                    **cfg_raw,
                },
                indent=2,
            )
        )

        _generate_plots(psf_data, pupil, result, out_dir)

        job.status = "completed"
        job.strehl_ratio = result.strehl_ratio
        job.rms_phase_rad = result.rms_phase_rad
        job.n_iterations = result.n_iterations
        job.elapsed_seconds = result.elapsed_seconds
        job.converged = result.converged
        job.cost_history_json = json.dumps(result.cost_history)
        job.output_dir = str(out_dir)
        job.completed_at = datetime.now(UTC)
        db.commit()

        logger.info(
            "Job %d completed: alg=%s strehl=%.4f rms=%.4f iter=%d time=%.2fs",
            job.id,
            job.algorithm,
            result.strehl_ratio,
            result.rms_phase_rad,
            result.n_iterations,
            result.elapsed_seconds,
        )
    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)
        job.completed_at = datetime.now(UTC)
        db.commit()
        logger.exception("Job %d failed", job.id)
    return job


def compare_algorithms(
    db: Session,
    user_id: int,
    fits_path: Path,
    grid_size: int = 128,
    max_iterations: int = 300,
    algorithm_keys: list[str] | None = None,
) -> list[Job]:
    """Run multiple algorithms on the same data and return all jobs."""
    from src.models.optics import PhaseRetrievalResult

    resolved_algorithm_keys = algorithm_keys
    if resolved_algorithm_keys is None:
        resolved_algorithm_keys = [
            a.value
            for a in [
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
        ]

    jobs: list[Job] = []
    phase_results: dict[str, PhaseRetrievalResult] = {}
    psf_data: PSFData | None = None
    pupil: PupilModel | None = None

    for key in resolved_algorithm_keys:
        job = Job(
            user_id=user_id,
            algorithm=key,
            fits_filename=fits_path.name,
            config_json=json.dumps({"max_iterations": max_iterations}),
        )
        db.add(job)
        db.flush()

        try:
            job.status = "running"
            db.commit()

            if psf_data is None:
                psf_data, pupil = _load_psf(fits_path, grid_size)
            if psf_data is None or pupil is None:  # should never happen, but guards against mypy
                raise RuntimeError("PSF / pupil failed to load")

            alg_cfg = AlgorithmConfig(
                name=AlgorithmName(key),
                max_iterations=max_iterations,
                random_seed=42,
            )
            retriever = AlgorithmRegistry.create(alg_cfg, pupil)
            result = retriever.run(psf_data)
            phase_results[key.upper()] = result

            out_dir = settings.output_dir / str(job.id)
            _generate_plots(psf_data, pupil, result, out_dir)

            job.status = "completed"
            job.strehl_ratio = result.strehl_ratio
            job.rms_phase_rad = result.rms_phase_rad
            job.n_iterations = result.n_iterations
            job.elapsed_seconds = result.elapsed_seconds
            job.converged = result.converged
            job.cost_history_json = json.dumps(result.cost_history)
            job.output_dir = str(out_dir)
            job.completed_at = datetime.now(UTC)
        except Exception as exc:
            job.status = "failed"
            job.error_message = str(exc)
            job.completed_at = datetime.now(UTC)
            logger.exception("Compare job %d (%s) failed", job.id, key)

        db.commit()
        jobs.append(job)

    # Generate comparison plots
    if len(phase_results) >= 2 and pupil is not None:
        cmp_dir = settings.output_dir / "compare" / str(jobs[0].id)
        cmp_dir.mkdir(parents=True, exist_ok=True)
        support = pupil.amplitude > 0

        try:
            fig = plot_algorithm_comparison(phase_results, support)
            save_figure(fig, cmp_dir / "comparison.png")
            plt.close(fig)
        except Exception:
            logger.exception("Failed to create comparison plot")

        try:
            fig = plot_strehl_rms_bar(phase_results)
            save_figure(fig, cmp_dir / "strehl_rms.png")
            plt.close(fig)
        except Exception:
            logger.exception("Failed to create Strehl/RMS bar chart")

    return jobs


def list_algorithms_with_defaults() -> list[AlgorithmInfo]:
    """Return available algorithms with recommended UI defaults."""
    return [
        AlgorithmInfo(
            key=key,
            name=key.upper(),
            defaults=ALGORITHM_DEFAULTS.get(key, AlgorithmDefaults()),
        )
        for key in AlgorithmRegistry.available()
    ]


def list_job_plots(job: Job) -> list[str]:
    """Return the plot filenames available for a completed job."""
    if not job.output_dir:
        return []
    out = Path(job.output_dir)
    if not out.exists():
        return []
    return sorted(p.name for p in out.glob("*.png"))


def list_comparison_plots(job_id: int) -> list[PlotReference]:
    """Return comparison plot references for a compare run keyed by its first job ID."""
    cmp_dir = settings.output_dir / "compare" / str(job_id)
    if not cmp_dir.exists():
        return []
    return [PlotReference(job_id=job_id, name=p.name) for p in sorted(cmp_dir.glob("*.png"))]
