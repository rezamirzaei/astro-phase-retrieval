"""Algorithm execution service — run, compare, and persist results."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib  # noqa: F401 — ensure it's importable; backend set in lifespan
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sqlalchemy.orm import Session

from src.algorithms.registry import AlgorithmRegistry
from src.benchmark import (
    available_benchmark_cases,
    default_benchmark_algorithms,
    run_benchmark,
)
from src.metrics.quality import zernike_decomposition
from src.models.config import (
    AlgorithmConfig,
    AlgorithmName,
    BetaSchedule,
    NoiseModel,
    default_hst_config,
)
from src.models.optics import PSFData, PupilModel
from src.pipeline import RetrievalPipeline
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
from web.schemas import (
    AlgorithmDefaults,
    AlgorithmInfo,
    BenchmarkAggregateRow,
    BenchmarkCaseInfo,
    BenchmarkResponse,
    BenchmarkStudyRow,
    PlotReference,
)

# NOTE: matplotlib.use("Agg") is called during app lifespan startup
# to avoid thread-safety issues.  Do NOT call it here at module level.

logger = logging.getLogger(__name__)


ALGORITHM_DEFAULTS: dict[str, AlgorithmDefaults] = {
    "er": AlgorithmDefaults(
        max_iterations=250,
        beta=0.85,
        beta_schedule=BetaSchedule.CONSTANT,
        momentum=0.0,
        tv_weight=0.0,
        noise_model=NoiseModel.GAUSSIAN,
        grid_size=128,
    ),
    "gs": AlgorithmDefaults(
        max_iterations=300,
        beta=0.9,
        beta_schedule=BetaSchedule.CONSTANT,
        momentum=0.0,
        tv_weight=0.0,
        noise_model=NoiseModel.GAUSSIAN,
        grid_size=128,
    ),
    "hio": AlgorithmDefaults(
        max_iterations=400,
        beta=0.9,
        beta_schedule=BetaSchedule.CONSTANT,
        momentum=0.25,
        tv_weight=0.0,
        noise_model=NoiseModel.GAUSSIAN,
        grid_size=128,
    ),
    "raar": AlgorithmDefaults(
        max_iterations=600,
        beta=0.9,
        beta_schedule=BetaSchedule.COSINE,
        momentum=0.1,
        tv_weight=0.0,
        noise_model=NoiseModel.GAUSSIAN,
        grid_size=128,
    ),
    "wf": AlgorithmDefaults(
        max_iterations=350,
        beta=0.75,
        beta_schedule=BetaSchedule.CONSTANT,
        momentum=0.0,
        tv_weight=0.0,
        noise_model=NoiseModel.POISSON,
        grid_size=128,
    ),
    "dr": AlgorithmDefaults(
        max_iterations=450,
        beta=0.9,
        beta_schedule=BetaSchedule.LINEAR,
        momentum=0.0,
        tv_weight=0.0,
        noise_model=NoiseModel.GAUSSIAN,
        grid_size=128,
    ),
    "admm": AlgorithmDefaults(
        max_iterations=300,
        beta=0.8,
        beta_schedule=BetaSchedule.CONSTANT,
        momentum=0.0,
        tv_weight=1e-4,
        noise_model=NoiseModel.GAUSSIAN,
        grid_size=128,
    ),
    "pinn": AlgorithmDefaults(
        max_iterations=250,
        beta=0.9,
        beta_schedule=BetaSchedule.CONSTANT,
        momentum=0.0,
        tv_weight=5e-5,
        noise_model=NoiseModel.GAUSSIAN,
        grid_size=128,
    ),
    "fista": AlgorithmDefaults(
        max_iterations=400,
        beta=0.9,
        beta_schedule=BetaSchedule.CONSTANT,
        momentum=0.0,
        tv_weight=0.0,
        noise_model=NoiseModel.GAUSSIAN,
        grid_size=128,
    ),
    "sparse_pr": AlgorithmDefaults(
        max_iterations=350,
        beta=0.75,
        beta_schedule=BetaSchedule.CONSTANT,
        momentum=0.0,
        tv_weight=0.0,
        noise_model=NoiseModel.GAUSSIAN,
        grid_size=128,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_algorithm_config(job: Job) -> AlgorithmConfig:
    """Create the validated shared algorithm config from a web job row.

    Delegates parsing/validation to Pydantic instead of manual str→float
    conversions — any type errors are caught at construction time.
    """
    cfg_raw: dict[str, Any] = json.loads(job.config_json)
    # Merge with defaults; let Pydantic handle type coercion & validation
    return AlgorithmConfig(
        name=AlgorithmName(job.algorithm),
        max_iterations=cfg_raw.get("max_iterations", 300),  # type: ignore[arg-type]
        tolerance=cfg_raw.get("tolerance", 1e-8),  # type: ignore[arg-type]
        beta=cfg_raw.get("beta", 0.9),  # type: ignore[arg-type]
        beta_schedule=cfg_raw.get("beta_schedule", "constant"),  # type: ignore[arg-type]
        momentum=cfg_raw.get("momentum", 0.0),  # type: ignore[arg-type]
        tv_weight=cfg_raw.get("tv_weight", 0.0),  # type: ignore[arg-type]
        noise_model=cfg_raw.get("noise_model", "gaussian"),  # type: ignore[arg-type]
        n_starts=cfg_raw.get("n_starts", 1),  # type: ignore[arg-type]
        admm_rho=cfg_raw.get("admm_rho", 1.0),  # type: ignore[arg-type]
        wf_step_size=cfg_raw.get("wf_step_size", 0.5),  # type: ignore[arg-type]
        wf_spectral_init=cfg_raw.get("wf_spectral_init", True),  # type: ignore[arg-type]
        spectral_init=cfg_raw.get("spectral_init", True),  # type: ignore[arg-type]
        regulariser=cfg_raw.get("regulariser", "none"),  # type: ignore[arg-type]
        proximal_weight=cfg_raw.get("proximal_weight", 1e-3),  # type: ignore[arg-type]
        sparsity_threshold=cfg_raw.get("sparsity_threshold", 0.1),  # type: ignore[arg-type]
        sparsity_keep_fraction=cfg_raw.get("sparsity_keep_fraction", 1.0),  # type: ignore[arg-type]
        random_seed=42,
    )


def _build_pipeline(
    grid_size: int,
    output_dir: Path,
    *,
    uncertainty_samples: int = 0,
) -> RetrievalPipeline:
    """Construct the shared retrieval pipeline used by the web application."""
    config = default_hst_config()
    data_cfg = config.data.model_copy(update={"cutout_size": grid_size})
    pupil_cfg = config.pupil.model_copy(update={"grid_size": grid_size})
    return RetrievalPipeline(
        config.model_copy(
            update={
                "data": data_cfg,
                "pupil": pupil_cfg,
                "output_dir": output_dir,
                "uncertainty_samples": uncertainty_samples,
            }
        )
    )


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
    zc = zernike_decomposition(res.recovered_phase, support)
    return [
        _save_plot(output_dir / "phase.png", lambda: plot_recovered_phase(res, support)),
        _save_plot(output_dir / "psf_comparison.png", lambda: plot_psf_comparison(psf_data, res)),
        _save_plot(output_dir / "convergence.png", lambda: plot_convergence(res)),
        _save_plot(output_dir / "radial.png", lambda: plot_radial_profile(psf_data, res, pupil)),
        _save_plot(output_dir / "summary.png", lambda: plot_summary(psf_data, pupil, res, zc)),
    ]


def _save_plot(path: Path, render: Callable[[], Figure]) -> str:
    """Render and persist a plot, ensuring the figure is always closed."""
    fig: Figure | None = None
    try:
        fig = render()
        save_figure(fig, path)
    finally:
        if fig is not None:
            plt.close(fig)
    return path.name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_algorithm(
    db: Session,
    job: Job,
    fits_path: Path,
    grid_size: int = 128,
) -> Job:
    """Execute one algorithm and update *job* in-place.

    .. note::

       When called via ``asyncio.to_thread`` the caller's ``db`` session
       crosses a thread boundary.  For SQLite this is fine (we set
       ``check_same_thread=False``).  For PostgreSQL, SQLAlchemy sessions
       are not fully thread-safe, but single-row commits on a dedicated
       session (no concurrent reads) are safe in practice.  A future
       refactor should create a thread-local session here instead.
    """
    try:
        job.status = "running"
        db.commit()

        out_dir = settings.output_dir / str(job.id)
        cfg_raw: dict[str, Any] = json.loads(job.config_json)
        pipeline = _build_pipeline(
            grid_size,
            out_dir,
            uncertainty_samples=int(cfg_raw.get("uncertainty_samples", 0) or 0),
        )
        pipeline_result = pipeline.run_from_file(
            fits_path,
            algorithm_config=_build_algorithm_config(job),
            output_dir=out_dir,
        )
        _generate_plots(
            pipeline_result.psf_data,
            pipeline_result.pupil,
            pipeline_result.result,
            out_dir,
        )

        job.status = "completed"
        job.strehl_ratio = pipeline_result.result.strehl_ratio
        job.rms_phase_rad = pipeline_result.result.rms_phase_rad
        job.n_iterations = pipeline_result.result.n_iterations
        job.elapsed_seconds = pipeline_result.result.elapsed_seconds
        job.converged = pipeline_result.result.converged
        job.cost_history_json = json.dumps(pipeline_result.result.cost_history)
        job.output_dir = str(out_dir)
        job.completed_at = datetime.now(UTC)
        db.commit()

        logger.info(
            "Job %d completed: alg=%s strehl=%.4f rms=%.4f iter=%d time=%.2fs",
            job.id,
            job.algorithm,
            pipeline_result.result.strehl_ratio,
            pipeline_result.result.rms_phase_rad,
            pipeline_result.result.n_iterations,
            pipeline_result.result.elapsed_seconds,
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
    base_pipeline = _build_pipeline(grid_size, settings.output_dir)

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
                psf_data, pupil = base_pipeline.load_inputs_from_file(fits_path)
            if psf_data is None or pupil is None:  # should never happen, but guards against mypy
                raise RuntimeError("PSF / pupil failed to load")

            alg_cfg = AlgorithmConfig(
                name=AlgorithmName(key),
                max_iterations=max_iterations,
                random_seed=42,
            )
            out_dir = settings.output_dir / str(job.id)
            pipeline_result = base_pipeline.run_from_psf(
                psf_data,
                pupil,
                algorithm_config=alg_cfg,
                output_dir=out_dir,
            )
            result = pipeline_result.result
            phase_results[key.upper()] = result

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

        _save_plot(
            cmp_dir / "comparison.png",
            lambda: plot_algorithm_comparison(phase_results, support),
        )
        _save_plot(cmp_dir / "strehl_rms.png", lambda: plot_strehl_rms_bar(phase_results))

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


def list_benchmark_cases() -> list[BenchmarkCaseInfo]:
    """Return the benchmark cases that can be launched from the web UI."""
    return [
        BenchmarkCaseInfo(key=key, description=case.description)
        for key, case in sorted(available_benchmark_cases().items())
    ]


def run_algorithm_benchmark(
    *,
    algorithm_keys: list[str] | None,
    case_keys: list[str] | None,
    max_iterations: int,
    beta: float,
    random_seed: int,
) -> BenchmarkResponse:
    """Run the synthetic benchmark suite and return a web-friendly summary."""
    available_cases = available_benchmark_cases()
    selected_cases = [
        available_cases[key]
        for key in (case_keys or list(available_cases))
        if key in available_cases
    ]
    if not selected_cases:
        raise ValueError("No valid benchmark cases were selected")

    selected_algorithms = (
        [AlgorithmName(key) for key in algorithm_keys]
        if algorithm_keys
        else default_benchmark_algorithms()
    )
    output_dir = settings.output_dir / "benchmarks" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S"))
    summary = run_benchmark(
        algorithms=selected_algorithms,
        cases=selected_cases,
        max_iterations=max_iterations,
        beta=beta,
        random_seed=random_seed,
        output_dir=output_dir,
    )
    return BenchmarkResponse(
        selected_algorithms=[algorithm.value for algorithm in selected_algorithms],
        selected_cases=[
            BenchmarkCaseInfo(key=case.key, description=case.description) for case in selected_cases
        ],
        aggregate=[BenchmarkAggregateRow(**row) for row in summary.aggregate],
        study=[BenchmarkStudyRow(**row) for row in summary.study],
        records_count=len(summary.records),
        artifacts=summary.artifacts,
    )


def list_job_plots(job: Job) -> list[str]:
    """Return the plot filenames available for a completed job."""
    if not job.output_dir:
        return []
    out = Path(job.output_dir)
    if not out.exists():
        return []
    return sorted(p.name for p in out.glob("*.png"))


def list_job_artifacts(job: Job) -> list[str]:
    """Return saved non-plot artifacts for a completed job."""
    if not job.output_dir:
        return []
    out = Path(job.output_dir)
    if not out.exists():
        return []
    return sorted(
        p.name
        for p in out.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".json", ".csv", ".md"}
        and p.name != "metadata.json"
    )


def list_comparison_plots(job_id: int) -> list[PlotReference]:
    """Return comparison plot references for a compare run keyed by its first job ID."""
    cmp_dir = settings.output_dir / "compare" / str(job_id)
    if not cmp_dir.exists():
        return []
    return [PlotReference(job_id=job_id, name=p.name) for p in sorted(cmp_dir.glob("*.png"))]
