"""Crystallography service — COD data, diffraction simulation, phase retrieval."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sqlalchemy.orm import Session

from src.data.crystallography import (
    available_cod_presets,
    parse_cif,
    run_crystallography_retrieval,
    simulate_diffraction,
)
from src.models.config import AlgorithmName
from src.models.crystallography import CrystallographyResult, DiffractionPattern
from src.visualization.crystallography_plots import (
    plot_crystal_summary,
    plot_crystallography_result,
    plot_diffraction_pattern,
    plot_electron_density,
    plot_r_factor_comparison,
)
from src.visualization.plots import save_figure
from web.config import settings
from web.models import CrystallographyJob
from web.schemas import CodPresetInfo

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def list_cod_presets() -> list[CodPresetInfo]:
    """Return available COD presets."""
    return [CodPresetInfo(key=k, description=v) for k, v in available_cod_presets().items()]


def list_cif_files() -> list[dict[str, object]]:
    """Return metadata for every CIF file under the crystallography data dir."""
    results: list[dict[str, object]] = []
    cif_dir = settings.data_dir / "crystallography"
    if not cif_dir.exists():
        return results
    results.extend(
        {
            "filename": p.name,
            "filepath": str(p),
            "size_bytes": p.stat().st_size,
        }
        for p in sorted(cif_dir.glob("*.cif"))
    )
    return results


def resolve_cif_path(filename: str) -> Path:
    """Find a CIF file by name under the crystallography data dir.

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    cif_dir = settings.data_dir / "crystallography"
    if cif_dir.exists():
        for p in cif_dir.glob("*.cif"):
            if p.name == filename:
                return p
    raise FileNotFoundError(f"No CIF file named '{filename}' in {cif_dir}")


def simulate_from_cif(
    cif_path: Path,
    grid_size: int = 128,
) -> tuple[DiffractionPattern, str]:
    """Parse a CIF and simulate diffraction, returning the pattern and formula."""
    crystal = parse_cif(cif_path)
    pattern = simulate_diffraction(crystal, grid_size=grid_size)

    # Save as .npy for potential reuse
    npy_dir = settings.data_dir / "crystallography"
    npy_dir.mkdir(parents=True, exist_ok=True)
    npy_path = npy_dir / f"{cif_path.stem}_{grid_size}.npy"
    np.save(str(npy_path), pattern.image)
    logger.info("Saved simulated diffraction → %s", npy_path)

    return pattern, crystal.formula


def run_crystallography_job(
    db: Session,
    job: CrystallographyJob,
    cif_path: Path,
    grid_size: int = 128,
) -> CrystallographyJob:
    """Execute crystallographic phase retrieval and update *job* in-place."""
    try:
        job.status = "running"
        db.commit()

        crystal = parse_cif(cif_path)
        pattern = simulate_diffraction(crystal, grid_size=grid_size)

        cfg_raw: dict[str, object] = json.loads(job.config_json)
        _g = cfg_raw.get

        result = run_crystallography_retrieval(
            diffraction=pattern,
            algorithm_name=job.algorithm,
            max_iterations=int(str(_g("max_iterations", 500))),
            beta=float(str(_g("beta", 0.9))),
            random_seed=42,
        )

        out_dir = settings.output_dir / f"cryst_{job.id}"
        _generate_crystallography_plots(pattern, result, out_dir)

        job.status = "completed"
        job.r_factor = result.r_factor
        job.n_iterations = result.n_iterations
        job.elapsed_seconds = result.elapsed_seconds
        job.converged = result.converged
        job.cost_history_json = json.dumps(result.cost_history)
        job.output_dir = str(out_dir)
        job.completed_at = datetime.now(UTC)
        db.commit()
    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)
        job.completed_at = datetime.now(UTC)
        db.commit()
        logger.exception("Crystallography job %d failed", job.id)
    return job


def compare_crystallography_algorithms(
    db: Session,
    user_id: int,
    cif_path: Path,
    grid_size: int = 128,
    max_iterations: int = 500,
    algorithm_keys: list[str] | None = None,
) -> list[CrystallographyJob]:
    """Run multiple algorithms on the same crystallographic data."""
    if algorithm_keys is None:
        algorithm_keys = [
            AlgorithmName.ERROR_REDUCTION.value,
            AlgorithmName.GERCHBERG_SAXTON.value,
            AlgorithmName.HYBRID_INPUT_OUTPUT.value,
            AlgorithmName.RAAR.value,
            AlgorithmName.WIRTINGER_FLOW.value,
            AlgorithmName.DOUGLAS_RACHFORD.value,
            AlgorithmName.ADMM.value,
            AlgorithmName.FISTA.value,
            AlgorithmName.SPARSE_PR.value,
        ]

    crystal = parse_cif(cif_path)
    pattern = simulate_diffraction(crystal, grid_size=grid_size)

    jobs: list[CrystallographyJob] = []
    cryst_results: dict[str, CrystallographyResult] = {}

    for key in algorithm_keys:
        job = CrystallographyJob(
            user_id=user_id,
            algorithm=key,
            cif_filename=cif_path.name,
            cod_id=crystal.cod_id,
            formula=crystal.formula,
            config_json=json.dumps({"max_iterations": max_iterations}),
        )
        db.add(job)
        db.flush()

        try:
            job.status = "running"
            db.commit()

            result = run_crystallography_retrieval(
                diffraction=pattern,
                algorithm_name=key,
                max_iterations=max_iterations,
                random_seed=42,
            )
            cryst_results[key.upper()] = result

            out_dir = settings.output_dir / f"cryst_{job.id}"
            _generate_crystallography_plots(pattern, result, out_dir)

            job.status = "completed"
            job.r_factor = result.r_factor
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
            logger.exception("Cryst compare job %d (%s) failed", job.id, key)

        db.commit()
        jobs.append(job)

    # Comparison plot
    if len(cryst_results) >= 2:
        cmp_dir = settings.output_dir / "cryst_compare" / str(jobs[0].id)
        cmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            fig = plot_r_factor_comparison(cryst_results)
            save_figure(fig, cmp_dir / "r_factor_comparison.png")
            plt.close(fig)
        except Exception:
            logger.exception("Failed to create R-factor comparison plot")

    return jobs


def _generate_crystallography_plots(
    pattern: DiffractionPattern,
    result: CrystallographyResult,
    output_dir: Path,
) -> list[str]:
    """Generate plots for a crystallography run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []

    try:
        fig = plot_diffraction_pattern(pattern)
        save_figure(fig, output_dir / "diffraction.png")
        plt.close(fig)
        names.append("diffraction.png")
    except Exception:
        logger.exception("Failed to generate diffraction plot")

    try:
        fig = plot_crystallography_result(pattern, result)
        save_figure(fig, output_dir / "result.png")
        plt.close(fig)
        names.append("result.png")
    except Exception:
        logger.exception("Failed to generate result plot")

    try:
        fig = plot_electron_density(result)
        save_figure(fig, output_dir / "electron_density.png")
        plt.close(fig)
        names.append("electron_density.png")
    except Exception:
        logger.exception("Failed to generate electron density plot")

    try:
        fig = plot_crystal_summary(pattern, result)
        save_figure(fig, output_dir / "summary.png")
        plt.close(fig)
        names.append("summary.png")
    except Exception:
        logger.exception("Failed to generate summary plot")

    return names


def list_crystallography_job_plots(job: CrystallographyJob) -> list[str]:
    """Return the plot filenames available for a completed crystallography job."""
    if not job.output_dir:
        return []
    out = Path(job.output_dir)
    if not out.exists():
        return []
    return sorted(p.name for p in out.glob("*.png"))
