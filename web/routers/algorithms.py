"""Algorithm execution endpoints — run single, compare all."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from web.dependencies import CurrentUser, DbSession
from web.models import Job
from web.schemas import AlgorithmRunRequest, CompareRequest, CompareResponse, JobResponse
from web.services.algorithm_service import (
    compare_algorithms,
    list_job_plots,
    run_algorithm,
)
from web.services.data_service import resolve_fits_path

router = APIRouter(prefix="/api/algorithms", tags=["algorithms"])


@router.get("/", response_model=list[dict[str, str]])
def list_algorithms(_user: CurrentUser) -> list[dict[str, str]]:
    """List available phase-retrieval algorithms with descriptions."""
    from src.algorithms.registry import AlgorithmRegistry

    return [
        {"key": k, "name": k.upper()}
        for k in AlgorithmRegistry.available()
    ]


@router.post("/run", response_model=JobResponse)
def run_single(body: AlgorithmRunRequest, user: CurrentUser, db: DbSession) -> JobResponse:
    """Run a single algorithm on a data file."""
    try:
        fits_path = resolve_fits_path(body.fits_filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    job = Job(
        user_id=user.id,
        algorithm=body.algorithm,
        fits_filename=body.fits_filename,
        config_json=json.dumps(
            {
                "max_iterations": body.max_iterations,
                "beta": body.beta,
                "beta_schedule": body.beta_schedule,
                "momentum": body.momentum,
                "tv_weight": body.tv_weight,
                "noise_model": body.noise_model,
            }
        ),
    )
    db.add(job)
    db.flush()

    job = run_algorithm(db, job, fits_path, grid_size=body.grid_size)
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error_message or "Unknown error")

    plots = list_job_plots(job)
    resp = JobResponse.model_validate(job)
    resp.plots = plots
    return resp


@router.post("/compare", response_model=CompareResponse)
def compare(body: CompareRequest, user: CurrentUser, db: DbSession) -> CompareResponse:
    """Run all (or selected) algorithms on the same data."""
    try:
        fits_path = resolve_fits_path(body.fits_filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    jobs = compare_algorithms(
        db,
        user_id=user.id,
        fits_path=fits_path,
        grid_size=body.grid_size,
        max_iterations=body.max_iterations,
        algorithm_keys=body.algorithms,
    )

    results: list[JobResponse] = []
    for j in jobs:
        resp = JobResponse.model_validate(j)
        resp.plots = list_job_plots(j)
        results.append(resp)

    return CompareResponse(results=results, comparison_plots=["comparison.png", "strehl_rms.png"])

