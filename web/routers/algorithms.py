"""Algorithm execution endpoints — run single, compare all."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException

from web.concurrency import get_job_semaphore
from web.dependencies import CurrentUser, DbSession
from web.models import Job
from web.schemas import (
    AlgorithmInfo,
    AlgorithmRunRequest,
    BenchmarkCaseInfo,
    BenchmarkResponse,
    BenchmarkRunRequest,
    CompareRequest,
    CompareResponse,
    JobResponse,
)
from web.services.algorithm_service import (
    compare_algorithms,
    list_benchmark_cases,
    list_algorithms_with_defaults,
    list_comparison_plots,
    list_job_artifacts,
    list_job_plots,
    run_algorithm_benchmark,
    run_algorithm,
)
from web.services.data_service import resolve_fits_path

router = APIRouter(prefix="/api/algorithms", tags=["algorithms"])


@router.get("/", response_model=list[AlgorithmInfo])
def list_algorithms(_user: CurrentUser) -> list[AlgorithmInfo]:
    """List available phase-retrieval algorithms with descriptions."""
    return list_algorithms_with_defaults()


@router.post("/run", response_model=JobResponse)
async def run_single(body: AlgorithmRunRequest, user: CurrentUser, db: DbSession) -> JobResponse:
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
                "tolerance": body.tolerance,
                "beta": body.beta,
                "beta_schedule": body.beta_schedule,
                "momentum": body.momentum,
                "tv_weight": body.tv_weight,
                "noise_model": body.noise_model,
                "n_starts": body.n_starts,
                "uncertainty_samples": body.uncertainty_samples,
                "admm_rho": body.admm_rho,
                "wf_step_size": body.wf_step_size,
                "wf_spectral_init": body.wf_spectral_init,
                "spectral_init": body.spectral_init,
                "regulariser": body.regulariser,
                "proximal_weight": body.proximal_weight,
                "sparsity_threshold": body.sparsity_threshold,
                "sparsity_keep_fraction": body.sparsity_keep_fraction,
            }
        ),
    )
    db.add(job)
    db.flush()

    async with get_job_semaphore():
        job = await asyncio.to_thread(run_algorithm, db, job, fits_path, grid_size=body.grid_size)
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error_message or "Unknown error")

    plots = list_job_plots(job)
    resp = JobResponse.model_validate(job)
    resp.plots = plots
    resp.artifacts = list_job_artifacts(job)
    return resp


@router.post("/compare", response_model=CompareResponse)
async def compare(body: CompareRequest, user: CurrentUser, db: DbSession) -> CompareResponse:
    """Run all (or selected) algorithms on the same data."""
    try:
        fits_path = resolve_fits_path(body.fits_filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async with get_job_semaphore():
        jobs = await asyncio.to_thread(
            compare_algorithms,
            db,
            user_id=user.id,
            fits_path=fits_path,
            grid_size=body.grid_size,
            max_iterations=body.max_iterations,
            algorithm_keys=[a.value for a in body.algorithms] if body.algorithms else None,
        )

    results: list[JobResponse] = []
    for j in jobs:
        resp = JobResponse.model_validate(j)
        resp.plots = list_job_plots(j)
        resp.artifacts = list_job_artifacts(j)
        results.append(resp)

    comparison_plots = list_comparison_plots(jobs[0].id) if jobs else []
    return CompareResponse(results=results, comparison_plots=comparison_plots)


@router.get("/benchmark/cases", response_model=list[BenchmarkCaseInfo])
def benchmark_cases(_user: CurrentUser) -> list[BenchmarkCaseInfo]:
    """List the synthetic benchmark cases available in the web UI."""
    return list_benchmark_cases()


@router.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark(
    body: BenchmarkRunRequest,
    _user: CurrentUser,
) -> BenchmarkResponse:
    """Run the deterministic synthetic benchmark suite through the web API."""
    async with get_job_semaphore():
        return await asyncio.to_thread(
            run_algorithm_benchmark,
            algorithm_keys=[algorithm.value for algorithm in body.algorithms]
            if body.algorithms
            else None,
            case_keys=body.cases,
            max_iterations=body.max_iterations,
            beta=body.beta,
            random_seed=body.random_seed,
        )
