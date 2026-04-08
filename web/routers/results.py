"""Result browsing and plot serving endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select

from web.config import settings
from web.dependencies import CurrentUser, DbSession
from web.models import Job
from web.schemas import DashboardStats, JobResponse
from web.services.algorithm_service import list_job_plots

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("/", response_model=list[JobResponse])
def list_results(user: CurrentUser, db: DbSession) -> list[JobResponse]:
    """Return all jobs belonging to the current user, newest first."""
    rows = db.execute(
        select(Job).where(Job.user_id == user.id).order_by(Job.created_at.desc())
    ).scalars().all()
    out: list[JobResponse] = []
    for j in rows:
        resp = JobResponse.model_validate(j)
        resp.plots = list_job_plots(j)
        out.append(resp)
    return out


@router.get("/dashboard", response_model=DashboardStats)
def dashboard(user: CurrentUser, db: DbSession) -> DashboardStats:
    """Aggregate stats for the dashboard view."""
    rows = (
        db.execute(select(Job).where(Job.user_id == user.id).order_by(Job.created_at.desc()))
        .scalars()
        .all()
    )
    completed = [j for j in rows if j.status == "completed"]
    strehls = [j.strehl_ratio for j in completed if j.strehl_ratio is not None]
    algos = sorted({j.algorithm for j in rows})
    recent = rows[:5]

    recent_responses: list[JobResponse] = []
    for j in recent:
        resp = JobResponse.model_validate(j)
        resp.plots = list_job_plots(j)
        recent_responses.append(resp)

    return DashboardStats(
        total_runs=len(rows),
        completed_runs=len(completed),
        best_strehl=max(strehls) if strehls else None,
        algorithms_used=algos,
        recent_jobs=recent_responses,
    )


@router.get("/{job_id}", response_model=JobResponse)
def get_result(job_id: int, user: CurrentUser, db: DbSession) -> JobResponse:
    """Get a single result by ID."""
    job = db.get(Job, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not found")
    resp = JobResponse.model_validate(job)
    resp.plots = list_job_plots(job)
    return resp


@router.get("/{job_id}/plots/{plot_name}")
def get_plot(job_id: int, plot_name: str, user: CurrentUser, db: DbSession) -> FileResponse:
    """Serve a plot PNG for a specific result."""
    job = db.get(Job, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Result not found")
    if not job.output_dir:
        raise HTTPException(status_code=404, detail="No plots available")

    plot_path = Path(job.output_dir) / plot_name
    if not plot_path.exists() or not plot_path.name.endswith(".png"):
        # Try comparison directory
        cmp_path = settings.output_dir / "compare" / str(job_id) / plot_name
        if cmp_path.exists():
            plot_path = cmp_path
        else:
            raise HTTPException(status_code=404, detail=f"Plot '{plot_name}' not found")

    return FileResponse(plot_path, media_type="image/png")


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_result(job_id: int, user: CurrentUser, db: DbSession) -> None:
    """Delete a result and its generated plots."""
    import shutil

    job = db.get(Job, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Result not found")
    if job.output_dir and Path(job.output_dir).exists():
        shutil.rmtree(job.output_dir, ignore_errors=True)
    db.delete(job)
    db.commit()

