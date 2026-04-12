"""Result browsing, plot serving, and export endpoints."""

from __future__ import annotations

import asyncio
import io
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select

from web.config import settings
from web.dependencies import CurrentUser, DbSession
from web.models import Job
from web.schemas import DashboardStats, JobResponse
from web.services.algorithm_service import list_job_plots

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("/", response_model=list[JobResponse])
def list_results(
    user: CurrentUser,
    db: DbSession,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> list[JobResponse]:
    """Return all jobs belonging to the current user, newest first."""
    rows = (
        db.execute(
            select(Job)
            .where(Job.user_id == user.id)
            .order_by(Job.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        .scalars()
        .all()
    )
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


@router.get("/{job_id}/export")
async def export_result(job_id: int, user: CurrentUser, db: DbSession) -> StreamingResponse:
    """Download a ZIP archive of all plots + config.json for a completed job.

    Returns a ``application/zip`` stream containing:

    * ``config.json``   — full algorithm config snapshot
    * ``*.png``         — all generated plot files
    * ``metadata.json`` — job metrics (Strehl, RMS, converged, …)

    This enables one-click reproducibility: you have everything needed to
    re-run the retrieval with identical settings.
    """
    import json

    job = db.get(Job, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Result not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Job has not completed yet")
    if not job.output_dir or not Path(job.output_dir).exists():
        raise HTTPException(status_code=404, detail="Output files not found")

    out_dir = Path(job.output_dir)

    def _build_zip() -> io.BytesIO:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for png in sorted(out_dir.glob("*.png")):
                zf.write(png, arcname=png.name)
            config_file = out_dir / "config.json"
            if config_file.exists():
                zf.write(config_file, arcname="config.json")
            metadata = {
                "job_id": job.id,
                "algorithm": job.algorithm,
                "status": job.status,
                "fits_filename": job.fits_filename,
                "strehl_ratio": job.strehl_ratio,
                "rms_phase_rad": job.rms_phase_rad,
                "n_iterations": job.n_iterations,
                "elapsed_seconds": job.elapsed_seconds,
                "converged": job.converged,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        buf.seek(0)
        return buf

    buf = await asyncio.to_thread(_build_zip)
    filename = f"phase_retrieval_job_{job_id}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
