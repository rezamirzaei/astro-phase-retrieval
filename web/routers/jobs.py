"""Background job management endpoints — submit, poll, list, cancel.

These endpoints work with the ``job_queue`` service to provide a
non-blocking async job submission model:

    1. ``GET  /api/jobs/{job_id}``  — poll status + progress
    2. ``GET  /api/jobs/``          — list all active/recent jobs
    3. ``POST /api/jobs/{job_id}/cancel`` — request cancellation
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from web.dependencies import CurrentUser
from web.services.job_queue import cancel_job, get_job_status, list_active_jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}")
def poll_job(job_id: str, _user: CurrentUser) -> dict:
    """Poll the status and progress of a background job."""
    status = get_job_status(job_id)
    if status.get("error") == "not_found":
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return status


@router.get("/")
def list_jobs(_user: CurrentUser) -> list[dict]:
    """List all tracked background jobs (active and recently completed)."""
    return list_active_jobs()


@router.post("/{job_id}/cancel")
def cancel_background_job(job_id: str, _user: CurrentUser) -> dict:
    """Request cancellation of a queued or running background job.

    Cancellation is best-effort — a job that is mid-computation may
    complete before the cancellation signal is processed.
    """
    ok = cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found or already finished")
    return {"job_id": job_id, "state": "cancelled", "message": "Cancellation requested"}
