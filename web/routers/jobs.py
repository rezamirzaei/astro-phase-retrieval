"""Background job management endpoints — submit, poll, list, cancel.

These endpoints work with the ``job_queue`` service to provide a
non-blocking async job submission model:

    1. ``POST /api/v1/jobs/submit`` — fire-and-forget, returns ``job_id``
    2. ``GET  /api/v1/jobs/{job_id}`` — poll status + progress
    3. ``GET  /api/v1/jobs/``         — list all active/recent jobs
"""

from __future__ import annotations

from fastapi import APIRouter

from web.dependencies import CurrentUser
from web.services.job_queue import get_job_status, list_active_jobs

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("/{job_id}")
def poll_job(job_id: str, _user: CurrentUser) -> dict:
    """Poll the status and progress of a background job."""
    return get_job_status(job_id)


@router.get("/")
def list_jobs(_user: CurrentUser) -> list[dict]:
    """List all tracked background jobs (active and recently completed)."""
    return list_active_jobs()

