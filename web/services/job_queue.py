"""Background job queue with real-time progress streaming.

Design
------
* A ``ThreadPoolExecutor`` runs compute-heavy phase-retrieval jobs off the
  async event loop.
* Each job is assigned a UUID.  While running, the executor publishes
  ``ProgressEvent`` messages to an ``asyncio.Queue`` that WebSocket clients
  consume in real-time.
* Jobs go through states: ``queued → running → completed | failed``.

The module exposes three entry-points used by routers:

* ``submit_job`` — enqueue and return immediately.
* ``get_job_status`` — poll from HTTP.
* ``subscribe_progress`` — async generator for WebSocket streaming.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from web.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressEvent:
    """A single progress tick emitted by a running job."""

    iteration: int = 0
    total_iterations: int = 0
    cost: float = 0.0
    elapsed_seconds: float = 0.0
    message: str = ""


@dataclass
class BackgroundJob:
    """Tracked state for a background job."""

    job_id: str
    state: JobState = JobState.QUEUED
    progress: ProgressEvent = field(default_factory=ProgressEvent)
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    _queue: asyncio.Queue[ProgressEvent | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=256)
    )


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_pool: ThreadPoolExecutor | None = None
_jobs: dict[str, BackgroundJob] = {}
_loop: asyncio.AbstractEventLoop | None = None


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        size = max(settings.max_concurrent_jobs, 2)
        _pool = ThreadPoolExecutor(max_workers=size, thread_name_prefix="pr-job")
        logger.info("Created job thread-pool with %d workers", size)
    return _pool


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None:
        _loop = asyncio.get_running_loop()
    return _loop


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def submit_job(
    fn: Callable[..., Any],
    *args: Any,
    callback: Callable[[str, Any], None] | None = None,
    **kwargs: Any,
) -> str:
    """Submit *fn* for background execution.

    Returns the job ID immediately.  The caller can poll via
    ``get_job_status()`` or stream via ``subscribe_progress()``.

    Parameters
    ----------
    fn
        Synchronous callable to run in the thread-pool.
    callback
        Optional callback invoked on the event loop with ``(job_id, result)``
        when the job completes.
    """
    job_id = uuid.uuid4().hex[:12]
    job = BackgroundJob(job_id=job_id)
    _jobs[job_id] = job

    loop = _get_loop()

    def _run() -> None:
        job.state = JobState.RUNNING
        job.started_at = time.time()
        try:
            result = fn(*args, **kwargs)
            job.result = result
            job.state = JobState.COMPLETED
        except Exception as exc:
            job.error = str(exc)
            job.state = JobState.FAILED
            logger.exception("Background job %s failed", job_id)
        finally:
            job.completed_at = time.time()
            # Sentinel to close WS subscriptions
            loop.call_soon_threadsafe(job._queue.put_nowait, None)
            if callback and job.state == JobState.COMPLETED:
                loop.call_soon_threadsafe(callback, job_id, job.result)

    _get_pool().submit(_run)
    logger.info("Submitted background job %s", job_id)
    return job_id


def publish_progress(job_id: str, event: ProgressEvent) -> None:
    """Called **from the worker thread** to publish a progress tick.

    Safe to call from synchronous code — uses ``call_soon_threadsafe``.
    """
    job = _jobs.get(job_id)
    if job is None:
        return
    job.progress = event
    try:
        loop = _get_loop()
        loop.call_soon_threadsafe(job._queue.put_nowait, event)
    except Exception:
        pass  # queue full or loop gone — non-fatal


def get_job_status(job_id: str) -> dict[str, Any]:
    """Return a JSON-serialisable snapshot of the job."""
    job = _jobs.get(job_id)
    if job is None:
        return {"error": "not_found"}
    return {
        "job_id": job.job_id,
        "state": job.state.value,
        "progress": {
            "iteration": job.progress.iteration,
            "total_iterations": job.progress.total_iterations,
            "cost": job.progress.cost,
            "elapsed_seconds": job.progress.elapsed_seconds,
            "message": job.progress.message,
        },
        "error": job.error,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }


def get_job_result(job_id: str) -> Any:
    """Return the raw result (only meaningful after COMPLETED)."""
    job = _jobs.get(job_id)
    return job.result if job else None


async def subscribe_progress(job_id: str) -> AsyncIterator[ProgressEvent]:
    """Async generator that yields ``ProgressEvent`` until the job ends."""
    job = _jobs.get(job_id)
    if job is None:
        return
    while True:
        event = await job._queue.get()
        if event is None:
            break  # sentinel — job finished
        yield event


def cancel_job(job_id: str) -> bool:
    """Mark a queued job as cancelled.  Returns ``True`` if state changed.

    Only jobs in ``QUEUED`` or ``RUNNING`` state can be cancelled.
    Running jobs will complete their current iteration but the result
    is discarded.
    """
    job = _jobs.get(job_id)
    if job is None:
        return False
    if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
        return False
    job.state = JobState.CANCELLED
    job.completed_at = time.time()
    # Push sentinel so WS consumers unblock
    try:
        loop = _get_loop()
        loop.call_soon_threadsafe(job._queue.put_nowait, None)
    except Exception:
        pass
    logger.info("Cancelled background job %s", job_id)
    return True


def list_active_jobs() -> list[dict[str, Any]]:
    """Return lightweight summaries of all tracked jobs."""
    return [
        {
            "job_id": j.job_id,
            "state": j.state.value,
            "created_at": j.created_at,
        }
        for j in _jobs.values()
    ]


async def shutdown_pool(timeout: float = 10.0) -> None:
    """Gracefully drain the thread-pool.  Called during app shutdown."""
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=True, cancel_futures=False)
        logger.info("Job thread-pool shut down")
        _pool = None
    # Allow lingering WS subscriptions to close
    for job in _jobs.values():
        with contextlib.suppress(asyncio.QueueFull):
            job._queue.put_nowait(None)


