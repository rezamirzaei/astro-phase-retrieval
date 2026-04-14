"""Background job queue with real-time progress streaming.

Design
------
* A ``ThreadPoolExecutor`` runs compute-heavy phase-retrieval jobs off the
  async event loop.
* Each job is assigned a UUID.  While running, the executor publishes
  ``ProgressEvent`` messages to an ``asyncio.Queue`` that WebSocket clients
  consume in real-time.
* Jobs go through states: ``queued → running → completed | failed``.

Thread-safety
-------------
* ``_jobs`` dict mutations are protected by ``_jobs_lock`` (a
  ``threading.Lock``), safe for both sync worker threads and the async
  event loop.
* Each ``BackgroundJob`` carries a ``threading.Event`` (``cancel_event``)
  that running functions can poll for cooperative cancellation.
* Completed jobs are evicted after ``_JOB_TTL_SECONDS`` to prevent
  unbounded memory growth.

The module exposes three entry-points used by routers:

* ``submit_job`` — enqueue and return immediately.
* ``get_job_status`` — poll from HTTP.
* ``subscribe_progress`` — async generator for WebSocket streaming.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from typing import Any

from web.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Completed / failed / cancelled jobs are evicted after this many seconds.
_JOB_TTL_SECONDS: float = 3600.0  # 1 hour

# Maximum tracked jobs (hard cap to prevent OOM under adversarial load).
_MAX_TRACKED_JOBS: int = 10_000


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
    loop: asyncio.AbstractEventLoop | None = None

    # Cooperative cancellation signal — worker threads should poll this.
    cancel_event: threading.Event = field(default_factory=threading.Event)

    # The asyncio queue is created lazily via ``_ensure_queue`` so that the
    # dataclass can be safely instantiated outside an async context.
    _queue: asyncio.Queue[ProgressEvent | None] | None = field(default=None, repr=False)
    _queue_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _ensure_queue(self) -> asyncio.Queue[ProgressEvent | None]:
        """Lazily create the asyncio.Queue on first access (thread-safe)."""
        if self._queue is None:
            with self._queue_lock:
                if self._queue is None:
                    self._queue = asyncio.Queue(maxsize=256)
        assert self._queue is not None  # guaranteed by double-checked init above
        return self._queue


# ---------------------------------------------------------------------------
# Global state (protected by _jobs_lock)
# ---------------------------------------------------------------------------
_pool: ThreadPoolExecutor | None = None
_jobs: dict[str, BackgroundJob] = {}
_jobs_lock = threading.Lock()


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    pool = _pool
    if pool is None:
        size = max(settings.max_concurrent_jobs, 2)
        pool = ThreadPoolExecutor(max_workers=size, thread_name_prefix="pr-job")
        _pool = pool
        logger.info("Created job thread-pool with %d workers", size)
    return pool


def _evict_stale_jobs() -> None:
    """Remove completed/failed/cancelled jobs older than the TTL.

    Must be called under ``_jobs_lock``.
    """
    now = time.time()
    terminal_states = {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}
    stale = [
        jid
        for jid, j in _jobs.items()
        if j.state in terminal_states
        and j.completed_at is not None
        and (now - j.completed_at) > _JOB_TTL_SECONDS
    ]
    for jid in stale:
        del _jobs[jid]
    if stale:
        logger.debug("Evicted %d stale jobs", len(stale))


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

    # Capture the running event loop at submission time (always called from
    # an async context, e.g. a FastAPI route).
    loop = asyncio.get_running_loop()
    job.loop = loop

    with _jobs_lock:
        _evict_stale_jobs()
        if len(_jobs) >= _MAX_TRACKED_JOBS:
            raise RuntimeError(f"Job queue capacity exceeded ({_MAX_TRACKED_JOBS} tracked jobs)")
        _jobs[job_id] = job

    def _run() -> None:
        with _jobs_lock:
            job.state = JobState.RUNNING
            job.started_at = time.time()
        try:
            # Check cancellation before starting heavy work
            if job.cancel_event.is_set():
                with _jobs_lock:
                    job.state = JobState.CANCELLED
                return

            result = fn(*args, **kwargs)

            with _jobs_lock:
                if job.cancel_event.is_set():
                    job.state = JobState.CANCELLED
                else:
                    job.result = result
                    job.state = JobState.COMPLETED
        except Exception as exc:
            with _jobs_lock:
                job.error = str(exc)
                job.state = JobState.FAILED
            logger.exception("Background job %s failed", job_id)
        finally:
            with _jobs_lock:
                job.completed_at = time.time()
            # Sentinel to close WS subscriptions
            _safe_put_sentinel(job)
            if callback and job.state == JobState.COMPLETED:
                loop.call_soon_threadsafe(callback, job_id, job.result)

    _get_pool().submit(_run)
    logger.info("Submitted background job %s", job_id)
    return job_id


def _put_queue_item(
    q: asyncio.Queue[ProgressEvent | None],
    item: ProgressEvent | None,
    *,
    job_id: str,
    item_name: str,
    force: bool = False,
) -> None:
    """Push an item to the asyncio queue from the loop thread."""
    if force and q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            logger.debug("Progress queue emptied before forcing %s for job %s", item_name, job_id)
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        logger.warning("Progress queue full; dropped %s for job %s", item_name, job_id)


def _schedule_queue_item(
    job: BackgroundJob,
    item: ProgressEvent | None,
    *,
    item_name: str,
    force: bool = False,
) -> None:
    """Schedule a queue write on the owning event loop."""
    if job.loop is None:
        logger.warning("No event loop recorded for background job %s", job.job_id)
        return
    q = job._ensure_queue()
    try:
        job.loop.call_soon_threadsafe(
            partial(
                _put_queue_item,
                q,
                item,
                job_id=job.job_id,
                item_name=item_name,
                force=force,
            )
        )
    except RuntimeError:
        logger.warning("Event loop already closed for background job %s", job.job_id)


def _safe_put_sentinel(job: BackgroundJob) -> None:
    """Thread-safe push of the completion sentinel to the job's queue."""
    _schedule_queue_item(job, None, item_name="completion sentinel", force=True)


def publish_progress(job_id: str, event: ProgressEvent) -> None:
    """Called **from the worker thread** to publish a progress tick.

    Safe to call from synchronous code — uses ``call_soon_threadsafe``.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return
    job.progress = event
    _schedule_queue_item(job, event, item_name="progress event")


def is_job_cancelled(job_id: str) -> bool:
    """Check whether the given job has been cancelled.

    Algorithm functions can poll this periodically to support cooperative
    cancellation::

        for i in range(max_iterations):
            if is_job_cancelled(job_id):
                break
            # ... compute ...
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return False
    return job.cancel_event.is_set()


def get_job_status(job_id: str) -> dict[str, Any]:
    """Return a JSON-serialisable snapshot of the job."""
    with _jobs_lock:
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
    with _jobs_lock:
        job = _jobs.get(job_id)
    return job.result if job else None


async def subscribe_progress(job_id: str) -> AsyncIterator[ProgressEvent]:
    """Async generator that yields ``ProgressEvent`` until the job ends."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return
    q = job._ensure_queue()
    while True:
        event = await q.get()
        if event is None:
            break  # sentinel — job finished
        yield event


def cancel_job(job_id: str) -> bool:
    """Mark a queued or running job as cancelled.  Returns ``True`` if state changed.

    Sets the ``cancel_event`` so that cooperative worker code can detect
    cancellation promptly.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
            return False
        job.state = JobState.CANCELLED
        job.completed_at = time.time()
        job.cancel_event.set()

    # Push sentinel so WS consumers unblock
    _safe_put_sentinel(job)
    logger.info("Cancelled background job %s", job_id)
    return True


def list_active_jobs() -> list[dict[str, Any]]:
    """Return lightweight summaries of all tracked jobs."""
    with _jobs_lock:
        snapshot = list(_jobs.values())
    return [
        {
            "job_id": j.job_id,
            "state": j.state.value,
            "created_at": j.created_at,
        }
        for j in snapshot
    ]


async def shutdown_pool(timeout: float = 10.0) -> None:
    """Gracefully drain the thread-pool.  Called during app shutdown.

    Respects the *timeout* parameter: after *timeout* seconds, remaining
    futures are cancelled to avoid blocking shutdown indefinitely.
    """
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=False)
        # Wait up to `timeout` for threads to finish, then force
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not any(t.is_alive() for t in _pool._threads):
                break
            await asyncio.sleep(0.2)
        else:
            logger.warning(
                "Job thread-pool did not drain within %.1fs — cancelling remaining",
                timeout,
            )
            _pool.shutdown(wait=False, cancel_futures=True)
        logger.info("Job thread-pool shut down")
        _pool = None

    # Allow lingering WS subscriptions to close
    with _jobs_lock:
        for job in _jobs.values():
            _safe_put_sentinel(job)
