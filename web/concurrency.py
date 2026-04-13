"""Concurrency control for heavy algorithm jobs.

The semaphore limits the number of algorithm runs that can execute in
parallel, preventing resource exhaustion on the server.  It is created
lazily on first access (inside the running event loop).
"""

from __future__ import annotations

import asyncio

from web.config import settings

_job_semaphore: asyncio.Semaphore | None = None


def get_job_semaphore() -> asyncio.Semaphore:
    """Return the global job semaphore (created lazily inside the event loop)."""
    global _job_semaphore
    if _job_semaphore is None:
        limit = settings.max_concurrent_jobs if settings.max_concurrent_jobs > 0 else 128
        _job_semaphore = asyncio.Semaphore(limit)
    return _job_semaphore
