"""WebSocket endpoint for real-time job progress streaming.

Usage from JavaScript::

    const ws = new WebSocket(`ws://host/api/ws/jobs/${jobId}?token=${jwt}`);
    ws.onmessage = (e) => {
        const ev = JSON.parse(e.data);
        // ev = {iteration, total_iterations, cost, elapsed_seconds, message}
    };

Authentication: the JWT access token is passed as a ``token`` query
parameter (WebSocket does not support Authorization headers natively).
"""

from __future__ import annotations

import contextlib
import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from web.security import decode_access_token
from web.services.job_queue import get_job_status, subscribe_progress

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)


@router.websocket("/api/ws/jobs/{job_id}")
async def ws_job_progress(
    websocket: WebSocket,
    job_id: str,
    token: str = Query(...),
) -> None:
    """Stream real-time progress events for a background job.

    The connection is authenticated via a ``token`` query parameter
    containing a valid JWT access token.  Unauthenticated connections
    are immediately rejected with code 4001.
    """
    # ── Auth ────────────────────────────────────────────────────────
    payload = decode_access_token(token)
    if payload.get("type") != "access" or not payload.get("sub"):
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    # ── Validate job exists ─────────────────────────────────────────
    status = get_job_status(job_id)
    if status.get("error") == "not_found":
        await websocket.close(code=4004, reason="Job not found")
        return

    await websocket.accept()
    logger.info("WS connected for job %s", job_id)

    try:
        # Send initial snapshot
        await websocket.send_text(json.dumps(status))

        # Stream incremental progress
        async for event in subscribe_progress(job_id):
            msg = {
                "iteration": event.iteration,
                "total_iterations": event.total_iterations,
                "cost": event.cost,
                "elapsed_seconds": event.elapsed_seconds,
                "message": event.message,
            }
            await websocket.send_text(json.dumps(msg))

        # Job finished — send final status then close
        final = get_job_status(job_id)
        await websocket.send_text(json.dumps({**final, "final": True}))
        await websocket.close(code=1000)

    except WebSocketDisconnect:
        logger.info("WS client disconnected for job %s", job_id)
    except Exception:
        logger.exception("WS error for job %s", job_id)
        with contextlib.suppress(Exception):
            await websocket.close(code=1011, reason="Internal error")

