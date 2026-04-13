"""Custom middleware stack — request tracing, structured logging, timing.

Middleware is applied in **reverse registration order**, so the first
middleware registered wraps all subsequent ones:

1. ``RequestIDMiddleware`` — assigns a unique ``X-Request-ID`` to every
   request and makes it available via a ``contextvars.ContextVar`` for
   structured logging.
2. ``RequestLoggingMiddleware`` — logs every request/response cycle with
   method, path, status code, latency, and the request ID.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("web.middleware")

# ---------------------------------------------------------------------------
# Context variable: accessible from ANY point in the request call-stack
# ---------------------------------------------------------------------------
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign / propagate ``X-Request-ID`` and store it in a ContextVar.

    If the incoming request already carries ``X-Request-ID`` (e.g. from an
    API gateway), it is reused.  Otherwise a UUID-4 is generated.

    The ID is:
    * stored in ``request.state.request_id``
    * injected into the **response** ``X-Request-ID`` header
    * pushed into ``request_id_ctx`` so that structured logs include it
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = rid
        request_id_ctx.set(rid)

        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request/response with structured fields.

    Emits: method, path, status, latency_ms, request_id, client_ip.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000.0

        rid = getattr(request.state, "request_id", request_id_ctx.get(""))
        client_ip = request.client.host if request.client else "-"

        # Skip noisy health-check logs
        if request.url.path not in ("/api/health", "/api/readiness"):
            logger.info(
                "%s %s → %s  (%.1f ms)  [rid=%s ip=%s]",
                request.method,
                request.url.path,
                response.status_code,
                latency_ms,
                rid,
                client_ip,
            )

        return response

