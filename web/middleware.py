"""Custom middleware stack — request tracing, structured logging, timing.

Pure ASGI middleware implementations that avoid ``BaseHTTPMiddleware``'s
response-body buffering problem (which breaks streaming and WebSocket).

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

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("web.middleware")

# ---------------------------------------------------------------------------
# Context variable: accessible from ANY point in the request call-stack
# ---------------------------------------------------------------------------
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class RequestIDMiddleware:
    """Assign / propagate ``X-Request-ID`` and store it in a ContextVar.

    If the incoming request already carries ``X-Request-ID`` (e.g. from an
    API gateway), it is reused.  Otherwise a UUID-4 is generated.

    Pure ASGI implementation — no response-body buffering.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Extract or generate request ID
        headers = dict(scope.get("headers", []))
        rid = headers.get(b"x-request-id", b"").decode() or uuid.uuid4().hex
        request_id_ctx.set(rid)

        # Store in scope state for downstream access
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = rid

        async def send_with_rid(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers_list: list[tuple[bytes, bytes]] = list(message.get("headers", []))
                headers_list.append((b"x-request-id", rid.encode()))
                message["headers"] = headers_list
            await send(message)

        await self.app(scope, receive, send_with_rid)


class RequestLoggingMiddleware:
    """Log every HTTP request/response with structured fields.

    Emits: method, path, status, latency_ms, request_id, client_ip.

    Pure ASGI implementation — no response-body buffering.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        method = scope.get("method", "?")
        path = scope.get("path", "?")
        status_code: int | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            rid = scope.get("state", {}).get("request_id", request_id_ctx.get(""))
            client = scope.get("client")
            client_ip = client[0] if client else "-"

            # Skip noisy health-check logs
            if path not in ("/api/health", "/api/readiness"):
                logger.info(
                    "%s %s → %s  (%.1f ms)  [rid=%s ip=%s]",
                    method,
                    path,
                    status_code,
                    latency_ms,
                    rid,
                    client_ip,
                )
