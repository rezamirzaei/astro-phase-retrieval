"""FastAPI application entry-point — Phase Retrieval Web API.

Run with::

    uvicorn web.main:app --reload

Environment variables (see ``web/config.py`` for full list):

* ``PR_SECRET_KEY``      — JWT signing secret (required in production)
* ``PR_ADMIN_PASSWORD``  — optional seed admin password for local/bootstrap use
* ``PR_DATABASE_URL``    — SQLAlchemy DB URL (default: SQLite)
* ``PR_CORS_ORIGINS``    — comma-separated allowed origins
"""

from __future__ import annotations

import logging  # noqa: F401 — used by logging.config
import logging.config
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from web.config import settings
from web.database import Base, SessionLocal, engine
from web.middleware import RequestIDMiddleware, RequestLoggingMiddleware
from web.models import User
from web.routers import algorithms, auth, crystallography, data, explain, results, studies
from web.routers import jobs as jobs_router
from web.routers import ws as ws_router
from web.schemas import HealthDetail, ReadinessDetail
from web.security import hash_password
from web.services.job_queue import shutdown_pool

# ---------------------------------------------------------------------------
# Structured JSON logging (machine-readable for log aggregators)
# ---------------------------------------------------------------------------
_LOG_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "logging.Formatter",
            "fmt": (
                '{"time":"%(asctime)s","level":"%(levelname)s",'
                '"logger":"%(name)s","msg":"%(message)s"}'
            ),
        },
        "plain": {
            "format": "%(asctime)s  %(name)-32s  %(levelname)-8s  %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "plain",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "uvicorn.access": {"propagate": False},
        "src": {"level": "INFO", "propagate": True},
        "web": {"level": "INFO", "propagate": True},
    },
}

logging.config.dictConfig(_LOG_CONFIG)
logger = logging.getLogger(__name__)

_startup_time: float = 0.0


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Create DB tables on startup, seed admin, and drain resources on shutdown."""
    global _startup_time
    _startup_time = time.time()

    # ── Startup ─────────────────────────────────────────────────────
    # Set matplotlib backend early, before any pyplot import, to avoid
    # thread-safety issues when called later from worker threads.
    import matplotlib
    matplotlib.use("Agg")

    Base.metadata.create_all(bind=engine)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "uploads").mkdir(parents=True, exist_ok=True)

    # Seed admin account only when explicitly configured.
    db = SessionLocal()
    try:
        admin_pw = settings.admin_password
        if admin_pw and not db.query(User).filter(User.username == "admin").first():
            admin = User(
                email="admin@phase-retrieval.local",
                username="admin",
                hashed_password=hash_password(admin_pw),
                is_active=True,
            )
            db.add(admin)
            db.commit()
            logger.info("Seeded admin account (username=admin)")
        elif not admin_pw:
            logger.info("Admin seeding disabled: PR_ADMIN_PASSWORD not configured")
    finally:
        db.close()

    logger.info(
        "Phase Retrieval API started — DB=%s  output_dir=%s",
        settings.database_url.split("?")[0],
        settings.output_dir,
    )
    yield

    # ── Shutdown ────────────────────────────────────────────────────
    logger.info("Phase Retrieval API shutting down — draining job queue…")
    await shutdown_pool(timeout=settings.shutdown_timeout_seconds)
    engine.dispose()
    logger.info("Phase Retrieval API shutdown complete")


# ---------------------------------------------------------------------------
# OpenAPI tag descriptions — enriched documentation
# ---------------------------------------------------------------------------

tags_metadata = [
    {
        "name": "auth",
        "description": "User registration, login, JWT refresh, and account introspection.",
    },
    {
        "name": "data",
        "description": (
            "Data management — list / upload FITS & NPY files, generate synthetic PSFs, "
            "and download curated observation presets from MAST."
        ),
    },
    {
        "name": "algorithms",
        "description": (
            "Run phase-retrieval algorithms (GS, HIO, RAAR, WF, FISTA, ADMM, PINN, …), "
            "compare multiple algorithms, and benchmark on synthetic test suites."
        ),
    },
    {
        "name": "results",
        "description": (
            "Browse results, view / download plots, export reproducibility archives, "
            "and aggregate dashboard statistics."
        ),
    },
    {
        "name": "studies",
        "description": "Multi-observation validation campaigns and robustness studies.",
    },
    {
        "name": "crystallography",
        "description": (
            "X-ray crystallography phase retrieval — load CIF files from COD, "
            "simulate diffraction patterns, and solve the crystallographic phase problem."
        ),
    },
    {
        "name": "explain",
        "description": (
            "Educational endpoints — algorithm theory, metric descriptions, science primer."
        ),
    },
    {
        "name": "jobs",
        "description": "Background job queue — submit, poll, list asynchronous compute jobs.",
    },
    {
        "name": "websocket",
        "description": "Real-time WebSocket streaming of algorithm progress.",
    },
    {
        "name": "health",
        "description": "Liveness, readiness, and version probes for orchestration / monitoring.",
    },
]

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Phase Retrieval — Web API",
    version="3.0.0",
    description=(
        "Production-grade REST + WebSocket API for astronomical wavefront sensing "
        "and X-ray crystallography.  Run state-of-the-art phase-retrieval algorithms, "
        "compare results in real-time, and explore the underlying science.\n\n"
        "**Key features:**\n"
        "- 10 algorithms (GS, HIO, RAAR, WF, FISTA, ADMM, DR, SparsePR, PINN, PD)\n"
        "- Background job queue with WebSocket progress streaming\n"
        "- Paginated results with one-click ZIP export\n"
        "- File upload for custom FITS/NPY datasets\n"
        "- JWT auth with refresh tokens, rate limiting, and audit logging\n"
        "- X-ray crystallography workflow (COD → diffraction → phase retrieval)\n"
    ),
    contact={"name": "Reza Mirzaeifard"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
    openapi_tags=tags_metadata,
    responses={
        401: {"description": "Authentication required — provide a valid JWT bearer token."},
        422: {"description": "Validation error — check the request body against the schema."},
    },
)

# ---------------------------------------------------------------------------
# Middleware stack (applied in reverse order)
# ---------------------------------------------------------------------------

# 1. CORS — must be outermost so pre-flight responses include CORS headers
# noinspection PyTypeChecker
app.add_middleware(  # type: ignore[arg-type]
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID", "Content-Disposition"],
)

# 2. Request ID
app.add_middleware(RequestIDMiddleware)

# 3. Request logging
app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# Security headers — defence-in-depth
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_security_headers(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
    """Inject hardening headers into every response."""
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
    if "/api/auth/" in request.url.path:
        response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------------------
# Global exception handler — catch-all for unhandled errors
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a clean JSON 500 for any unhandled exception.

    Prevents stack-trace leakage to the client while logging full details
    server-side for debugging.
    """
    rid = getattr(request.state, "request_id", "unknown")
    logger.exception("Unhandled exception [rid=%s]: %s", rid, exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": rid,
        },
    )


# ---------------------------------------------------------------------------
# Mount all routers
# ---------------------------------------------------------------------------

# Core API
app.include_router(auth.router)
app.include_router(data.router)
app.include_router(algorithms.router)
app.include_router(results.router)
app.include_router(studies.router)
app.include_router(explain.router)
app.include_router(crystallography.router)

# New in v3.0
app.include_router(jobs_router.router)
app.include_router(ws_router.router)


# ---------------------------------------------------------------------------
# Health / readiness / version — unversioned, no auth
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthDetail, tags=["health"])
def health_check() -> HealthDetail:
    """Liveness probe — always returns ``status: ok`` if the process is alive."""
    uptime = time.time() - _startup_time if _startup_time else 0.0
    return HealthDetail(status="ok", version=app.version, uptime_seconds=round(uptime, 1))


@app.get("/api/readiness", response_model=ReadinessDetail, tags=["health"])
def readiness_check() -> ReadinessDetail:
    """Readiness probe — verifies database connectivity and disk write access.

    Returns 200 with per-subsystem status.  Kubernetes / load-balancers
    should use this to decide whether to route traffic.
    """
    detail = ReadinessDetail(version=app.version)

    # Check DB
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
        detail.db = "ok"
    except SQLAlchemyError as exc:
        detail.db = f"error: {exc}"

    # Check disk write
    try:
        test_file = settings.output_dir / ".readiness_check"
        test_file.write_text("ok")
        test_file.unlink()
        detail.disk = "ok"
    except OSError as exc:
        detail.disk = f"error: {exc}"

    return detail


@app.get("/api/version", tags=["health"])
def version() -> dict[str, str]:
    """Return the API version and Python runtime info."""

    return {
        "api_version": app.version,
        "python": sys.version.split()[0],
    }
