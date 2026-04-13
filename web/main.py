"""FastAPI application entry-point.

Run with::

    uvicorn web.main:app --reload

Environment variables (see ``web/config.py`` for full list):

* ``PR_SECRET_KEY``      — JWT signing secret (required in production)
* ``PR_ADMIN_PASSWORD``  — seed admin password (default: ``admin123``)
* ``PR_DATABASE_URL``    — SQLAlchemy DB URL (default: SQLite)
"""

from __future__ import annotations

import logging
import logging.config
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from web.config import settings
from web.database import Base, SessionLocal, engine
from web.models import User
from web.routers import algorithms, auth, crystallography, data, explain, results
from web.security import hash_password

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


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Create DB tables on startup and seed an admin account."""
    Base.metadata.create_all(bind=engine)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    # Seed admin account — password is read from PR_ADMIN_PASSWORD env var
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.username == "admin").first():
            admin_pw = settings.admin_password  # env-configurable, never hardcoded
            admin = User(
                email="admin@phase-retrieval.local",
                username="admin",
                hashed_password=hash_password(admin_pw),
                is_active=True,
            )
            db.add(admin)
            db.commit()
            logger.info("Seeded admin account (username=admin)")
    finally:
        db.close()

    logger.info(
        "Phase Retrieval API started — DB=%s  output_dir=%s",
        settings.database_url.split("?")[0],
        settings.output_dir,
    )
    yield
    logger.info("Phase Retrieval API shutting down")


app = FastAPI(
    title="Phase Retrieval — Web API",
    version="2.3.0",
    description=(
        "REST API for astronomical wavefront sensing and X-ray crystallography: "
        "run state-of-the-art phase-retrieval algorithms, compare results, "
        "and explore the underlying science."
    ),
    contact={"name": "Reza Mirzaeifard"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Security headers middleware — defence-in-depth
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
    # HSTS — enable when serving behind TLS
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    # Prevent caching of auth responses
    if "/api/auth/" in request.url.path:
        response.headers["Cache-Control"] = "no-store"
    return response


# Mount all routers
app.include_router(auth.router)
app.include_router(data.router)
app.include_router(algorithms.router)
app.include_router(results.router)
app.include_router(explain.router)
app.include_router(crystallography.router)


@app.get("/api/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Liveness probe — returns ``{"status": "ok"}``."""
    return {"status": "ok"}


@app.get("/api/version", tags=["health"])
def version() -> dict[str, str]:
    """Return the API version and Python runtime info."""
    import sys

    return {
        "api_version": app.version,
        "python": sys.version.split()[0],
    }
