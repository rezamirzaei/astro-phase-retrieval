"""FastAPI application entry-point.

Run with::

    uvicorn web.main:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.config import settings
from web.database import Base, SessionLocal, engine
from web.models import User
from web.routers import algorithms, auth, crystallography, data, explain, results
from web.security import hash_password


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Create DB tables on startup and seed an admin account."""
    Base.metadata.create_all(bind=engine)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    # Seed hardcoded admin account
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.username == "admin").first():
            admin = User(
                email="admin@phase-retrieval.local",
                username="admin",
                hashed_password=hash_password("admin123"),
                is_active=True,
            )
            db.add(admin)
            db.commit()
    finally:
        db.close()

    yield


app = FastAPI(
    title="Phase Retrieval — Web API",
    version="2.0.2",
    description=(
        "REST API for astronomical wavefront sensing: "
        "run algorithms, compare results, and explore the science."
    ),
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

# Mount all routers
app.include_router(auth.router)
app.include_router(data.router)
app.include_router(algorithms.router)
app.include_router(results.router)
app.include_router(explain.router)
app.include_router(crystallography.router)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
