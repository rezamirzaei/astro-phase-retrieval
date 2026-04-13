"""SQLAlchemy engine, session factory, and ORM base class.

Connection pooling is configured for production robustness:
* ``pool_pre_ping=True`` — test connections before checkout to handle
  stale / timed-out connections transparently.
* Pool size is 5 by default (SQLAlchemy default) with overflow up to 10.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from web.config import settings

_connect_args: dict[str, bool] = {}
_engine_kwargs: dict = {
    "pool_pre_ping": True,
}

if settings.database_url.startswith("sqlite"):
    _connect_args["check_same_thread"] = False
else:
    # Production DB (PostgreSQL) — configure connection pool
    _engine_kwargs.update(
        pool_size=5,
        max_overflow=10,
        pool_recycle=1800,  # recycle connections after 30 min
    )

engine = create_engine(settings.database_url, connect_args=_connect_args, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
