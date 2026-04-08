"""SQLAlchemy ORM models for the web application."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from web.database import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    """Registered user account."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    jobs: Mapped[list[Job]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Job(Base):
    """One algorithm-run record (pending → running → completed | failed)."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    algorithm: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    fits_filename: Mapped[str] = mapped_column(String(500))
    config_json: Mapped[str] = mapped_column(Text, default="{}")

    # Result metrics (populated on completion)
    strehl_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    rms_phase_rad: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_iterations: Mapped[int | None] = mapped_column(Integer, nullable=True)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    converged: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    cost_history_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_dir: Mapped[str | None] = mapped_column(String(500), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="jobs")

