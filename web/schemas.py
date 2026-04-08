"""Pydantic v2 request / response schemas for every API endpoint."""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class UserCreate(BaseModel):
    """POST /api/auth/register body."""

    email: str = Field(max_length=255)
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def _validate_email(cls, v: str) -> str:
        if not re.match(r"^[\w.+-]+@[\w.-]+\.\w+$", v):
            raise ValueError("Invalid email address")
        return v.lower().strip()

    @field_validator("username")
    @classmethod
    def _validate_username(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Username must be alphanumeric (plus _ and -)")
        return v.strip()


class UserResponse(BaseModel):
    """Serialised user (never includes password)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    username: str
    is_active: bool
    created_at: datetime


class Token(BaseModel):
    """JWT response."""

    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    """POST /api/auth/login body."""

    username: str
    password: str


# ---------------------------------------------------------------------------
# Algorithm execution
# ---------------------------------------------------------------------------


class AlgorithmRunRequest(BaseModel):
    """POST /api/algorithms/run body."""

    fits_filename: str
    algorithm: str
    max_iterations: int = Field(default=300, ge=1, le=10_000)
    beta: float = Field(default=0.9, gt=0, le=1.0)
    beta_schedule: str = Field(default="constant")
    momentum: float = Field(default=0.0, ge=0, le=0.99)
    tv_weight: float = Field(default=0.0, ge=0)
    noise_model: str = Field(default="gaussian")
    grid_size: int = Field(default=128, ge=64, le=512)


class CompareRequest(BaseModel):
    """POST /api/algorithms/compare body."""

    fits_filename: str
    max_iterations: int = Field(default=300, ge=1, le=10_000)
    grid_size: int = Field(default=128, ge=64, le=512)
    algorithms: list[str] | None = None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class JobResponse(BaseModel):
    """Serialised algorithm-run result."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    algorithm: str
    status: str
    fits_filename: str
    strehl_ratio: float | None = None
    rms_phase_rad: float | None = None
    n_iterations: int | None = None
    elapsed_seconds: float | None = None
    converged: bool | None = None
    created_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    plots: list[str] = []


class CompareResponse(BaseModel):
    """Response for /api/algorithms/compare."""

    results: list[JobResponse]
    comparison_plots: list[str] = []


# ---------------------------------------------------------------------------
# Data management
# ---------------------------------------------------------------------------


class PresetInfo(BaseModel):
    key: str
    description: str


class FitsFileInfo(BaseModel):
    filename: str
    filepath: str
    size_bytes: int


class SyntheticRequest(BaseModel):
    """POST /api/data/synthetic body."""

    name: str = Field(default="synthetic", max_length=100)
    grid_size: int = Field(default=128, ge=64, le=512)
    aberration_rms: float = Field(default=0.5, ge=0.1, le=3.0)
    telescope: str = Field(default="hst")
    filter_name: str = Field(default="F606W")


# ---------------------------------------------------------------------------
# Explain / educational
# ---------------------------------------------------------------------------


class AlgorithmExplain(BaseModel):
    key: str
    name: str
    category: str
    description: str
    reference: str


class MetricExplain(BaseModel):
    name: str
    description: str
    unit: str


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class DashboardStats(BaseModel):
    total_runs: int
    completed_runs: int
    best_strehl: float | None
    algorithms_used: list[str]
    recent_jobs: list[JobResponse]

