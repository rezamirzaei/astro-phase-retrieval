"""Pydantic v2 request / response schemas for every API endpoint.

Design principles:
* Enum-typed fields give 422 validation at the API boundary — invalid algorithm
  names or noise models are rejected before reaching the service layer.
* ``AlgorithmParams`` is the single source of truth for shared hyper-parameters;
  ``AlgorithmRunRequest`` and ``AlgorithmDefaults`` both inherit from it to
  avoid drift.
* All ``from_attributes=True`` models handle ORM-to-schema conversion.
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.models.config import AlgorithmName, BetaSchedule, NoiseModel

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
    """JWT response (access + refresh token pair)."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    """POST /api/auth/refresh body."""

    refresh_token: str


class LoginRequest(BaseModel):
    """POST /api/auth/login body."""

    username: str
    password: str


# ---------------------------------------------------------------------------
# Algorithm execution — shared base and request/response models
# ---------------------------------------------------------------------------


class AlgorithmParams(BaseModel):
    """Shared algorithm hyper-parameters — the single source of truth.

    Both ``AlgorithmRunRequest`` and ``AlgorithmDefaults`` inherit from here
    so the two schemas stay in sync automatically.
    """

    max_iterations: int = Field(default=300, ge=1, le=10_000)
    beta: float = Field(default=0.9, gt=0, le=1.0)
    beta_schedule: BetaSchedule = Field(default=BetaSchedule.CONSTANT)
    momentum: float = Field(default=0.0, ge=0, le=0.99)
    tv_weight: float = Field(default=0.0, ge=0, description="TV regularisation weight (0 = off)")
    noise_model: NoiseModel = Field(default=NoiseModel.GAUSSIAN)
    grid_size: int = Field(default=128, ge=64, le=512)


class AlgorithmRunRequest(AlgorithmParams):
    """POST /api/algorithms/run body."""

    fits_filename: str
    algorithm: AlgorithmName


class AlgorithmDefaults(AlgorithmParams):
    """Recommended defaults for a specific algorithm (returned by GET /api/algorithms/)."""


class AlgorithmInfo(BaseModel):
    """Metadata for an available algorithm."""

    key: str
    name: str
    defaults: AlgorithmDefaults


class CompareRequest(BaseModel):
    """POST /api/algorithms/compare body."""

    fits_filename: str
    max_iterations: int = Field(default=300, ge=1, le=10_000)
    grid_size: int = Field(default=128, ge=64, le=512)
    algorithms: list[AlgorithmName] | None = None


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
    plots: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class PlotReference(BaseModel):
    """Reference to a protected plot asset served by the API."""

    job_id: int
    name: str


class CompareResponse(BaseModel):
    """Response for /api/algorithms/compare."""

    results: list[JobResponse]
    comparison_plots: list[PlotReference] = Field(default_factory=list)


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


# ---------------------------------------------------------------------------
# Crystallography
# ---------------------------------------------------------------------------


class CodPresetInfo(BaseModel):
    key: str
    description: str


class CifFileInfo(BaseModel):
    filename: str
    filepath: str
    size_bytes: int


class CrystallographyRunRequest(BaseModel):
    """POST /api/crystallography/run body."""

    cif_filename: str
    algorithm: AlgorithmName = AlgorithmName.HYBRID_INPUT_OUTPUT
    max_iterations: int = Field(default=500, ge=1, le=10_000)
    beta: float = Field(default=0.9, gt=0, le=1.0)
    grid_size: int = Field(default=128, ge=64, le=512)


class CrystallographyJobResponse(BaseModel):
    """Serialised crystallography job result."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    algorithm: str
    status: str
    cif_filename: str
    cod_id: str = ""
    formula: str = ""
    r_factor: float | None = None
    n_iterations: int | None = None
    elapsed_seconds: float | None = None
    converged: bool | None = None
    created_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    plots: list[str] = Field(default_factory=list)


class CrystallographyCompareRequest(BaseModel):
    """POST /api/crystallography/compare body."""

    cif_filename: str
    max_iterations: int = Field(default=500, ge=1, le=10_000)
    grid_size: int = Field(default=128, ge=64, le=512)
    algorithms: list[AlgorithmName] | None = None


class CrystallographyCompareResponse(BaseModel):
    """Response for /api/crystallography/compare."""

    results: list[CrystallographyJobResponse]


class SimulateDiffractionRequest(BaseModel):
    """POST /api/crystallography/simulate body."""

    cif_filename: str
    grid_size: int = Field(default=128, ge=64, le=512)
