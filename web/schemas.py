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
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.models.config import (
    AlgorithmName,
    BetaSchedule,
    NoiseModel,
    Regulariser,
    TelescopeType,
)

# ---------------------------------------------------------------------------
# Generics for paginated responses
# ---------------------------------------------------------------------------
T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Wrapper for paginated list endpoints.

    Example response::

        {
          "items": [...],
          "total": 142,
          "skip": 0,
          "limit": 50
        }
    """

    items: list[T]
    total: int
    skip: int
    limit: int

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
    tolerance: float = Field(default=1e-8, gt=0)
    beta: float = Field(default=0.9, gt=0, le=1.0)
    beta_schedule: BetaSchedule = Field(default=BetaSchedule.CONSTANT)
    momentum: float = Field(default=0.0, ge=0, le=0.99)
    tv_weight: float = Field(default=0.0, ge=0, description="TV regularisation weight (0 = off)")
    noise_model: NoiseModel = Field(default=NoiseModel.GAUSSIAN)
    n_starts: int = Field(default=1, ge=1, le=32)
    uncertainty_samples: int = Field(default=0, ge=0, le=32)
    admm_rho: float = Field(default=1.0, gt=0)
    wf_step_size: float = Field(default=0.5, gt=0, le=10.0)
    wf_spectral_init: bool = True
    spectral_init: bool = True
    regulariser: Regulariser = Field(default=Regulariser.NONE)
    proximal_weight: float = Field(default=1e-3, ge=0)
    sparsity_threshold: float = Field(default=0.1, ge=0, le=1.0)
    sparsity_keep_fraction: float = Field(default=1.0, gt=0, le=1.0)
    grid_size: int = Field(default=128, ge=64, le=512)


class AlgorithmRunRequest(AlgorithmParams):
    """POST /api/algorithms/run body."""

    fits_filename: str = Field(..., examples=["test_synth_64.npy"])
    algorithm: AlgorithmName = Field(..., examples=["raar"])

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "fits_filename": "test_synth_64.npy",
                    "algorithm": "raar",
                    "max_iterations": 300,
                    "beta": 0.9,
                    "grid_size": 128,
                }
            ]
        }
    )


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


class ValidationCampaignRequest(AlgorithmParams):
    """POST /api/studies/validation-campaign body."""

    fits_filenames: list[str] | None = None
    algorithm: AlgorithmName = Field(default=AlgorithmName.RAAR)


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
    verification_supported: bool = False
    baseline_key: str | None = None


class FitsFileInfo(BaseModel):
    filename: str
    filepath: str
    size_bytes: int


class SyntheticRequest(BaseModel):
    """POST /api/data/synthetic body."""

    name: str = Field(default="synthetic", max_length=100, examples=["my_psf"])
    grid_size: int = Field(default=128, ge=64, le=512)
    aberration_rms: float = Field(default=0.5, ge=0.1, le=3.0)
    n_zernike: int = Field(default=15, ge=3, le=50)
    telescope: TelescopeType = Field(default=TelescopeType.HST, examples=["hst", "jwst"])
    filter_name: str = Field(default="F606W", examples=["F606W", "F814W"])
    photon_count: float = Field(
        default=0.0, ge=0.0, description="Total photon count (0 = noiseless)"
    )
    read_noise_std: float = Field(default=0.0, ge=0.0, description="Gaussian read noise σ")
    center_offset_row_pixels: float = Field(default=0.0, ge=-8.0, le=8.0)
    center_offset_col_pixels: float = Field(default=0.0, ge=-8.0, le=8.0)
    background_level: float = Field(default=0.0, ge=0.0)
    bandwidth_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    spectral_samples: int = Field(default=1, ge=1, le=15)
    spectral_weighting: str = Field(default="delta", pattern="^(delta|gaussian|uniform)$")
    field_defocus_waves: float = Field(default=0.0, ge=-5.0, le=5.0)
    detector_sigma_pixels: float = Field(default=0.0, ge=0.0, le=10.0)
    jitter_sigma_pixels: float = Field(default=0.0, ge=0.0, le=10.0)
    pixel_integration_width: float = Field(default=1.0, ge=0.5, le=4.0)
    random_seed: int = Field(default=42)


class BenchmarkCaseInfo(BaseModel):
    key: str
    description: str


class BenchmarkAggregateRow(BaseModel):
    algorithm: str
    n_cases: int
    mean_score: float
    mean_ssim: float
    mean_phase_rms_error_rad: float
    mean_radial_profile_error: float
    mean_encircled_energy_error: float
    mean_elapsed_seconds: float
    converged_fraction: float


class BenchmarkStudyRow(BaseModel):
    algorithm: str
    clean_mean_score: float
    stress_mean_score: float
    robustness_drop: float
    failure_rate: float
    convergence_stability: float
    worst_case: str


class BenchmarkRunRequest(BaseModel):
    algorithms: list[AlgorithmName] | None = None
    cases: list[str] | None = None
    max_iterations: int = Field(default=80, ge=1, le=10_000)
    beta: float = Field(default=0.9, gt=0, le=1.0)
    random_seed: int = Field(default=42)


class BenchmarkResponse(BaseModel):
    selected_algorithms: list[str]
    selected_cases: list[BenchmarkCaseInfo]
    aggregate: list[BenchmarkAggregateRow]
    study: list[BenchmarkStudyRow]
    records_count: int
    artifacts: dict[str, str] = Field(default_factory=dict)


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


class ArtifactContentResponse(BaseModel):
    name: str
    format: str
    content: object | str


class ValidationCampaignResponse(BaseModel):
    campaign_id: str
    selected_files: list[str]
    summary: dict[str, Any]
    records: list[dict[str, Any]]
    consistency: dict[str, Any]
    reference_summary: dict[str, Any]
    artifacts: list[str]


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

    cif_filename: str = Field(..., examples=["test_nacl.cif"])
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


# ---------------------------------------------------------------------------
# Health & readiness
# ---------------------------------------------------------------------------


class HealthDetail(BaseModel):
    """Response for ``GET /api/health``."""

    status: str = "ok"
    version: str = ""
    uptime_seconds: float = 0.0


class ReadinessDetail(BaseModel):
    """Response for ``GET /api/readiness`` — checks downstream deps."""

    db: str = "ok"
    disk: str = "ok"
    version: str = ""


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------


class UploadedFileResponse(BaseModel):
    """Response after successfully uploading a FITS/NPY file."""

    filename: str
    size_bytes: int
    message: str = "Upload successful"


# ---------------------------------------------------------------------------
# Background job (queue-based)
# ---------------------------------------------------------------------------


class BackgroundJobResponse(BaseModel):
    """Response when a job is submitted to the background queue."""

    job_id: str
    state: str = "queued"
    message: str = "Job submitted"


