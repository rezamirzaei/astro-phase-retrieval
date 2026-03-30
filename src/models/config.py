"""Pydantic configuration models for every stage of the pipeline."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TelescopeType(str, Enum):
    """Supported telescope pupil geometries."""
    HST = "hst"
    JWST = "jwst"
    GENERIC_CIRCULAR = "generic_circular"


class AlgorithmName(str, Enum):
    """Registered phase-retrieval algorithm identifiers."""
    ERROR_REDUCTION = "er"
    GERCHBERG_SAXTON = "gs"
    HYBRID_INPUT_OUTPUT = "hio"
    RAAR = "raar"
    PHASE_DIVERSITY = "phase_diversity"
    WIRTINGER_FLOW = "wf"
    DOUGLAS_RACHFORD = "dr"
    ADMM = "admm"


class BetaSchedule(str, Enum):
    """Adaptive β scheduling strategies."""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"


class NoiseModel(str, Enum):
    """Noise model for focal-plane projection."""
    GAUSSIAN = "gaussian"
    POISSON = "poisson"


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    """Where to find / download the observation data."""

    data_dir: Path = Field(default=Path("data"), description="Root directory for downloaded FITS files")
    obs_id: str = Field(
        default="JDOX-HST-WFC3",
        description="MAST observation identifier or preset name",
    )
    detector: Literal["WFC3/UVIS", "WFC3/IR", "ACS/WFC", "NIRCam"] = Field(
        default="WFC3/UVIS",
        description="Detector used for the observation",
    )
    cutout_size: int = Field(default=128, ge=32, le=1024, description="PSF cutout half-width in pixels")
    filter_name: str = Field(default="F606W", description="Optical filter bandpass name")

    @field_validator("cutout_size")
    @classmethod
    def _power_of_two(cls, v: int) -> int:
        if v & (v - 1) != 0:
            raise ValueError(f"cutout_size must be a power of 2, got {v}")
        return v


# ---------------------------------------------------------------------------
# Pupil / optics configuration
# ---------------------------------------------------------------------------

class PupilConfig(BaseModel):
    """Telescope pupil model parameters."""

    telescope: TelescopeType = Field(default=TelescopeType.HST)
    grid_size: int = Field(default=256, ge=64, le=2048, description="Pupil-plane grid side length (px)")
    primary_radius: float = Field(default=1.2, gt=0, description="Primary mirror radius (m)")
    secondary_radius: float = Field(default=0.396, ge=0, description="Secondary mirror obstruction radius (m)")
    spider_width: float = Field(default=0.0254, ge=0, description="Spider vane width (m)")
    n_spiders: int = Field(default=4, ge=0, le=8, description="Number of spider vanes")
    wavelength_m: float = Field(default=606e-9, gt=0, description="Observation wavelength (m)")
    pixel_scale_arcsec: float = Field(default=0.04, gt=0, description="Detector pixel scale (arcsec/px)")

    @field_validator("grid_size")
    @classmethod
    def _power_of_two(cls, v: int) -> int:
        if v & (v - 1) != 0:
            raise ValueError(f"grid_size must be a power of 2, got {v}")
        return v


# ---------------------------------------------------------------------------
# Algorithm configuration
# ---------------------------------------------------------------------------

class AlgorithmConfig(BaseModel):
    """Phase-retrieval algorithm hyper-parameters."""

    name: AlgorithmName = Field(default=AlgorithmName.HYBRID_INPUT_OUTPUT)
    max_iterations: int = Field(default=300, ge=1, le=100_000)
    tolerance: float = Field(default=1e-8, gt=0, description="Convergence tolerance on cost-function change")
    beta: float = Field(default=0.9, gt=0, le=1.0, description="HIO / RAAR feedback parameter β")
    beta_schedule: BetaSchedule = Field(
        default=BetaSchedule.CONSTANT,
        description="Adaptive β scheduling: constant, linear ramp-down, or cosine annealing",
    )
    beta_min: float = Field(default=0.5, ge=0, le=1.0, description="Minimum β for adaptive schedules")
    defocus_waves: float = Field(
        default=1.0,
        description="Defocus amount (waves) for phase-diversity second image",
    )
    use_sw_constraint: bool = Field(
        default=True,
        description="Apply shrink-wrap (dynamic support) refinement",
    )
    support_threshold: float = Field(
        default=0.04,
        ge=0,
        le=1,
        description="Fraction of max amplitude for automatic support estimation",
    )
    random_seed: int | None = Field(default=42, description="RNG seed for reproducibility")

    # ── State-of-the-art enhancements ─────────────────────────────────
    momentum: float = Field(
        default=0.0,
        ge=0,
        le=0.99,
        description="Nesterov/heavy-ball momentum coefficient (0 = off)",
    )
    tv_weight: float = Field(
        default=0.0,
        ge=0,
        description="Total-variation regularization weight on recovered phase (0 = off)",
    )
    noise_model: NoiseModel = Field(
        default=NoiseModel.GAUSSIAN,
        description="Noise model for focal-plane amplitude projection",
    )
    n_starts: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of random restarts (multi-start); best result is returned",
    )
    wf_step_size: float = Field(
        default=0.5,
        gt=0,
        le=10.0,
        description="Wirtinger Flow step size (learning rate)",
    )
    wf_spectral_init: bool = Field(
        default=True,
        description="Use spectral initialization for Wirtinger Flow",
    )


# ---------------------------------------------------------------------------
# Pipeline (master) configuration
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """Top-level configuration aggregating all sub-configs."""

    data: DataConfig = Field(default_factory=DataConfig)
    pupil: PupilConfig = Field(default_factory=PupilConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    output_dir: Path = Field(default=Path("outputs"))

    @model_validator(mode="after")
    def _sync_wavelength_with_filter(self) -> "PipelineConfig":
        """Auto-set wavelength from filter name when using defaults."""
        _filter_to_wavelength = {
            "F606W": 606e-9,
            "F814W": 814e-9,
            "F200W": 2.0e-6,
            "F150W": 1.5e-6,
            "F110W": 1.1e-6,
            "F160W": 1.6e-6,
            "F438W": 438e-9,
        }
        wl = _filter_to_wavelength.get(self.data.filter_name)
        if wl is not None:
            self.pupil.wavelength_m = wl
        return self


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def default_hst_config() -> PipelineConfig:
    """Return a sensible default config for HST/WFC3/UVIS phase retrieval."""
    return PipelineConfig(
        data=DataConfig(
            detector="WFC3/UVIS",
            filter_name="F606W",
            cutout_size=128,
        ),
        pupil=PupilConfig(
            telescope=TelescopeType.HST,
            grid_size=256,
            primary_radius=1.2,
            secondary_radius=0.396,
            spider_width=0.0254,
            n_spiders=4,
            pixel_scale_arcsec=0.04,
        ),
        algorithm=AlgorithmConfig(
            name=AlgorithmName.HYBRID_INPUT_OUTPUT,
            max_iterations=500,
            beta=0.9,
        ),
    )



