"""Pydantic models for optical data containers and algorithm results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.models.config import AlgorithmName


# ---------------------------------------------------------------------------
# Helpers – allow numpy arrays inside pydantic models
# ---------------------------------------------------------------------------

class _NumpyModel(BaseModel):
    """Base model that permits arbitrary types (numpy arrays)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# PSF data container
# ---------------------------------------------------------------------------

class PSFData(_NumpyModel):
    """Validated container for an observed PSF image."""

    image: np.ndarray = Field(..., description="2-D intensity array (background-subtracted, normalised)")
    pixel_scale_arcsec: float = Field(..., gt=0)
    wavelength_m: float = Field(..., gt=0)
    filter_name: str
    telescope: str
    obs_id: str = ""

    @field_validator("image")
    @classmethod
    def _check_2d(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2:
            raise ValueError(f"PSF image must be 2-D, got {v.ndim}-D")
        if v.shape[0] != v.shape[1]:
            raise ValueError(f"PSF image must be square, got shape {v.shape}")
        return v


class PSFPair(_NumpyModel):
    """Focused + defocused PSF pair for phase-diversity retrieval."""

    focused: PSFData
    defocused: PSFData

    @field_validator("defocused")
    @classmethod
    def _same_shape(cls, v: PSFData, info) -> PSFData:
        foc = info.data.get("focused")
        if foc is not None and v.image.shape != foc.image.shape:
            raise ValueError("Focused and defocused images must have the same shape")
        return v


# ---------------------------------------------------------------------------
# Pupil model result
# ---------------------------------------------------------------------------

class PupilModel(_NumpyModel):
    """Computed telescope pupil amplitude mask."""

    amplitude: np.ndarray = Field(..., description="2-D binary/soft pupil amplitude")
    grid_size: int = Field(..., gt=0)

    @field_validator("amplitude")
    @classmethod
    def _check_square(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2 or v.shape[0] != v.shape[1]:
            raise ValueError("Pupil amplitude must be a square 2-D array")
        return v


# ---------------------------------------------------------------------------
# Phase-retrieval result
# ---------------------------------------------------------------------------

class PhaseRetrievalResult(_NumpyModel):
    """Complete output from a phase-retrieval run."""

    algorithm: AlgorithmName
    recovered_phase: np.ndarray = Field(..., description="Recovered pupil-plane phase (radians)")
    recovered_amplitude: np.ndarray = Field(..., description="Recovered pupil-plane amplitude")
    reconstructed_psf: np.ndarray = Field(..., description="Forward-modelled PSF from recovered wavefront")
    cost_history: list[float] = Field(default_factory=list, description="Cost function vs. iteration")
    n_iterations: int = Field(..., ge=1)
    converged: bool = False
    elapsed_seconds: float = Field(default=0.0, ge=0)
    rms_phase_rad: float = Field(default=0.0, ge=0, description="RMS wavefront error in radians")
    strehl_ratio: float = Field(default=0.0, ge=0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("recovered_phase", "recovered_amplitude", "reconstructed_psf")
    @classmethod
    def _check_2d(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2:
            raise ValueError(f"Array must be 2-D, got {v.ndim}-D")
        return v



