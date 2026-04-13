"""Pydantic models for optical data containers and algorithm results."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import Field, ValidationInfo, field_validator

from src.models._base import NumpyModel
from src.models.config import AlgorithmName

# ---------------------------------------------------------------------------
# PSF data container
# ---------------------------------------------------------------------------


class PSFData(NumpyModel):
    """Validated container for an observed PSF image."""

    image: np.ndarray = Field(
        ..., description="2-D intensity array (background-subtracted, normalised)"
    )
    pixel_scale_arcsec: float = Field(..., gt=0)
    wavelength_m: float = Field(..., gt=0)
    filter_name: str
    telescope: str
    obs_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("image")
    @classmethod
    def _check_2d(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2:
            raise ValueError(f"PSF image must be 2-D, got {v.ndim}-D")
        if v.shape[0] != v.shape[1]:
            raise ValueError(f"PSF image must be square, got shape {v.shape}")
        return v


class PSFPair(NumpyModel):
    """Focused + defocused PSF pair for phase-diversity retrieval."""

    focused: PSFData
    defocused: PSFData

    @field_validator("defocused")
    @classmethod
    def _same_shape(cls, v: PSFData, info: ValidationInfo) -> PSFData:
        foc = info.data.get("focused")
        if foc is not None and v.image.shape != foc.image.shape:
            raise ValueError("Focused and defocused images must have the same shape")
        return v


# ---------------------------------------------------------------------------
# Pupil model result
# ---------------------------------------------------------------------------


class PupilModel(NumpyModel):
    """Computed telescope pupil amplitude mask."""

    amplitude: np.ndarray = Field(..., description="2-D binary/soft pupil amplitude")
    grid_size: int = Field(..., gt=0)
    wavelength_m: float = Field(default=606e-9, gt=0)
    bandwidth_fraction: float = Field(default=0.0, ge=0.0)
    spectral_samples: int = Field(default=1, ge=1)
    spectral_weighting: str = Field(default="delta")
    field_defocus_waves: float = Field(default=0.0)
    detector_sigma_pixels: float = Field(default=0.0, ge=0.0)
    jitter_sigma_pixels: float = Field(default=0.0, ge=0.0)
    pixel_integration_width: float = Field(default=1.0, gt=0.0)

    @field_validator("amplitude")
    @classmethod
    def _check_square(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2 or v.shape[0] != v.shape[1]:
            raise ValueError("Pupil amplitude must be a square 2-D array")
        return v

    def forward_model_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments for :func:`forward_model` / :func:`compute_strehl_ratio`.

        This eliminates copy-pasting 8 keyword arguments at every call-site.
        """
        return {
            "wavelength_m": self.wavelength_m,
            "bandwidth_fraction": self.bandwidth_fraction,
            "spectral_samples": self.spectral_samples,
            "spectral_weighting": self.spectral_weighting,
            "field_defocus_waves": self.field_defocus_waves,
            "detector_sigma_pixels": self.detector_sigma_pixels,
            "jitter_sigma_pixels": self.jitter_sigma_pixels,
            "pixel_integration_width": self.pixel_integration_width,
        }


# ---------------------------------------------------------------------------
# Phase-retrieval result
# ---------------------------------------------------------------------------


class PhaseRetrievalResult(NumpyModel):
    """Complete output from a phase-retrieval run."""

    algorithm: AlgorithmName
    recovered_phase: np.ndarray = Field(..., description="Recovered pupil-plane phase (radians)")
    recovered_amplitude: np.ndarray = Field(..., description="Recovered pupil-plane amplitude")
    reconstructed_psf: np.ndarray = Field(
        ..., description="Forward-modelled PSF from recovered wavefront"
    )
    cost_history: list[float] = Field(
        default_factory=list, description="Cost function vs. iteration"
    )
    n_iterations: int = Field(..., ge=1)
    converged: bool = False
    elapsed_seconds: float = Field(default=0.0, ge=0)
    rms_phase_rad: float = Field(default=0.0, ge=0, description="RMS wavefront error in radians")
    strehl_ratio: float = Field(default=0.0, ge=0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("recovered_phase", "recovered_amplitude", "reconstructed_psf")
    @classmethod
    def _check_2d(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2:
            raise ValueError(f"Array must be 2-D, got {v.ndim}-D")
        return v
