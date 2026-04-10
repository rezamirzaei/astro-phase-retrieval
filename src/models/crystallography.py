"""Pydantic models for crystallographic phase retrieval."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.models._base import NumpyModel
from src.models.config import AlgorithmName

# ---------------------------------------------------------------------------
# Atom site
# ---------------------------------------------------------------------------


class AtomSite(BaseModel):
    """A single atom site within a crystal unit cell."""

    label: str = Field(..., description="Atom label (e.g. 'Na1', 'Cl1')")
    symbol: str = Field(..., description="Chemical element symbol (e.g. 'Na', 'Cl')")
    x: float = Field(..., description="Fractional coordinate x")
    y: float = Field(..., description="Fractional coordinate y")
    z: float = Field(..., description="Fractional coordinate z")
    occupancy: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Crystal structure
# ---------------------------------------------------------------------------


class CrystalStructure(BaseModel):
    """Parsed crystal structure from a CIF file."""

    cod_id: str = Field(default="", description="Crystallography Open Database ID")
    formula: str = Field(default="", description="Chemical formula (e.g. 'NaCl')")
    space_group: str = Field(default="P 1", description="Space group symbol")
    a: float = Field(..., gt=0, description="Unit cell a (Å)")
    b: float = Field(..., gt=0, description="Unit cell b (Å)")
    c: float = Field(..., gt=0, description="Unit cell c (Å)")
    alpha: float = Field(default=90.0, ge=0, le=180, description="Unit cell α (degrees)")
    beta: float = Field(default=90.0, ge=0, le=180, description="Unit cell β (degrees)")
    gamma: float = Field(default=90.0, ge=0, le=180, description="Unit cell γ (degrees)")
    atoms: list[AtomSite] = Field(default_factory=list, description="Atom sites")


# ---------------------------------------------------------------------------
# Diffraction pattern
# ---------------------------------------------------------------------------


class DiffractionPattern(NumpyModel):
    """2-D diffraction intensity pattern for phase retrieval."""

    image: np.ndarray = Field(
        ..., description="2-D diffraction intensity array (normalised)"
    )
    wavelength_angstrom: float = Field(default=1.5418, gt=0, description="X-ray wavelength (Å)")
    d_max: float = Field(default=20.0, gt=0, description="Maximum d-spacing (Å)")
    space_group: str = Field(default="P 1", description="Space group symbol")
    source_id: str = Field(default="", description="Identifier for the data source")

    @field_validator("image")
    @classmethod
    def _check_2d_square(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2:
            raise ValueError(f"Diffraction image must be 2-D, got {v.ndim}-D")
        if v.shape[0] != v.shape[1]:
            raise ValueError(f"Diffraction image must be square, got shape {v.shape}")
        return v


# ---------------------------------------------------------------------------
# Crystallography phase-retrieval result
# ---------------------------------------------------------------------------


class CrystallographyResult(NumpyModel):
    """Complete output from a crystallographic phase-retrieval run."""

    algorithm: AlgorithmName
    recovered_phase: np.ndarray = Field(
        ..., description="Recovered phase (radians)"
    )
    recovered_amplitude: np.ndarray = Field(
        ..., description="Recovered amplitude (electron density proxy)"
    )
    reconstructed_diffraction: np.ndarray = Field(
        ..., description="Forward-modelled diffraction pattern"
    )
    electron_density: np.ndarray = Field(
        ..., description="Real-space electron density map"
    )
    cost_history: list[float] = Field(
        default_factory=list, description="Cost function vs. iteration"
    )
    n_iterations: int = Field(..., ge=1)
    converged: bool = False
    elapsed_seconds: float = Field(default=0.0, ge=0)
    r_factor: float = Field(default=0.0, ge=0, description="Crystallographic R-factor")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("recovered_phase", "recovered_amplitude", "reconstructed_diffraction",
                     "electron_density")
    @classmethod
    def _check_2d(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2:
            raise ValueError(f"Array must be 2-D, got {v.ndim}-D")
        return v


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class CrystallographyConfig(BaseModel):
    """Configuration for crystallographic phase retrieval."""

    grid_size: int = Field(default=128, ge=64, le=1024)
    wavelength_angstrom: float = Field(default=1.5418, gt=0, description="X-ray wavelength (Å)")
    algorithm: AlgorithmName = Field(default=AlgorithmName.HYBRID_INPUT_OUTPUT)
    max_iterations: int = Field(default=500, ge=1, le=100_000)
    beta: float = Field(default=0.9, gt=0, le=1.0)
    random_seed: int | None = Field(default=42)

    @field_validator("grid_size")
    @classmethod
    def _power_of_two(cls, v: int) -> int:
        if v & (v - 1) != 0:
            raise ValueError(f"grid_size must be a power of 2, got {v}")
        return v

