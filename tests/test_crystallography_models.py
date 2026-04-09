"""Tests for crystallography data models."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.config import AlgorithmName
from src.models.crystallography import (
    AtomSite,
    CrystallographyConfig,
    CrystallographyResult,
    CrystalStructure,
    DiffractionPattern,
)


class TestAtomSite:
    def test_construction(self) -> None:
        atom = AtomSite(label="Na1", symbol="Na", x=0.0, y=0.0, z=0.0)
        assert atom.label == "Na1"
        assert atom.symbol == "Na"
        assert atom.occupancy == 1.0

    def test_occupancy_bounds(self) -> None:
        atom = AtomSite(label="A1", symbol="C", x=0.0, y=0.0, z=0.0, occupancy=0.5)
        assert atom.occupancy == 0.5
        with pytest.raises(ValueError):
            AtomSite(label="A1", symbol="C", x=0.0, y=0.0, z=0.0, occupancy=1.5)
        with pytest.raises(ValueError):
            AtomSite(label="A1", symbol="C", x=0.0, y=0.0, z=0.0, occupancy=-0.1)


class TestCrystalStructure:
    def test_construction(self) -> None:
        cs = CrystalStructure(a=5.0, b=5.0, c=5.0)
        assert cs.a == 5.0
        assert cs.space_group == "P 1"
        assert cs.atoms == []

    def test_with_atoms(self) -> None:
        atoms = [
            AtomSite(label="Na1", symbol="Na", x=0.0, y=0.0, z=0.0),
            AtomSite(label="Cl1", symbol="Cl", x=0.5, y=0.5, z=0.5),
        ]
        cs = CrystalStructure(a=5.64, b=5.64, c=5.64, atoms=atoms)
        assert len(cs.atoms) == 2

    def test_invalid_cell(self) -> None:
        with pytest.raises(ValueError):
            CrystalStructure(a=-1.0, b=5.0, c=5.0)


class TestDiffractionPattern:
    def test_construction(self) -> None:
        img = np.random.default_rng(0).random((64, 64))
        dp = DiffractionPattern(image=img, wavelength_angstrom=1.5418)
        assert dp.image.shape == (64, 64)
        assert dp.wavelength_angstrom == pytest.approx(1.5418)

    def test_must_be_square(self) -> None:
        with pytest.raises(ValueError, match="square"):
            DiffractionPattern(image=np.zeros((64, 32)))

    def test_must_be_2d(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            DiffractionPattern(image=np.zeros((64,)))


class TestCrystallographyResult:
    def test_construction(self) -> None:
        n = 64
        result = CrystallographyResult(
            algorithm=AlgorithmName.HYBRID_INPUT_OUTPUT,
            recovered_phase=np.zeros((n, n)),
            recovered_amplitude=np.ones((n, n)),
            reconstructed_diffraction=np.zeros((n, n)),
            electron_density=np.zeros((n, n)),
            n_iterations=100,
            r_factor=0.15,
        )
        assert result.r_factor == pytest.approx(0.15)
        assert result.n_iterations == 100

    def test_invalid_arrays(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            CrystallographyResult(
                algorithm=AlgorithmName.ERROR_REDUCTION,
                recovered_phase=np.zeros((64,)),
                recovered_amplitude=np.ones((64, 64)),
                reconstructed_diffraction=np.zeros((64, 64)),
                electron_density=np.zeros((64, 64)),
                n_iterations=10,
            )


class TestCrystallographyConfig:
    def test_defaults(self) -> None:
        cfg = CrystallographyConfig()
        assert cfg.grid_size == 128
        assert cfg.wavelength_angstrom == pytest.approx(1.5418)

    def test_grid_power_of_two(self) -> None:
        with pytest.raises(ValueError, match="power of 2"):
            CrystallographyConfig(grid_size=100)

    def test_valid_grid(self) -> None:
        cfg = CrystallographyConfig(grid_size=256)
        assert cfg.grid_size == 256






