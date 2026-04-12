"""Tests for crystallography visualization functions."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.models.config import AlgorithmName
from src.models.crystallography import CrystallographyResult, DiffractionPattern
from src.visualization.crystallography_plots import (
    plot_crystal_summary,
    plot_crystallography_comparison,
    plot_crystallography_phase,
    plot_crystallography_result,
    plot_diffraction_pattern,
    plot_electron_density,
    plot_r_factor_comparison,
)

matplotlib.use("Agg")

GRID = 64


@pytest.fixture()
def diffraction_pattern() -> DiffractionPattern:
    rng = np.random.default_rng(42)
    img = rng.random((GRID, GRID))
    img /= img.sum()
    return DiffractionPattern(
        image=img,
        wavelength_angstrom=1.5418,
        space_group="F m -3 m",
        source_id="1000041",
    )


@pytest.fixture()
def crystallography_result() -> CrystallographyResult:
    rng = np.random.default_rng(42)
    return CrystallographyResult(
        algorithm=AlgorithmName.HYBRID_INPUT_OUTPUT,
        recovered_phase=rng.uniform(-1, 1, (GRID, GRID)),
        recovered_amplitude=np.ones((GRID, GRID)),
        reconstructed_diffraction=rng.random((GRID, GRID)),
        electron_density=rng.random((GRID, GRID)),
        cost_history=[0.5, 0.3, 0.2, 0.15, 0.1],
        n_iterations=5,
        r_factor=0.15,
        elapsed_seconds=1.5,
    )


class TestPlotDiffractionPattern:
    def test_returns_figure(self, diffraction_pattern: DiffractionPattern) -> None:
        fig = plot_diffraction_pattern(diffraction_pattern)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_linear_scale(self, diffraction_pattern: DiffractionPattern) -> None:
        fig = plot_diffraction_pattern(diffraction_pattern, log_scale=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_axes(self, diffraction_pattern: DiffractionPattern) -> None:
        _, ax = plt.subplots()
        fig = plot_diffraction_pattern(diffraction_pattern, ax=ax)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotElectronDensity:
    def test_returns_figure(self, crystallography_result: CrystallographyResult) -> None:
        fig = plot_electron_density(crystallography_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_axes(self, crystallography_result: CrystallographyResult) -> None:
        _, ax = plt.subplots()
        fig = plot_electron_density(crystallography_result, ax=ax)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotCrystallographyPhase:
    def test_returns_figure(self, crystallography_result: CrystallographyResult) -> None:
        fig = plot_crystallography_phase(crystallography_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_axes(self, crystallography_result: CrystallographyResult) -> None:
        _, ax = plt.subplots()
        fig = plot_crystallography_phase(crystallography_result, ax=ax)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotCrystallographyResult:
    def test_returns_figure(
        self,
        diffraction_pattern: DiffractionPattern,
        crystallography_result: CrystallographyResult,
    ) -> None:
        fig = plot_crystallography_result(diffraction_pattern, crystallography_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotCrystalSummary:
    def test_returns_figure(
        self,
        diffraction_pattern: DiffractionPattern,
        crystallography_result: CrystallographyResult,
    ) -> None:
        fig = plot_crystal_summary(diffraction_pattern, crystallography_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotRFactorComparison:
    def test_returns_figure(self, crystallography_result: CrystallographyResult) -> None:
        results = {
            "ER": crystallography_result,
            "HIO": crystallography_result,
        }
        fig = plot_r_factor_comparison(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotCrystallographyComparison:
    def test_returns_figure(
        self,
        diffraction_pattern: DiffractionPattern,
        crystallography_result: CrystallographyResult,
    ) -> None:
        results = {
            "ER": crystallography_result,
            "HIO": crystallography_result,
        }
        fig = plot_crystallography_comparison(diffraction_pattern, results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_algorithm(
        self,
        diffraction_pattern: DiffractionPattern,
        crystallography_result: CrystallographyResult,
    ) -> None:
        results = {"ER": crystallography_result}
        fig = plot_crystallography_comparison(diffraction_pattern, results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
