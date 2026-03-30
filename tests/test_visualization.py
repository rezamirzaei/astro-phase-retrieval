"""Smoke tests for visualization functions (non-interactive Agg backend)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.models.config import AlgorithmName  # noqa: E402
from src.models.optics import PSFData, PhaseRetrievalResult, PupilModel  # noqa: E402
from src.visualization.plots import (  # noqa: E402
    plot_convergence,
    plot_observed_psf,
    plot_pupil,
    plot_recovered_phase,
    plot_reconstructed_psf,
    plot_summary,
    plot_zernike_bar,
)


@pytest.fixture()
def dummy_result(pupil: PupilModel) -> PhaseRetrievalResult:
    n = pupil.grid_size
    return PhaseRetrievalResult(
        algorithm=AlgorithmName.ERROR_REDUCTION,
        recovered_phase=np.random.default_rng(0).uniform(-0.5, 0.5, (n, n)),
        recovered_amplitude=pupil.amplitude,
        reconstructed_psf=np.random.default_rng(0).random((n, n)),
        cost_history=[1.0, 0.5, 0.3, 0.2, 0.15],
        n_iterations=5,
        converged=False,
        strehl_ratio=0.85,
        rms_phase_rad=0.3,
    )


class TestPlots:
    def test_plot_pupil(self, pupil: PupilModel) -> None:
        fig = plot_pupil(pupil)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_observed_psf(self, psf_data: PSFData) -> None:
        fig = plot_observed_psf(psf_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_recovered_phase(self, dummy_result, support) -> None:
        fig = plot_recovered_phase(dummy_result, support)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_reconstructed_psf(self, dummy_result) -> None:
        fig = plot_reconstructed_psf(dummy_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_convergence(self, dummy_result) -> None:
        fig = plot_convergence(dummy_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_zernike_bar(self) -> None:
        coeffs = {2: 0.1, 3: -0.2, 4: 0.5, 5: -0.05, 6: 0.03}
        fig = plot_zernike_bar(coeffs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_summary(self, psf_data, pupil, dummy_result) -> None:
        coeffs = {j: 0.01 * j for j in range(2, 12)}
        fig = plot_summary(psf_data, pupil, dummy_result, coeffs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
