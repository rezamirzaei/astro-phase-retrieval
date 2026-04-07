"""Smoke tests for visualization functions (non-interactive Agg backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.models.config import AlgorithmName  # noqa: E402
from src.models.optics import PhaseRetrievalResult, PSFData, PupilModel  # noqa: E402
from src.visualization.plots import (  # noqa: E402
    _azimuthal_average,
    plot_algorithm_comparison,
    plot_algorithm_dashboard,
    plot_convergence,
    plot_encircled_energy,
    plot_multi_observation_grid,
    plot_multi_observation_radial,
    plot_observed_psf,
    plot_pinn_benchmark,
    plot_psf_comparison,
    plot_psf_cross_sections,
    plot_psf_residual,
    plot_pupil,
    plot_radial_profile,
    plot_reconstructed_psf,
    plot_recovered_phase,
    plot_strehl_rms_bar,
    plot_summary,
    plot_wavefront_3d,
    plot_zernike_bar,
    plot_zernike_polar,
    save_figure,
    set_style,
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
        elapsed_seconds=1.5,
    )


@pytest.fixture()
def dummy_results(
    dummy_result: PhaseRetrievalResult,
) -> dict[str, PhaseRetrievalResult]:
    return {
        "ER": dummy_result,
        "HIO": dummy_result.model_copy(
            update={
                "algorithm": AlgorithmName.HYBRID_INPUT_OUTPUT,
                "strehl_ratio": 0.80,
                "rms_phase_rad": 0.35,
            }
        ),
    }


@pytest.fixture()
def zernike_coeffs() -> dict[int, float]:
    return {2: 0.1, 3: -0.2, 4: 0.5, 5: -0.05, 6: 0.03}


@pytest.fixture()
def observations(
    psf_data: PSFData,
    dummy_result: PhaseRetrievalResult,
    support: np.ndarray,
) -> list[dict]:
    return [
        {
            "label": "Obs1",
            "psf": psf_data,
            "result": dummy_result,
            "support": support,
        },
        {
            "label": "Obs2",
            "psf": psf_data,
            "result": dummy_result,
            "support": support,
        },
    ]


class TestPlots:
    def test_plot_pupil(self, pupil: PupilModel) -> None:
        fig = plot_pupil(pupil)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_observed_psf(self, psf_data: PSFData) -> None:
        fig = plot_observed_psf(psf_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_observed_psf_linear(self, psf_data: PSFData) -> None:
        fig = plot_observed_psf(psf_data, log_scale=False)
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

    def test_plot_reconstructed_psf_linear(self, dummy_result) -> None:
        fig = plot_reconstructed_psf(dummy_result, log_scale=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_reconstructed_psf_with_ax(self, dummy_result) -> None:
        """Pass an existing ax to exercise the `fig = ax.figure` branch."""
        fig_ext, ax_ext = plt.subplots()
        fig = plot_reconstructed_psf(dummy_result, ax=ax_ext)
        assert fig is fig_ext
        plt.close(fig)

    def test_plot_convergence(self, dummy_result) -> None:
        fig = plot_convergence(dummy_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_zernike_bar(self, zernike_coeffs) -> None:
        fig = plot_zernike_bar(zernike_coeffs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_summary(self, psf_data, pupil, dummy_result, zernike_coeffs) -> None:
        fig = plot_summary(psf_data, pupil, dummy_result, zernike_coeffs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_summary_no_zernike(self, psf_data, pupil, dummy_result) -> None:
        fig = plot_summary(psf_data, pupil, dummy_result, zernike_coeffs=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pinn_benchmark(self, psf_data, dummy_result) -> None:
        results = {
            "RAAR": dummy_result.model_copy(update={"algorithm": AlgorithmName.RAAR}),
            "WF": dummy_result.model_copy(update={"algorithm": AlgorithmName.WIRTINGER_FLOW}),
            "PINN": dummy_result.model_copy(update={"algorithm": AlgorithmName.PINN}),
        }
        fig = plot_pinn_benchmark(psf_data, results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pinn_benchmark_empty(self, psf_data) -> None:
        """Empty results dict triggers the unused-axes-hiding loop."""
        fig = plot_pinn_benchmark(psf_data, {})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_psf_residual(self, psf_data, dummy_result) -> None:
        fig = plot_psf_residual(psf_data, dummy_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_psf_comparison(self, psf_data, dummy_result) -> None:
        fig = plot_psf_comparison(psf_data, dummy_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_psf_comparison_linear(self, psf_data, dummy_result) -> None:
        fig = plot_psf_comparison(psf_data, dummy_result, log_scale=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_algorithm_comparison(self, dummy_results, support) -> None:
        fig = plot_algorithm_comparison(dummy_results, support)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_algorithm_comparison_single(self, dummy_result, support) -> None:
        fig = plot_algorithm_comparison({"ER": dummy_result}, support)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_radial_profile(self, psf_data, dummy_result, pupil) -> None:
        fig = plot_radial_profile(psf_data, dummy_result, pupil)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_psf_cross_sections(self, psf_data, dummy_result) -> None:
        fig = plot_psf_cross_sections(psf_data, dummy_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_wavefront_3d(self, dummy_result, support) -> None:
        fig = plot_wavefront_3d(dummy_result, support)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_encircled_energy(self, psf_data, dummy_result, pupil) -> None:
        fig = plot_encircled_energy(psf_data, dummy_result, pupil)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_zernike_polar(self, zernike_coeffs) -> None:
        fig = plot_zernike_polar(zernike_coeffs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_algorithm_dashboard(self, psf_data, dummy_results, support, pupil) -> None:
        fig = plot_algorithm_dashboard(psf_data, dummy_results, support, pupil)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_algorithm_dashboard_single(self, psf_data, dummy_result, support, pupil) -> None:
        fig = plot_algorithm_dashboard(psf_data, {"ER": dummy_result}, support, pupil)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_strehl_rms_bar(self, dummy_results) -> None:
        fig = plot_strehl_rms_bar(dummy_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_multi_observation_grid(self, observations) -> None:
        fig = plot_multi_observation_grid(observations)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_multi_observation_grid_single(self, observations) -> None:
        fig = plot_multi_observation_grid(observations[:1])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_multi_observation_radial(self, observations) -> None:
        fig = plot_multi_observation_radial(observations)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSaveFigure:
    def test_save_png(self, tmp_path, dummy_result) -> None:
        fig = plot_convergence(dummy_result)
        out = tmp_path / "sub" / "test.png"
        save_figure(fig, out)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)


class TestHelpers:
    def test_set_style_does_not_raise(self) -> None:
        set_style()

    def test_azimuthal_average_shape(self) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((64, 64))
        img[32, 32] = 10.0
        radii, profile = _azimuthal_average(img)
        assert len(radii) == len(profile)
        assert len(radii) > 0
        assert radii[0] == 0.0

    def test_azimuthal_average_peak_brightest(self) -> None:
        img = np.zeros((64, 64))
        img[32, 32] = 1.0
        radii, profile = _azimuthal_average(img)
        assert profile[0] >= profile[-1]


class TestEdgeCases:
    """Test edge cases in plot functions for full coverage of vmax==0 guards."""

    def test_recovered_phase_zero(self, pupil: PupilModel, support: np.ndarray) -> None:
        """Zero phase everywhere → vmax==0, should hit the guard."""
        n = pupil.grid_size
        result = PhaseRetrievalResult(
            algorithm=AlgorithmName.ERROR_REDUCTION,
            recovered_phase=np.zeros((n, n)),
            recovered_amplitude=pupil.amplitude,
            reconstructed_psf=np.ones((n, n)) / (n * n),
            cost_history=[1.0],
            n_iterations=1,
            strehl_ratio=1.0,
            rms_phase_rad=0.0,
            elapsed_seconds=0.1,
        )
        fig = plot_recovered_phase(result, support)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_reconstructed_psf_zero(self, pupil: PupilModel) -> None:
        """Zero PSF → vmax==0 guard in log scale."""
        n = pupil.grid_size
        result = PhaseRetrievalResult(
            algorithm=AlgorithmName.ERROR_REDUCTION,
            recovered_phase=np.zeros((n, n)),
            recovered_amplitude=pupil.amplitude,
            reconstructed_psf=np.zeros((n, n)),
            cost_history=[1.0],
            n_iterations=1,
            strehl_ratio=0.0,
            rms_phase_rad=0.0,
            elapsed_seconds=0.1,
        )
        fig = plot_reconstructed_psf(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_psf_residual_zero(self, pupil: PupilModel) -> None:
        """Identical PSFs → zero residual → vmax==0 guard."""
        n = pupil.grid_size
        img = np.ones((n, n)) / (n * n)
        psf = PSFData(
            image=img,
            pixel_scale_arcsec=0.04,
            wavelength_m=606e-9,
            filter_name="T",
            telescope="test",
        )
        result = PhaseRetrievalResult(
            algorithm=AlgorithmName.ERROR_REDUCTION,
            recovered_phase=np.zeros((n, n)),
            recovered_amplitude=pupil.amplitude,
            reconstructed_psf=img.copy(),
            cost_history=[1.0],
            n_iterations=1,
            strehl_ratio=1.0,
            rms_phase_rad=0.0,
            elapsed_seconds=0.1,
        )
        fig = plot_psf_residual(psf, result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_psf_comparison_zero_residual(self, pupil: PupilModel) -> None:
        """Identical PSFs → zero residual in comparison view."""
        n = pupil.grid_size
        img = np.ones((n, n)) / (n * n)
        psf = PSFData(
            image=img,
            pixel_scale_arcsec=0.04,
            wavelength_m=606e-9,
            filter_name="T",
            telescope="test",
        )
        result = PhaseRetrievalResult(
            algorithm=AlgorithmName.ERROR_REDUCTION,
            recovered_phase=np.zeros((n, n)),
            recovered_amplitude=pupil.amplitude,
            reconstructed_psf=img.copy(),
            cost_history=[1.0],
            n_iterations=1,
            strehl_ratio=1.0,
            rms_phase_rad=0.0,
            elapsed_seconds=0.1,
        )
        fig = plot_psf_comparison(psf, result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

