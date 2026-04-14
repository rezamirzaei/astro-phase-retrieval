"""Tests for v2.2.0: FISTA, Sparse PR, pipeline, synthetic data, validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from src.algorithms.registry import AlgorithmRegistry
from src.data.synthetic import SyntheticDataset, generate_synthetic_psf
from src.models.config import (
    AlgorithmConfig,
    AlgorithmName,
    PipelineConfig,
    Regulariser,
)
from src.models.optics import PSFData, PupilModel

# ---------------------------------------------------------------------------
# FISTA algorithm tests
# ---------------------------------------------------------------------------


class TestFISTA:
    """Tests for the FISTA proximal-gradient algorithm."""

    def test_fista_runs_and_converges(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.FISTA,
            max_iterations=60,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1
        assert result.cost_history[-1] <= result.cost_history[0]
        assert 0.0 <= result.strehl_ratio <= 1.0

    def test_fista_with_tv_regulariser(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.FISTA,
            max_iterations=40,
            regulariser=Regulariser.TV,
            proximal_weight=0.01,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1
        assert result.rms_phase_rad >= 0.0

    def test_fista_with_l1_regulariser(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.FISTA,
            max_iterations=40,
            regulariser=Regulariser.L1_WAVELET,
            proximal_weight=0.005,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1

    def test_fista_result_fields(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.FISTA,
            max_iterations=30,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.algorithm == AlgorithmName.FISTA
        assert result.recovered_phase.shape == (pupil.grid_size, pupil.grid_size)
        assert result.reconstructed_psf.shape == (pupil.grid_size, pupil.grid_size)

    def test_fista_phase_zero_outside_support(
        self, pupil: PupilModel, psf_data: PSFData, support: np.ndarray
    ) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.FISTA,
            max_iterations=30,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        np.testing.assert_array_equal(result.recovered_phase[~support], 0.0)


# ---------------------------------------------------------------------------
# Sparse Phase Retrieval tests
# ---------------------------------------------------------------------------


class TestSparsePR:
    """Tests for the Thresholded Wirtinger Flow algorithm."""

    def test_sparse_pr_runs(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.SPARSE_PR,
            max_iterations=60,
            sparsity_threshold=0.1,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1
        assert result.cost_history[-1] <= result.cost_history[0]

    def test_sparse_pr_with_soft_threshold(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.SPARSE_PR,
            max_iterations=40,
            regulariser=Regulariser.L1_WAVELET,
            sparsity_threshold=0.2,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1

    def test_sparse_pr_result_shape(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.SPARSE_PR,
            max_iterations=30,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.recovered_phase.shape == (pupil.grid_size, pupil.grid_size)

    def test_sparse_pr_keep_fraction_limits_active_support(
        self,
        pupil: PupilModel,
        psf_data: PSFData,
    ) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.SPARSE_PR,
            max_iterations=20,
            sparsity_keep_fraction=0.05,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        support = pupil.amplitude > 0
        active_fraction = float(np.mean(np.abs(result.recovered_phase[support]) > 1e-6))
        assert active_fraction <= 0.20


# ---------------------------------------------------------------------------
# Synthetic data generator tests
# ---------------------------------------------------------------------------


class TestSyntheticDataGenerator:
    """Tests for the synthetic data generation module."""

    def test_generate_basic(self) -> None:
        ds = generate_synthetic_psf(grid_size=64, rms_aberration=0.5, random_seed=42)
        assert isinstance(ds, SyntheticDataset)
        assert ds.psf_data.image.shape == (64, 64)
        assert ds.pupil.grid_size == 64
        assert ds.true_phase.shape == (64, 64)
        assert ds.true_psf.shape == (64, 64)

    def test_generate_with_poisson_noise(self) -> None:
        ds = generate_synthetic_psf(
            grid_size=64, rms_aberration=0.3, photon_count=1e4, random_seed=7
        )
        assert ds.noise_level > 0
        # Noisy PSF should differ from noiseless
        assert not np.allclose(ds.psf_data.image, ds.true_psf, atol=1e-6)

    def test_generate_with_read_noise(self) -> None:
        ds = generate_synthetic_psf(
            grid_size=64, rms_aberration=0.3, read_noise_std=0.001, random_seed=7
        )
        assert ds.noise_level > 0

    def test_psf_normalised(self) -> None:
        ds = generate_synthetic_psf(grid_size=64, random_seed=42)
        np.testing.assert_allclose(ds.psf_data.image.sum(), 1.0, atol=1e-6)

    def test_rms_matches_target(self) -> None:
        target_rms = 0.8
        ds = generate_synthetic_psf(grid_size=64, rms_aberration=target_rms, random_seed=42)
        support = ds.pupil.amplitude > 0
        vals = ds.true_phase[support]
        vals = vals - vals.mean()
        actual_rms = float(np.sqrt(np.mean(vals**2)))
        np.testing.assert_allclose(actual_rms, target_rms, rtol=0.05)

    def test_reproducible_with_seed(self) -> None:
        ds1 = generate_synthetic_psf(grid_size=64, random_seed=123)
        ds2 = generate_synthetic_psf(grid_size=64, random_seed=123)
        np.testing.assert_array_equal(ds1.psf_data.image, ds2.psf_data.image)

    def test_different_seeds_differ(self) -> None:
        ds1 = generate_synthetic_psf(grid_size=64, random_seed=1)
        ds2 = generate_synthetic_psf(grid_size=64, random_seed=2)
        assert not np.allclose(ds1.true_phase, ds2.true_phase)

    def test_generate_with_broadband_and_detector_effects(self) -> None:
        ds = generate_synthetic_psf(
            grid_size=64,
            rms_aberration=0.4,
            bandwidth_fraction=0.12,
            spectral_samples=5,
            spectral_weighting="gaussian",
            detector_sigma_pixels=0.25,
            jitter_sigma_pixels=0.15,
            pixel_integration_width=1.2,
            random_seed=42,
        )
        assert ds.bandwidth_fraction == pytest.approx(0.12)
        assert ds.spectral_weighting == "gaussian"
        assert ds.detector_sigma_pixels == pytest.approx(0.25)
        assert ds.jitter_sigma_pixels == pytest.approx(0.15)
        assert ds.pixel_integration_width == pytest.approx(1.2)
        assert ds.psf_data.metadata["spectral_weighting"] == "gaussian"


# ---------------------------------------------------------------------------
# Pipeline orchestrator tests
# ---------------------------------------------------------------------------


class TestPipelineOrchestrator:
    """Tests for the reusable pipeline orchestrator."""

    def test_pipeline_run_from_psf(self, pupil: PupilModel, psf_data: PSFData) -> None:
        from src.pipeline import PipelineResult, RetrievalPipeline

        config = PipelineConfig()
        pipeline = RetrievalPipeline(config)
        alg_cfg = AlgorithmConfig(
            name=AlgorithmName.ERROR_REDUCTION,
            max_iterations=30,
            random_seed=42,
        )
        result = pipeline.run_from_psf(psf_data, pupil, alg_cfg)
        assert isinstance(result, PipelineResult)
        assert result.result.n_iterations >= 1
        assert result.ssim > 0.0
        assert len(result.zernike_coefficients) > 0

    def test_pipeline_saves_outputs(
        self, pupil: PupilModel, psf_data: PSFData, tmp_path: Path
    ) -> None:
        from src.pipeline import RetrievalPipeline

        config = PipelineConfig()
        pipeline = RetrievalPipeline(config)
        alg_cfg = AlgorithmConfig(
            name=AlgorithmName.ERROR_REDUCTION,
            max_iterations=20,
            random_seed=42,
        )
        pipeline.run_from_psf(psf_data, pupil, alg_cfg, output_dir=tmp_path)
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "result.json").exists()

        # Verify JSON is valid and contains expected fields
        summary = json.loads((tmp_path / "result.json").read_text())
        assert "strehl_ratio" in summary
        assert "rms_phase_rad" in summary
        assert "ssim" in summary


# ---------------------------------------------------------------------------
# Cross-field validator tests
# ---------------------------------------------------------------------------


class TestCrossFieldValidation:
    """Tests for AlgorithmConfig cross-field validators."""

    def test_admm_rho_warning_non_admm(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="src.models.config"):
            AlgorithmConfig(
                name=AlgorithmName.HYBRID_INPUT_OUTPUT,
                admm_rho=5.0,
            )
            assert any("admm_rho" in msg for msg in caplog.messages)

    def test_admm_rho_no_warning_for_admm(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="src.models.config"):
            AlgorithmConfig(
                name=AlgorithmName.ADMM,
                admm_rho=5.0,
            )
            rho_warnings = [msg for msg in caplog.messages if "admm_rho" in msg]
            assert len(rho_warnings) == 0

    def test_regulariser_warning_non_fista(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="src.models.config"):
            AlgorithmConfig(
                name=AlgorithmName.HYBRID_INPUT_OUTPUT,
                regulariser=Regulariser.TV,
            )
            assert any("regulariser" in msg for msg in caplog.messages)

    def test_sw_sigma_start_lt_end_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="src.models.config"):
            AlgorithmConfig(
                name=AlgorithmName.HYBRID_INPUT_OUTPUT,
                sw_sigma_start=1.0,
                sw_sigma_end=3.0,
            )
            assert any("sw_sigma_start" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Registry completeness tests
# ---------------------------------------------------------------------------


class TestRegistryCompleteness:
    """Ensure all declared algorithm names are registered."""

    def test_all_algorithms_registered(self) -> None:
        available = AlgorithmRegistry.available()
        assert "fista" in available
        assert "sparse_pr" in available
        assert len(available) == len(AlgorithmName)

    def test_fista_in_available(self) -> None:
        assert "fista" in AlgorithmRegistry.available()

    def test_sparse_pr_in_available(self) -> None:
        assert "sparse_pr" in AlgorithmRegistry.available()


# ---------------------------------------------------------------------------
# ER-finish fraction tests
# ---------------------------------------------------------------------------


class TestERFinishFraction:
    """Test configurable ER-finish fraction."""

    def test_custom_er_fraction(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.HYBRID_INPUT_OUTPUT,
            max_iterations=50,
            er_finish_fraction=0.2,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1

    def test_zero_er_fraction(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.RAAR,
            max_iterations=30,
            er_finish_fraction=0.0,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1


# ---------------------------------------------------------------------------
# Shared NumpyModel base tests
# ---------------------------------------------------------------------------


class TestNumpyModelBase:
    """Test that NumpyModel base class works correctly."""

    def test_import(self) -> None:
        from src.models._base import NumpyModel

        assert NumpyModel.model_config["arbitrary_types_allowed"] is True

    def test_psf_data_still_works(self) -> None:
        psf = PSFData(
            image=np.zeros((32, 32)),
            pixel_scale_arcsec=0.04,
            wavelength_m=606e-9,
            filter_name="SYNTH",
            telescope="test",
        )
        assert psf.image.shape == (32, 32)
