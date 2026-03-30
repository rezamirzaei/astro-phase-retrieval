"""Tests for phase-retrieval algorithms on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.multi_start import multi_start_run
from src.algorithms.registry import AlgorithmRegistry
from src.models.config import AlgorithmConfig, AlgorithmName, BetaSchedule, NoiseModel
from src.models.optics import PSFData, PupilModel


# ── Helpers ───────────────────────────────────────────────────────────────

_ALGORITHMS = [
    AlgorithmName.ERROR_REDUCTION,
    AlgorithmName.GERCHBERG_SAXTON,
    AlgorithmName.HYBRID_INPUT_OUTPUT,
    AlgorithmName.RAAR,
    AlgorithmName.WIRTINGER_FLOW,
    AlgorithmName.DOUGLAS_RACHFORD,
    AlgorithmName.ADMM,
]


# ── Core algorithm tests ─────────────────────────────────────────────────


class TestAlgorithmsConverge:
    """Every algorithm should decrease cost on a noiseless synthetic problem."""

    @pytest.mark.parametrize("alg_name", _ALGORITHMS)
    def test_cost_decreases(
        self,
        alg_name: AlgorithmName,
        pupil: PupilModel,
        psf_data: PSFData,
    ) -> None:
        cfg = AlgorithmConfig(
            name=alg_name,
            max_iterations=60,
            beta=0.9,
            random_seed=42,
        )
        retriever = AlgorithmRegistry.create(cfg, pupil)
        result = retriever.run(psf_data)

        # Cost should decrease from start to end
        assert result.cost_history[-1] < result.cost_history[0]

    @pytest.mark.parametrize("alg_name", _ALGORITHMS)
    def test_result_fields(
        self,
        alg_name: AlgorithmName,
        pupil: PupilModel,
        psf_data: PSFData,
    ) -> None:
        cfg = AlgorithmConfig(
            name=alg_name,
            max_iterations=30,
            beta=0.9,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)

        assert result.algorithm == alg_name
        assert result.n_iterations >= 1
        assert result.elapsed_seconds >= 0
        assert 0.0 <= result.strehl_ratio <= 1.0
        assert result.rms_phase_rad >= 0.0
        assert result.recovered_phase.shape == (pupil.grid_size, pupil.grid_size)
        assert result.reconstructed_psf.shape == (pupil.grid_size, pupil.grid_size)

    @pytest.mark.parametrize("alg_name", _ALGORITHMS)
    def test_phase_zero_outside_support(
        self,
        alg_name: AlgorithmName,
        pupil: PupilModel,
        psf_data: PSFData,
        support: np.ndarray,
    ) -> None:
        cfg = AlgorithmConfig(
            name=alg_name,
            max_iterations=30,
            beta=0.9,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        np.testing.assert_array_equal(result.recovered_phase[~support], 0.0)


# ── State-of-the-art enhancements ────────────────────────────────────────


class TestEnhancements:
    """Test momentum, TV regularization, adaptive β, noise model."""

    def test_momentum_runs(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.HYBRID_INPUT_OUTPUT,
            max_iterations=30,
            momentum=0.5,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1

    def test_tv_regularization_runs(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.ERROR_REDUCTION,
            max_iterations=30,
            tv_weight=0.01,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1

    def test_cosine_beta_schedule(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.RAAR,
            max_iterations=30,
            beta_schedule=BetaSchedule.COSINE,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1

    def test_linear_beta_schedule(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.HYBRID_INPUT_OUTPUT,
            max_iterations=30,
            beta_schedule=BetaSchedule.LINEAR,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1

    def test_poisson_noise_model(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.GERCHBERG_SAXTON,
            max_iterations=30,
            noise_model=NoiseModel.POISSON,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1


# ── Multi-start ──────────────────────────────────────────────────────────


class TestMultiStart:
    def test_multi_start_returns_best(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.ERROR_REDUCTION,
            max_iterations=20,
            random_seed=42,
            n_starts=3,
        )
        result = multi_start_run(cfg, pupil, psf_data)
        assert result.n_iterations >= 1
        assert 0.0 <= result.strehl_ratio <= 1.0

    def test_single_start_passthrough(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.ERROR_REDUCTION,
            max_iterations=20,
            random_seed=42,
            n_starts=1,
        )
        result = multi_start_run(cfg, pupil, psf_data)
        assert result.n_iterations >= 1


# ── Registry ──────────────────────────────────────────────────────────────


class TestAlgorithmRegistry:
    def test_available_lists_all(self) -> None:
        avail = AlgorithmRegistry.available()
        assert "er" in avail
        assert "hio" in avail
        assert "raar" in avail
        assert "wf" in avail
        assert "dr" in avail
        assert "admm" in avail

    def test_unknown_algorithm_raises(self, pupil: PupilModel) -> None:
        cfg = AlgorithmConfig(name=AlgorithmName.HYBRID_INPUT_OUTPUT)
        # Manually break the name to test error path
        cfg.name = "nonexistent"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown algorithm"):
            AlgorithmRegistry.create(cfg, pupil)
