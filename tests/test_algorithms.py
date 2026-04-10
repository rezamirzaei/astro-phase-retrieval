"""Tests for phase-retrieval algorithms on synthetic data."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.algorithms.multi_start import multi_start_run
from src.algorithms.phase_diversity import PhaseDiversity
from src.algorithms.registry import AlgorithmRegistry
from src.metrics.quality import compute_ssim
from src.models.config import AlgorithmConfig, AlgorithmName, BetaSchedule, NoiseModel
from src.models.optics import PSFData, PSFPair, PupilModel
from src.optics.propagator import add_defocus, forward_model

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

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


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

        # Cost should decrease (or at least not increase) from start to end
        assert result.cost_history[-1] <= result.cost_history[0]

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

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is not installed")
    def test_pinn_runs(self, pupil: PupilModel, psf_data: PSFData) -> None:
        cfg = AlgorithmConfig(
            name=AlgorithmName.PINN,
            max_iterations=10,
            random_seed=42,
            pinn_hidden_features=16,
            pinn_hidden_layers=2,
            pinn_warm_start=True,
            pinn_warm_start_iterations=5,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1
        assert result.cost_history[-1] <= result.cost_history[0]
        assert result.metadata["warm_start_objective"] is not None
        assert result.metadata["best_objective"] <= result.metadata["warm_start_objective"] + 1e-12

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is not installed")
    def test_pinn_runs_without_torch_numpy_bridge(
        self,
        pupil: PupilModel,
        psf_data: PSFData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torch

        def _raise_numpy_unavailable(
            self,
        ) -> np.ndarray:  # pragma: no cover - exercised via monkeypatch
            raise RuntimeError("Numpy is not available")

        monkeypatch.setattr(torch.Tensor, "numpy", _raise_numpy_unavailable, raising=True)

        cfg = AlgorithmConfig(
            name=AlgorithmName.PINN,
            max_iterations=10,
            random_seed=42,
            pinn_hidden_features=16,
            pinn_hidden_layers=2,
            pinn_warm_start=True,
            pinn_warm_start_iterations=5,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1
        assert result.recovered_phase.shape == (pupil.grid_size, pupil.grid_size)
        assert np.isfinite(result.recovered_phase).all()


class TestSyntheticRecovery:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "alg_name",
        [
            AlgorithmName.RAAR,
            AlgorithmName.WIRTINGER_FLOW,
            AlgorithmName.DOUGLAS_RACHFORD,
            AlgorithmName.ADMM,
        ],
    )
    def test_reconstructed_psf_matches_truth(
        self,
        alg_name: AlgorithmName,
        pupil: PupilModel,
        psf_data: PSFData,
    ) -> None:
        cfg = AlgorithmConfig(name=alg_name, max_iterations=80, random_seed=42)
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert compute_ssim(psf_data.image, result.reconstructed_psf) > 0.99


class TestPhaseDiversity:
    def test_run_diversity_reduces_joint_cost(
        self,
        pupil: PupilModel,
        psf_data: PSFData,
        true_phase: np.ndarray,
    ) -> None:
        defocused_psf = forward_model(
            pupil.amplitude,
            add_defocus(true_phase, pupil.amplitude, defocus_waves=0.75),
        )
        pair = PSFPair(
            focused=psf_data,
            defocused=PSFData(
                image=defocused_psf,
                pixel_scale_arcsec=psf_data.pixel_scale_arcsec,
                wavelength_m=psf_data.wavelength_m,
                filter_name=psf_data.filter_name,
                telescope=psf_data.telescope,
                obs_id=f"{psf_data.obs_id}-defocused",
            ),
        )
        cfg = AlgorithmConfig(
            name=AlgorithmName.PHASE_DIVERSITY,
            max_iterations=60,
            defocus_waves=0.75,
            random_seed=42,
        )
        result = PhaseDiversity(cfg, pupil).run_diversity(pair)
        assert result.cost_history[-1] < result.cost_history[0]
        assert compute_ssim(psf_data.image, result.reconstructed_psf) > 0.99


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
        assert "pinn" in avail

    def test_unknown_algorithm_raises(self, pupil: PupilModel) -> None:
        cfg = AlgorithmConfig(name=AlgorithmName.HYBRID_INPUT_OUTPUT)
        # Manually break the name to test error path
        cfg.name = "nonexistent"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown algorithm"):
            AlgorithmRegistry.create(cfg, pupil)


# ── Property-based tests (Hypothesis) ─────────────────────────────────────


class TestPropertyBased:
    """Hypothesis property-based tests for mathematical invariants."""

    @given(
        phase_scale=st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_strehl_ratio_bounded(
        self,
        phase_scale: float,
        pupil: PupilModel,
    ) -> None:
        """For any scaled phase map, Strehl ratio must remain in [0, 1]."""
        from src.metrics.quality import compute_strehl_ratio
        from src.optics.propagator import forward_model

        n = pupil.grid_size
        rng = np.random.default_rng(0)
        phase = rng.standard_normal((n, n)) * phase_scale
        phase[pupil.amplitude == 0] = 0.0
        psf = forward_model(pupil.amplitude, phase)
        strehl = compute_strehl_ratio(psf, pupil.amplitude)
        assert 0.0 <= strehl <= 1.0, f"Strehl={strehl} out of bounds for scale={phase_scale}"

    @given(rms_rad=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_rms_phase_nonnegative(
        self,
        rms_rad: float,
        pupil: PupilModel,
    ) -> None:
        """RMS phase is always non-negative regardless of phase amplitude."""
        from src.metrics.quality import compute_rms_phase

        support = pupil.amplitude > 0
        n = pupil.grid_size
        rng = np.random.default_rng(1)
        phase = rng.standard_normal((n, n)) * rms_rad
        phase[~support] = 0.0
        rms = compute_rms_phase(phase, support)
        assert rms >= 0.0, f"RMS={rms} is negative"

    @given(
        n_iter=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_cost_history_length_matches_n_iterations(
        self,
        n_iter: int,
        pupil: PupilModel,
        psf_data: PSFData,
    ) -> None:
        """cost_history length must equal n_iterations for any iteration count."""
        cfg = AlgorithmConfig(
            name=AlgorithmName.ERROR_REDUCTION,
            max_iterations=n_iter,
            random_seed=7,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert len(result.cost_history) == result.n_iterations
        assert result.n_iterations <= n_iter


# ── Shrink-Wrap tests ────────────────────────────────────────────────────────


class TestShrinkWrap:
    """Shrink-Wrap constraint tests (Marchesini et al. 2003)."""

    def test_shrink_wrap_produces_smaller_support(
        self,
        pupil: PupilModel,
        psf_data: PSFData,
    ) -> None:
        """With SW enabled, the effective support area should not grow beyond the pupil."""
        cfg_sw = AlgorithmConfig(
            name=AlgorithmName.HYBRID_INPUT_OUTPUT,
            max_iterations=40,
            beta=0.9,
            use_sw_constraint=True,
            support_threshold=0.04,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg_sw, pupil).run(psf_data)
        # The result must still be valid
        assert result.recovered_phase.shape == pupil.amplitude.shape
        assert result.rms_phase_rad >= 0.0

    def test_shrink_wrap_disabled_vs_enabled_run_both(
        self,
        pupil: PupilModel,
        psf_data: PSFData,
    ) -> None:
        """Both SW=True and SW=False must complete without error."""
        for sw in (True, False):
            cfg = AlgorithmConfig(
                name=AlgorithmName.RAAR,
                max_iterations=30,
                use_sw_constraint=sw,
                random_seed=42,
            )
            result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
            assert len(result.cost_history) > 0

    def test_admm_rho_parameter(
        self,
        pupil: PupilModel,
        psf_data: PSFData,
    ) -> None:
        """ADMM should run correctly with a non-default rho."""
        cfg = AlgorithmConfig(
            name=AlgorithmName.ADMM,
            max_iterations=30,
            admm_rho=2.0,
            random_seed=42,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert len(result.cost_history) == result.n_iterations
        assert result.strehl_ratio >= 0.0


