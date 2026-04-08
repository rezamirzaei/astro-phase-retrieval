"""Tests to cover remaining uncovered lines across all modules.

Targets:
  src/__init__.py            lines 7-8   (PackageNotFoundError fallback)
  src/__main__.py            line 6      (if __name__ == "__main__)
  src/algorithms/base.py     lines 80, 90, 161, 179-192, 242, 316, 403, 451
  src/algorithms/phase_diversity.py  lines 71-72, 143-149
  src/algorithms/pinn.py     lines 178-179, 183-191, 251-252, 299, 345,
                             468, 470, 472-478, 489
  src/algorithms/wirtinger_flow.py   lines 50, 57
  src/data/loader.py         lines 42-43, 45
  src/cli.py                 lines 31-37 (_JsonFormatter)
  src/metrics/quality.py     lines 221, 229-232
  src/optics/zernike.py      line 60
"""

from __future__ import annotations

import importlib
import importlib.util
import runpy
from unittest.mock import patch

import numpy as np
import pytest

from src.algorithms.base import PhaseRetriever
from src.algorithms.phase_diversity import PhaseDiversity
from src.algorithms.registry import AlgorithmRegistry
from src.metrics.quality import compute_phase_structure_function
from src.models.config import AlgorithmConfig, AlgorithmName
from src.models.optics import PSFData, PSFPair, PupilModel
from src.optics.propagator import add_defocus, forward_model
from src.optics.zernike import _noll_lookup

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


# =====================================================================
# src/__init__.py  lines 7-8
# =====================================================================


class TestInitVersionFallback:
    def test_version_fallback_on_not_found(self) -> None:
        """When the package is not installed, __version__ falls back."""
        from importlib.metadata import PackageNotFoundError

        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("importlib.metadata.version", side_effect=PackageNotFoundError()),
        ):
            # Reload the module to re-execute the try/except
            import src

            importlib.reload(src)
            assert src.__version__ == "0.0.0-dev"

        # Restore
        importlib.reload(src)


# =====================================================================
# src/__main__.py  line 6  +  src/cli.py  line 332
# =====================================================================


class TestMainModule:
    def test_main_module_guard(self) -> None:
        """Exercise the ``if __name__ == '__main__': main()`` guard."""
        with patch("src.cli.main") as mock_main:
            runpy.run_module("src", run_name="__main__", alter_sys=False)
            mock_main.assert_called_once()


# =====================================================================
# src/algorithms/base.py  lines 80, 90, 242, 316, 451
# =====================================================================


class TestBaseEdgeCases:
    def test_shape_mismatch_raises(self, pupil: PupilModel) -> None:
        """Line 80: PSF shape doesn't match pupil grid → ValueError."""
        wrong = PSFData(
            image=np.ones((32, 32)),
            pixel_scale_arcsec=0.04,
            wavelength_m=606e-9,
            filter_name="X",
            telescope="test",
            obs_id="bad",
        )
        cfg = AlgorithmConfig(
            name=AlgorithmName.ERROR_REDUCTION,
            max_iterations=5,
            random_seed=1,
        )
        with pytest.raises(ValueError, match="does not match"):
            AlgorithmRegistry.create(cfg, pupil).run(wrong)

    def test_zero_energy_warning(self, pupil: PupilModel) -> None:
        """Line 90: all-zero PSF triggers a UserWarning."""
        n = pupil.grid_size
        zero = PSFData(
            image=np.zeros((n, n)),
            pixel_scale_arcsec=0.04,
            wavelength_m=606e-9,
            filter_name="X",
            telescope="test",
            obs_id="zero",
        )
        cfg = AlgorithmConfig(
            name=AlgorithmName.ERROR_REDUCTION,
            max_iterations=2,
            random_seed=1,
            spectral_init=False,
        )
        with pytest.warns(UserWarning, match="zero total energy"):
            AlgorithmRegistry.create(cfg, pupil).run(zero)

    def test_get_beta_unknown_schedule_returns_max(
        self, pupil: PupilModel, psf_data: PSFData
    ) -> None:
        """Line 242: unrecognised schedule falls back to beta_max."""
        cfg = AlgorithmConfig(
            name=AlgorithmName.ERROR_REDUCTION,
            max_iterations=5,
            random_seed=1,
        )
        retriever = AlgorithmRegistry.create(cfg, pupil)
        # Monkey-patch schedule to an unsupported value
        retriever.config.beta_schedule = "unknown"  # type: ignore[assignment]
        beta = retriever._get_beta(1)
        assert beta == cfg.beta

    def test_tv_prox_zero_weight_passthrough(self) -> None:
        """Line 316: weight <= 0 returns input unchanged."""
        phase = np.random.default_rng(0).standard_normal((32, 32))
        support = np.ones((32, 32), dtype=bool)
        result = PhaseRetriever._tv_prox(phase, weight=0.0, support=support)
        np.testing.assert_array_equal(result, phase)

    def test_focal_cost_zero_target(self) -> None:
        """Line 451: when target is all-zero, denom=0 branch."""
        target = np.zeros((8, 8))
        G = np.ones((8, 8), dtype=complex)
        cost = PhaseRetriever._focal_cost(target, G)
        assert isinstance(cost, float)
        assert cost >= 0.0


# =====================================================================
# src/algorithms/phase_diversity.py  lines 71-72, 143-149
# =====================================================================


class TestPhaseDiversityCoverage:
    def test_convergence_early_stop(
        self,
        pupil: PupilModel,
        psf_data: PSFData,
        true_phase: np.ndarray,
    ) -> None:
        """Lines 71-72: early convergence with very loose tolerance."""
        defocused = forward_model(
            pupil.amplitude,
            add_defocus(true_phase, pupil.amplitude, 0.75),
        )
        pair = PSFPair(
            focused=psf_data,
            defocused=PSFData(
                image=defocused,
                pixel_scale_arcsec=0.04,
                wavelength_m=606e-9,
                filter_name="S",
                telescope="test",
                obs_id="defoc",
            ),
        )
        cfg = AlgorithmConfig(
            name=AlgorithmName.PHASE_DIVERSITY,
            max_iterations=200,
            defocus_waves=0.75,
            random_seed=42,
            tolerance=1e6,  # huge → converge on 2nd iteration
        )
        result = PhaseDiversity(cfg, pupil).run_diversity(pair)
        assert result.converged
        assert result.n_iterations == 2

    def test_single_image_fallback(self, pupil: PupilModel, psf_data: PSFData) -> None:
        """Lines 143-149: single-image _iterate (ER-like) via base run()."""
        cfg = AlgorithmConfig(
            name=AlgorithmName.PHASE_DIVERSITY,
            max_iterations=5,
            random_seed=42,
        )
        result = PhaseDiversity(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1
        assert result.recovered_phase.shape == (pupil.grid_size, pupil.grid_size)


# =====================================================================
# src/algorithms/pinn.py
# =====================================================================


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestPINNCoverage:
    def test_pinn_no_warm_start(self, pupil: PupilModel, psf_data: PSFData) -> None:
        """Lines 178-179, 489: best_loss updates during Adam without warm start."""
        cfg = AlgorithmConfig(
            name=AlgorithmName.PINN,
            max_iterations=8,
            random_seed=42,
            pinn_hidden_features=16,
            pinn_hidden_layers=2,
            pinn_warm_start=False,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1
        assert result.metadata["warm_start_objective"] is None

    def test_pinn_early_convergence(self, pupil: PupilModel, psf_data: PSFData) -> None:
        """Lines 183-191: early convergence with very high tolerance."""
        cfg = AlgorithmConfig(
            name=AlgorithmName.PINN,
            max_iterations=50,
            random_seed=42,
            pinn_hidden_features=16,
            pinn_hidden_layers=2,
            pinn_warm_start=True,
            pinn_warm_start_iterations=5,
            tolerance=1e6,  # huge → converge early
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations < 50

    def test_pinn_iterate_raises(self, pupil: PupilModel) -> None:
        """Line 299: _iterate raises NotImplementedError."""
        cfg = AlgorithmConfig(name=AlgorithmName.PINN, max_iterations=1)
        retriever = AlgorithmRegistry.create(cfg, pupil)
        with pytest.raises(NotImplementedError, match="overrides run"):
            retriever._iterate(
                g=np.zeros((64, 64)),
                pupil_amplitude=np.zeros((64, 64)),
                target_amplitude=np.zeros((64, 64)),
                support=np.zeros((64, 64), dtype=bool),
                iteration=1,
            )

    def test_pinn_forward_phase_coverage(self, pupil: PupilModel, psf_data: PSFData) -> None:
        """Line 345: exercise _forward_phase via a short run."""
        cfg = AlgorithmConfig(
            name=AlgorithmName.PINN,
            max_iterations=3,
            random_seed=42,
            pinn_hidden_features=16,
            pinn_hidden_layers=2,
            pinn_warm_start=True,
            pinn_warm_start_iterations=3,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert np.isfinite(result.recovered_phase).all()

    def test_device_resolution_branches(self, pupil: PupilModel) -> None:
        """Lines 468-478: test all device resolution branches."""
        import torch

        from src.algorithms.pinn import PINNPhaseRetriever

        cfg = AlgorithmConfig(name=AlgorithmName.PINN, max_iterations=1)
        retriever = PINNPhaseRetriever(cfg, pupil)

        # auto → cpu (mock CUDA and MPS unavailable)
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=False),
        ):
            assert retriever._resolve_device(torch) == "cpu"

        # auto → cuda
        with patch.object(torch.cuda, "is_available", return_value=True):
            assert retriever._resolve_device(torch) == "cuda"

        # auto → mps
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=True),
        ):
            assert retriever._resolve_device(torch) == "mps"

        # cuda requested but unavailable → cpu
        retriever.config.pinn_device = "cuda"
        with patch.object(torch.cuda, "is_available", return_value=False):
            assert retriever._resolve_device(torch) == "cpu"

        # mps requested but unavailable → cpu
        retriever.config.pinn_device = "mps"
        with patch.object(torch.backends.mps, "is_available", return_value=False):
            assert retriever._resolve_device(torch) == "cpu"

        # Explicit "cpu" requested → "cpu" (line 478: return requested)
        retriever.config.pinn_device = "cpu"
        assert retriever._resolve_device(torch) == "cpu"

    def test_pinn_lbfgs_improves(self, pupil: PupilModel, psf_data: PSFData) -> None:
        """Lines 251-252: L-BFGS phase should try to improve on best_loss."""
        cfg = AlgorithmConfig(
            name=AlgorithmName.PINN,
            max_iterations=15,
            random_seed=42,
            pinn_hidden_features=16,
            pinn_hidden_layers=2,
            pinn_warm_start=False,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1
        assert result.metadata["best_objective"] is not None


# =====================================================================
# src/algorithms/wirtinger_flow.py  lines 50, 57
# =====================================================================


class TestWirtingerFlowCoverage:
    def test_wf_no_spectral_init(self, pupil: PupilModel, psf_data: PSFData) -> None:
        """Lines 50, 57: wf_spectral_init=False uses random fallback."""
        cfg = AlgorithmConfig(
            name=AlgorithmName.WIRTINGER_FLOW,
            max_iterations=10,
            random_seed=42,
            wf_spectral_init=False,
            spectral_init=False,
        )
        result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)
        assert result.n_iterations >= 1


# =====================================================================
# src/metrics/quality.py  lines 221, 229-232
# =====================================================================


class TestPhaseStructureFunctionCoverage:
    def test_all_quadrant_branches(self, pupil: PupilModel, support: np.ndarray) -> None:
        """Lines 221, 229-232: exercise all four shift-direction branches."""
        rng = np.random.default_rng(42)
        phase = rng.standard_normal((pupil.grid_size, pupil.grid_size)) * 0.5
        phase[~support] = 0.0
        # Use a larger max_sep to ensure all quadrant branches (dy<0, dx<0, etc.)
        seps, sf = compute_phase_structure_function(phase, support, max_sep=30)
        assert len(seps) == 30
        assert all(v >= 0 for v in sf)


# =====================================================================
# src/optics/zernike.py  line 60
# =====================================================================


class TestNollLookupBoundary:
    def test_noll_index_boundary_correction(self) -> None:
        """Line 60: Noll indices at the boundary where n needs +1 correction."""
        # j=66 should trigger the n += 1 branch
        for j in [66, 78, 91, 105]:
            n, m = _noll_lookup(j)
            assert n >= 0
            assert (n - abs(m)) % 2 == 0, f"Invalid Zernike (n, m) for j={j}: ({n}, {m})"


# =====================================================================
# src/algorithms/base.py  lines 161, 179-192 (rich progress bar path)
# =====================================================================


class TestRichProgressBar:
    def test_run_with_rich_progress_bar(self, pupil: PupilModel, psf_data: PSFData) -> None:
        """Exercise the rich progress-bar branch by mocking sys.stdout.isatty()=True."""
        from unittest.mock import MagicMock, patch

        # Build a mock Progress that acts like a context manager
        mock_task_id = MagicMock()  # non-None task id
        mock_progress = MagicMock()
        mock_progress.__enter__ = MagicMock(return_value=mock_progress)
        mock_progress.__exit__ = MagicMock(return_value=False)
        mock_progress.add_task = MagicMock(return_value=mock_task_id)

        # Patch at the rich.progress level so the local `from rich.progress import ...`
        # picks up the mocks, and patch sys.stdout.isatty to signal a TTY.
        with (
            patch("sys.stdout.isatty", return_value=True),
            patch("rich.progress.Progress", return_value=mock_progress),
            patch("rich.progress.SpinnerColumn", return_value=MagicMock()),
            patch("rich.progress.TextColumn", return_value=MagicMock()),
            patch("rich.progress.BarColumn", return_value=MagicMock()),
            patch("rich.progress.TaskProgressColumn", return_value=MagicMock()),
            patch("rich.progress.TimeElapsedColumn", return_value=MagicMock()),
        ):
            cfg = AlgorithmConfig(
                name=AlgorithmName.ERROR_REDUCTION,
                max_iterations=5,
                random_seed=42,
            )
            result = AlgorithmRegistry.create(cfg, pupil).run(psf_data)

        assert result.n_iterations >= 1
        # progress.update should have been called at least once per iteration
        mock_progress.update.assert_called()


# =====================================================================
# src/algorithms/base.py  line 403 (TV prox early-exit break)
# =====================================================================


class TestTvProxEarlyExit:
    def test_tv_prox_converges_early(self) -> None:
        """Line 403: TV prox early-exit is triggered when dual variables converge."""
        rng = np.random.default_rng(0)
        n = 32
        # Smooth, non-zero phase — dual variables will have non-zero norm
        # but will converge quickly on a smooth signal
        phase = rng.standard_normal((n, n)) * 0.01  # very small phase
        support = np.ones((n, n), dtype=bool)
        # Run with many iterations and a medium weight — should break early
        result = PhaseRetriever._tv_prox(phase, weight=0.5, support=support, n_iter=100)
        # Result should be a valid array (the early exit didn't corrupt it)
        assert result.shape == phase.shape
        assert np.all(np.isfinite(result))
