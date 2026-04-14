"""Mock-based tests for PINNPhaseRetriever — **no PyTorch dependency required**.

Exercises every method of ``src.algorithms.pinn.PINNPhaseRetriever`` using
lightweight numpy-backed fake tensors so that full code coverage is achieved
even in CI environments that do not install the ``pinn`` optional extra.
"""

from __future__ import annotations

import sys
from typing import Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.algorithms.pinn import PINNPhaseRetriever, _TorchModules
from src.models.config import AlgorithmConfig, AlgorithmName

# =====================================================================
# Numpy-backed fake tensor & mock torch / nn namespaces
# =====================================================================


def _uw(x: object) -> np.ndarray:
    """Unwrap ``_FT`` to its underlying numpy array, or pass through."""
    return x._d if isinstance(x, _FT) else x  # type: ignore[return-value]


class _FT:
    """Numpy-backed fake tensor that supports the torch Tensor API surface
    used by :class:`PINNPhaseRetriever`."""

    def __init__(self, data: object, dtype: np.dtype | None = None) -> None:
        if isinstance(data, _FT):
            data = data._d
        arr = np.asarray(data)
        resolved_dtype = dtype or np.float64
        # Avoid ComplexWarning when casting complex → real:
        # take the real part explicitly instead of letting numpy silently discard
        # the imaginary component.
        if np.issubdtype(arr.dtype, np.complexfloating) and not np.issubdtype(
            resolved_dtype, np.complexfloating
        ):
            arr = arr.real
        self._d: np.ndarray = np.asarray(arr, dtype=resolved_dtype)

    # -- arithmetic ----------------------------------------------------
    def __add__(self, o: object) -> _FT:
        return _FT(self._d + _uw(o))

    def __radd__(self, o: object) -> _FT:
        return _FT(_uw(o) + self._d)

    def __sub__(self, o: object) -> _FT:
        return _FT(self._d - _uw(o))

    def __rsub__(self, o: object) -> _FT:
        return _FT(_uw(o) - self._d)

    def __mul__(self, o: object) -> _FT:
        return _FT(self._d * _uw(o))

    def __rmul__(self, o: object) -> _FT:
        return _FT(_uw(o) * self._d)

    def __truediv__(self, o: object) -> _FT:
        return _FT(self._d / _uw(o))

    def __pow__(self, o: object) -> _FT:
        return _FT(self._d ** _uw(o))

    def __neg__(self) -> _FT:
        return _FT(-self._d)

    def __matmul__(self, o: object) -> _FT:
        return _FT(self._d @ _uw(o))

    def __getitem__(self, k: object) -> _FT:
        return _FT(self._d[k])  # type: ignore[call-overload]

    def __float__(self) -> float:
        return float(np.asarray(self._d).flat[0])

    # -- torch Tensor API ----------------------------------------------
    def reshape(self, *a: int) -> _FT:
        return _FT(self._d.reshape(*a))

    def square(self) -> _FT:
        return _FT(self._d**2)

    def mean(self) -> _FT:
        return _FT(np.mean(self._d))

    def sum(self) -> _FT:
        return _FT(np.sum(self._d))

    def clamp_min(self, v: float) -> _FT:
        return _FT(np.maximum(self._d, v))

    def clamp(self, min: float | None = None, max: float | None = None) -> _FT:
        return _FT(np.clip(self._d, min, max))

    def detach(self) -> _FT:
        return self

    def cpu(self) -> _FT:
        return self

    def tolist(self) -> list:  # type: ignore[type-arg]
        return self._d.tolist()  # type: ignore[no-any-return]

    def to(self, *_a: object, **_kw: object) -> _FT:
        return self

    def backward(self) -> None:
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        return self._d.shape


# -- Fake nn -----------------------------------------------------------


class _FakeNN:
    """Minimal fake ``torch.nn`` namespace."""

    class Linear:
        def __init__(self, in_f: int, out_f: int) -> None:
            rng = np.random.default_rng(0)
            self.weight = _FT(rng.standard_normal((out_f, in_f)) * 0.01)
            self.bias = _FT(np.zeros(out_f))

        def __call__(self, x: _FT) -> _FT:
            return _FT(_uw(x) @ _uw(self.weight).T + _uw(self.bias))

    class GELU:
        def __call__(self, x: _FT) -> _FT:
            d = _uw(x)
            return _FT(d * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (d + 0.044715 * d**3))))

    class Sequential:
        def __init__(self, *layers: object) -> None:
            self._layers = layers

        def __call__(self, x: _FT) -> _FT:
            r: object = x
            for lay in self._layers:
                r = lay(r)  # type: ignore[operator]
            return r  # type: ignore[return-value]

        def parameters(self):  # noqa: ANN201
            for lay in self._layers:
                if hasattr(lay, "weight"):
                    yield lay.weight  # type: ignore[attr-defined]
                    yield lay.bias  # type: ignore[attr-defined]

        def modules(self):  # noqa: ANN201
            yield self
            yield from self._layers

        def to(self, *_a: object, **_kw: object) -> _FakeNN.Sequential:
            return self

    class init:  # noqa: N801
        @staticmethod
        def xavier_uniform_(_t: object) -> None:
            pass

        @staticmethod
        def zeros_(_t: object) -> None:
            pass

    class utils:  # noqa: N801
        @staticmethod
        def clip_grad_norm_(_params: object, _max_norm: float) -> None:
            pass


# -- Fake torch --------------------------------------------------------


class _FakeTorch:
    """Minimal fake ``torch`` namespace backed by numpy."""

    float32 = np.float32
    complex64 = np.complex64

    @staticmethod
    def manual_seed(_s: int) -> None:
        pass

    @staticmethod
    def tensor(data: object, dtype: np.dtype | None = None, device: str | None = None) -> _FT:
        return _FT(data, dtype=dtype)

    @staticmethod
    def randn(
        *shape: int,
        generator: object = None,
        device: str | None = None,
        dtype: np.dtype | None = None,
    ) -> _FT:
        return _FT(np.random.default_rng(42).standard_normal(shape), dtype=dtype)

    # element-wise ops
    @staticmethod
    def sin(x: _FT) -> _FT:
        return _FT(np.sin(_uw(x)))

    @staticmethod
    def cos(x: _FT) -> _FT:
        return _FT(np.cos(_uw(x)))

    @staticmethod
    def tanh(x: _FT) -> _FT:
        return _FT(np.tanh(_uw(x)))

    @staticmethod
    def sqrt(x: _FT) -> _FT:
        return _FT(np.sqrt(np.maximum(_uw(x), 0.0)))

    @staticmethod
    def log1p(x: _FT) -> _FT:
        return _FT(np.log1p(_uw(x)))

    @staticmethod
    def mean(x: _FT) -> _FT:
        return _FT(np.mean(_uw(x)))

    @staticmethod
    def abs(x: _FT) -> _FT:
        return _FT(np.abs(_uw(x)))

    @staticmethod
    def exp(x: _FT) -> _FT:
        return _FT(np.exp(_uw(x)))

    @staticmethod
    def cat(tensors: list[_FT], dim: int = -1) -> _FT:
        return _FT(np.concatenate([_uw(t) for t in tensors], axis=dim))

    @staticmethod
    def linspace(
        start: float, end: float, steps: int, device: str | None = None, dtype: object = None
    ) -> _FT:
        return _FT(np.linspace(start, end, steps), dtype=dtype)  # type: ignore[arg-type]

    @staticmethod
    def meshgrid(*tensors: _FT, indexing: Literal["xy", "ij"] = "ij") -> list[_FT]:
        unwrapped = [_uw(t) for t in tensors]
        grids = np.meshgrid(*unwrapped, indexing=indexing)
        return [_FT(r) for r in grids]

    @staticmethod
    def stack(tensors: list[_FT], dim: int = -1) -> _FT:
        return _FT(np.stack([_uw(t) for t in tensors], axis=dim))

    class Generator:
        def __init__(self, device: str | None = None) -> None:
            pass

        def manual_seed(self, _s: int) -> None:
            pass

    class fft:  # noqa: N801
        @staticmethod
        def fft2(x: _FT) -> _FT:
            return _FT(np.fft.fft2(_uw(x)))

        @staticmethod
        def fftshift(x: _FT) -> _FT:
            return _FT(np.fft.fftshift(_uw(x)))

        @staticmethod
        def ifftshift(x: _FT) -> _FT:
            return _FT(np.fft.ifftshift(_uw(x)))

    class no_grad:  # noqa: N801
        def __enter__(self) -> _FakeTorch.no_grad:
            return self

        def __exit__(self, *_a: object) -> None:
            pass

    class cuda:  # noqa: N801
        @staticmethod
        def is_available() -> bool:
            return False

    class backends:  # noqa: N801
        class mps:  # noqa: N801
            @staticmethod
            def is_available() -> bool:
                return False

    class optim:  # noqa: N801
        class Adam:
            def __init__(self, params: object, lr: float = 0.001) -> None:
                self._p = list(params)  # type: ignore[call-overload]

            def zero_grad(self, set_to_none: bool = False) -> None:
                pass

            def step(self, closure: object = None) -> None:
                pass

        class LBFGS:
            def __init__(
                self,
                params: object,
                *,
                lr: float = 1,
                max_iter: int = 20,
                line_search_fn: str | None = None,
                history_size: int = 10,
                tolerance_grad: float = 1e-5,
                tolerance_change: float = 1e-9,
            ) -> None:
                self._p = list(params)  # type: ignore[call-overload]

            def zero_grad(self) -> None:
                pass

            def step(self, closure: object = None) -> None:
                if closure is not None:
                    closure()  # type: ignore[operator]

        class lr_scheduler:  # noqa: N801
            class CosineAnnealingWarmRestarts:
                def __init__(
                    self,
                    optimizer: object,
                    T_0: int = 1,
                    T_mult: int = 1,
                    eta_min: float = 0,
                ) -> None:
                    pass

                def step(self) -> None:
                    pass


# =====================================================================
# Helpers
# =====================================================================

_FAKE_MODULES = _TorchModules(torch=_FakeTorch, nn=_FakeNN)


def _cfg(**overrides: object) -> AlgorithmConfig:
    """Build a small-footprint PINN config."""
    defaults: dict[str, object] = dict(
        name=AlgorithmName.PINN,
        max_iterations=10,
        random_seed=42,
        pinn_hidden_features=8,
        pinn_hidden_layers=1,
        pinn_fourier_features=8,
        pinn_warm_start=False,
    )
    defaults.update(overrides)
    return AlgorithmConfig(**defaults)


def _run(retriever: PINNPhaseRetriever, psf_data: object):  # noqa: ANN201
    with patch.object(PINNPhaseRetriever, "_import_torch", return_value=_FAKE_MODULES):
        return retriever.run(psf_data)  # type: ignore[arg-type]


# =====================================================================
# Tests
# =====================================================================


class TestPINNRunMocked:
    """Exercise the full ``run()`` method with numpy-backed fakes."""

    def test_run_no_warm_start_with_lbfgs(self, pupil, psf_data) -> None:
        """Adam (8 iters, no convergence) → L-BFGS (budget 2)."""
        cfg = _cfg(max_iterations=10, pinn_warm_start=False)
        result = _run(PINNPhaseRetriever(cfg, pupil), psf_data)
        assert result.n_iterations >= 1
        assert result.metadata["warm_start_objective"] is None
        assert not result.metadata["fallback_to_warm_start"]
        assert result.metadata["adam_iterations"] == 8
        assert result.metadata["lbfgs_budget"] == 2
        assert np.isfinite(result.recovered_phase).all()

    def test_run_with_warm_start(self, pupil, psf_data) -> None:
        """Warm-start path: RAAR → neural field refinement."""
        cfg = _cfg(max_iterations=10, pinn_warm_start=True, pinn_warm_start_iterations=5)
        result = _run(PINNPhaseRetriever(cfg, pupil), psf_data)
        assert result.n_iterations >= 1
        assert result.metadata["warm_start_objective"] is not None
        assert result.metadata["warm_start"] is True

    def test_run_convergence(self, pupil, psf_data) -> None:
        """Enough Adam iterations (48) to trigger the convergence window check.

        Since mock weights never change, all loss values are identical →
        ``rel_change == 0 < tolerance`` → converges.
        """
        cfg = _cfg(max_iterations=60, pinn_warm_start=False)
        result = _run(PINNPhaseRetriever(cfg, pupil), psf_data)
        assert result.converged

    def test_run_no_grad_clip(self, pupil, psf_data) -> None:
        """Gradient clipping disabled (pinn_grad_clip=0)."""
        cfg = _cfg(max_iterations=5, pinn_grad_clip=0.0, pinn_warm_start=False)
        result = _run(PINNPhaseRetriever(cfg, pupil), psf_data)
        assert result.n_iterations >= 1

    def test_run_no_random_seed(self, pupil, psf_data) -> None:
        """No fixed random seed — exercises the ``is None`` branches."""
        cfg = _cfg(max_iterations=5, random_seed=None, pinn_warm_start=False)
        result = _run(PINNPhaseRetriever(cfg, pupil), psf_data)
        assert result.n_iterations >= 1

    def test_result_fields(self, pupil, psf_data) -> None:
        """All expected metadata keys are present."""
        cfg = _cfg(max_iterations=5, pinn_warm_start=False)
        result = _run(PINNPhaseRetriever(cfg, pupil), psf_data)
        for key in (
            "solver",
            "device",
            "final_objective",
            "best_objective",
            "hidden_features",
            "hidden_layers",
            "fourier_features",
            "fourier_sigma",
            "learning_rate",
            "sqrt_weight",
            "log_weight",
            "warm_start",
            "warm_start_iterations",
            "residual_scale",
            "warm_start_objective",
            "fallback_to_warm_start",
            "adam_iterations",
            "lbfgs_budget",
        ):
            assert key in result.metadata, f"missing metadata key: {key}"


class TestPINNHelpersMocked:
    """Targeted tests for individual helper methods."""

    # -- _iterate() ----------------------------------------------------

    def test_iterate_raises(self, pupil) -> None:
        with pytest.raises(NotImplementedError, match="overrides run"):
            PINNPhaseRetriever(_cfg(), pupil)._iterate(
                g=np.zeros((64, 64)),
                pupil_amplitude=np.zeros((64, 64)),
                target_amplitude=np.zeros((64, 64)),
                support=np.zeros((64, 64), dtype=bool),
                iteration=1,
            )

    # -- _resolve_device() ---------------------------------------------

    def test_resolve_device_auto_cpu(self, pupil) -> None:
        r = PINNPhaseRetriever(_cfg(), pupil)
        assert r._resolve_device(_FakeTorch) == "cpu"

    def test_resolve_device_auto_cuda(self, pupil) -> None:
        class _T(_FakeTorch):
            class cuda:  # noqa: N801
                @staticmethod
                def is_available() -> bool:
                    return True

            class backends(_FakeTorch.backends):  # noqa: N801
                pass

        assert PINNPhaseRetriever(_cfg(), pupil)._resolve_device(_T) == "cuda"

    def test_resolve_device_auto_mps(self, pupil) -> None:
        class _T(_FakeTorch):
            class cuda:  # noqa: N801
                @staticmethod
                def is_available() -> bool:
                    return False

            class backends:  # noqa: N801
                class mps:  # noqa: N801
                    @staticmethod
                    def is_available() -> bool:
                        return True

        assert PINNPhaseRetriever(_cfg(), pupil)._resolve_device(_T) == "mps"

    def test_resolve_device_cuda_unavailable(self, pupil) -> None:
        r = PINNPhaseRetriever(_cfg(pinn_device="cuda"), pupil)
        assert r._resolve_device(_FakeTorch) == "cpu"

    def test_resolve_device_mps_unavailable(self, pupil) -> None:
        r = PINNPhaseRetriever(_cfg(pinn_device="mps"), pupil)
        assert r._resolve_device(_FakeTorch) == "cpu"

    def test_resolve_device_mps_no_attr(self, pupil) -> None:
        """backends has no ``mps`` attribute at all."""

        class _T(_FakeTorch):
            class cuda:  # noqa: N801
                @staticmethod
                def is_available() -> bool:
                    return False

            class backends:  # noqa: N801
                pass  # no mps

        r = PINNPhaseRetriever(_cfg(pinn_device="mps"), pupil)
        assert r._resolve_device(_T) == "cpu"

    def test_resolve_device_explicit_cpu(self, pupil) -> None:
        r = PINNPhaseRetriever(_cfg(pinn_device="cpu"), pupil)
        assert r._resolve_device(_FakeTorch) == "cpu"

    # -- _tensor_to_numpy() --------------------------------------------

    def test_tensor_to_numpy(self) -> None:
        ft = _FT(np.array([[1.0, 2.0], [3.0, 4.0]]))
        result = PINNPhaseRetriever._tensor_to_numpy(ft)
        np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])
        assert result.dtype == np.float64

    # -- _objective_value() --------------------------------------------

    def test_objective_value_identical(self, pupil) -> None:
        r = PINNPhaseRetriever(_cfg(), pupil)
        n = pupil.grid_size
        arr = np.random.default_rng(0).random((n, n)).astype(np.float32)
        arr /= arr.sum()
        val = r._objective_value(target_np=arr, reconstructed_psf=arr.copy())
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_objective_value_nonzero(self, pupil) -> None:
        r = PINNPhaseRetriever(_cfg(), pupil)
        n = pupil.grid_size
        a = np.random.default_rng(0).random((n, n)).astype(np.float32)
        a /= a.sum()
        b = np.random.default_rng(1).random((n, n)).astype(np.float32)
        b /= b.sum()
        assert r._objective_value(target_np=a, reconstructed_psf=b) > 0.0

    # -- _warm_start_phase() -------------------------------------------

    def test_warm_start_disabled(self, pupil, psf_data) -> None:
        r = PINNPhaseRetriever(_cfg(pinn_warm_start=False), pupil)
        phase, result = r._warm_start_phase(psf_data)
        assert result is None
        np.testing.assert_array_equal(phase, 0.0)

    def test_warm_start_enabled(self, pupil, psf_data) -> None:
        r = PINNPhaseRetriever(_cfg(pinn_warm_start=True, pinn_warm_start_iterations=5), pupil)
        phase, result = r._warm_start_phase(psf_data)
        assert result is not None
        assert phase.shape == (pupil.grid_size, pupil.grid_size)
        assert result.algorithm.value == "raar"

    # -- _coordinate_features() ----------------------------------------

    def test_coordinate_features(self) -> None:
        coords = PINNPhaseRetriever._coordinate_features(
            _FakeTorch, 8, device="cpu", dtype=np.float32
        )
        assert coords.shape == (64, 5)  # 8*8 pixels, 5 features

    # -- _smoothness_penalty() -----------------------------------------

    def test_smoothness_penalty(self) -> None:
        phase = _FT(np.random.default_rng(0).standard_normal((8, 8)))
        support = _FT(np.ones((8, 8)))
        penalty = PINNPhaseRetriever._smoothness_penalty(phase, support)
        assert float(penalty) >= 0.0

    # -- _build_phase_field() ------------------------------------------

    def test_build_phase_field(self, pupil) -> None:
        r = PINNPhaseRetriever(_cfg(pinn_hidden_features=8, pinn_hidden_layers=2), pupil)
        model = r._build_phase_field(_FakeNN, n_fourier=8, device="cpu", dtype=np.float32)
        # Should be callable: (N, 16) → (N, 1)
        x = _FT(np.random.default_rng(0).standard_normal((4, 16)))
        out = model(x)
        assert out.shape == (4, 1)

    # -- _forward_phase() ---------------------------------------------

    def test_forward_phase(self, pupil) -> None:
        n = 8
        r = PINNPhaseRetriever(
            _cfg(
                pinn_hidden_features=8,
                pinn_hidden_layers=1,
                pinn_fourier_features=8,
                pinn_residual_scale=0.5,
            ),
            pupil,
        )
        model = r._build_phase_field(_FakeNN, n_fourier=8, device="cpu", dtype=np.float32)
        coords = PINNPhaseRetriever._coordinate_features(
            _FakeTorch, n, device="cpu", dtype=np.float32
        )
        B = _FT(np.random.default_rng(0).standard_normal((5, 8)))
        base = _FT(np.zeros((n, n)))
        support = _FT(np.ones((n, n)))
        phase = r._forward_phase(model, coords, B, base, support, n, _FakeTorch)
        assert phase.shape == (n, n)

    # -- _forward_psf() -----------------------------------------------

    def test_forward_psf(self, pupil) -> None:
        n = 8
        r = PINNPhaseRetriever(_cfg(), pupil)
        phase = _FT(np.zeros((n, n)))
        amp = _FT(np.ones((n, n)))
        psf = r._forward_psf(phase, amp, _FakeTorch)
        assert psf.shape == (n, n)
        assert float(psf.sum()) == pytest.approx(1.0, abs=1e-6)

    # -- _composite_loss() --------------------------------------------

    def test_composite_loss(self, pupil) -> None:
        n = 8
        r = PINNPhaseRetriever(_cfg(), pupil)
        pred = _FT(np.random.default_rng(0).random((n, n)))
        pred = pred / pred.sum()
        target = _FT(np.random.default_rng(1).random((n, n)))
        target = target / target.sum()
        phase = _FT(np.zeros((n, n)))
        support = _FT(np.ones((n, n)))
        loss, objective = r._composite_loss(pred, target, phase, support, _FakeTorch)
        assert float(objective) >= 0.0

    # -- _import_torch() (mocked sys.modules) --------------------------

    def test_import_torch_via_sys_modules(self) -> None:
        """Exercise the try-branch of _import_torch using mocked modules."""
        mock_torch = MagicMock()
        mock_nn = MagicMock()
        mock_torch.nn = mock_nn
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_nn}):
            result = PINNPhaseRetriever._import_torch()
        assert result.torch is mock_torch
        assert result.nn is mock_nn
