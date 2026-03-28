"""
Unit tests — Adam and L-BFGS gradient optimisers on simple 2-D potentials.

We test TorchParameterOptimizer by wrapping simple mathematical functions
in an object that exposes the same interface as TorchCostFunction
(callable with a torch.Tensor, plus .device and .dtype attributes).

Test cases
----------
1. **Convex quadratic** – unique global minimum: f(x,y) = (x-3)² + (y+2)²
   Both Adam and L-BFGS should converge to (3, -2) from any starting point.

2. **Rosenbrock banana** – single minimum with a long curved valley:
   f(x,y) = (1-x)² + 100(y-x²)²
   Minimum at (1, 1). L-BFGS is expected to converge; Adam may need many steps.

3. **Flat ridge** – f(x,y) = (x+y)²  (degenerate, minimum is a line x+y=0)
   The optimiser should still reduce cost close to 0.
"""

import numpy as np
import pytest
import torch

from core.torch_optimizer import TorchParameterOptimizer

# ── toy cost functions ────────────────────────────────────────────────────────


class _TorchCostAdapter:
    """
    Wraps a plain Python callable f(params_tensor) → scalar tensor so it
    presents the full interface that TorchParameterOptimizer expects.
    """

    def __init__(self, fn, device=None, dtype=torch.float64):
        self._fn = fn
        self.device = device or torch.device("cpu")
        self.dtype = dtype

    def __call__(self, params: torch.Tensor) -> torch.Tensor:
        return self._fn(params)

    def total_cost_numpy(self, params_np: np.ndarray) -> float:
        params = torch.tensor(params_np, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            return float(self(params))


def _quadratic(params: torch.Tensor) -> torch.Tensor:
    """f(x,y) = (x-3)² + (y+2)²  — minimum at (3, -2), value 0."""
    return (params[0] - 3.0) ** 2 + (params[1] + 2.0) ** 2


def _rosenbrock(params: torch.Tensor) -> torch.Tensor:
    """f(x,y) = (1-x)² + 100(y-x²)²  — minimum at (1,1), value 0."""
    x, y = params[0], params[1]
    return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2


def _flat_ridge(params: torch.Tensor) -> torch.Tensor:
    """f(x,y) = (x+y)²  — minimum is the line x+y=0."""
    return (params[0] + params[1]) ** 2


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_optimizer(fn) -> TorchParameterOptimizer:
    return TorchParameterOptimizer(_TorchCostAdapter(fn))


_BOUNDS_2D = [(-10.0, 10.0), (-10.0, 10.0)]


# ── quadratic tests ───────────────────────────────────────────────────────────


class TestQuadraticConvergence:

    def test_adam_finds_minimum(self):
        opt = _make_optimizer(_quadratic)
        result = opt.optimize(
            initial_guess=np.array([0.0, 0.0]),
            bounds=_BOUNDS_2D,
            n_adam=2000,
            adam_lr=0.05,
            n_lbfgs=0,
        )
        assert result.fun == pytest.approx(0.0, abs=1e-3)
        assert result.x[0] == pytest.approx(3.0, abs=0.05)
        assert result.x[1] == pytest.approx(-2.0, abs=0.05)

    def test_lbfgs_finds_minimum(self):
        opt = _make_optimizer(_quadratic)
        result = opt.optimize(
            initial_guess=np.array([0.0, 0.0]),
            bounds=_BOUNDS_2D,
            n_adam=0,
            n_lbfgs=50,
        )
        assert result.fun == pytest.approx(0.0, abs=1e-8)
        assert result.x[0] == pytest.approx(3.0, abs=1e-4)
        assert result.x[1] == pytest.approx(-2.0, abs=1e-4)

    def test_adam_then_lbfgs_finds_minimum(self):
        """The combined pipeline should converge tightly."""
        opt = _make_optimizer(_quadratic)
        result = opt.optimize(
            initial_guess=np.array([-5.0, 5.0]),
            bounds=_BOUNDS_2D,
            n_adam=500,
            adam_lr=0.1,
            n_lbfgs=50,
        )
        assert result.fun == pytest.approx(0.0, abs=1e-8)

    @pytest.mark.parametrize(
        "x0,y0",
        [
            (-8.0, 8.0),
            (8.0, -8.0),
            (0.0, 0.0),
            (9.0, 9.0),
        ],
    )
    def test_lbfgs_converges_from_many_starts(self, x0, y0):
        opt = _make_optimizer(_quadratic)
        result = opt.optimize(
            initial_guess=np.array([x0, y0]),
            bounds=_BOUNDS_2D,
            n_adam=0,
            n_lbfgs=100,
        )
        assert result.fun == pytest.approx(
            0.0, abs=1e-6
        ), f"Failed from ({x0},{y0}): cost={result.fun}"

    def test_result_has_correct_attributes(self):
        opt = _make_optimizer(_quadratic)
        result = opt.optimize(
            initial_guess=np.array([0.0, 0.0]),
            bounds=_BOUNDS_2D,
        )
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "nit")
        assert isinstance(result.x, np.ndarray)
        assert isinstance(result.fun, float)

    def test_cost_decreases_during_adam(self):
        """Track cost per step; must be non-increasing on average."""
        cost_log = []

        def _fn_with_log(params):
            loss = _quadratic(params)
            cost_log.append(loss.item())
            return loss

        opt = _make_optimizer(_fn_with_log)
        opt.optimize(
            initial_guess=np.array([0.0, 0.0]),
            bounds=_BOUNDS_2D,
            n_adam=100,
            adam_lr=0.1,
            n_lbfgs=0,
        )
        assert cost_log[-1] < cost_log[0], "Cost did not decrease during Adam"


# ── Rosenbrock tests ──────────────────────────────────────────────────────────


class TestRosenbrockConvergence:

    def test_lbfgs_solves_rosenbrock(self):
        """L-BFGS should solve the Rosenbrock function from a near-minimum start."""
        opt = _make_optimizer(_rosenbrock)
        result = opt.optimize(
            initial_guess=np.array([0.9, 0.8]),  # close to (1,1)
            bounds=[(-2.0, 2.0), (-2.0, 2.0)],
            n_adam=0,
            n_lbfgs=200,
        )
        assert result.fun == pytest.approx(0.0, abs=1e-4)
        assert result.x[0] == pytest.approx(1.0, abs=0.01)
        assert result.x[1] == pytest.approx(1.0, abs=0.01)

    def test_adam_plus_lbfgs_solves_rosenbrock(self):
        opt = _make_optimizer(_rosenbrock)
        result = opt.optimize(
            initial_guess=np.array([0.0, 0.0]),
            bounds=[(-2.0, 2.0), (-2.0, 2.0)],
            n_adam=3000,
            adam_lr=0.01,
            n_lbfgs=200,
        )
        assert result.fun < 0.1  # tolerate imperfect Adam for banana


# ── flat-ridge tests ──────────────────────────────────────────────────────────


class TestFlatRidgeConvergence:

    def test_lbfgs_reduces_cost_to_zero(self):
        opt = _make_optimizer(_flat_ridge)
        result = opt.optimize(
            initial_guess=np.array([3.0, 3.0]),
            bounds=_BOUNDS_2D,
            n_adam=0,
            n_lbfgs=100,
        )
        assert result.fun == pytest.approx(0.0, abs=1e-6)


# ── callback tests ────────────────────────────────────────────────────────────


class TestProgressCallback:

    def test_callback_is_called(self):
        calls = []

        def _cb(iteration, cost, convergence):
            calls.append((iteration, cost))

        opt = _make_optimizer(_quadratic)
        opt.optimize(
            initial_guess=np.array([0.0, 0.0]),
            bounds=_BOUNDS_2D,
            n_adam=50,
            n_lbfgs=20,
            progress_callback=_cb,
        )
        assert len(calls) > 0

    def test_callback_iteration_increases(self):
        iters = []

        def _cb(iteration, cost, convergence):
            iters.append(iteration)

        opt = _make_optimizer(_quadratic)
        opt.optimize(
            initial_guess=np.array([0.0, 0.0]),
            bounds=_BOUNDS_2D,
            n_adam=50,
            n_lbfgs=20,
            progress_callback=_cb,
        )
        # iterations should be monotonically increasing
        assert all(iters[i] <= iters[i + 1] for i in range(len(iters) - 1))
