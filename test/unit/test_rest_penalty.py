"""
Unit tests — resting-inactivation penalty  rest_lambda · I_rest².

I_rest is the inactivated fraction of the steady state at v_rest. With no
experimental data the data cost is zero, so the total cost is the penalty
alone and can be compared against a direct computation.

Checks, for the scipy CostFunction and (if torch) TorchCostFunction:
1. rest_lambda = 0 adds nothing.
2. The penalty equals rest_lambda · I_rest² exactly.
3. v_rest is honoured (different voltage -> different penalty).
4. The torch penalty is differentiable.
"""

import numpy as np
import pytest

from core.msm_builder import make_11state_preset
from core.optimizer import CostFunction
from core.simulator import ProtocolSimulator, stationary_distribution

MSM = make_11state_preset()
PARAMS = MSM.free_initial_guess
NO_DATA = {"activation": None, "inactivation": None, "cs_inactivation": None, "recovery": None}
INACT = [i for i, s in enumerate(MSM.states) if s.state_type == "inactivated"]


def _i_rest(v: float) -> float:
    Q = ProtocolSimulator(PARAMS, msm_def=MSM)._build_Q(v)
    return float(stationary_distribution(Q)[INACT].sum())


class TestSciPyRestPenalty:
    def _cost(self, lam, v_rest=-90.0):
        return CostFunction(NO_DATA, msm_def=MSM, rest_lambda=lam, v_rest=v_rest)

    def test_zero_lambda_no_penalty(self):
        assert self._cost(0.0).total_cost(PARAMS) == 0.0

    def test_penalty_equals_formula(self):
        expected = 0.1 * _i_rest(-90.0) ** 2
        assert expected > 0
        assert self._cost(0.1).total_cost(PARAMS) == pytest.approx(expected, rel=1e-12)

    def test_v_rest_is_used(self):
        a = self._cost(1.0, -90.0).total_cost(PARAMS)
        b = self._cost(1.0, -60.0).total_cost(PARAMS)
        assert b == pytest.approx(_i_rest(-60.0) ** 2, rel=1e-12)
        assert a != pytest.approx(b)


torch = pytest.importorskip("torch")


class TestTorchRestPenalty:
    from core.torch_optimizer import TorchCostFunction

    def _cost(self, lam, v_rest=-90.0):
        return self.TorchCostFunction(
            NO_DATA, msm_def=MSM, rest_lambda=lam, v_rest=v_rest,
            device=torch.device("cpu"), dtype=torch.float64,
        )

    def test_zero_lambda_no_penalty(self):
        assert self._cost(0.0).total_cost_numpy(PARAMS) == 0.0

    def test_matches_scipy(self):
        for v in (-90.0, -60.0):
            t = self._cost(0.1, v).total_cost_numpy(PARAMS)
            s = CostFunction(NO_DATA, msm_def=MSM, rest_lambda=0.1, v_rest=v).total_cost(PARAMS)
            assert t == pytest.approx(s, rel=1e-10)

    def test_differentiable(self):
        p = torch.tensor(PARAMS, dtype=torch.float64, requires_grad=True)
        self._cost(1.0)(p).backward()
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
