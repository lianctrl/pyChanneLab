"""
Unit tests — L2 regularization in cost functions.

Checks that:
1. reg_lambda=0 leaves the cost unchanged vs. the unregularized path.
2. reg_lambda>0 strictly increases the cost.
3. The penalty equals lambda * sum(theta^2) exactly.
4. The same semantics hold for the scipy CostFunction and
   (when torch is available) TorchCostFunction.
"""

import numpy as np
import pytest

from core.optimizer import CostFunction
from core.msm_builder import MSMDefinition, ParamSpec, StateSpec, TransitionSpec


# ── minimal 2-state MSM fixture ───────────────────────────────────────────────


def _two_state_msm() -> MSMDefinition:
    """C ↔ O with constant (voltage-independent) rates k_CO and k_OC."""
    states = [
        StateSpec(name="C", state_type="closed"),
        StateSpec(name="O", state_type="open"),
    ]
    params = [
        ParamSpec(name="k_CO", initial_value=4.0, lower_bound=0.1, upper_bound=20.0),
        ParamSpec(name="k_OC", initial_value=1.0, lower_bound=0.1, upper_bound=20.0),
    ]
    transitions = [
        TransitionSpec(from_state="C", to_state="O", rate_expr="k_CO"),
        TransitionSpec(from_state="O", to_state="C", rate_expr="k_OC"),
    ]
    return MSMDefinition(states=states, parameters=params, transitions=transitions)


def _minimal_exp_data(msm: MSMDefinition) -> dict:
    """Tiny synthetic activation dataset (3 voltage points)."""
    from core.simulator import ProtocolSimulator
    from core.config import ActivationConfig

    params = np.array([p.initial_value for p in msm.parameters])
    act_cfg = ActivationConfig(v_min=-60.0, v_max=0.0, v_step=30.0)
    sim = ProtocolSimulator(params, msm_def=msm, act_cfg=act_cfg)
    voltages = sim.act_proto.get_test_voltages()
    y = sim.run_activation(voltages)
    return {
        "activation": (voltages, y, None),
        "inactivation": None,
        "cs_inactivation": None,
        "recovery": None,
    }


# ── scipy CostFunction tests ──────────────────────────────────────────────────


class TestSciPyCostFunction:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.msm = _two_state_msm()
        self.exp = _minimal_exp_data(self.msm)
        self.params = np.array([p.initial_value for p in self.msm.parameters])

    def _cost(self, reg_lambda: float) -> CostFunction:
        return CostFunction(
            self.exp,
            msm_def=self.msm,
            reg_lambda=reg_lambda,
        )

    def test_zero_lambda_no_change(self):
        """reg_lambda=0 must give the same cost as the default (no regularization)."""
        cf_default = CostFunction(self.exp, msm_def=self.msm)
        cf_zero = self._cost(0.0)
        assert cf_default.total_cost(self.params) == pytest.approx(
            cf_zero.total_cost(self.params), rel=1e-12
        )

    def test_positive_lambda_increases_cost(self):
        """reg_lambda>0 must give a strictly higher cost than reg_lambda=0."""
        base = self._cost(0.0).total_cost(self.params)
        regularized = self._cost(1e-3).total_cost(self.params)
        assert regularized > base

    def test_penalty_equals_lambda_sum_sq(self):
        """The extra cost must equal exactly lambda * sum(theta^2)."""
        lam = 0.05
        base = self._cost(0.0).total_cost(self.params)
        reg = self._cost(lam).total_cost(self.params)
        expected_penalty = lam * float(np.sum(self.params ** 2))
        assert (reg - base) == pytest.approx(expected_penalty, rel=1e-9)

    def test_penalty_scales_with_lambda(self):
        """Doubling lambda must double the added penalty."""
        base = self._cost(0.0).total_cost(self.params)
        c1 = self._cost(1e-4).total_cost(self.params) - base
        c2 = self._cost(2e-4).total_cost(self.params) - base
        assert c2 == pytest.approx(2.0 * c1, rel=1e-9)

    def test_zero_params_zero_penalty(self):
        """Parameters of all zeros must add zero regularization penalty."""
        zero_params = np.zeros_like(self.params)
        # Replace with zero params but still need a valid MSM — just verify
        # the penalty contribution is 0 regardless of the data term.
        base = self._cost(0.0).total_cost(zero_params)
        reg = self._cost(1.0).total_cost(zero_params)
        assert reg == pytest.approx(base, rel=1e-12)


# ── torch TorchCostFunction tests (skipped if torch unavailable) ──────────────

torch = pytest.importorskip("torch", reason="torch not installed")


class TestTorchCostFunction:
    @pytest.fixture(autouse=True)
    def setup(self):
        from core.torch_optimizer import TorchCostFunction
        from core.torch_simulator import get_device, preferred_dtype

        self.msm = _two_state_msm()
        self.exp = _minimal_exp_data(self.msm)
        self.params_np = np.array([p.initial_value for p in self.msm.parameters])
        self.device = get_device()
        self.dtype = preferred_dtype(self.device)
        self.TorchCostFunction = TorchCostFunction

    def _cost(self, reg_lambda: float):
        return self.TorchCostFunction(
            self.exp,
            msm_def=self.msm,
            device=self.device,
            dtype=self.dtype,
            reg_lambda=reg_lambda,
        )

    def _eval(self, cf, params_np: np.ndarray) -> float:
        p = torch.tensor(params_np, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            return float(cf(p))

    def test_zero_lambda_no_change(self):
        cf_default = self.TorchCostFunction(
            self.exp, msm_def=self.msm, device=self.device, dtype=self.dtype
        )
        cf_zero = self._cost(0.0)
        assert self._eval(cf_default, self.params_np) == pytest.approx(
            self._eval(cf_zero, self.params_np), rel=1e-6
        )

    def test_positive_lambda_increases_cost(self):
        base = self._eval(self._cost(0.0), self.params_np)
        regularized = self._eval(self._cost(1e-3), self.params_np)
        assert regularized > base

    def test_penalty_equals_lambda_sum_sq(self):
        lam = 0.05
        base = self._eval(self._cost(0.0), self.params_np)
        reg = self._eval(self._cost(lam), self.params_np)
        expected_penalty = lam * float(np.sum(self.params_np ** 2))
        assert (reg - base) == pytest.approx(expected_penalty, rel=1e-5)

    def test_penalty_scales_with_lambda(self):
        base = self._eval(self._cost(0.0), self.params_np)
        c1 = self._eval(self._cost(1e-4), self.params_np) - base
        c2 = self._eval(self._cost(2e-4), self.params_np) - base
        assert c2 == pytest.approx(2.0 * c1, rel=1e-5)

    def test_gradient_flows_through_penalty(self):
        """Regularization must be differentiable — gradient must be nonzero."""
        cf = self._cost(1.0)
        p = torch.tensor(
            self.params_np, dtype=self.dtype, device=self.device, requires_grad=True
        )
        loss = cf(p)
        loss.backward()
        assert p.grad is not None
        assert p.grad.abs().sum().item() > 0.0
