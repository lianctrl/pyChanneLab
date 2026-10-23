"""
Unit tests — scale-invariant, prior-anchored L2 regularization.

The penalty is  λ · Σ_i d_i²  with  d_i = log(θ_i) − log(θ_prior,i)  for
log-scaled params and  (θ_i − θ_prior,i)/(ub_i − lb_i)  for linear params, so it
is 0 at the prior (initial guess) and acts as a soft pull toward it.

Checks:
1. reg_lambda=0 leaves the cost unchanged vs. the unregularized path.
2. The penalty is exactly 0 when evaluated at the prior, for any λ.
3. Off the prior, reg_lambda>0 strictly increases the cost.
4. The extra cost equals λ · prior_penalty(...) and scales linearly with λ.
5. Same semantics for the scipy CostFunction and (if torch) TorchCostFunction,
   with the torch penalty differentiable off the prior.
"""

import numpy as np
import pytest

from core.optimizer import CostFunction
from core.msm_builder import MSMDefinition, ParamSpec, StateSpec, TransitionSpec
from core.param_transform import prior_penalty_np


def _reg_args(msm: MSMDefinition):
    """(prior, lb, ub, log_mask) for the free params of *msm*."""
    fb = msm.free_bounds
    return (
        msm.free_initial_guess,
        [b[0] for b in fb],
        [b[1] for b in fb],
        msm.free_log_mask,
    )


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

    @property
    def _pert(self):
        """Params off the prior: double each rate (log fold-change ln2 each)."""
        return self.params * 2.0

    def test_zero_lambda_no_change(self):
        """reg_lambda=0 must give the same cost as the default (no regularization)."""
        cf_default = CostFunction(self.exp, msm_def=self.msm)
        cf_zero = self._cost(0.0)
        assert cf_default.total_cost(self._pert) == pytest.approx(
            cf_zero.total_cost(self._pert), rel=1e-12
        )

    def test_penalty_zero_at_prior(self):
        """At the prior (initial guess) the penalty is 0 for any λ."""
        base = self._cost(0.0).total_cost(self.params)
        reg = self._cost(1.0).total_cost(self.params)
        assert reg == pytest.approx(base, rel=1e-12)

    def test_positive_lambda_increases_cost_off_prior(self):
        """Off the prior, reg_lambda>0 must give a strictly higher cost."""
        base = self._cost(0.0).total_cost(self._pert)
        regularized = self._cost(1e-2).total_cost(self._pert)
        assert regularized > base

    def test_penalty_equals_formula(self):
        """The extra cost must equal λ · prior_penalty(...)."""
        lam = 0.05
        base = self._cost(0.0).total_cost(self._pert)
        reg = self._cost(lam).total_cost(self._pert)
        expected = lam * prior_penalty_np(self._pert, *_reg_args(self.msm))
        assert (reg - base) == pytest.approx(expected, rel=1e-9)
        # closed form: both params doubled → 2·(ln2)² for two log params
        assert expected == pytest.approx(lam * 2.0 * np.log(2.0) ** 2, rel=1e-9)

    def test_penalty_scales_with_lambda(self):
        """Doubling lambda must double the added penalty."""
        base = self._cost(0.0).total_cost(self._pert)
        c1 = self._cost(1e-4).total_cost(self._pert) - base
        c2 = self._cost(2e-4).total_cost(self._pert) - base
        assert c2 == pytest.approx(2.0 * c1, rel=1e-9)


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

    @property
    def _pert(self):
        return self.params_np * 2.0

    def test_zero_lambda_no_change(self):
        cf_default = self.TorchCostFunction(
            self.exp, msm_def=self.msm, device=self.device, dtype=self.dtype
        )
        cf_zero = self._cost(0.0)
        assert self._eval(cf_default, self._pert) == pytest.approx(
            self._eval(cf_zero, self._pert), rel=1e-6
        )

    def test_penalty_zero_at_prior(self):
        base = self._eval(self._cost(0.0), self.params_np)
        reg = self._eval(self._cost(1.0), self.params_np)
        assert reg == pytest.approx(base, rel=1e-6)

    def test_positive_lambda_increases_cost_off_prior(self):
        base = self._eval(self._cost(0.0), self._pert)
        regularized = self._eval(self._cost(1e-2), self._pert)
        assert regularized > base

    def test_penalty_equals_formula(self):
        lam = 0.05
        base = self._eval(self._cost(0.0), self._pert)
        reg = self._eval(self._cost(lam), self._pert)
        expected = lam * prior_penalty_np(self._pert, *_reg_args(self.msm))
        assert (reg - base) == pytest.approx(expected, rel=1e-5)
        assert expected == pytest.approx(lam * 2.0 * np.log(2.0) ** 2, rel=1e-6)

    def test_penalty_scales_with_lambda(self):
        base = self._eval(self._cost(0.0), self._pert)
        c1 = self._eval(self._cost(1e-4), self._pert) - base
        c2 = self._eval(self._cost(2e-4), self._pert) - base
        assert c2 == pytest.approx(2.0 * c1, rel=1e-5)

    def test_gradient_flows_through_penalty(self):
        """Regularization must be differentiable — gradient nonzero off the prior."""
        cf = self._cost(1.0)
        p = torch.tensor(
            self._pert, dtype=self.dtype, device=self.device, requires_grad=True
        )
        loss = cf(p)
        loss.backward()
        assert p.grad is not None
        assert p.grad.abs().sum().item() > 0.0
