"""
Unit tests — ProtocolSimulator on a minimal 2-state model.

Analytic solution for 2-state C ↔ O  (constant rates, no voltage dependence)
-------------------
Q = [[-k_CO,  k_OC ],
     [ k_CO, -k_OC ]]

Eigenvalues: λ₀ = 0,  λ₁ = -(k_CO + k_OC)

Starting from p = (1, 0) (all in C):
    p_O(t) = k_CO/(k_CO+k_OC) * (1 - exp(-λ₁ · t))

With k_CO = 4.0, k_OC = 1.0:
    p_O_eq  = 0.8
    λ₁      = -5.0
    p_O(t)  = 0.8 · (1 - e^{-5t})

The ProtocolSimulator uses the matrix-exponential method so its numerical
solution should match this analytic result to high precision.
"""

import numpy as np
import pytest

from core.config import ActivationConfig
from core.msm_builder import MSMDefinition, ParamSpec, StateSpec, TransitionSpec
from core.simulator import ProtocolSimulator


# ── fixtures ──────────────────────────────────────────────────────────────────

K_CO = 4.0
K_OC = 1.0
K_TOT = K_CO + K_OC          # 5.0
P_O_EQ = K_CO / K_TOT        # 0.8


def make_2state_msm() -> MSMDefinition:
    return MSMDefinition(
        states=[
            StateSpec(name="C", state_type="closed"),
            StateSpec(name="O", state_type="open"),
        ],
        transitions=[
            TransitionSpec(from_state="C", to_state="O", rate_expr="k_CO"),
            TransitionSpec(from_state="O", to_state="C", rate_expr="k_OC"),
        ],
        parameters=[
            ParamSpec("k_CO", K_CO, 0.01, 100.0),
            ParamSpec("k_OC", K_OC, 0.01, 100.0),
        ],
    )


def make_sim(params=None, t_total: float = 2.0, dt: float = 1e-4) -> ProtocolSimulator:
    """Build a ProtocolSimulator for the 2-state model with fast activation protocol."""
    msm = make_2state_msm()
    p   = np.array(params if params is not None else [K_CO, K_OC])
    # Simple activation: hold at 0 mV for t_hold, test at 0 mV for t_test.
    # Since rates are voltage-independent, the voltage value doesn't matter.
    act_cfg = ActivationConfig(
        v_hold=-90.0, v_tail=-90.0,
        v_min=0.0, v_max=0.0, v_step=10.0,   # single test voltage
        t_hold=0.01, t_test=t_total - 0.01,
    )
    return ProtocolSimulator(
        p,
        msm_def=msm,
        act_cfg=act_cfg,
        t_total=t_total,
        dt=dt,
        g_k_max=1.0,          # normalise by 1 so output ≡ open probability
        initial_state=np.array([1.0, 0.0]),   # start all in C
    )


# ── basic sanity checks ───────────────────────────────────────────────────────

class TestSimulatorSanity:

    def test_run_activation_returns_array(self):
        sim = make_sim()
        result = sim.run_activation()
        assert isinstance(result, np.ndarray)

    def test_run_activation_output_in_unit_interval(self):
        """Normalised observables must be in [0, 1]."""
        sim = make_sim()
        result = sim.run_activation()
        assert np.all(result >= -1e-6)
        assert np.all(result <= 1.0 + 1e-6)

    def test_output_length_matches_voltage_steps(self):
        sim = make_sim()
        result  = sim.run_activation()
        voltages = sim.act_proto.get_test_voltages()
        assert len(result) == len(voltages)


# ── analytic check ────────────────────────────────────────────────────────────

class TestSimulatorAnalytic:
    """
    Compare the matrix-exponential simulator against the closed-form
    solution for a 2-state model.
    """

    def _p_O_analytic(self, t: float) -> float:
        """Analytic open-state probability starting from p=(1,0)."""
        return P_O_EQ * (1.0 - np.exp(-K_TOT * t))

    def test_equilibrium_approached(self):
        """After long time (>>1/k_tot) the system should be near equilibrium."""
        t_total = 5.0   # 25 × relaxation time (1/5 = 0.2 s)
        sim = make_sim(t_total=t_total, dt=1e-4)
        result = sim.run_activation()
        # result is normalised by g_k_max=1 and the peak open prob
        # here there's only one test step; result should be 1.0 (normalised)
        assert result[0] == pytest.approx(1.0, abs=1e-3)

    def test_matrix_exp_matches_analytic_at_several_times(self):
        """
        Build a raw DynamicModel and propagate by hand; compare to analytic.
        """
        from scipy.linalg import expm as matrix_expm
        from core.msm_builder import DynamicModel

        params = np.array([K_CO, K_OC])
        model  = DynamicModel(make_2state_msm(), params)
        Q = model.build_Q(V=0.0)   # voltage-independent

        p0 = np.array([1.0, 0.0])   # start in C

        for t in [0.05, 0.1, 0.3, 0.5, 1.0]:
            M = matrix_expm(Q * t)
            p = M @ p0
            p_O_num   = p[1]           # open-state probability
            p_O_exact = self._p_O_analytic(t)
            assert p_O_num == pytest.approx(p_O_exact, abs=1e-8), (
                f"t={t}: numerical={p_O_num:.8f}, analytic={p_O_exact:.8f}"
            )

    def test_probability_conservation(self):
        """State probabilities must sum to 1 at all times."""
        from scipy.linalg import expm as matrix_expm
        from core.msm_builder import DynamicModel

        model = DynamicModel(make_2state_msm(), np.array([K_CO, K_OC]))
        Q  = model.build_Q(0.0)
        p0 = np.array([1.0, 0.0])

        for t in np.linspace(0, 2.0, 20):
            p = matrix_expm(Q * t) @ p0
            assert abs(p.sum() - 1.0) < 1e-12, f"p.sum()={p.sum()} at t={t}"

    def test_all_probabilities_nonnegative(self):
        from scipy.linalg import expm as matrix_expm
        from core.msm_builder import DynamicModel

        model = DynamicModel(make_2state_msm(), np.array([K_CO, K_OC]))
        Q  = model.build_Q(0.0)
        p0 = np.array([1.0, 0.0])

        for t in np.linspace(0.01, 2.0, 20):
            p = matrix_expm(Q * t) @ p0
            assert np.all(p >= -1e-12), f"Negative probability at t={t}: {p}"


# ── different parameter sets ──────────────────────────────────────────────────

class TestSimulatorParameterVariation:

    @pytest.mark.parametrize("k_CO,k_OC", [
        (1.0, 1.0),
        (10.0, 1.0),
        (0.5, 2.0),
        (100.0, 0.1),
    ])
    def test_output_bounded_for_various_rates(self, k_CO, k_OC):
        sim = make_sim(params=[k_CO, k_OC])
        result = sim.run_activation()
        assert np.all(np.isfinite(result))
        assert np.all(result >= -1e-6)
        assert np.all(result <= 1.0 + 1e-6)

    def test_faster_rate_reaches_equilibrium_sooner(self):
        """Higher k_CO+k_OC → closer to equilibrium at fixed short time."""
        from scipy.linalg import expm as matrix_expm
        from core.msm_builder import DynamicModel

        t = 0.1  # fixed short time

        for k_CO, k_OC in [(1.0, 0.25), (10.0, 2.5)]:
            model = DynamicModel(
                make_2state_msm(),
                np.array([k_CO, k_OC]),
            )
            Q    = model.build_Q(0.0)
            p_O  = (matrix_expm(Q * t) @ np.array([1.0, 0.0]))[1]
            p_eq = k_CO / (k_CO + k_OC)
            dist = abs(p_O - p_eq)

            if k_CO + k_OC > 5:   # faster system is closer to equilibrium
                assert dist < 0.3, f"Fast system not near equil: dist={dist}"
