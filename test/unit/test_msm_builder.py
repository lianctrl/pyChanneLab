"""
Unit tests — MSM construction and Q-matrix structure.

All tests use a minimal 2-state model  C ↔ O  with constant rates so that
the expected Q matrix and equilibrium can be computed by hand.

Model:
    states     : C (closed), O (open)
    transitions: C → O  rate = k_CO
                 O → C  rate = k_OC
    Q matrix   :  [[-k_CO,  k_OC ],
                   [ k_CO, -k_OC ]]

With k_CO = 4.0 and k_OC = 1.0:
    equilibrium  p_O = k_CO / (k_CO + k_OC) = 0.8
"""

import numpy as np
import pytest

from core.msm_builder import (
    DynamicModel,
    MSMDefinition,
    ParamSpec,
    StateSpec,
    TransitionSpec,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def make_2state_msm(k_CO: float = 4.0, k_OC: float = 1.0) -> MSMDefinition:
    """Return a minimal 2-state MSMDefinition with constant (voltage-independent) rates."""
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
            ParamSpec(
                name="k_CO", initial_value=k_CO, lower_bound=0.01, upper_bound=100.0
            ),
            ParamSpec(
                name="k_OC", initial_value=k_OC, lower_bound=0.01, upper_bound=100.0
            ),
        ],
    )


# ── MSMDefinition construction ────────────────────────────────────────────────


class TestMSMDefinition:

    def test_state_names(self):
        msm = make_2state_msm()
        assert msm.state_names == ["C", "O"]

    def test_n_states(self):
        msm = make_2state_msm()
        assert msm.n_states == 2

    def test_open_state_indices(self):
        msm = make_2state_msm()
        assert msm.open_state_indices == [1]  # "O" is index 1

    def test_param_names(self):
        msm = make_2state_msm()
        assert msm.param_names == ["k_CO", "k_OC"]

    def test_initial_guess(self):
        msm = make_2state_msm(k_CO=4.0, k_OC=1.0)
        np.testing.assert_allclose(msm.initial_guess, [4.0, 1.0])

    def test_bounds(self):
        msm = make_2state_msm()
        assert list(msm.bounds) == [(0.01, 100.0), (0.01, 100.0)]

    def test_validate_passes(self):
        msm = make_2state_msm()
        errors = msm.validate()
        assert errors == [], f"Unexpected validation errors: {errors}"

    def test_validate_missing_state_in_transition(self):
        msm = MSMDefinition(
            states=[StateSpec("C", "closed"), StateSpec("O", "open")],
            transitions=[
                TransitionSpec("C", "X", "k_CO"),  # 'X' does not exist
                TransitionSpec("O", "C", "k_OC"),
            ],
            parameters=[
                ParamSpec("k_CO", 4.0, 0.01, 100.0),
                ParamSpec("k_OC", 1.0, 0.01, 100.0),
            ],
        )
        errors = msm.validate()
        assert len(errors) > 0

    def test_validate_no_open_state(self):
        msm = MSMDefinition(
            states=[StateSpec("A", "closed"), StateSpec("B", "closed")],
            transitions=[
                TransitionSpec("A", "B", "k_AB"),
                TransitionSpec("B", "A", "k_BA"),
            ],
            parameters=[
                ParamSpec("k_AB", 1.0, 0.0, 10.0),
                ParamSpec("k_BA", 1.0, 0.0, 10.0),
            ],
        )
        errors = msm.validate()
        assert any("open" in e.lower() for e in errors)

    def test_default_initial_conditions_sums_to_one(self):
        msm = make_2state_msm()
        ic = msm.default_initial_conditions
        assert abs(ic.sum() - 1.0) < 1e-9

    def test_default_initial_conditions_closed_only(self):
        """Default IC should put all probability on closed states."""
        msm = make_2state_msm()
        ic = msm.default_initial_conditions
        assert ic[0] == pytest.approx(1.0)  # C
        assert ic[1] == pytest.approx(0.0)  # O


# ── Frozen-parameter handling ─────────────────────────────────────────────────


class TestFrozenParams:

    def test_frozen_excluded_from_free_bounds(self):
        msm = MSMDefinition(
            states=[StateSpec("C", "closed"), StateSpec("O", "open")],
            transitions=[
                TransitionSpec("C", "O", "k_CO"),
                TransitionSpec("O", "C", "k_OC"),
            ],
            parameters=[
                ParamSpec("k_CO", 4.0, 0.01, 100.0, frozen=True),
                ParamSpec("k_OC", 1.0, 0.01, 100.0, frozen=False),
            ],
        )
        free_bounds = list(msm.free_bounds)
        assert len(free_bounds) == 1  # only k_OC is free

    def test_expand_params_inserts_frozen_value(self):
        msm = MSMDefinition(
            states=[StateSpec("C", "closed"), StateSpec("O", "open")],
            transitions=[
                TransitionSpec("C", "O", "k_CO"),
                TransitionSpec("O", "C", "k_OC"),
            ],
            parameters=[
                ParamSpec("k_CO", 4.0, 0.01, 100.0, frozen=True),
                ParamSpec("k_OC", 1.0, 0.01, 100.0, frozen=False),
            ],
        )
        free = np.array([2.5])  # only k_OC
        full = msm.expand_params(free)
        assert full[0] == pytest.approx(4.0)  # frozen k_CO
        assert full[1] == pytest.approx(2.5)  # free k_OC


# ── Q-matrix structure ────────────────────────────────────────────────────────


class TestQMatrix:

    def _build_Q(
        self, k_CO: float = 4.0, k_OC: float = 1.0, V: float = 0.0
    ) -> np.ndarray:
        msm = make_2state_msm(k_CO, k_OC)
        model = DynamicModel(msm, np.array([k_CO, k_OC]))
        return model.build_Q(V)

    def test_Q_shape(self):
        Q = self._build_Q()
        assert Q.shape == (2, 2)

    def test_Q_off_diagonal_nonnegative(self):
        """All off-diagonal elements must be ≥ 0 (rates are non-negative)."""
        Q = self._build_Q()
        n = Q.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert Q[i, j] >= 0.0, f"Q[{i},{j}] = {Q[i,j]} is negative"

    def test_Q_columns_sum_to_zero(self):
        """Each column of a generator matrix sums to 0 (probability is conserved)."""
        Q = self._build_Q()
        np.testing.assert_allclose(Q.sum(axis=0), 0.0, atol=1e-12)

    def test_Q_values(self):
        k_CO, k_OC = 4.0, 1.0
        Q = self._build_Q(k_CO, k_OC)
        # Column convention: Q[j, i] = rate i → j
        # C→O : Q[1, 0] = k_CO
        # O→C : Q[0, 1] = k_OC
        # diagonal: Q[0,0] = -k_CO, Q[1,1] = -k_OC
        assert Q[1, 0] == pytest.approx(k_CO)
        assert Q[0, 1] == pytest.approx(k_OC)
        assert Q[0, 0] == pytest.approx(-k_CO)
        assert Q[1, 1] == pytest.approx(-k_OC)

    def test_Q_constant_rate_voltage_independent(self):
        """Rates with no V-dependence should be the same at any voltage."""
        Q0 = self._build_Q(V=0.0)
        Q50 = self._build_Q(V=50.0)
        np.testing.assert_allclose(Q0, Q50, atol=1e-12)


# ── Serialisation ─────────────────────────────────────────────────────────────


class TestSerialisation:

    def test_round_trip_json(self):
        msm = make_2state_msm(k_CO=3.0, k_OC=2.0)
        restored = MSMDefinition.from_json(msm.to_json())
        assert restored.state_names == msm.state_names
        assert restored.param_names == msm.param_names
        np.testing.assert_allclose(restored.initial_guess, msm.initial_guess)

    def test_round_trip_dict(self):
        msm = make_2state_msm()
        restored = MSMDefinition.from_dict(msm.to_dict())
        assert len(restored.states) == len(msm.states)
        assert len(restored.transitions) == len(msm.transitions)
        assert len(restored.parameters) == len(msm.parameters)
