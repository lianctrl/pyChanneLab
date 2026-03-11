"""
User-defined Markov State Model builder.

The user specifies:
  - States     — name + type (closed / inactivated / open)
  - Transitions — from-state, to-state, rate expression
  - Parameters  — name, initial value, bounds

The DynamicModel class then assembles the ODEs at run-time from those
transition rate expressions, so no hardcoded equations are needed.

Expression syntax
-----------------
Rate expressions are Python snippets evaluated with these names in scope:
  V           — current voltage (mV)
  EXP_FACTOR  — (e / k_B T) * 1e-3  (physical constant, ≈0.0388 mV⁻¹)
  exp, log, sqrt, abs, pi   — from math
  <param_name> — value of any parameter defined in the parameters list

Example expressions:
  "4 * alpha_0 * exp(alpha_1 * V * EXP_FACTOR)"      # voltage-activated
  "k_CI * f**4"                                        # constant × coupling
  "k_IC / f**4"
"""

import math
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from core.config import EXP_FACTOR


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StateSpec:
    name: str
    state_type: str   # "closed" | "inactivated" | "open"


@dataclass
class TransitionSpec:
    from_state: str
    to_state: str
    rate_expr: str    # evaluated Python expression


@dataclass
class ParamSpec:
    name: str
    initial_value: float
    lower_bound: float
    upper_bound: float


@dataclass
class MSMDefinition:
    """Complete, self-contained description of a Markov State Model."""
    states:      list   = field(default_factory=list)   # list[StateSpec]
    transitions: list   = field(default_factory=list)   # list[TransitionSpec]
    parameters:  list   = field(default_factory=list)   # list[ParamSpec]

    # ---- convenience ---------------------------------------------------

    @property
    def state_names(self) -> list:
        return [s.name for s in self.states]

    @property
    def param_names(self) -> list:
        return [p.name for p in self.parameters]

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def open_state_indices(self) -> list:
        return [i for i, s in enumerate(self.states) if s.state_type == "open"]

    @property
    def initial_guess(self) -> np.ndarray:
        return np.array([p.initial_value for p in self.parameters], dtype=float)

    @property
    def bounds(self) -> tuple:
        return tuple((p.lower_bound, p.upper_bound) for p in self.parameters)

    @property
    def default_initial_conditions(self) -> np.ndarray:
        """Uniform distribution over closed states; 0 everywhere else."""
        ic = np.zeros(self.n_states)
        closed = [i for i, s in enumerate(self.states) if s.state_type == "closed"]
        if closed:
            ic[closed] = 1.0 / len(closed)
        else:
            ic[0] = 1.0
        return ic

    # ---- validation ----------------------------------------------------

    def validate(self) -> list:
        """Return a list of human-readable error strings (empty = OK)."""
        errors = []
        names = self.state_names

        if len(names) != len(set(names)):
            errors.append("Duplicate state names detected.")

        if not self.open_state_indices:
            errors.append("Model must have at least one 'open' state.")

        if len(self.parameters) == 0:
            errors.append("Model must define at least one parameter.")

        edge_set = {(t.from_state, t.to_state) for t in self.transitions}
        for tr in self.transitions:
            if tr.from_state not in names:
                errors.append(f"Transition references unknown state '{tr.from_state}'.")
            if tr.to_state not in names:
                errors.append(f"Transition references unknown state '{tr.to_state}'.")

        # Evaluate every expression with dummy V=0 to catch syntax errors
        test_ns = _build_namespace(V=0.0, param_names=self.param_names,
                                   param_values=self.initial_guess)
        for tr in self.transitions:
            ok, msg = _eval_expr(tr.rate_expr, test_ns)
            if not ok:
                errors.append(
                    f"Bad expression [{tr.from_state}→{tr.to_state}]: {msg}"
                )

        return errors

    # ---- serialisation -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "states":      [{"name": s.name, "state_type": s.state_type}
                            for s in self.states],
            "transitions": [{"from_state": t.from_state, "to_state": t.to_state,
                             "rate_expr": t.rate_expr}
                            for t in self.transitions],
            "parameters":  [{"name": p.name, "initial_value": p.initial_value,
                             "lower_bound": p.lower_bound, "upper_bound": p.upper_bound}
                            for p in self.parameters],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "MSMDefinition":
        return cls(
            states      = [StateSpec(**s)      for s in d.get("states", [])],
            transitions = [TransitionSpec(**t) for t in d.get("transitions", [])],
            parameters  = [ParamSpec(**p)      for p in d.get("parameters", [])],
        )

    @classmethod
    def from_json(cls, text: str) -> "MSMDefinition":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------

def _build_namespace(V: float, param_names: list, param_values: np.ndarray) -> dict:
    ns = {
        "V":          V,
        "EXP_FACTOR": EXP_FACTOR,
        "exp":   math.exp,
        "log":   math.log,
        "sqrt":  math.sqrt,
        "abs":   abs,
        "pi":    math.pi,
        "__builtins__": {},
    }
    ns.update(dict(zip(param_names, param_values)))
    return ns


def _eval_expr(expr: str, namespace: dict):
    """Return (True, "") on success or (False, error_message)."""
    try:
        result = eval(expr, namespace)
        if not isinstance(result, (int, float)):
            return False, "expression must return a scalar number"
        if not np.isfinite(float(result)):
            return False, f"evaluated to non-finite value: {result}"
        return True, ""
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Dynamic ODE model built from an MSMDefinition
# ---------------------------------------------------------------------------

class DynamicModel:
    """
    Assembles ODE right-hand sides at run-time from an MSMDefinition.

    Same interface as IonChannelModel:
        equations(state, t, voltage_func) -> tuple of derivatives
    """

    def __init__(self, definition: MSMDefinition, parameters: np.ndarray):
        self.defn   = definition
        self.params = np.asarray(parameters, dtype=float)

        # Map state name → column index
        self._idx = {s: i for i, s in enumerate(definition.state_names)}

        # Pre-resolve transition indices for speed
        self._transitions = [
            (self._idx[tr.from_state], self._idx[tr.to_state], tr.rate_expr)
            for tr in definition.transitions
        ]

        # Static part of the evaluation namespace (updated each call with V)
        self._base_ns = _build_namespace(
            V=0.0,
            param_names=definition.param_names,
            param_values=self.params,
        )

    def equations(self, state, t: float, voltage_func: Callable) -> tuple:
        V  = voltage_func(t)
        ns = {**self._base_ns, "V": V}

        derivs = [0.0] * self.defn.n_states
        for (i, j, expr) in self._transitions:
            rate = eval(expr, ns)   # noqa: S307 — intentional, scientific tool
            flux = rate * state[i]
            derivs[i] -= flux
            derivs[j] += flux

        return tuple(derivs)


# ---------------------------------------------------------------------------
# Layout helper for the network diagram
# ---------------------------------------------------------------------------

def compute_layout(definition: MSMDefinition) -> dict:
    """
    Heuristic 2-D layout for the network diagram:
      closed states     → top row
      inactivated states → bottom row
      open states        → right column (vertically centred)
    Returns {state_name: (x, y)}.
    """
    by_type = {"closed": [], "inactivated": [], "open": []}
    for s in definition.states:
        bucket = by_type.get(s.state_type, by_type["open"])
        bucket.append(s.name)

    layout = {}
    spacing = 1.8

    for i, name in enumerate(by_type["closed"]):
        layout[name] = (i * spacing, 2.0)

    for i, name in enumerate(by_type["inactivated"]):
        layout[name] = (i * spacing, 0.0)

    right_x = max(
        len(by_type["closed"]) - 1,
        len(by_type["inactivated"]) - 1,
    ) * spacing + spacing

    for i, name in enumerate(by_type["open"]):
        y = 1.0 + i * spacing
        layout[name] = (right_x, y)

    return layout


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

def make_11state_preset() -> MSMDefinition:
    """
    The classic 11-state Kv channel model:
        C0 ↔ C1 ↔ C2 ↔ C3 ↔ C4 ↔ O
        |    |    |    |    |
        I0 ↔ I1 ↔ I2 ↔ I3 ↔ I4
    """
    states = [
        StateSpec("C0", "closed"), StateSpec("C1", "closed"),
        StateSpec("C2", "closed"), StateSpec("C3", "closed"),
        StateSpec("C4", "closed"),
        StateSpec("I0", "inactivated"), StateSpec("I1", "inactivated"),
        StateSpec("I2", "inactivated"), StateSpec("I3", "inactivated"),
        StateSpec("I4", "inactivated"),
        StateSpec("O",  "open"),
    ]

    # Helper expressions
    _alpha = "alpha_0 * exp(alpha_1 * V * EXP_FACTOR)"
    _beta  = "beta_0  * exp(-beta_1  * V * EXP_FACTOR)"

    transitions = [
        # ---- C-ladder (horizontal) ----
        TransitionSpec("C0", "C1", f"4 * {_alpha}"),
        TransitionSpec("C1", "C0", f"1 * {_beta}"),
        TransitionSpec("C1", "C2", f"3 * {_alpha}"),
        TransitionSpec("C2", "C1", f"2 * {_beta}"),
        TransitionSpec("C2", "C3", f"2 * {_alpha}"),
        TransitionSpec("C3", "C2", f"3 * {_beta}"),
        TransitionSpec("C3", "C4", f"1 * {_alpha}"),
        TransitionSpec("C4", "C3", f"4 * {_beta}"),

        # ---- I-ladder (horizontal, scaled by f) ----
        TransitionSpec("I0", "I1", f"4 * {_alpha} / f"),
        TransitionSpec("I1", "I0", f"1 * {_beta}  * f"),
        TransitionSpec("I1", "I2", f"3 * {_alpha} / f"),
        TransitionSpec("I2", "I1", f"2 * {_beta}  * f"),
        TransitionSpec("I2", "I3", f"2 * {_alpha} / f"),
        TransitionSpec("I3", "I2", f"3 * {_beta}  * f"),
        TransitionSpec("I3", "I4", f"1 * {_alpha} / f"),
        TransitionSpec("I4", "I3", f"4 * {_beta}  * f"),

        # ---- C ↔ I vertical transitions ----
        TransitionSpec("C0", "I0", "k_CI * f**4"),
        TransitionSpec("I0", "C0", "k_IC / f**4"),
        TransitionSpec("C1", "I1", "k_CI * f**3"),
        TransitionSpec("I1", "C1", "k_IC / f**3"),
        TransitionSpec("C2", "I2", "k_CI * f**2"),
        TransitionSpec("I2", "C2", "k_IC / f**2"),
        TransitionSpec("C3", "I3", "k_CI * f"),
        TransitionSpec("I3", "C3", "k_IC / f"),
        TransitionSpec("C4", "I4", "k_CI"),
        TransitionSpec("I4", "C4", "k_IC"),

        # ---- C4 ↔ O ----
        TransitionSpec("C4", "O", "k_CO_0 * exp( k_CO_1 * V * EXP_FACTOR)"),
        TransitionSpec("O",  "C4", "k_OC_0 * exp(-k_OC_1 * V * EXP_FACTOR)"),
    ]

    parameters = [
        ParamSpec("alpha_0", 204.0,  0.0, 2000.0),
        ParamSpec("alpha_1", 2.07,   0.0,    5.0),
        ParamSpec("beta_0",  21.2,   0.0,  100.0),
        ParamSpec("beta_1",  2.8e-3, 0.0,    5.0),
        ParamSpec("k_CO_0",  2.8,    0.0, 1000.0),
        ParamSpec("k_CO_1",  0.343,  0.0,    5.0),
        ParamSpec("k_OC_0",  37.5,   0.0, 1000.0),
        ParamSpec("k_OC_1",  1.0e-2, 0.0,    5.0),
        ParamSpec("k_CI",    94.5,   0.0, 2000.0),
        ParamSpec("k_IC",    0.24,   0.0,  100.0),
        ParamSpec("f",       0.44,   0.0,    1.0),
    ]

    return MSMDefinition(states=states, transitions=transitions, parameters=parameters)


PRESETS = {
    "11-state Kv channel (C0–C4, I0–I4, O)": make_11state_preset,
}
