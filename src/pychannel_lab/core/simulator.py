"""
Matrix-exponential simulator for all four voltage-clamp protocols.

P(t + dt) = expm(Q(V) · dt) @ P(t)

Exact for piecewise-constant voltage within each dt step.
Accepts either:
  - a plain parameter array (uses the hardcoded IonChannelModel for speed)
  - an MSMDefinition  (builds a DynamicModel at run-time, supports any topology)
"""

import numpy as np
from scipy.linalg import expm as _matrix_expm
from typing import Callable, Tuple

from core.config import (
    INITIAL_CONDITIONS,
    TIME_PARAMS,
    G_K_MAX,
    ActivationConfig,
    InactivationConfig,
    CSInactivationConfig,
    RecoveryConfig,
)
from core.model import IonChannelModel, build_Q as _build_Q_11state
from core.protocols import (
    ActivationProtocol,
    InactivationProtocol,
    CSInactivationProtocol,
    RecoveryProtocol,
)


class ProtocolSimulator:
    """
    Run voltage-clamp protocols and extract normalised observables.

    Parameters
    ----------
    parameters : array-like
        Kinetic parameters (length must match IonChannelModel or msm_def).
    msm_def : MSMDefinition, optional
        If provided, a DynamicModel is built from this definition instead of
        the default 11-state IonChannelModel.
    act_cfg, inact_cfg, csi_cfg, rec_cfg : protocol config dataclasses
        Override default protocol timing.  Pass None for defaults.
    t_total, dt : float
        Simulation time grid.
    g_k_max : float
        Maximum channel conductance (nS).
    initial_state : array-like, optional
        Initial probability distribution.  Defaults to INITIAL_CONDITIONS
        (11-state) or msm_def.default_initial_conditions.
    """

    def __init__(
        self,
        parameters,
        msm_def=None,
        act_cfg: ActivationConfig = None,
        inact_cfg: InactivationConfig = None,
        csi_cfg: CSInactivationConfig = None,
        rec_cfg: RecoveryConfig = None,
        t_total: float = TIME_PARAMS["tend"],
        dt: float = TIME_PARAMS["dt"],
        g_k_max: float = G_K_MAX,
        initial_state=None,
    ):
        self.params = np.asarray(parameters, dtype=float)
        self.t_total = t_total
        self.dt = dt
        self.g_k_max = g_k_max

        if msm_def is not None:
            from core.msm_builder import DynamicModel

            self.model = DynamicModel(msm_def, self.params)
            self._open_idx = msm_def.open_state_indices  # list of ints
            self.s0 = (
                msm_def.default_initial_conditions
                if initial_state is None
                else np.asarray(initial_state, dtype=float)
            )
        else:
            self.model = IonChannelModel(self.params)
            self._open_idx = [10]  # default 11-state model: O is index 10
            self.s0 = (
                INITIAL_CONDITIONS
                if initial_state is None
                else np.asarray(initial_state, dtype=float)
            )

        self.act_proto = ActivationProtocol(act_cfg)
        self.inact_proto = InactivationProtocol(inact_cfg)
        self.csi_proto = CSInactivationProtocol(csi_cfg)
        self.rec_proto = RecoveryProtocol(rec_cfg)

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _time_array(self) -> np.ndarray:
        return np.arange(0.0, self.t_total + self.dt, self.dt)

    def _build_Q(self, V: float) -> np.ndarray:
        """Return the generator matrix Q at voltage V (dispatches to model type)."""
        if isinstance(self.model, IonChannelModel):
            return _build_Q_11state(self.params, V)
        return self.model.build_Q(V)

    def _simulate(self, voltage_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """Step-wise matrix-exponential propagation: P(t+dt) = expm(Q(V)·dt) @ P(t).

        expm(Q(V)·dt) is cached per unique voltage value.  All four protocols are
        piecewise-constant (2-4 voltage levels per run), so only 2-4 expm calls are
        needed instead of one per timestep.
        """
        t = self._time_array()
        dt = self.dt
        P = self.s0.copy()
        states = np.zeros((len(t), len(P)))
        states[0] = P

        _expm_cache: dict = {}
        prev_V = None
        mat = None

        for k in range(1, len(t)):
            V = voltage_func(t[k - 1])
            if V != prev_V:
                if V not in _expm_cache:
                    _expm_cache[V] = _matrix_expm(self._build_Q(V) * dt)
                mat = _expm_cache[V]
                prev_V = V
            P = mat @ P
            states[k] = P
        return t, states

    def _idx(self, t_sec: float) -> int:
        return int(round(t_sec / self.dt))

    def _peak_open(self, states: np.ndarray, i_start: int, i_end: int) -> float:
        """Peak total open-state probability in the window [i_start, i_end)."""
        open_prob = states[i_start:i_end, self._open_idx].sum(axis=1)
        return float(np.max(open_prob))

    def _open_at(self, states: np.ndarray, i: int) -> float:
        return float(states[i, self._open_idx].sum())

    # ------------------------------------------------------------------
    # Protocol runners — return normalised arrays
    # ------------------------------------------------------------------

    def run_activation(self, test_voltages: np.ndarray = None) -> np.ndarray:
        proto = self.act_proto
        if test_voltages is None:
            test_voltages = proto.get_test_voltages()

        i0 = self._idx(proto.t_pulse_start)
        i1 = self._idx(proto.t_pulse_end)

        conductances = np.zeros(len(test_voltages))
        for i, V in enumerate(test_voltages):
            _, states = self._simulate(proto.get_voltage_function(V))
            conductances[i] = self.g_k_max * self._peak_open(states, i0, i1)

        mx = np.max(conductances)
        return conductances / mx if mx > 0 else conductances

    def run_inactivation(self, test_voltages: np.ndarray = None) -> np.ndarray:
        proto = self.inact_proto
        if test_voltages is None:
            test_voltages = proto.get_test_voltages()

        v_depo = proto.cfg.v_depo
        v_hold = proto.cfg.v_hold
        i_test = self._idx(proto.t_test_start)

        currents = np.zeros(len(test_voltages))
        for i, V in enumerate(test_voltages):
            _, states = self._simulate(proto.get_voltage_function(V))
            baseline = self._open_at(states, i_test - 1)
            peak = float(np.max(states[i_test:, self._open_idx].sum(axis=1)))
            g = self.g_k_max * (peak - baseline)
            currents[i] = g * (v_depo - v_hold)

        mx = np.max(currents)
        return currents / mx if mx > 0 else currents

    def run_cs_inactivation(self, test_times: np.ndarray = None) -> np.ndarray:
        proto = self.csi_proto
        if test_times is None:
            test_times = proto.get_test_times()

        i_prep = self._idx(proto.cfg.t_initial)
        v_depo = proto.cfg.v_depo
        v_prep = proto.cfg.v_prep

        currents = np.zeros(len(test_times))
        for i, t_pulse in enumerate(test_times):
            _, states = self._simulate(proto.get_voltage_function(t_pulse))
            i_pulse = self._idx(t_pulse)
            baseline_max = float(
                np.max(states[0 : i_prep + 1, self._open_idx].sum(axis=1))
            )
            if baseline_max == 0:
                baseline_max = 1e-12
            peak_after = float(np.max(states[i_pulse:, self._open_idx].sum(axis=1)))
            g = self.g_k_max * peak_after / baseline_max
            currents[i] = g * (v_depo - v_prep)

        mx = np.max(currents)
        return currents / mx if mx > 0 else currents

    def run_recovery(self, test_times: np.ndarray = None) -> np.ndarray:
        proto = self.rec_proto
        if test_times is None:
            test_times = proto.get_test_times()

        cfg = proto.cfg
        i_pre_start = self._idx(cfg.t_prep)
        i_pre_end = self._idx(cfg.t_pulse)
        v_depo, v_hold = cfg.v_depo, cfg.v_hold

        ratios = np.zeros(len(test_times))
        for i, t_rec_end in enumerate(test_times):
            _, states = self._simulate(proto.get_voltage_function(t_rec_end))
            g_pre = self.g_k_max * self._peak_open(states, i_pre_start, i_pre_end)
            g_test = self.g_k_max * float(
                np.max(states[self._idx(t_rec_end) :, self._open_idx].sum(axis=1))
            )
            I_pre = g_pre * (v_depo - v_hold)
            I_test = g_test * (v_depo - v_hold)
            ratios[i] = I_test / I_pre if I_pre > 0 else 0.0

        return ratios
