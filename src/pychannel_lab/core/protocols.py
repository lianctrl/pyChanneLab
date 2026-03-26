"""
Voltage-clamp protocols.  Each class is instantiated with a config dataclass
so the user can freely adjust timing and voltage parameters.
"""

import numpy as np
from typing import Callable
from core.config import (
    ActivationConfig,
    InactivationConfig,
    CSInactivationConfig,
    RecoveryConfig,
)


class ActivationProtocol:
    """
    Step protocol sweeping test voltages for activation (G/V curve).

    Timeline for each sweep:
        [0, t_hold)        → v_hold
        [t_hold, t_hold+t_test) → v_test
        [t_hold+t_test, …) → v_tail
    """

    def __init__(self, cfg: ActivationConfig = None):
        self.cfg = cfg or ActivationConfig()

    @property
    def t_pulse_start(self) -> float:
        return self.cfg.t_hold

    @property
    def t_pulse_end(self) -> float:
        return self.cfg.t_hold + self.cfg.t_test

    def get_test_voltages(self) -> np.ndarray:
        c = self.cfg
        return np.arange(c.v_min, c.v_max + c.v_step, c.v_step)

    def get_voltage_function(self, v_test: float) -> Callable:
        t0, t1 = self.t_pulse_start, self.t_pulse_end
        v_hold, v_tail = self.cfg.v_hold, self.cfg.v_tail

        def voltage(t: float) -> float:
            if t < t0:
                return v_hold
            elif t < t1:
                return v_test
            else:
                return v_tail

        return voltage


class InactivationProtocol:
    """
    Conditioning-pulse protocol for steady-state inactivation (h∞/V).

    Timeline:
        [0, t_hold)                   → v_hold
        [t_hold, t_hold+t_cond)       → v_cond  (swept)
        [t_hold+t_cond, …)            → v_depo
    """

    def __init__(self, cfg: InactivationConfig = None):
        self.cfg = cfg or InactivationConfig()

    @property
    def t_test_start(self) -> float:
        return self.cfg.t_hold + self.cfg.t_cond

    def get_test_voltages(self) -> np.ndarray:
        c = self.cfg
        return np.arange(c.v_min, c.v_max + c.v_step, c.v_step)

    def get_voltage_function(self, v_cond: float) -> Callable:
        t0 = self.cfg.t_hold
        t1 = self.t_test_start
        v_hold, v_depo = self.cfg.v_hold, self.cfg.v_depo

        def voltage(t: float) -> float:
            if t < t0:
                return v_hold
            elif t < t1:
                return v_cond
            else:
                return v_depo

        return voltage


class CSInactivationProtocol:
    """
    Closed-state inactivation protocol (variable-duration prepulse).

    Timeline:
        [0, t_initial)         → v_hold
        [t_initial, t_pulse)   → v_prep  (prepulse, swept duration)
        [t_pulse, t_test_end)  → v_depo
        [t_test_end, …)        → v_hold
    """

    def __init__(self, cfg: CSInactivationConfig = None):
        self.cfg = cfg or CSInactivationConfig()

    def get_test_times(self) -> np.ndarray:
        """Absolute end-times of the prepulse (t_initial + prepulse_duration)."""
        c = self.cfg
        durations = np.arange(
            c.min_pulse, c.max_pulse + c.pulse_increment, c.pulse_increment
        )
        return c.t_initial + durations

    def get_voltage_function(self, t_pulse: float) -> Callable:
        c = self.cfg

        def voltage(t: float) -> float:
            if t < c.t_initial:
                return c.v_hold
            elif t <= t_pulse:
                return c.v_prep
            elif t <= c.t_test_end:
                return c.v_depo
            else:
                return c.v_hold

        return voltage


class RecoveryProtocol:
    """
    Recovery-from-inactivation protocol (variable recovery interval).

    Timeline:
        [0, t_prep)             → v_hold
        [t_prep, t_pulse)       → v_depo  (inactivating pulse)
        [t_pulse, t_rec_end)    → v_hold  (recovery interval, swept)
        [t_rec_end, t_end)      → v_depo  (test pulse)
        [t_end, …)              → v_depo
    """

    def __init__(self, cfg: RecoveryConfig = None):
        self.cfg = cfg or RecoveryConfig()

    def get_test_times(self) -> np.ndarray:
        """Absolute start-times of the test pulse (t_pulse + interval)."""
        c = self.cfg
        intervals = np.arange(
            c.min_interval, c.max_interval + c.interval_increment, c.interval_increment
        )
        return c.t_pulse + intervals

    def get_voltage_function(self, t_rec_end: float) -> Callable:
        c = self.cfg

        def voltage(t: float) -> float:
            if t < c.t_prep:
                return c.v_hold
            elif t <= c.t_pulse:
                return c.v_depo
            elif t < t_rec_end:
                return c.v_hold
            else:
                return c.v_depo

        return voltage
