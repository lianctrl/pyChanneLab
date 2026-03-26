"""
Physical constants, default parameters, and protocol configuration dataclasses.
"""

from dataclasses import dataclass, field
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
TEMPERATURE = 291.0  # K (18 °C)
ELEMENTARY_CHARGE = 1.602176634e-19  # C
BOLTZMANN_CONSTANT = 1.380649e-23  # J K^-1
EXP_FACTOR = (ELEMENTARY_CHARGE / (BOLTZMANN_CONSTANT * TEMPERATURE)) * 1e-3

# Channel properties
G_K_MAX = 33.2  # nS — maximum potassium conductance

# ---------------------------------------------------------------------------
# Model state initial conditions  (11 states: C0-C4, I0-I4, O)
# ---------------------------------------------------------------------------
INITIAL_CONDITIONS = np.array(
    [
        0.4390,  # C0
        0.2588,  # C1
        0.0572,  # C2
        0.0056,  # C3
        0.0002,  # C4
        0.0128,  # I0
        0.0553,  # I1
        0.0894,  # I2
        0.0642,  # I3
        0.0172,  # I4
        0.0001,  # O
    ]
)

# ---------------------------------------------------------------------------
# Default optimisation initial guess and bounds
# ---------------------------------------------------------------------------
PARAMETER_NAMES = [
    "alpha_0",
    "alpha_1",
    "beta_0",
    "beta_1",
    "k_CO_0",
    "k_CO_1",
    "k_OC_0",
    "k_OC_1",
    "k_CI",
    "k_IC",
    "f",
]

INITIAL_GUESS = np.array(
    [
        204.0,  # alpha_0
        2.07,  # alpha_1
        21.2,  # beta_0
        2.8e-3,  # beta_1
        2.8,  # k_CO_0
        0.343,  # k_CO_1
        37.5,  # k_OC_0
        1.0e-2,  # k_OC_1
        94.5,  # k_CI
        0.24,  # k_IC
        0.44,  # f
    ]
)

PARAMETER_BOUNDS = (
    (0.0, 2000.0),  # alpha_0
    (0.0, 5.0),  # alpha_1
    (0.0, 100.0),  # beta_0
    (0.0, 5.0),  # beta_1
    (0.0, 1000.0),  # k_CO_0
    (0.0, 5.0),  # k_CO_1
    (0.0, 1000.0),  # k_OC_0
    (0.0, 5.0),  # k_OC_1
    (0.0, 2000.0),  # k_CI
    (0.0, 100.0),  # k_IC
    (0.0, 1.0),  # f
)

# Optimisation engine settings
OPTIMIZATION_SETTINGS = {
    "maxiter": 5000,
    "workers": -1,
}

# Simulation time grid
TIME_PARAMS = {
    "tini": 0.0,
    "tend": 3.0,
    "dt": 1e-5,
}

# ---------------------------------------------------------------------------
# Protocol configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ActivationConfig:
    """Activation (G/V) protocol parameters."""

    v_hold: float = -90.0  # mV — holding potential
    v_tail: float = -50.0  # mV — tail potential after test pulse
    v_min: float = -90.0  # mV — first test voltage
    v_max: float = 60.0  # mV — last test voltage
    v_step: float = 10.0  # mV — voltage step
    t_hold: float = 0.50  # s  — duration of holding phase
    t_test: float = 0.05  # s  — duration of test pulse


@dataclass
class InactivationConfig:
    """Steady-state inactivation (h∞/V) protocol parameters."""

    v_hold: float = -90.0  # mV — holding potential
    v_depo: float = 60.0  # mV — depolarising test pulse
    v_min: float = -90.0  # mV — first conditioning voltage
    v_max: float = 60.0  # mV — last conditioning voltage
    v_step: float = 10.0  # mV
    t_hold: float = 0.50  # s  — initial holding
    t_cond: float = 1.00  # s  — conditioning pulse duration
    # test pulse lasts until end of simulation


@dataclass
class CSInactivationConfig:
    """Closed-state inactivation (prepulse) protocol parameters."""

    v_hold: float = -90.0  # mV
    v_prep: float = -50.0  # mV — prepulse potential
    v_depo: float = 60.0  # mV — test depolarisation
    t_initial: float = 0.10  # s  — initial hold before prepulse
    min_pulse: float = 0.010  # s  — shortest prepulse (after t_initial)
    max_pulse: float = 0.580  # s  — longest  prepulse (after t_initial)
    pulse_increment: float = 0.030  # s
    t_test_end: float = 1.150  # s  — absolute end of test pulse


@dataclass
class RecoveryConfig:
    """Recovery-from-inactivation protocol parameters."""

    v_hold: float = -90.0  # mV
    v_depo: float = 60.0  # mV — inactivating / test pulse voltage
    t_prep: float = 0.50  # s  — hold before inactivating pulse
    t_pulse: float = 1.50  # s  — absolute start of recovery interval
    min_interval: float = 0.000  # s  — shortest recovery interval
    max_interval: float = 0.570  # s  — longest  recovery interval
    interval_increment: float = 0.030  # s
    t_end: float = 2.650  # s  — absolute end of test pulse
