"""
Configuration file for ion channel modeling
Contains all physical constants, experimental parameters, and initial conditions
"""

import numpy as np

# Physical constants
TEMPERATURE = 291.0  # K (18°C)
ELEMENTARY_CHARGE = 1.602176634e-19  # C
BOLTZMANN_CONSTANT = 1.380649e-23  # J*K^-1

# Derived constant
EXP_FACTOR = (ELEMENTARY_CHARGE / (BOLTZMANN_CONSTANT * TEMPERATURE)) * 1e-3

# Channel properties
G_K_MAX = 33.2  # nS - Maximum potassium conductance
C_M = 1.0  # microF cm^-2 - Membrane capacitance

# Initial state conditions (11 states: C0-C4, I0-I4, O)
INITIAL_CONDITIONS = np.array([
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
    0.0001   # O
])

# Data folder
DATA_FOLDER = "NoKChIP/"

# Voltage protocol parameters
class VoltageProtocols:
    """Voltage parameters for different experimental protocols"""
    
    # Common voltages
    V_HOLD = -90.0  # mV
    V_DEPO = 60.0   # mV
    V_PREP = -50.0  # mV
    
    # Activation protocol
    ACT_V_MAX = 60.0       # mV
    ACT_INCREMENT = 10.0   # mV
    ACT_T_START = 0.50     # s
    ACT_T_END = 0.55       # s
    
    # Inactivation protocol
    INACT_V_MAX = 60.0     # mV
    INACT_INCREMENT = 10.0 # mV
    INACT_T_TEST = 1.50    # s
    
    # CS Inactivation protocol
    CSI_MIN_PULSE = 0.010  # s
    CSI_MAX_PULSE = 0.580  # s
    CSI_INCREMENT = 0.030  # s
    CSI_T_PREP = 0.10      # s
    CSI_T_END = 1.150      # s
    
    # Recovery protocol
    REC_MIN_PULSE = 0.000  # s
    REC_MAX_PULSE = 0.570  # s
    REC_INCREMENT = 0.030  # s
    REC_T_PREP = 0.50      # s
    REC_T_PULSE = 1.50     # s
    REC_T_END = 2.650      # s

# Time discretization
TIME_PARAMS = {
    'tini': 0.0,
    'tend': 3.0,
    'dt': 1e-5
}

# Optimization parameters
INITIAL_GUESS = np.array([
    204,    # alpha_0
    2.07,   # alpha_1
    21.2,   # beta_0
    2.8e-3, # beta_1
    2.8,    # k_CO_0
    0.343,  # k_CO_1
    37.5,   # k_OC_0
    1.0e-2, # k_OC_1
    94.5,   # k_CI
    0.24,   # k_IC
    0.44    # f
])

PARAMETER_BOUNDS = (
    (0.0, 2000.0),  # alpha_0
    (0.0, 5.0),     # alpha_1
    (0.0, 100.0),   # beta_0
    (0.0, 5.0),     # beta_1
    (0.0, 1000.0),  # k_CO_0
    (0.0, 5.0),     # k_CO_1
    (0.0, 1000.0),  # k_OC_0
    (0.0, 5.0),     # k_OC_1
    (0.0, 2000.0),  # k_CI
    (0.0, 100.0),   # k_IC
    (0.0, 1.0)      # f
)

# Optimization settings
OPTIMIZATION_SETTINGS = {
    'maxiter': 5000,
    'workers': -1  # Use all available cores
}
