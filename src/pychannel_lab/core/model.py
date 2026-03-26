"""
11-state Markov model for ion channel kinetics.

States: C0-C4 (closed), I0-I4 (inactivated), O (open)
"""

import numpy as np
from core.config import EXP_FACTOR


class IonChannelModel:
    """
    11-state Markov model.

    Parameters
    ----------
    parameters : array-like, shape (11,)
        [alpha_0, alpha_1, beta_0, beta_1,
         k_CO_0,  k_CO_1,  k_OC_0, k_OC_1,
         k_CI,    k_IC,    f]
    """

    def __init__(self, parameters: np.ndarray):
        self.params = np.asarray(parameters, dtype=float)

    # ------------------------------------------------------------------
    def _rates(self, V: float):
        p = self.params
        alpha = p[0] * np.exp(p[1] * V * EXP_FACTOR)
        beta = p[2] * np.exp(-p[3] * V * EXP_FACTOR)
        k_CO = p[4] * np.exp(p[5] * V * EXP_FACTOR)
        k_OC = p[6] * np.exp(-p[7] * V * EXP_FACTOR)
        return alpha, beta, k_CO, k_OC

    # ------------------------------------------------------------------
    def equations(self, state, t: float, voltage_func):
        """ODE right-hand side.  Returns tuple of 11 derivatives."""
        C0, C1, C2, C3, C4 = state[0:5]
        I0, I1, I2, I3, I4 = state[5:10]
        O = state[10]

        V = voltage_func(t)
        alpha, beta, k_CO, k_OC = self._rates(V)
        k_CI = self.params[8]
        k_IC = self.params[9]
        f = self.params[10]

        # Closed states
        dC0 = beta * C1 + (k_IC / f**4) * I0 - (k_CI * f**4 + 4 * alpha) * C0
        dC1 = (
            4 * alpha * C0
            + 2 * beta * C2
            + (k_IC / f**3) * I1
            - (k_CI * f**3 + beta + 3 * alpha) * C1
        )
        dC2 = (
            3 * alpha * C1
            + 3 * beta * C3
            + (k_IC / f**2) * I2
            - (k_CI * f**2 + 2 * beta + 2 * alpha) * C2
        )
        dC3 = (
            2 * alpha * C2
            + 4 * beta * C4
            + (k_IC / f) * I3
            - (k_CI * f + 3 * beta + alpha) * C3
        )
        dC4 = alpha * C3 + k_OC * O + k_IC * I4 - (k_CI + k_CO + 4 * beta) * C4

        # Inactivated states
        dI0 = beta * f * I1 + k_CI * f**4 * C0 - (k_IC / f**4 + 4 * (alpha / f)) * I0
        dI1 = (
            4 * (alpha / f) * I0
            + 2 * beta * f * I2
            + k_CI * f**3 * C1
            - (k_IC / f**3 + beta * f + 3 * (alpha / f)) * I1
        )
        dI2 = (
            3 * (alpha / f) * I1
            + 3 * beta * f * I3
            + k_CI * f**2 * C2
            - (k_IC / f**2 + 2 * beta * f + 2 * (alpha / f)) * I2
        )
        dI3 = (
            2 * (alpha / f) * I2
            + 4 * beta * f * I4
            + k_CI * f * C3
            - (k_IC / f + 3 * beta * f + (alpha / f)) * I3
        )
        dI4 = (alpha / f) * I3 + k_CI * C4 - (k_IC + 4 * beta * f) * I4

        # Open state
        dO = k_CO * C4 - k_OC * O

        return (dC0, dC1, dC2, dC3, dC4, dI0, dI1, dI2, dI3, dI4, dO)
