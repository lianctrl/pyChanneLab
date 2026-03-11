"""
Ion channel kinetic model
Contains the ordinary differential equations for the 11-state model
"""

import numpy as np
from typing import List, Tuple
from config import EXP_FACTOR


class IonChannelModel:
    """
    11-state Markov model for ion channel kinetics
    States: C0, C1, C2, C3, C4 (closed), I0, I1, I2, I3, I4 (inactivated), O (open)
    """
    
    def __init__(self, parameters: np.ndarray):
        """
        Initialize the model with kinetic parameters
        
        Parameters
        ----------
        parameters : np.ndarray
            Array of 11 parameters:
            [alpha_0, alpha_1, beta_0, beta_1, k_CO_0, k_CO_1, 
             k_OC_0, k_OC_1, k_CI, k_IC, f]
        """
        self.params = parameters
        
    def _calculate_rate_constants(self, V: float) -> Tuple[float, float, float, float]:
        """
        Calculate voltage-dependent rate constants
        
        Parameters
        ----------
        V : float
            Membrane voltage (mV)
            
        Returns
        -------
        alpha, beta, k_CO, k_OC : tuple of floats
            Voltage-dependent rate constants (s^-1)
        """
        alpha = self.params[0] * np.exp(self.params[1] * (V * EXP_FACTOR))
        beta = self.params[2] * np.exp(-self.params[3] * (V * EXP_FACTOR))
        k_CO = self.params[4] * np.exp(self.params[5] * (V * EXP_FACTOR))
        k_OC = self.params[6] * np.exp(-self.params[7] * (V * EXP_FACTOR))
        
        return alpha, beta, k_CO, k_OC
    
    def equations(self, state: List[float], t: float, voltage_func) -> Tuple:
        """
        System of ODEs for the 11-state model
        
        Parameters
        ----------
        state : list
            Current state vector [C0, C1, C2, C3, C4, I0, I1, I2, I3, I4, O]
        t : float
            Current time (s)
        voltage_func : callable
            Function that returns voltage at time t
            
        Returns
        -------
        derivatives : tuple
            Time derivatives of all 11 states
        """
        # Unpack states
        C0, C1, C2, C3, C4 = state[0:5]
        I0, I1, I2, I3, I4 = state[5:10]
        O = state[10]
        
        # Get voltage at current time
        V = voltage_func(t)
        
        # Calculate voltage-dependent rates
        alpha, beta, k_CO, k_OC = self._calculate_rate_constants(V)
        
        # Voltage-independent rates
        k_CI = self.params[8]
        k_IC = self.params[9]
        f = self.params[10]
        
        # Closed states (C0-C4)
        dC0dt = (beta * C1 + 
                 (k_IC / f**4) * I0 - 
                 (k_CI * f**4 + 4 * alpha) * C0)
        
        dC1dt = (4 * alpha * C0 + 2 * beta * C2 + 
                 (k_IC / f**3) * I1 - 
                 (k_CI * f**3 + beta + 3 * alpha) * C1)
        
        dC2dt = (3 * alpha * C1 + 3 * beta * C3 + 
                 (k_IC / f**2) * I2 - 
                 (k_CI * f**2 + 2 * beta + 2 * alpha) * C2)
        
        dC3dt = (2 * alpha * C2 + 4 * beta * C4 + 
                 (k_IC / f) * I3 - 
                 (k_CI * f + 3 * beta + alpha) * C3)
        
        dC4dt = (alpha * C3 + k_OC * O + k_IC * I4 - 
                 (k_CI + k_CO + 4 * beta) * C4)
        
        # Inactivated states (I0-I4)
        dI0dt = (beta * f * I1 + 
                 k_CI * f**4 * C0 - 
                 (k_IC / f**4 + 4 * (alpha / f)) * I0)
        
        dI1dt = (4 * (alpha / f) * I0 + 2 * beta * f * I2 + 
                 k_CI * f**3 * C1 - 
                 (k_IC / f**3 + beta * f + 3 * (alpha / f)) * I1)
        
        dI2dt = (3 * (alpha / f) * I1 + 3 * beta * f * I3 + 
                 k_CI * f**2 * C2 - 
                 (k_IC / f**2 + 2 * beta * f + 2 * (alpha / f)) * I2)
        
        dI3dt = (2 * (alpha / f) * I2 + 4 * beta * f * I4 + 
                 k_CI * f * C3 - 
                 (k_IC / f + 3 * beta * f + (alpha / f)) * I3)
        
        dI4dt = ((alpha / f) * I3 + k_CI * C4 - 
                 (k_IC + 4 * beta * f) * I4)
        
        # Open state (O)
        dOdt = k_CO * C4 - k_OC * O
        
        return (dC0dt, dC1dt, dC2dt, dC3dt, dC4dt, 
                dI0dt, dI1dt, dI2dt, dI3dt, dI4dt, dOdt)
