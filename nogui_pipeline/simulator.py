"""
Simulation module for running voltage protocols
"""

import numpy as np
from scipy.integrate import odeint
from typing import Tuple, Callable
from model import IonChannelModel
from config import INITIAL_CONDITIONS, TIME_PARAMS, G_K_MAX, VoltageProtocols as VP


class ProtocolSimulator:
    """Simulator for running voltage protocols and extracting measurements"""
    
    def __init__(self, parameters: np.ndarray):
        """
        Initialize simulator with model parameters
        
        Parameters
        ----------
        parameters : np.ndarray
            Array of 11 kinetic parameters
        """
        self.model = IonChannelModel(parameters)
        self.params = parameters
        
    def _generate_time_array(self) -> np.ndarray:
        """Generate time array for simulation"""
        return np.arange(
            TIME_PARAMS['tini'],
            TIME_PARAMS['tend'] + TIME_PARAMS['dt'],
            TIME_PARAMS['dt']
        )
    
    def simulate(self, voltage_func: Callable, initial_state: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run simulation with given voltage protocol
        
        Parameters
        ----------
        voltage_func : callable
            Function V(t) that returns voltage at time t
        initial_state : np.ndarray, optional
            Initial conditions. If None, uses default from config
            
        Returns
        -------
        t : np.ndarray
            Time array
        states : np.ndarray
            State trajectories (11 columns for 11 states)
        """
        if initial_state is None:
            initial_state = INITIAL_CONDITIONS
            
        t = self._generate_time_array()
        
        # Solve ODEs
        states = odeint(
            lambda state, time: self.model.equations(state, time, voltage_func),
            initial_state,
            t
        )
        
        return t, states
    
    def extract_peak_conductance(self, states: np.ndarray, t: np.ndarray, 
                                 t_start: float, t_end: float) -> float:
        """
        Extract peak conductance (open state) in a time window
        
        Parameters
        ----------
        states : np.ndarray
            State trajectories
        t : np.ndarray
            Time array
        t_start : float
            Start of measurement window (s)
        t_end : float
            End of measurement window (s)
            
        Returns
        -------
        g_peak : float
            Peak conductance (nS)
        """
        dt = TIME_PARAMS['dt']
        idx_start = int(t_start / dt)
        idx_end = int(t_end / dt)
        
        # Open state is column 10
        open_probability = np.max(states[idx_start:idx_end, 10])
        
        return G_K_MAX * open_probability
    
    def extract_conductance_change(self, states: np.ndarray, t: np.ndarray,
                                   t_start: float) -> float:
        """
        Extract conductance change from a baseline
        
        Parameters
        ----------
        states : np.ndarray
            State trajectories
        t : np.ndarray
            Time array
        t_start : float
            Start time for measuring change
            
        Returns
        -------
        g_change : float
            Change in conductance (nS)
        """
        dt = TIME_PARAMS['dt']
        idx_start = int(t_start / dt)
        
        baseline = states[idx_start - 1, 10]
        peak_after = np.max(states[idx_start:, 10])
        
        return G_K_MAX * (peak_after - baseline)
    
    def run_activation_protocol(self, test_voltages: np.ndarray) -> np.ndarray:
        """
        Run activation protocol for multiple test voltages
        
        Parameters
        ----------
        test_voltages : np.ndarray
            Array of test voltages (mV)
            
        Returns
        -------
        normalized_conductance : np.ndarray
            Normalized conductance (g/g_max)
        """
        from protocols import ActivationProtocol
        
        conductances = np.zeros(len(test_voltages))
        t = self._generate_time_array()
        
        for i, V_test in enumerate(test_voltages):
            voltage_func = ActivationProtocol.get_voltage_function(V_test)
            _, states = self.simulate(voltage_func)
            
            # Measure peak during test pulse
            conductances[i] = self.extract_peak_conductance(
                states, t, VP.ACT_T_START, VP.ACT_T_END
            )
        
        # Normalize
        return conductances / np.max(conductances)
    
    def run_inactivation_protocol(self, test_voltages: np.ndarray) -> np.ndarray:
        """
        Run inactivation protocol for multiple conditioning voltages
        
        Parameters
        ----------
        test_voltages : np.ndarray
            Array of conditioning voltages (mV)
            
        Returns
        -------
        normalized_current : np.ndarray
            Normalized peak current (I/I_max)
        """
        from protocols import InactivationProtocol
        
        currents = np.zeros(len(test_voltages))
        t = self._generate_time_array()
        
        for i, V_test in enumerate(test_voltages):
            voltage_func = InactivationProtocol.get_voltage_function(V_test)
            _, states = self.simulate(voltage_func)
            
            # Measure conductance change at test pulse
            g = self.extract_conductance_change(states, t, VP.INACT_T_TEST)
            
            # Convert to current: I = g * (V_test - V_hold)
            currents[i] = g * (VP.V_DEPO - VP.V_HOLD)
        
        # Normalize
        return currents / np.max(currents)
    
    def run_cs_inactivation_protocol(self, test_times: np.ndarray) -> np.ndarray:
        """
        Run Cole-Moore shift protocol for multiple prepulse durations
        
        Parameters
        ----------
        test_times : np.ndarray
            Array of prepulse durations (s)
            
        Returns
        -------
        normalized_current : np.ndarray
            Normalized peak current (I/I_max)
        """
        from protocols import CSInactivationProtocol
        
        currents = np.zeros(len(test_times))
        t = self._generate_time_array()
        
        for i, t_pulse in enumerate(test_times):
            voltage_func = CSInactivationProtocol.get_voltage_function(t_pulse)
            _, states = self.simulate(voltage_func)
            
            dt = TIME_PARAMS['dt']
            idx_pulse = int(t_pulse / dt)
            
            # Measure peak after prepulse relative to baseline
            baseline_max = np.max(states[0:int(VP.CSI_T_PREP / dt) + 1, 10])
            peak_after = np.max(states[idx_pulse:, 10])
            
            g = G_K_MAX * peak_after / baseline_max
            currents[i] = g * (VP.V_DEPO - VP.V_PREP)
        
        # Normalize
        return currents / np.max(currents)
    
    def run_recovery_protocol(self, test_times: np.ndarray) -> np.ndarray:
        """
        Run recovery from inactivation protocol
        
        Parameters
        ----------
        test_times : np.ndarray
            Array of recovery interval durations (s)
            
        Returns
        -------
        normalized_current : np.ndarray
            Normalized recovery (I_test / I_prepulse)
        """
        from protocols import RecoveryProtocol
        
        recovery_ratios = np.zeros(len(test_times))
        t = self._generate_time_array()
        dt = TIME_PARAMS['dt']
        
        for i, t_pulse in enumerate(test_times):
            voltage_func = RecoveryProtocol.get_voltage_function(t_pulse)
            _, states = self.simulate(voltage_func)
            
            # Measure peak during prepulse
            idx_prep_start = int(VP.REC_T_PREP / dt)
            idx_prep_end = int(VP.REC_T_PULSE / dt)
            g_prepulse = G_K_MAX * np.max(states[idx_prep_start:idx_prep_end, 10])
            
            # Measure peak during test pulse
            idx_test = int(t_pulse / dt)
            g_test = G_K_MAX * np.max(states[idx_test:, 10])
            
            # Calculate currents
            I_prepulse = g_prepulse * (VP.V_DEPO - VP.V_HOLD)
            I_test = g_test * (VP.V_DEPO - VP.V_HOLD)
            
            # Recovery ratio
            recovery_ratios[i] = I_test / I_prepulse if I_prepulse > 0 else 0
        
        return recovery_ratios
