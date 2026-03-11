"""
Voltage protocols for different experimental procedures
"""

import numpy as np
from typing import Callable
from config import VoltageProtocols as VP


class VoltageProtocol:
    """Base class for voltage protocols"""
    
    @staticmethod
    def get_voltage_function(protocol_params) -> Callable:
        """
        Return a function that gives voltage at any time t
        
        Parameters
        ----------
        protocol_params : dict
            Protocol-specific parameters
            
        Returns
        -------
        voltage_func : callable
            Function V(t) that returns voltage at time t
        """
        raise NotImplementedError


class ActivationProtocol(VoltageProtocol):
    """Activation voltage protocol"""
    
    @staticmethod
    def get_test_voltages() -> np.ndarray:
        """Generate array of test voltages"""
        return np.arange(VP.V_HOLD, VP.ACT_V_MAX + VP.ACT_INCREMENT, VP.ACT_INCREMENT)
    
    @staticmethod
    def get_voltage_function(V_test: float) -> Callable:
        """
        Voltage protocol: -90 mV hold → V_test → -50 mV
        
        Timeline:
        - 0 to 0.5 s: -90 mV (hold)
        - 0.5 to 0.55 s: V_test
        - 0.55 s onward: -50 mV
        """
        def voltage(t: float) -> float:
            if t < 0.5:
                return VP.V_HOLD
            elif t < 0.55:
                return V_test
            else:
                return -50.0
        return voltage


class InactivationProtocol(VoltageProtocol):
    """Inactivation voltage protocol"""
    
    @staticmethod
    def get_test_voltages() -> np.ndarray:
        """Generate array of test voltages"""
        return np.arange(VP.V_HOLD, VP.INACT_V_MAX + VP.INACT_INCREMENT, VP.INACT_INCREMENT)
    
    @staticmethod
    def get_voltage_function(V_test: float) -> Callable:
        """
        Voltage protocol: -90 mV hold → V_test → 60 mV test
        
        Timeline:
        - 0 to 0.5 s: -90 mV (hold)
        - 0.5 to 1.5 s: V_test (conditioning)
        - 1.5 s onward: 60 mV (test pulse)
        """
        def voltage(t: float) -> float:
            if t < 0.5:
                return VP.V_HOLD
            elif t < 1.5:
                return V_test
            else:
                return VP.V_DEPO
        return voltage


class CSInactivationProtocol(VoltageProtocol):
    """Cole-Moore shift inactivation protocol"""
    
    @staticmethod
    def get_test_times() -> np.ndarray:
        """Generate array of prepulse durations"""
        return np.arange(
            VP.CSI_MIN_PULSE, 
            VP.CSI_MAX_PULSE + VP.CSI_INCREMENT, 
            VP.CSI_INCREMENT
        )
    
    @staticmethod
    def get_voltage_function(t_pulse: float) -> Callable:
        """
        Voltage protocol: -90 mV → -50 mV prepulse → 60 mV test → -90 mV
        
        Timeline:
        - 0 to 0.1 s: -90 mV
        - 0.1 to t_pulse s: -50 mV (prepulse)
        - t_pulse to 1.15 s: 60 mV (test)
        - 1.15 s onward: -90 mV
        """
        def voltage(t: float) -> float:
            if t < 0.1:
                return VP.V_HOLD
            elif t <= t_pulse:
                return VP.V_PREP
            elif t <= 1.15:
                return VP.V_DEPO
            else:
                return VP.V_HOLD
        return voltage


class RecoveryProtocol(VoltageProtocol):
    """Recovery from inactivation protocol"""
    
    @staticmethod
    def get_test_times() -> np.ndarray:
        """Generate array of recovery interval durations"""
        return np.arange(
            VP.REC_MIN_PULSE,
            VP.REC_MAX_PULSE + VP.REC_INCREMENT,
            VP.REC_INCREMENT
        )
    
    @staticmethod
    def get_voltage_function(t_pulse: float) -> Callable:
        """
        Voltage protocol: -90 mV → 60 mV pulse → -90 mV recovery → 60 mV test
        
        Timeline:
        - 0 to 0.5 s: -90 mV (hold)
        - 0.5 to 1.5 s: 60 mV (inactivating pulse)
        - 1.5 to t_pulse s: -90 mV (recovery interval)
        - t_pulse to 2.65 s: 60 mV (test pulse)
        - 2.65 s onward: 60 mV
        """
        def voltage(t: float) -> float:
            if t < 0.5:
                return VP.V_HOLD
            elif t <= 1.5:
                return VP.V_DEPO
            elif t < t_pulse:
                return VP.V_HOLD
            else:
                return VP.V_DEPO
        return voltage
