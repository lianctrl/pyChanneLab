"""
Visualization utilities for comparing experimental and simulated data
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from simulator import ProtocolSimulator
from protocols import (ActivationProtocol, InactivationProtocol,
                       CSInactivationProtocol, RecoveryProtocol)


class DataVisualizer:
    """Visualize experimental vs simulated data"""
    
    def __init__(self, experimental_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        """
        Initialize visualizer
        
        Parameters
        ----------
        experimental_data : dict
            Dictionary with experimental data for each protocol
        """
        self.exp_data = experimental_data
        
    def plot_protocol_comparison(self, parameters: np.ndarray, 
                                 protocol_name: str,
                                 ax: Optional[plt.Axes] = None,
                                 show_errors: bool = True) -> plt.Axes:
        """
        Plot comparison of experimental vs simulated data for one protocol
        
        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
        protocol_name : str
            One of: 'activation', 'inactivation', 'cs_inactivation', 'recovery'
        ax : plt.Axes, optional
            Matplotlib axes to plot on
        show_errors : bool
            Whether to show error bars
            
        Returns
        -------
        ax : plt.Axes
            Matplotlib axes with plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get experimental data
        x_exp, y_exp, y_err = self.exp_data[protocol_name]
        
        # Simulate
        simulator = ProtocolSimulator(parameters)
        
        if protocol_name == 'activation':
            test_values = ActivationProtocol.get_test_voltages()
            y_sim = simulator.run_activation_protocol(test_values)
            xlabel = 'Voltage (mV)'
            ylabel = 'Normalized Conductance (g/g_max)'
            title = 'Activation Protocol'
            
        elif protocol_name == 'inactivation':
            test_values = InactivationProtocol.get_test_voltages()
            y_sim = simulator.run_inactivation_protocol(test_values)
            xlabel = 'Conditioning Voltage (mV)'
            ylabel = 'Normalized Current (I/I_max)'
            title = 'Inactivation Protocol'
            
        elif protocol_name == 'cs_inactivation':
            test_values = CSInactivationProtocol.get_test_times()
            y_sim = simulator.run_cs_inactivation_protocol(test_values)
            xlabel = 'Prepulse Duration (s)'
            ylabel = 'Normalized Current (I/I_max)'
            title = 'Cole-Moore Shift Protocol'
            
        elif protocol_name == 'recovery':
            test_values = RecoveryProtocol.get_test_times()
            y_sim = simulator.run_recovery_protocol(test_values)
            xlabel = 'Recovery Interval (s)'
            ylabel = 'Recovery Ratio (I_test/I_prepulse)'
            title = 'Recovery from Inactivation'
            
        else:
            raise ValueError(f"Unknown protocol: {protocol_name}")
        
        # Plot experimental data
        if show_errors and y_err is not None:
            ax.errorbar(x_exp, y_exp, yerr=y_err, fmt='o', 
                       label='Experimental', capsize=5, markersize=8,
                       color='black', ecolor='gray', alpha=0.7)
        else:
            ax.plot(x_exp, y_exp, 'o', label='Experimental', 
                   markersize=8, color='black', alpha=0.7)
        
        # Plot simulated data
        ax.plot(test_values, y_sim, '-', label='Simulated', 
               linewidth=2, color='red')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_all_protocols(self, parameters: np.ndarray, 
                          figsize: Tuple[int, int] = (16, 12),
                          save_path: Optional[str] = None):
        """
        Create a 2x2 grid of all protocol comparisons
        
        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            If provided, save figure to this path
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        protocols = ['activation', 'inactivation', 'cs_inactivation', 'recovery']
        
        for ax, protocol in zip(axes, protocols):
            self.plot_protocol_comparison(parameters, protocol, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    def plot_state_trajectories(self, parameters: np.ndarray,
                               protocol_name: str,
                               test_value: float,
                               figsize: Tuple[int, int] = (12, 8)):
        """
        Plot time courses of all states for a specific protocol
        
        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
        protocol_name : str
            Protocol to simulate
        test_value : float
            Test voltage (mV) or time (s) depending on protocol
        figsize : tuple
            Figure size
        """
        from config import TIME_PARAMS
        
        # Get voltage function
        if protocol_name == 'activation':
            voltage_func = ActivationProtocol.get_voltage_function(test_value)
        elif protocol_name == 'inactivation':
            voltage_func = InactivationProtocol.get_voltage_function(test_value)
        elif protocol_name == 'cs_inactivation':
            voltage_func = CSInactivationProtocol.get_voltage_function(test_value)
        elif protocol_name == 'recovery':
            voltage_func = RecoveryProtocol.get_voltage_function(test_value)
        else:
            raise ValueError(f"Unknown protocol: {protocol_name}")
        
        # Simulate
        simulator = ProtocolSimulator(parameters)
        t, states = simulator.simulate(voltage_func)
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot voltage
        voltages = np.array([voltage_func(ti) for ti in t])
        ax1.plot(t, voltages, 'k-', linewidth=2)
        ax1.set_ylabel('Voltage (mV)', fontsize=12)
        ax1.set_title(f'{protocol_name.replace("_", " ").title()} - State Trajectories', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot closed states
        for i in range(5):
            ax2.plot(t, states[:, i], label=f'C{i}', linewidth=2)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title('Closed States', fontsize=12)
        ax2.legend(loc='best', ncol=5)
        ax2.grid(True, alpha=0.3)
        
        # Plot inactivated and open states
        for i in range(5, 10):
            ax3.plot(t, states[:, i], label=f'I{i-5}', linewidth=2)
        ax3.plot(t, states[:, 10], 'k-', label='O', linewidth=3)
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('Probability', fontsize=12)
        ax3.set_title('Inactivated and Open States', fontsize=12)
        ax3.legend(loc='best', ncol=6)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def plot_parameter_evolution(params_history: list, 
                            figsize: Tuple[int, int] = (14, 10)):
    """
    Plot evolution of parameters during optimization
    
    Parameters
    ----------
    params_history : list
        List of parameter arrays at each iteration
    figsize : tuple
        Figure size
    """
    param_names = [
        'alpha_0', 'alpha_1', 'beta_0', 'beta_1',
        'k_CO_0', 'k_CO_1', 'k_OC_0', 'k_OC_1',
        'k_CI', 'k_IC', 'f'
    ]
    
    params_array = np.array(params_history)
    iterations = np.arange(len(params_history))
    
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(iterations, params_array[:, i], 'b-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    axes[-1].remove()
    
    plt.suptitle('Parameter Evolution During Optimization', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
