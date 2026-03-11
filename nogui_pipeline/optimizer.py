"""
Optimization module for parameter fitting
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Dict, Tuple
from simulator import ProtocolSimulator
from protocols import (ActivationProtocol, InactivationProtocol, 
                       CSInactivationProtocol, RecoveryProtocol)


class CostFunction:
    """
    Cost function for parameter optimization
    Compares simulated data with experimental data
    """
    
    def __init__(self, experimental_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        """
        Initialize cost function with experimental data
        
        Parameters
        ----------
        experimental_data : dict
            Dictionary with keys: 'activation', 'inactivation', 'cs_inactivation', 'recovery'
            Each value is a tuple of (x_data, y_data, y_err)
        """
        self.exp_data = experimental_data
        
    def _mean_squared_error(self, predicted: np.ndarray, observed: np.ndarray) -> float:
        """
        Calculate normalized mean squared error
        
        Parameters
        ----------
        predicted : np.ndarray
            Predicted values from simulation
        observed : np.ndarray
            Observed experimental values
            
        Returns
        -------
        mse : float
            Mean squared error
        """
        return np.mean((predicted - observed) ** 2)
    
    def activation_cost(self, parameters: np.ndarray) -> float:
        """Calculate cost for activation protocol"""
        simulator = ProtocolSimulator(parameters)
        test_voltages = ActivationProtocol.get_test_voltages()
        
        predicted = simulator.run_activation_protocol(test_voltages)
        _, observed, _ = self.exp_data['activation']
        
        return self._mean_squared_error(predicted, observed)
    
    def inactivation_cost(self, parameters: np.ndarray) -> float:
        """Calculate cost for inactivation protocol"""
        simulator = ProtocolSimulator(parameters)
        test_voltages = InactivationProtocol.get_test_voltages()
        
        predicted = simulator.run_inactivation_protocol(test_voltages)
        _, observed, _ = self.exp_data['inactivation']
        
        return self._mean_squared_error(predicted, observed)
    
    def cs_inactivation_cost(self, parameters: np.ndarray) -> float:
        """Calculate cost for CS inactivation protocol"""
        simulator = ProtocolSimulator(parameters)
        test_times = CSInactivationProtocol.get_test_times()
        
        predicted = simulator.run_cs_inactivation_protocol(test_times)
        _, observed, _ = self.exp_data['cs_inactivation']
        
        return self._mean_squared_error(predicted, observed)
    
    def recovery_cost(self, parameters: np.ndarray) -> float:
        """Calculate cost for recovery protocol"""
        simulator = ProtocolSimulator(parameters)
        test_times = RecoveryProtocol.get_test_times()
        
        predicted = simulator.run_recovery_protocol(test_times)
        _, observed, _ = self.exp_data['recovery']
        
        return self._mean_squared_error(predicted, observed)
    
    def total_cost(self, parameters: np.ndarray) -> float:
        costs = {
            'activation': self.activation_cost(parameters),
            'inactivation': self.inactivation_cost(parameters),
            'cs_inactivation': self.cs_inactivation_cost(parameters),
            'recovery': self.recovery_cost(parameters)
        }
    
        weights = {
            'activation': 1.0,
            'inactivation': 1.0,
            'cs_inactivation': 2.0,
            'recovery': 2.0
        }
    
        return sum(weights[key] * costs[key] for key in costs.keys())
    
    def get_individual_costs(self, parameters: np.ndarray) -> Dict[str, float]:
        """
        Get cost breakdown for each protocol
        
        Parameters
        ----------
        parameters : np.ndarray
            Array of 11 kinetic parameters
            
        Returns
        -------
        costs : dict
            Dictionary of individual protocol costs
        """
        return {
            'activation': self.activation_cost(parameters),
            'inactivation': self.inactivation_cost(parameters),
            'cs_inactivation': self.cs_inactivation_cost(parameters),
            'recovery': self.recovery_cost(parameters)
        }


class ParameterOptimizer:
    """
    Parameter optimizer using global or local optimization algorithms
    """
    
    def __init__(self, cost_function: CostFunction):
        """
        Initialize optimizer
        
        Parameters
        ----------
        cost_function : CostFunction
            Cost function object for optimization
        """
        self.cost_func = cost_function
        
    def optimize_global(self, bounds: tuple, initial_guess: np.ndarray = None,
                       maxiter: int = 5000, workers: int = -1,
                       callback=None) -> dict:
        """
        Global optimization using differential evolution
        
        Parameters
        ----------
        bounds : tuple
            Parameter bounds as tuple of (min, max) pairs
        initial_guess : np.ndarray, optional
            Initial parameter guess (used to seed population)
        maxiter : int
            Maximum number of iterations
        workers : int
            Number of parallel workers (-1 for all cores)
        callback : callable, optional
            Callback function called after each iteration
            
        Returns
        -------
        result : dict
            Optimization result with keys 'x' (parameters), 'fun' (final cost), etc.
        """
        # Create custom callback that prints progress
        iteration = [0]
        
        def progress_callback(xk, convergence):
            iteration[0] += 1
            if iteration[0] % 100 == 0:
                cost = self.cost_func.total_cost(xk)
                print(f"Iteration {iteration[0]}: Cost = {cost:.6f}")
            if callback:
                callback(xk, convergence)
        
        print("Starting global optimization (differential evolution)...")
        print(f"Maximum iterations: {maxiter}")
        print(f"Workers: {workers if workers > 0 else 'all available cores'}")
        
        result = differential_evolution(
            self.cost_func.total_cost,
            bounds=bounds,
            maxiter=maxiter,
            workers=workers,
            callback=progress_callback,
            seed=42,  # For reproducibility
            disp=True
        )
        
        return result
    
    def optimize_local(self, initial_guess: np.ndarray, bounds: tuple,
                      method: str = 'L-BFGS-B', maxiter: int = 15000) -> dict:
        """
        Local optimization from initial guess
        
        Parameters
        ----------
        initial_guess : np.ndarray
            Initial parameter values
        bounds : tuple
            Parameter bounds
        method : str
            Optimization method ('L-BFGS-B', 'TNC', etc.)
        maxiter : int
            Maximum number of iterations
            
        Returns
        -------
        result : dict
            Optimization result
        """
        print(f"Starting local optimization ({method})...")
        print(f"Maximum iterations: {maxiter}")
        
        result = minimize(
            self.cost_func.total_cost,
            initial_guess,
            bounds=bounds,
            method=method,
            options={'maxiter': maxiter, 'maxfev': 50000, 'disp': True}
        )
        
        return result
    
    def evaluate_parameters(self, parameters: np.ndarray, verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate a parameter set and return cost breakdown
        
        Parameters
        ----------
        parameters : np.ndarray
            Parameter values to evaluate
        verbose : bool
            If True, print detailed cost breakdown
            
        Returns
        -------
        costs : dict
            Dictionary of costs for each protocol plus total
        """
        costs = self.cost_func.get_individual_costs(parameters)
        costs['total'] = sum(costs.values())
        
        if verbose:
            print("\n" + "="*50)
            print("COST BREAKDOWN")
            print("="*50)
            for protocol, cost in costs.items():
                print(f"{protocol:20s}: {cost:.6f}")
            print("="*50)
        
        return costs
