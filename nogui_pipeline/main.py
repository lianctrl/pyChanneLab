"""
Main script for ion channel parameter optimization
"""

import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path

from config import INITIAL_GUESS, PARAMETER_BOUNDS, OPTIMIZATION_SETTINGS
from data_loader import EXPERIMENTAL_DATA
from optimizer import CostFunction, ParameterOptimizer


def save_results(result, output_dir: str = "results"):
    """
    Save optimization results to file
    
    Parameters
    ----------
    result : dict
        Optimization result dictionary
    output_dir : str
        Directory to save results
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/optimization_result_{timestamp}.json"
    
    # Prepare data for JSON serialization
    save_data = {
        'timestamp': timestamp,
        'success': bool(result.get('success', False)),
        'final_cost': float(result['fun']),
        'parameters': result['x'].tolist(),
        'parameter_names': [
            'alpha_0', 'alpha_1', 'beta_0', 'beta_1',
            'k_CO_0', 'k_CO_1', 'k_OC_0', 'k_OC_1',
            'k_CI', 'k_IC', 'f'
        ],
        'iterations': int(result.get('nit', 0)),
        'function_evaluations': int(result.get('nfev', 0)),
        'message': str(result.get('message', ''))
    }
    
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    # Also save as numpy file for easy loading
    np_filename = f"{output_dir}/parameters_{timestamp}.npy"
    np.save(np_filename, result['x'])
    print(f"Parameters saved to: {np_filename}")


def print_parameter_comparison(initial: np.ndarray, final: np.ndarray):
    """Print comparison of initial and final parameters"""
    param_names = [
        'alpha_0', 'alpha_1', 'beta_0', 'beta_1',
        'k_CO_0', 'k_CO_1', 'k_OC_0', 'k_OC_1',
        'k_CI', 'k_IC', 'f'
    ]
    
    print("\n" + "="*70)
    print("PARAMETER COMPARISON")
    print("="*70)
    print(f"{'Parameter':<12} {'Initial':>15} {'Final':>15} {'Change (%)':>15}")
    print("-"*70)
    
    for name, init_val, final_val in zip(param_names, initial, final):
        change = ((final_val - init_val) / init_val * 100) if init_val != 0 else np.inf
        print(f"{name:<12} {init_val:>15.6f} {final_val:>15.6f} {change:>14.2f}%")
    
    print("="*70)


def main():
    """Main optimization routine"""
    parser = argparse.ArgumentParser(
        description='Optimize ion channel kinetic parameters'
    )
    parser.add_argument(
        '--method',
        choices=['global', 'local', 'both'],
        default='global',
        help='Optimization method (default: global)'
    )
    parser.add_argument(
        '--maxiter',
        type=int,
        default=OPTIMIZATION_SETTINGS['maxiter'],
        help=f"Maximum iterations (default: {OPTIMIZATION_SETTINGS['maxiter']})"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=OPTIMIZATION_SETTINGS['workers'],
        help=f"Number of workers (default: {OPTIMIZATION_SETTINGS['workers']}, -1 for all cores)"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    parser.add_argument(
        '--initial-params',
        type=str,
        default=None,
        help='Path to .npy file with initial parameters (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load initial parameters
    if args.initial_params:
        initial_guess = np.load(args.initial_params)
        print(f"Loaded initial parameters from: {args.initial_params}")
    else:
        initial_guess = INITIAL_GUESS
        print("Using default initial parameters from config")
    
    # Create cost function and optimizer
    cost_func = CostFunction(EXPERIMENTAL_DATA)
    optimizer = ParameterOptimizer(cost_func)
    
    # Evaluate initial parameters
    print("\n" + "="*70)
    print("INITIAL PARAMETER EVALUATION")
    print("="*70)
    optimizer.evaluate_parameters(initial_guess, verbose=True)
    
    # Run optimization
    if args.method in ['global', 'both']:
        print("\n" + "="*70)
        print("STARTING GLOBAL OPTIMIZATION")
        print("="*70)
        
        result = optimizer.optimize_global(
            bounds=PARAMETER_BOUNDS,
            initial_guess=initial_guess,
            maxiter=args.maxiter,
            workers=args.workers
        )
        
        print("\n" + "="*70)
        print("GLOBAL OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Success: {result.success}")
        print(f"Final cost: {result.fun:.6f}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
        
        # Evaluate final parameters
        optimizer.evaluate_parameters(result.x, verbose=True)
        print_parameter_comparison(initial_guess, result.x)
        
        # Save results
        save_results(result, args.output_dir)
        
        # Update initial guess for local optimization if running both
        if args.method == 'both':
            initial_guess = result.x
    
    if args.method in ['local', 'both']:
        print("\n" + "="*70)
        print("STARTING LOCAL OPTIMIZATION")
        print("="*70)
        
        result = optimizer.optimize_local(
            initial_guess=initial_guess,
            bounds=PARAMETER_BOUNDS,
            maxiter=15000
        )
        
        print("\n" + "="*70)
        print("LOCAL OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Success: {result.success}")
        print(f"Final cost: {result.fun:.6f}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
        
        # Evaluate final parameters
        optimizer.evaluate_parameters(result.x, verbose=True)
        print_parameter_comparison(INITIAL_GUESS, result.x)
        
        # Save results
        save_results(result, args.output_dir)


if __name__ == "__main__":
    main()
