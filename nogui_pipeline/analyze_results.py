"""
Utility script for analyzing optimization results
"""

import numpy as np
import argparse
from pathlib import Path

from config import INITIAL_GUESS
from data_loader import EXPERIMENTAL_DATA
from optimizer import CostFunction, ParameterOptimizer
from visualizer import DataVisualizer


def load_parameters(filepath: str) -> np.ndarray:
    """Load parameters from .npy file"""
    return np.load(filepath)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize optimization results'
    )
    parser.add_argument(
        'params_file',
        type=str,
        help='Path to .npy file with optimized parameters'
    )
    parser.add_argument(
        '--compare-initial',
        action='store_true',
        help='Compare with initial parameters'
    )
    parser.add_argument(
        '--plot-trajectories',
        action='store_true',
        help='Plot state trajectories for each protocol'
    )
    parser.add_argument(
        '--save-plots',
        type=str,
        default=None,
        help='Directory to save plots'
    )
    
    args = parser.parse_args()
    
    # Load parameters
    print(f"Loading parameters from: {args.params_file}")
    parameters = load_parameters(args.params_file)
    
    # Create evaluator and visualizer
    cost_func = CostFunction(EXPERIMENTAL_DATA)
    optimizer = ParameterOptimizer(cost_func)
    visualizer = DataVisualizer(EXPERIMENTAL_DATA)
    
    # Evaluate parameters
    print("\n" + "="*70)
    print("PARAMETER EVALUATION")
    print("="*70)
    costs = optimizer.evaluate_parameters(parameters, verbose=True)
    
    # Compare with initial if requested
    if args.compare_initial:
        print("\n" + "="*70)
        print("INITIAL PARAMETERS EVALUATION")
        print("="*70)
        initial_costs = optimizer.evaluate_parameters(INITIAL_GUESS, verbose=True)
        
        print("\n" + "="*70)
        print("IMPROVEMENT SUMMARY")
        print("="*70)
        for protocol in ['activation', 'inactivation', 'cs_inactivation', 'recovery', 'total']:
            improvement = (initial_costs[protocol] - costs[protocol]) / initial_costs[protocol] * 100
            print(f"{protocol:20s}: {improvement:>6.2f}% improvement")
        print("="*70)
    
    # Plot comparisons
    save_path = None
    if args.save_plots:
        Path(args.save_plots).mkdir(exist_ok=True)
        save_path = f"{args.save_plots}/protocol_comparison.png"
    
    print("\nGenerating comparison plots...")
    visualizer.plot_all_protocols(parameters, save_path=save_path)
    
    # Plot trajectories if requested
    if args.plot_trajectories:
        print("\nGenerating state trajectory plots...")
        
        # Example test values for each protocol
        test_values = {
            'activation': 0.0,      # 0 mV
            'inactivation': -30.0,  # -30 mV conditioning
            'cs_inactivation': 0.3, # 300 ms prepulse
            'recovery': 2.0         # 500 ms recovery (1.5 + 0.5)
        }
        
        for protocol, test_val in test_values.items():
            print(f"  Plotting {protocol}...")
            visualizer.plot_state_trajectories(parameters, protocol, test_val)


if __name__ == "__main__":
    main()
