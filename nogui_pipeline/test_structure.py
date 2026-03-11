"""
Test script to verify the refactored code structure
Run this to check if all modules work together correctly
"""

import numpy as np
from config import INITIAL_GUESS, PARAMETER_BOUNDS, INITIAL_CONDITIONS
from model import IonChannelModel
from protocols import ActivationProtocol
from simulator import ProtocolSimulator


def test_model():
    """Test that the model can be instantiated and equations run"""
    print("Testing model instantiation...")
    model = IonChannelModel(INITIAL_GUESS)
    
    # Test voltage function
    def test_voltage(t):
        return -90.0 if t < 0.5 else 0.0
    
    # Test equations
    derivatives = model.equations(INITIAL_CONDITIONS, 0.0, test_voltage)
    assert len(derivatives) == 11, "Should return 11 derivatives"
    print("✓ Model test passed")


def test_protocols():
    """Test that protocols generate correct voltage functions"""
    print("\nTesting voltage protocols...")
    
    # Test activation protocol
    V_test = ActivationProtocol.get_test_voltages()
    assert len(V_test) > 0, "Should generate test voltages"
    
    voltage_func = ActivationProtocol.get_voltage_function(0.0)
    assert callable(voltage_func), "Should return a callable function"
    assert voltage_func(0.0) == -90.0, "Should start at holding voltage"
    
    print("✓ Protocol test passed")


def test_simulator():
    """Test that simulator can run protocols"""
    print("\nTesting simulator...")
    
    simulator = ProtocolSimulator(INITIAL_GUESS)
    
    # Test activation protocol
    test_voltages = ActivationProtocol.get_test_voltages()
    result = simulator.run_activation_protocol(test_voltages)
    
    assert len(result) == len(test_voltages), "Should return result for each voltage"
    assert np.all(result >= 0) and np.all(result <= 1), "Results should be normalized"
    
    print("✓ Simulator test passed")


def test_parameter_bounds():
    """Test that initial guess is within bounds"""
    print("\nTesting parameter bounds...")
    
    for i, (guess, (lower, upper)) in enumerate(zip(INITIAL_GUESS, PARAMETER_BOUNDS)):
        assert lower <= guess <= upper, f"Parameter {i} outside bounds: {guess} not in [{lower}, {upper}]"
    
    print("✓ Parameter bounds test passed")


def main():
    """Run all tests"""
    print("="*60)
    print("RUNNING REFACTORED CODE TESTS")
    print("="*60)
    
    try:
        test_model()
        test_protocols()
        test_simulator()
        test_parameter_bounds()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nThe refactored code structure is working correctly!")
        print("You can now run: python main.py --method global")
        
    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED ✗")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
