''' Module where useful functions are declared to be used in MSM

here a list of possible functions the user can choose the rate to be dependent

'''

import numpy as np
from scipy.constants import e
from scipy.constants import k

# constant activation function
def constant(r: float) -> float:
    return r

# linear activation function
def linear(r0: float, r1: float, V: float) -> float:
    return r0*V+r1

# parabolic activation function
def quadratic(r0: float, r1: float, r2: float, V: float) -> float:
    return r0*V**2+r1*V+r2

# negative exponential activation function
def nexponential(r0: float, r1: float, V: float, T: float) -> float:
    factor = e/(k*T)
    return r0*np.exp(-r1*V*factor)

# positive exponential activation function
def pexponential(r0: float, r1: float, V: float, T: float) -> float:
    factor = e/(k*T)
    return r0*np.exp(r1*V*factor)
