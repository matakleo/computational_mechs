import numpy as np
import scipy as sc

nsteps=50
max_load=15
radius=0.5

def N(x):
    return np.array(0.2*x**3-2.1*x**2+6*x)-max_load


def other_eq(x):
    return np.array(0.6*x**2 - 4.2*x + 6)

for nstep in range(nsteps):
    sc.optimize.fsolve()