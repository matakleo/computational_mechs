import numpy as np
import matplotlib.pyplot as plt

# Define the function N(d)
def N(d):
    return 0.2 * d**3 - 2.1 * d**2 + 6 * d

# Define the fixed point iteration function
def fixed_point_iteration(g, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        x_new = g(x)
        if abs(x_new - x) < tol:
            return x_new, i+1
        x = x_new
    return x, max_iter

# Define the function for the fixed point iteration
def g(x, d, F):
    return x + F/N(d)

# Set the initial displacement value and the force increment
x0 = 0.0
delta_F = 0.1

# Create arrays to store the displacement and force values
d_values = np.arange(0, 2.6, 0.1)
x_values = np.zeros_like(d_values)

# Use the fixed point iteration to compute the displacement values for each force increment
for i, d in enumerate(d_values):
    F = i * delta_F
    g_func = lambda x: g(x, d, F)
    x, _ = fixed_point_iteration(g_func, x0)
    x_values[i] = x

# Plot the load-displacement curve
plt.plot(x_values, np.arange(0, 2.6, 0.1))
plt.xlabel('Displacement')
plt.ylabel('Load')
plt.title('Load-Displacement Curve')
plt.show()
