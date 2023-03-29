import numpy as np
import matplotlib.pyplot as plt

# Define the function N(d) and its derivative N'(d)
def N(d):
    return 0.2*d*3 - 2.1*d*2 + 6*d

def N_prime(d):
    return 0.6*d**2 - 4.2*d + 6

# Define the tolerance and maximum number of iterations
tol = 1e-8
max_iter = 1000

# Define the load control parameters
delta_F = 0.1
d_range = np.arange(0, 2.6, delta_F)

# Define the initial displacement and force
d0 = 0
F0 = N(d0)

# Initialize the arrays to store the results
d = [d0]
F = [F0]

# Use the Newton-Raphson method to obtain the force-displacement curve
for i in range(len(d_range)-1):
    F_target = N(d_range[i+1])
    d_curr = d[-1]
    F_curr = F[-1]
    iter_count = 0
    while abs(F_target - F_curr) > tol and iter_count < max_iter:
        d_next = d_curr - (F_curr - F_target)/N_prime(d_curr)
        F_next = N(d_next)
        d_curr = d_next
        F_curr = F_next
        iter_count += 1
    d.append(d_curr)
    F.append(F_curr)

# Plot the force-displacement curve
plt.plot(d, F, '-o')
plt.xlabel('Displacement (mm)')
plt.ylabel('Force (kN)')
plt.title('Force-Displacement Curve (Newton-Raphson Method)')
plt.show()