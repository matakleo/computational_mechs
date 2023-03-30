import numpy as np
import matplotlib.pyplot as plt

# define the function that describes the relationship between load and displacement
def N(d):
    return 0.2*d**3 - 2.1*d**2 + 6*d

# define the fixed point iteration function
def fixed_point_iteration(g, x0, tol=1e-8, max_iter=100):
    x = x0
    for i in range(max_iter):
        x_new = g(x)
        print(x)
        if abs(x_new - x) < tol:
            return x_new, i+1
        x = x_new
    return x, max_iter

# define the function for the fixed point iteration
# def g(x):
#     return x + 0.05*(N(x)/100)
def g(x):
    return x + 0.05*(N(x)/100)

# set the range of displacements
d = np.arange(0, 8, 0.5)

# use the fixed point iteration to solve for the displacement at each point in the range
x = np.zeros_like(d)
for i in range(len(d)):
    x[i], _ = fixed_point_iteration(g, d[i])

# plot the load-displacement curve
plt.plot(x, N(x))
plt.xlabel("Displacement (mm)")
plt.ylabel("Load (N)")
plt.title("Load-Displacement Curve")
plt.show()
