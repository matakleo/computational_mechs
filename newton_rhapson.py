import numpy as np
import matplotlib.pyplot as plt

# define the function that describes the relationship between load and displacement
def N(d):
    return 0.2*d**3 - 2.1*d**2 + 6*d

def g(x):
    return x + 0.05*(N(x)/100)

# define the derivative of g(x) for the Newton-Raphson method
def g_prime(x):
    return 1 + 0.05*0.6*x**2 - 0.05*4.2*x

# define the Newton-Raphson method function
def newton_raphson(f, f_prime, x0, tol=1e-8, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        fx_prime = f_prime(x)
        x_new = x - fx/fx_prime
        if abs(x_new - x) < tol:
            return x_new, i+1
        x = x_new
    return x, max_iter

# set the range of displacements
d = np.linspace(0, 8.1, 10)

# use the Newton-Raphson method to solve for the displacement at each point in the range
x = np.zeros_like(d)
for i in range(len(d)):
    x[i], _ = newton_raphson(lambda x: g(x) - x, g_prime, d[i])

# plot the load-displacement curve
plt.plot(x, N(x))
plt.xlabel("Displacement (mm)")
plt.ylabel("Load (N)")
plt.title("Load-Displacement Curve")
plt.show()
