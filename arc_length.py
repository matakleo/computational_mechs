import numpy as np
import matplotlib.pyplot as plt

# define the function that describes the relationship between load and displacement
def N(d):
    return 0.2*d**3 - 2.1*d**2 + 6*d

def g(x):
    return x + 0.05*(N(x)/100)    

# define the derivative of g(x) for the arc-length method
def g_prime(x):
    return 1 + 0.05*0.2*x**2 - 0.05*4.2*x + 0.05*0.6*x

# define the arc-length method function
def arc_length_method(f, f_prime, x0, s, tol=1e-8, max_iter=100):
    x = x0
    fx = f(x)
    fx_prime = f_prime(x)
    for i in range(max_iter):
        # compute the tangent direction
        t = fx_prime / np.sqrt(fx_prime**2 + s**2)

        # compute the next approximation
        x_new = x + t * s
        fx_new = f(x_new)

        # check for convergence
        if abs(fx_new) < tol:
            return x_new, i+1

        # compute the corrected direction
        t_new = (fx_new - fx) / (s * np.sqrt(fx_prime**2 + s**2))

        # compute the corrected step size
        ds = - fx_new / (t_new + 1e-8)

        # update the approximation and other variables
        x = x_new
        fx = fx_new
        fx_prime = f_prime(x)
        s += ds

    return x, max_iter

# set the range of displacements
d = np.linspace(0, 2.6, 10)

# use the arc-length method to solve for the displacement at each point in the range
x = np.zeros_like(d)
for i in range(len(d)):
    x[i], _ = arc_length_method(lambda x: g(x) - x, g_prime, d[i], s=0.1)

# plot the load-displacement curve
plt.plot(x, N(x))
plt.xlabel("Displacement (mm)")
plt.ylabel("Load (N)")
plt.title("Load-Displacement Curve")
plt.show()
