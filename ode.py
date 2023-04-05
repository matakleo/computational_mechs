import numpy as np

def midpoint_method_system(f, t0, y0, tf, n):
    """
    This function uses the midpoint method to solve a system of first-order ODEs
    y' = f(t, y), where y is a vector of length m, from t0 to tf with n steps.
    The initial condition is y(t0) = y0.
    """
    h = (tf - t0) / n  # step size
    t = np.linspace(t0, tf, n+1)  # time points
    y = np.zeros((n+1, len(y0)))  # initialize solution array
    y[0] = y0  # initial condition
    
    for i in range(n):
        # compute the midpoint
        tn = (t[i] + t[i+1]) / 2
        yn = y[i] + h/2 * f(t[i], y[i])
        
        # update the solution using the midpoint approximation
        y[i+1] = y[i] + h * f(tn, yn)
    
    return t, y


# example usage
f = lambda t, y: np.array([y[1], -y[0]])  # define the system of ODEs y' = f(t, y)
t0, y0 = 0, np.array([0, 1])  # initial conditions
tf, n = 10, 1000  # final time and number of steps
t, y = midpoint_method_system(f, t0, y0, tf, n)

# plot the solution
import matplotlib.pyplot as plt
plt.plot(t, y[:, 0], label='y1')
plt.plot(t, y[:, 1], label='y2')
plt.legend()
plt.show()
