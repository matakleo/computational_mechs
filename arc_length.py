import numpy as np
import matplotlib.pyplot as plt

# define the function that describes the relationship between load and displacement
def N(d):
    return 0.2*d**3 - 2.1*d**2 + 6*d

# define the function that describes the relationship between displacement and arc length
def F(d, l):
    return np.sqrt((d[1]-d[0])**2 + (N(d[1])-N(d[0]))**2) - l

# define the partial derivative of F with respect to displacement d for the arc-length method
def F_d(d, l):
    return -(d[1]-d[0])*N(d[0])/((d[1]-d[0])**2 + (N(d[1])-N(d[0]))**2)**0.5

# define the partial derivative of F with respect to arc length l for the arc-length method
def F_l(d, l):
    return -1

# define the arc-length method function
def arc_length(f, f_d, f_l, d0, l0, step_size, tol=1e-8, max_iter=100):
    d = d0
    l = l0
    for i in range(max_iter):
        fd = f_d(d, l)
        fl = f_l(d, l)
        delta_d = step_size*fd / np.sqrt(fd**2 + step_size**2*fl**2)
        delta_l = step_size*fl / np.sqrt(fd**2 + step_size**2*fl**2)
        d_new = d + delta_d
        l_new = l + delta_l
        if np.sqrt((d_new - d)**2 + (l_new - l)**2) < tol:
            return d_new, l_new, i+1
        d = d_new
        l = l_new
    return d, l, max_iter

# set the range of forces
F = np.linspace(0, 15, 10)

# initialize the displacement and arc-length
d = np.zeros_like(F)
l = np.zeros_like(F)
d[0] = 0
l[0] = 0

# use the arc-length method to solve for the displacement at each point in the range
for i in range(len(F)-1):
    d[i+1], l[i+1], _ = arc_length(lambda d,l: F(d,l), lambda d,l: F_d(d,l), lambda d,l: F_l(d,l), [d[i], d[i+1]], l[i], 0.1)

# plot the load-displacement curve
plt.plot(d, N(d))
plt.xlabel("Displacement (mm)")
plt.ylabel("Load (N)")
plt.title("Load-Displacement Curve")
plt.show()
