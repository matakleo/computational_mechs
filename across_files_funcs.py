import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import *

def N(x):
    return np.array(0.2*x**3-2.1*x**2+6*x)


def Nprime(x):
    return np.array(0.6*x**2 - 4.2*x + 6)


def plot_exact_sol(boolean):
    if boolean:
        lst_to_plot=[]
        x=np.linspace(0,8.1,81)
        for d in x:
            lst_to_plot.append(N(d))
        plt.plot(x,lst_to_plot,label='exact solution')   
        plt.vlines(x = 2.5, ymin = 0, ymax = 5,
           colors = 'black',
           label = '2.5 point') 

    return


# define the fixed point iteration function
def Leos_fixed_point(g, x0, tol=1e-8, max_iter=100):
    """
    The function to solve a system of nonlinear equations using 
    fixed point method withing a given tolerance.

    >>> print(Leos_fixed_point(lambda x: np.array([0.5 * (x[0]**2 + 1.0)]),[0],0.001))
    (array([0.95708033]), 41, ('converged in', 41, 'iters.'))
    """


    #initialize x
    x = x0
    status='not converged'
    #loop over the max iter number, exit if sol found!
    for i in range(max_iter):
        x_new = g(x)
        # print(x,i)
        if abs(x_new - x) < tol:
            status=('converged in',i+1,'iters.')
            return x_new, i+1,status
        x = x_new
    return x, max_iter, status    





def Leos_Newton_Raphson(func,dfunc,x0, tol=1e-8, max_iter=100):
    """
    The function to solve a system of nonlinear equations using 
    Newton-Raphson method withing a given tolerance. Required both Function and it's derivative!

    def funcxsq(x):
        return x**2
    def func2x(x):
        return 2*x

    >>> print(Leos_Newton_Raphson(funcxsq,func2x,0.1,0.001))
    (0.0007812499999999999, 7, 'converged!')

    
    """
    x = x0
    status='not converged.'
    for i in range(max_iter):
        f = func(x)
        df = dfunc(x)
        delta_x = -f / df
        x = x + delta_x
        if abs(delta_x) < tol:
            status='converged!'
            return x, i+1, status  

    return x, max_iter, status  

# Test the function with an initial guess of x = 1


def Leos_modified_newton_raphson(func,dfunc,x0, tol=1e-8, max_iter=100):

    """
    The function to solve a system of nonlinear equations using 
    initial tangent Newton-Raphson method withing a given tolerance. Requires both Function and it's derivative!

    def funcxsq(x):
        return x**2
    def func2x(x):
        return 2*x

    >>> print(Leos_modified_newton_raphson(funcxsq,func2x,0.1,0.001))
    (0.012925493446806331, 11, 'converged!')

    
    """
    x = x0
    df = dfunc(x)
    status='not converged.'
    for i in range(max_iter):
        f = func(x)
        
        delta_x = -f / df
        x = x + delta_x
        if abs(delta_x) < tol:
            status='converged!'
            return x, i+1, status  

    return x, max_iter, status  

def funcxsq(x):
    return x**2
def func2x(x):
    return 2*x

