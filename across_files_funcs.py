import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def N(x):
    return 0.2*x**3-2.1*x**2+6*x


def Nprime(x):
    return 0.6*x**2 - 4.2*x + 6


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
def fixed_point_iteration(g, x0, tol=1e-8, max_iter=100):
    x = x0
    status='not converged'
    for i in range(max_iter):
        x_new = g(x)
        # print(x,i)
        if abs(x_new - x) < tol:
            status=('converged in',i,'iters.')
            return x_new, i+1,status
        x = x_new
    return x, max_iter, status    





def newton_raphson(func,dfunc,x0, tol=1e-8, max_iter=100):
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


def modified_newton_raphson(func,dfunc,x0, tol=1e-8, max_iter=100):
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

