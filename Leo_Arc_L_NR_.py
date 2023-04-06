import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import scipy.optimize as opt

    # Define function
def f(x):
    return 0.2*x**3 - 2.1*x**2 + 6*x

def plot_exact_sol(boolean):
    if boolean:
        lst_to_plot=[]
        x=np.linspace(0,8.1,81)
        for d in x:
            lst_to_plot.append(f(d))
        plt.plot(x,lst_to_plot)   
        plt.vlines(x = 2.5, ymin = 0, ymax = 5,
           colors = 'black',
           ) 

    return

def Arc_Length_Method(x_initial=[0,0],nsteps=100,tol=1e-8,F_ext=15,radius=0.1):
    """
    Arc length method to compute the load-displacement curve for a maximum external load of F.
    Uses scipy.optimize.fsolve by giving it system of equations to solve and the jacobian of the equations.

        >>>Arc_Length_Method(nsteps=45,tol=1e-8,F_ext=15,radius=0.2)
        displacement = 0.00 solved for y*F= 0.00 converged in 7 iters.
        displacement = 0.09 solved for y*F= 0.54 converged in 7 iters.
        displacement = 0.19 solved for y*F= 1.05 converged in 7 iters.
        .
        .
        .
        displacement = 7.63 solved for y*F= 12.40 converged in 7 iters.
        displacement = 7.72 solved for y*F= 13.18 converged in 7 iters.
        displacement = 7.80 solved for y*F= 13.99 converged in 7 iters.


    written by Leo M.
    """

    results = [x_initial]

    for i in range (nsteps):

        def eq_system(x):
            return f(x[0])-x[1]*F_ext, (x[1]-x_initial[1])**2+(x[0]-x_initial[0])**2- radius**2
        
        def jacob_eq(x):
            return [[0.6*x[0]**2 - 4.2*x[0] + 6,-F_ext],[2*(x[0]-x_initial[0]),2*(x[1]-x_initial[1])]]

            ##break the loop if gamma is to become more than 1! y E [0,1] !
        if opt.fsolve(eq_system,np.array(x_initial)+0.1,fprime=jacob_eq)[1]>1:
            break
        
        results.append(opt.fsolve(eq_system,np.array(x_initial)+0.1,fprime=jacob_eq))
        print('displacement = {:.2f}'.format(x_initial[0]),'solved for y*F= {:.2f}'.format(results[i][1]*F_ext),'converged in',opt.fsolve(eq_system,np.array(x_initial)+0.1,fprime=jacob_eq,full_output=1)[1]["nfev"],'iters.')
        x_initial = results[i+1]
        

        plt.scatter(results[i][0],results[i][1]*F_ext,color='red')
    return()
Arc_Length_Method()
plot_exact_sol(True)
plt.legend(['exact soln','arc length method'])
plt.show()