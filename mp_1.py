from math import *
import scipy.linalg as LA
import numpy as np


################################################################################################################
######################### Write a general program in Python for fixed-point metod. #############################
################################################################################################################


def Leo_fixed_point(func, x0, tol=1e-6, max_iter=100):
    """
    Computes the fixed point of the given function using fixed-point iteration method.

    Parameters:
    func: 
        The function whose fixed point is to be determined.
    x0: 
        Initial guess.
    tol: 
        Tolerance for the convergence criterion. Default, following your example is 1e-6.
    max_iter: int
        Maximum number of iterations allowed. Default is 100.

    Returns:
        Approximation of the fixed point of the given function.

    example:
    print(Leo_fixed_point(lambda x: np.array([0.5 * (x[0]**2 + 1.0)]),[0],0.001,333))   ---> (0.9570803282813485, 'Converged in 40 iterations.')
    
    """

    status='Not converged.'
    x_prev = x0
    for i in range(max_iter):
        x_next = np.array(func(x_prev))
        print('Iter =',i,' x=',x_next)
        if abs(x_next - x_prev) < tol:

            status=('Converged in '+str(i)+' iterations.')
            return float(x_next),status
        x_prev = x_next
    raise ValueError("Fixed point not found within the given number of iterations.")

# print(Leo_fixed_point(lambda x: np.array([0.5 * (x[0]**2 + 1.0)]),[0],0.001,333))




################################################################################################################
######################### Write a general program in Python for Newton-Raphson method. #########################
################################################################################################################


def Leo_newton_raphson(func, dfunc, x0, tol=1e-6, max_iter=100):
    """
    Computes the root of the given function using Newton-Raphson method.

    Parameters:
    func: The function whose root is to be determined.
    dfunc: The derivative of the function.
    x0: Initial guess for the root.
    tol: Tolerance for the convergence criterion. Default is 1e-6.
    max_iter: Maximum number of iterations allowed. Default is 100.
    Returns:
        Approximation of the root of the given function.
    """
    x_prev = x0
    for i in range(max_iter):
        fx = np.array(func(x_prev))
        print('fx ',float(fx))
        if abs(fx) < tol:
            return x_prev
        dfx = dfunc(x_prev)
        

        if dfx == 0:
            raise ValueError("Zero derivative. Newton-Raphson fails.")
        x_next = x_prev - fx / dfx
        print('x = ',float(x_next))
        if abs(x_next - x_prev) < tol:
            return x_next
        x_prev = x_next
    raise ValueError("Newton-Raphson fails to converge within the given number of iterations.")

# my_func = lambda x: [x[0]**2 - 3 * x[0] + 2.0]
# my_dfunc = lambda x: [[2.0 * x[0] - 3.0]]

# (Leo_newton_raphson(my_func,my_dfunc,[0.1],1.0e-10,100))



################################################################################################################
## Write  a  general  program  in  Python  for  modified  Newton-Raphson method with initial tangent.  ##########
################################################################################################################


def modified_newton_raphson(func, dfunc, x0, tol=1.0e-3, max_iter=100):
    """
    Computes the root of the given function using Modified Newton-Raphson method with initial tangent.

    Parameters:
    func: callable
        The function whose root is to be determined.
    dfunc: callable
        The derivative of the function.

    x0: float
        Initial guess for the root.
    tol: float
        Tolerance for the convergence criterion. Default is 1e-6.
    max_iter: int
        Maximum number of iterations allowed. Default is 100.

    Returns:
    float
        Approximation of the root of the given function.
    """
    x_prev = x0
    dfx = dfunc(x_prev)
    for i in range(max_iter):
        fx = np.array(func(x_prev))
        
        if abs(fx) < tol:
            return x_prev
        if dfx == 0 :
            raise ValueError("Zero derivative Modified Newton-Raphson method fails.")
        x_next = x_prev - (fx * dfx) 
        print('iteration',i,'residual',abs(x_next - x_prev))
        if abs(x_next - x_prev) < tol:
            return x_next
        x_prev = x_next
    raise ValueError("Modified Newton-Raphson method fails to converge within the given number of iterations.")


# my_func = lambda x: [x[0]**2 - 2. * x[0] + 1.]
# my_dfunc = lambda x: [[2.0 * x[0] - 2]]

# modified_newton_raphson(my_func,my_dfunc,[0.5])





