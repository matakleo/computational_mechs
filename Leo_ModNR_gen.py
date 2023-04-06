import numpy as np
import matplotlib.pyplot as plt
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
    
def N(x):
    return np.array(0.2*x**3-2.1*x**2+6*x)


def Nprime(x):
    return np.array(0.6*x**2 - 4.2*x + 6)





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

    writen by Leo M.    
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

# Define the initial values of d and lambda
d0 = 0
Forcing=np.arange(0,10,0.4)


for df in Forcing:
    
    def g(d):
        ##convert N(d)-F=0 to fixed point function
        return 0.2*d**3-2.1*d**2+6*d-df

    d,num_of_iters,status = Leos_modified_newton_raphson(g,Nprime,0.1)
    print('force ={:.2f}'.format(df),'displacement = {:.2f}'.format(d),num_of_iters,status)
    # if df<5.5:
    plt.scatter(d,df,color='r')

plot_exact_sol(True)
plt.legend(['Exact soln','Fixed point',])
plt.title('Initial tangent Newton-Raphson')
plt.show()