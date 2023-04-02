import numpy as np
import matplotlib.pyplot as plt
# from across_files_funcs import fixed_point_iteration,plot_exact_sol

def plot_exact_sol(boolean):
    if boolean:
        lst_to_plot=[]
        x=np.linspace(0,8.1,81)
        for d in x:
            lst_to_plot.append(N(d))
        plt.plot(x,lst_to_plot)   
        plt.vlines(x = 2.5, ymin = 0, ymax = 5,
           colors = 'black',
           ) 

    return

def N(x):
    return np.array(0.2*x**3-2.1*x**2+6*x)


def Nprime(x):
    return np.array(0.6*x**2 - 4.2*x + 6)    
# Define the initial values of d and lambda
d0 = 0
Forcing=np.arange(0,10,0.4)


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

for df in Forcing:
    
    def g(d):
        ##convert N(d)-F=0 to fixed point function
        return df/(0.2*d**2-2.1*d+6)

    d,num_of_iters,status = Leos_fixed_point(g, d0,)
    print('force ='+str(df),'displacement = '+str(d),num_of_iters,status)
  
    plt.scatter(d,df,color='r')

plot_exact_sol(True)
plt.legend(['Exact soln','Fixed point'])
plt.title('Fixed Point')
plt.show()