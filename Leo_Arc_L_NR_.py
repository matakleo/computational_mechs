import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import scipy.optimize as opt

# Define function
def f(x):
    return 0.2*x**3 - 2.1*x**2 + 6*x

F_ext=14
tol=0.001
r=0.5
n_total=85

x_initial=[0,0]

results = [x_initial]

for i in range (n_total):
    print(x_initial)    
    def eq_system(x):
        return f(x[0])-x[1]*14, (x[1]-x_initial[1])**2+(x[0]-x_initial[0])**2- 0.01
    
    def jacob_eq(x):
        return [[0.6*x[0]**2 - 4.2*x[0] + 6,-14],[2*(x[0]-x_initial[0]),2*(x[1]-x_initial[1])]]
    
    results.append(opt.fsolve(eq_system,np.array(x_initial)+0.1,fprime=jacob_eq))
    
    x_initial = results[i+1]
    

    plt.scatter(results[i][0],results[i][1]*F_ext)
plt.show()