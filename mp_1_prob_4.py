import numpy as np
import matplotlib.pyplot as plt

########## A simple problem that has limit points can be written as follows:    ##################

############################ N(d)−F= 0  ###############################################################

############################ N(d) = 0.2d^3−2.1d^2+ 6    #############################################
def plot_exact_solution(up_to_where):
    func_vals=[]
    steps=[]
    for step in np.arange(0,up_to_where,0.1):
        
        func_vals.append(myfunc(step))
        steps.append(step)


    return steps,func_vals

def Leo_fixed_point(func, x0,delF, tol=10e-8, max_iter=100):


    status='Not converged.'
    x_prev = np.array(x0,dtype=float)
    print(x_prev)
    for i in np.arange(0,2.6,0.001):
        x_next = np.array(func(i),dtype=float)

        print('Iter =',i,'d=',x_prev,' N(d) =',x_next,'diff=',x_next - delF)
        if abs(x_next - delF) < tol:
            print('solved!')

            return (x_prev)
        x_prev = x_next
        
def myfunc(x):
    return 0.2*x**3-2.1*x**2+6*x

right_boundary=2.6  

initial_displacement=0.01
initial_force=0.0

force_increment_step=0.1
displacement_range=np.arange(initial_displacement,right_boundary,force_increment_step)

Leo_fixed_point(myfunc,initial_displacement,force_increment_step,)

print(plot_exact_solution(2.6))
plt.plot(plot_exact_solution(2.6))
plt.show()
# d=Leo_fixed_point(func,[d0],0.1,)
# print('d=',d)
