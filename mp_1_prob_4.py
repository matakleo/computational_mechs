import numpy as np
import matplotlib.pyplot as plt

########## A simple problem that has limit points can be written as follows:    ##################

############################ N(d)−F= 0  ###############################################################

############################ N(d) = 0.2d^3−2.1d^2+ 6    #############################################
#tolerance <10e-8
#maxIter <100

##  a)
## Use load control with load step δF≤0.5 to obtain the force-displacement curve in the range of d ∈[0,2.5] using:
#(i)  Fixed-point method

# Given_Func=lambda d:[0.2*d[0]**3-2.1*d[0]+6]
# d0=0.1
# Leo_fixed_point(Given_Func, [d0], tol=1e-8, max_iter=100)




# def N(d):
#     return 0.2*d**3 - 2.1*d**2 + 6*d

# def get_displacement(F, d0):
#     # solve for the displacement using the current load value and the previous displacement value
#     d = F + d0
#     return d

# d0 = 0  # initial displacement value
# delta_F = 0.1  # load step
# tolerance = 1e-6  # tolerance for convergence


# load_values = [0]  # list to store load values
# displacement_values = [d0]  # list to store displacement values

# while displacement_values[-1] <= 2.5:  # stop when the displacement reaches 2.5
#     F_new = (displacement_values[-1] - d0) + delta_F  # apply a load of k(d_new - d0) + δF
#     load_values.append(F_new)
    
#     d_new = get_displacement(F_new, d0)  # solve for the new displacement using the current load value and the previous displacement value
#     displacement_values.append(d_new)
    
#     delta_d = displacement_values[-1] - displacement_values[-2]  # calculate the difference in displacement
    
#     if abs(delta_d) < tolerance:  # check for convergence
#         break
        
#     d0 = displacement_values[-1]  # update the previous displacement value

# # Plot the force-displacement curve
# plt.plot(displacement_values, load_values, 'bo-')
# plt.xlabel('Displacement (d)')
# plt.ylabel('Load (F)')
# plt.title('Force-Displacement Curve with Fixed-Point Method and Load Control')
# plt.show()



def Leo_fixed_point(func, x0,delF, tol=10e-2, max_iter=100):


    status='Not converged.'
    x_prev = np.array(x0,dtype=float)
    print(x_prev)
    for i in range(max_iter):
        x_next = np.array(func(x_prev),dtype=float)

        print('Iter =',i,'d=',x_prev,' N(d) =',x_next,'diff=',x_next - delF)
        if abs(x_next - delF) < tol:

            return (x_prev)
        x_prev = x_next
        
func=lambda x: np.array([0.2*x[0]**3 - 2.1*x[0]**2 + 6*x[0]])
d0=0.000001
for delF in [0.1,0.2,0.3,0.4] :
    print(delF)
    if delF==0.1:
        d=Leo_fixed_point(func,[d0],delF,)
        print('d=',d)
    else:
        d=Leo_fixed_point(func,[d],delF,)
        print('d=',d)

# d=Leo_fixed_point(func,[d0],0.1,)
# print('d=',d)