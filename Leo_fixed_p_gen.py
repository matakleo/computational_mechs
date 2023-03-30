import numpy as np
import matplotlib.pyplot as plt
from across_files_funcs import fixed_point_iteration,plot_exact_sol


# Define the initial values of d and lambda
d0 = 0
Forcing=np.arange(0,10,0.2)


for df in Forcing:
    
    def g(d):
        ##convert N(d)-F=0 to fixed point function
        return df/(0.2*d**2-2.1*d+6)

    d,num_of_iters,status = fixed_point_iteration(g, d0,)
    print(d,df)
    if df<5.5:
        plt.scatter(d,df,color='r')

plot_exact_sol(True)
plt.legend()
plt.show()