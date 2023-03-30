import numpy as np
import matplotlib.pyplot as plt
from across_files_funcs import modified_newton_raphson,plot_exact_sol,Nprime


# Define the initial values of d and lambda
d0 = 0
Forcing=np.arange(0,8,0.2)
def dN(d):
    return 0.6*d**2-4.2*d+6

for df in Forcing:
    
    def g(d):
        ##convert N(d)-F=0 to fixed point function
        return 0.2*d**3-2.1*d**2+6*d-df

    d,num_of_iters,status = modified_newton_raphson(g,Nprime,0.5)
    print(d,df)
    # if df<5.5:
    plt.scatter(d,df,color='r')

plot_exact_sol(True)
plt.legend()
plt.show()