import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
def fixed_point(f, x0, tolerance):
    while True:
        x = f(x)
        if abs(x - x0) < tolerance:
            return x

def f(x):
# Define the nonlinear equation
    return 0.2*x**3 - 2.1*x**2 + 6*x

def main():
    

    # Define the initial guess
    x0 = 0

    # Define the tolerance
    tolerance = 1e-6

    # Solve the equation
    x = fixed_point(f, x0, tolerance)

    # Define the load increment
    delF = 0.2

    # Define the load-displacement curve
    load = []
    displacement = []

    # Iterate through the displacement values
    for d in np.arange(0, 8):
        # Calculate the load
        load.append(delF * x)

        # Calculate the displacement
        displacement.append(d)

        # Update the guess
        x = f(x)

    # Plot the load-displacement curve
    plt.plot(displacement, load)

    # Add labels to the plot
    plt.xlabel("Displacement")
    plt.ylabel("Load")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()