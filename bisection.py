import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - x - 1

a = float(input("Enter the value of a: "))
b = float(input("Enter the value of b: "))

def plot_graphs(midpoints):
    iterations = range(1, len(midpoints) + 1)
    
    # Plot midpoint convergence
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, midpoints, 'o-', color='orange')
    plt.title("Bisection Method: Midpoint Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Midpoint")
    plt.grid()

    # Plot error reduction
    errors = [abs(midpoints[i] - midpoints[i - 1]) for i in range(1, len(midpoints))]
    plt.subplot(1, 2, 2)
    plt.plot(range(2, len(midpoints) + 1), errors, 'o-', color='blue')
    plt.title("Bisection Method: Error Reduction")
    plt.xlabel("Iteration")
    plt.ylabel("Error (Difference between midpoints)")
    plt.grid()

    plt.tight_layout()
    plt.show()
    
def bisection(func, a, b, tol=1e-5, max_iter=100):
    if func(a) * func(b) >= 0:
        print("Invalid interval: f(a) and f(b) must have opposite signs.")
        return None

    midpoints = []
    for _ in range(max_iter):
        midpoint = (a + b) / 2
        midpoints.append(midpoint)

        if abs(func(midpoint)) < tol or (b - a) / 2 < tol:
            plot_graphs(midpoints)
            return midpoint

        if func(a) * func(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint

    plot_graphs(midpoints)
    return (a + b) / 2

root = bisection(f, a, b)
if root is not None:
    print(f"Approximate root: {root}")
