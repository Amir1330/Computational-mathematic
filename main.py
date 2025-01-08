import numpy as np
import pandas as pd
from tabulate import tabulate

# Define the coefficient matrix A and the right-hand side vector B
A = np.array([
    [5, 2, 1],
    [-1, 4, 2],
    [2, -3, 10]
])

B = np.array([7, 3, -1])

# Cramer's Rule
def cramer_rule(A, B):
    det_A = np.linalg.det(A)
    if det_A == 0:
        return "No unique solution (det(A) = 0)"
    solutions = []
    for i in range(len(B)):
        Ai = np.copy(A)
        Ai[:, i] = B
        solutions.append(np.linalg.det(Ai) / det_A)
    return np.array(solutions)

# Gauss Elimination
def gauss_elimination(A, B):
    return np.linalg.solve(A, B)

# Jacobi Method with iterations
def jacobi_iterations(A, B, tol=1e-10, max_iterations=100):
    n = len(B)
    x = np.zeros_like(B, dtype=np.double)
    results = {"Iteration": [], "x_i": [], "y_i": [], "z_i": [], "Error xi": [], "Error yi": [], "Error zi": []}
    for k in range(max_iterations):
        x_new = np.zeros_like(x, dtype=np.double)
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (B[i] - s1 - s2) / A[i, i]
        errors = np.abs(x_new - x)
        results["Iteration"].append(k)
        results["x_i"].append(x_new[0])
        results["y_i"].append(x_new[1])
        results["z_i"].append(x_new[2])
        results["Error xi"].append(errors[0])
        results["Error yi"].append(errors[1])
        results["Error zi"].append(errors[2])
        if np.allclose(x, x_new, atol=tol):
            break
        x = x_new
    return pd.DataFrame(results)

# Gauss-Seidel Method with iterations
def gauss_seidel_iterations(A, B, tol=1e-10, max_iterations=100):
    n = len(B)
    x = np.zeros_like(B, dtype=np.double)
    results = {"Iteration": [], "x_i": [], "y_i": [], "z_i": [], "Error xi": [], "Error yi": [], "Error zi": []}
    for k in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (B[i] - s1 - s2) / A[i, i]
        errors = np.abs(x_new - x)
        results["Iteration"].append(k)
        results["x_i"].append(x_new[0])
        results["y_i"].append(x_new[1])
        results["z_i"].append(x_new[2])
        results["Error xi"].append(errors[0])
        results["Error yi"].append(errors[1])
        results["Error zi"].append(errors[2])
        if np.allclose(x, x_new, atol=tol):
            break
        x = x_new
    return pd.DataFrame(results)

# Solve using direct methods
cramer_solution = cramer_rule(A, B)
gauss_solution = gauss_elimination(A, B)

# Solve using iterative methods
jacobi_results = jacobi_iterations(A, B)
gauss_seidel_results = gauss_seidel_iterations(A, B)

# Adjust Pandas display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)       # Increase display width

# Display direct method results
print("Cramer's Rule Solution:", cramer_solution)
print("Gauss Elimination Solution:", gauss_solution)

# Display iterative method results using tabulate
print("\nJacobi Method Iterations:")
print(tabulate(jacobi_results, headers='keys', tablefmt='grid'))

print("\nGauss-Seidel Method Iterations:")
print(tabulate(gauss_seidel_results, headers='keys', tablefmt='grid'))

# Export results to CSV for better viewing
jacobi_results.to_csv('jacobi_iterations.csv', index=False)
gauss_seidel_results.to_csv('gauss_seidel_iterations.csv', index=False)

print("\nResults have been saved to 'jacobi_iterations.csv' and 'gauss_seidel_iterations.csv'.")
