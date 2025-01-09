def gauss_seidel(A, b, initial_guess, tolerance, max_iterations):
    n = len(A)
    x = initial_guess[:]

    for _ in range(max_iterations):
        x_new = x[:]

        for i in range(n):
            sum1 = 0
            for j in range(i):
                sum1 += A[i][j] * x_new[j]
            sum2 = 0
            for j in range(i + 1, n):
                sum2 += A[i][j] * x[j]
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]

        has_converged = True
        for i in range(n):
            if abs(x_new[i] - x[i]) >= tolerance:
                has_converged = False
                break

        if has_converged:
            return [round(val, 4) for val in x_new]

        x = x_new

# Example usage:
A = [
    [3, -5, 47,   20],
    [11, 16, 17,  10],
    [56, 22, 11, -18],
    [17, 66, -12,  7]
]

b = [18, 26, 34, 82]
initial_guess = [0, 0, 0, 0]
tolerance = 0.002
max_iterations = 100

solution = gauss_seidel(A, b, initial_guess, tolerance, max_iterations)
print(f"Solution: {solution}")
