from matplotlib import pyplot as plt
import sympy as sp

x = sp.symbols('x')
equation = input("Enter equation (in terms of x): ")
tolerance = float(input("Enter tolerance: "))

f_expr = sp.sympify(equation)
f_prime_expr = sp.diff(f_expr, x)

def evaluate_expression(expr, x_val):
    """Evaluate a sympy expression at a given x value."""
    return float(expr.subs(x, x_val))

def f(x_val):
    return evaluate_expression(f_expr, x_val)

def f_prime(x_val):
    return evaluate_expression(f_prime_expr, x_val)

    x = start
    while x <= end:
        if f(x) * f(x + step) < 0:
            return x, x + step
        x += step
    return find_interval(f, start=start - 10, end=end + 10, step=step)

def bisection_method(f, x1, x2, tolerance):
    iterations, midpoints, errors = [], [], []
    while x2 - x1 > tolerance:
        mid = (x1 + x2) / 2
        iterations.append(len(midpoints))
        midpoints.append(mid)
        errors.append(abs(x2 - x1))

        if f(mid) == 0 or abs(f(mid)) < tolerance:
            break
        elif f(mid) * f(x1) < 0:
            x2 = mid
        else:
            x1 = mid

    return midpoints, iterations, errors

def newton_raphson_method(f, f_prime, x0, tolerance):
    iterations, roots = [0], [x0]
    while True:
        x_next = x0 - f(x0) / f_prime(x0)
        iterations.append(len(roots))
        roots.append(x_next)

        if abs(x_next - x0) < tolerance:
            break

        x0 = x_next

    return roots, iterations

def iteration_method(f, f_prime, x0, tolerance, max_iter=100):
    iterations, roots = [0], [x0]
    for _ in range(max_iter):
        try:
            x_next = x0 - f(x0) / f_prime(x0)
            roots.append(x_next)
            iterations.append(len(roots) - 1)
            if abs(x_next - x0) < tolerance:
                break
            x0 = x_next
        except ZeroDivisionError:
            print("Division by zero encountered. Terminating iteration.")
            break

    return roots, iterations

def false_position_method(f, x1, x2, tolerance):
    iterations, roots, errors = [], [], []
    while abs(x2 - x1) > tolerance:
        x_root = x2 - (f(x2) * (x2 - x1)) / (f(x2) - f(x1))
        iterations.append(len(roots))
        roots.append(x_root)
        errors.append(abs(x2 - x1))

        if abs(f(x_root)) < tolerance:
            break
        elif f(x1) * f(x_root) < 0:
            x2 = x_root
        else:
            x1 = x_root

    return roots, iterations, errors

x1, x2 = find_interval(f)
print(f"Interval found: [{x1}, {x2}]")

bisection_results = bisection_method(f, x1, x2, tolerance)
newton_results = newton_raphson_method(f, f_prime, (x1 + x2) / 2, tolerance)
iteration_results = iteration_method(f, f_prime, (x1 + x2) / 2, tolerance)
false_position_results = false_position_method(f, x1, x2, tolerance)

# Display results
methods = ["Bisection", "Newton-Raphson", "Iteration", "False Position"]
results = [bisection_results, newton_results, iteration_results, false_position_results]

for method, (roots, iterations, *errors) in zip(methods, results):
    print(f"{method} Method: Approx. root: {roots[-1]:.6f}, Iterations: {len(iterations)}")

# Plot convergence
plt.figure(figsize=(12, 6))
for method, (roots, iterations, *errors) in zip(methods, results):
    plt.plot(iterations, roots, marker='o', label=f"{method} Method")

plt.xlabel("Iteration")
plt.ylabel("Root Estimate")
plt.title("Convergence of Root-Finding Methods")
plt.legend()
plt.grid()
plt.show()
