import operator
from functools import reduce
from typing import Dict
import numpy as np


def compute_forward_diff_table(x_vals: np.ndarray, y_vals: np.ndarray) -> Dict:
    """
    Generates a forward difference table for provided x and y values.
    """
    if len(x_vals) != len(y_vals):
        raise ValueError("x and y arrays must be of equal length")

    n = len(x_vals)
    diff_matrix = np.zeros((n, n))
    diff_matrix[:, 0] = y_vals

    for col in range(1, n):
        for row in range(n - col):
            diff_matrix[row, col] = diff_matrix[row + 1, col - 1] - diff_matrix[row, col - 1]

    table = []
    for i in range(n):
        row_data = {'x': x_vals[i], 'f': y_vals[i]}
        for j in range(1, n - i):
            row_data[f'Δ{j}f'] = diff_matrix[i, j]
        table.append(row_data)

    return {'table': table, 'diff_matrix': diff_matrix}


def newton_forward_interpolate(x_target: float, data: Dict) -> float:
    """
    Performs interpolation using Newton's Forward Interpolation method.
    """
    x_data = np.array([row['x'] for row in data['table']])
    diff_matrix = data['diff_matrix']
    h = x_data[1] - x_data[0]
    u = (x_target - x_data[0]) / h

    result = diff_matrix[0, 0]
    u_term = 1

    for i in range(1, len(x_data)):
        u_term *= (u - i + 1) / i
        result += u_term * diff_matrix[0, i]

    return result


def compute_backward_diff_table(x_vals: np.ndarray, y_vals: np.ndarray) -> Dict:
    """
    Generates a backward difference table for provided x and y values.
    """
    if len(x_vals) != len(y_vals):
        raise ValueError("x and y arrays must be of equal length")

    n = len(x_vals)
    diff_matrix = np.zeros((n, n))
    diff_matrix[:, 0] = y_vals

    for col in range(1, n):
        for row in range(n - 1, col - 1, -1):
            diff_matrix[row, col] = diff_matrix[row, col - 1] - diff_matrix[row - 1, col - 1]

    return {'diff_table': diff_matrix, 'x': x_vals, 'y': y_vals}


def newton_backward_interpolate(x_target: float, data: Dict) -> float:
    """
    Performs interpolation using Newton's Backward Interpolation method.
    """
    x_data = data['x']
    diff_matrix = data['diff_table']
    n = len(x_data)
    h = x_data[1] - x_data[0]
    u = (x_target - x_data[n - 1]) / h

    result = diff_matrix[n - 1, 0]
    u_term = 1

    for i in range(1, n):
        u_term *= (u + i - 1) / i
        result += u_term * diff_matrix[n - i - 1, i]

    return result


def central_diff_interpolate(x_target: float, x_data: np.ndarray, y_data: np.ndarray) -> float:
    """
    Performs interpolation using Central Difference Interpolation method.
    """
    h = x_data[1] - x_data[0]
    k = len(x_data) // 2
    delta_yk = (y_data[k + 1] - y_data[k - 1]) / 2
    return y_data[k] + ((x_target - x_data[k]) / h) * delta_yk


def lagrange_interpolate(x_target: float, x_data: np.ndarray, y_data: np.ndarray) -> float:
    """
    Performs interpolation using Lagrange Interpolation method.
    """
    n = len(x_data)
    result = 0.0

    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if j != i:
                term *= (x_target - x_data[j]) / (x_data[i] - x_data[j])
        result += term

    return result


def display_diff_table(result: Dict, method: str = 'forward') -> None:
    """Displays the difference table in a formatted manner."""
    table = result['table']
    symbol = 'Δ' if method == 'forward' else '∇'

    max_diff = max(len(row) - 2 for row in table)
    headers = ['x', 'f'] + [f'{symbol}{i}f' for i in range(1, max_diff + 1)]
    header_str = '|'.join(f'{h:^12}' for h in headers)
    print('-' * len(header_str))
    print(header_str)
    print('-' * len(header_str))

    for row in table:
        row_values = [f"{row['x']:^12.4f}", f"{row['f']:^12.4f}"]
        for i in range(1, max_diff + 1):
            key = f'{symbol}{i}f'
            row_values.append(f"{row[key]:^12.4f}" if key in row else ' ' * 12)
        print('|'.join(row_values))
    print('-' * len(header_str))


def product_func(x: float, terms: range) -> float:
    """Computes the product function of the form (2x + 1)(2x + 3)...(2x + n)."""
    return reduce(operator.mul, ((2 * x + i) for i in terms))


def execute_tasks():
    print("\n=== Tasks for Forward and Backward Difference ===")

    print("\nTask 1: Forward Difference Table")
    x1 = np.array([10, 20, 30, 40])
    y1 = np.array([1.1, 2.0, 4.4, 7.9])
    result1 = compute_forward_diff_table(x1, y1)
    display_diff_table(result1)

    print("\nTask 2: Forward Difference Table and Δ³f(2)")
    x2 = np.array([0, 1, 2, 3, 4])
    y2 = np.array([1.0, 1.5, 2.2, 3.1, 4.6])
    result2 = compute_forward_diff_table(x2, y2)
    display_diff_table(result2)
    print(f"Δ³f(2) = {result2['diff_matrix'][2, 3]}")

    print("\nTask 3: Polynomial y = x³ + x² - 2x + 1")
    def f3(x): return x ** 3 + x ** 2 - 2 * x + 1
    x3 = np.array([0, 1, 2, 3, 4, 5])
    y3 = np.array([f3(x) for x in x3])
    result3 = compute_forward_diff_table(x3, y3)
    display_diff_table(result3)
    actual_y6 = f3(6)
    interpolated_y6 = newton_forward_interpolate(6, result3)
    print(f"Actual y(6) = {actual_y6}")
    print(f"Interpolated y(6) = {interpolated_y6}")

    print("\nTask 4: f(x) = x³ + 5x - 7")
    def f4(x): return x ** 3 + 5 * x - 7
    x4 = np.array([-1, 0, 1, 2, 3, 4, 5])
    y4 = np.array([f4(x) for x in x4])
    result4 = compute_forward_diff_table(x4, y4)
    display_diff_table(result4)
    actual_y6 = f4(6)
    interpolated_y6 = newton_forward_interpolate(6, result4)
    print(f"Actual y(6) = {actual_y6}")
    print(f"Interpolated y(6) = {interpolated_y6}")

    print("\nTask 5: Extended Table")
    x5 = np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    y5 = np.array([2.6, 3.0, 3.4, 4.28, 7.08, 14.2, 29.0])
    result5 = compute_forward_diff_table(x5, y5)
    first_x = -0.4
    last_x = 1.2
    first_y = newton_forward_interpolate(first_x, result5)
    last_y = newton_forward_interpolate(last_x, result5)
    x_extended = np.insert(x5, 0, first_x)
    x_extended = np.append(x_extended, last_x)
    y_extended = np.insert(y5, 0, first_y)
    y_extended = np.append(y_extended, last_y)
    result_extended = compute_forward_diff_table(x_extended, y_extended)
    display_diff_table(result_extended)

    print("\nTask 6: Find Missing Term")
    x6 = np.array([2, 3, 5])
    y6 = np.array([45.0, 49.2, 59.6999])
    result6 = compute_forward_diff_table(x6, y6)
    missing_value = lagrange_interpolate(4, x6, y6)
    print(f"Missing value = {missing_value}")

    print("\nTask 7: Product Function")
    def f7(x): return product_func(x, range(1, 16, 2))
    x7 = np.array([0, 1, 2, 3, 4])
    y7 = np.array([f7(x) for x in x7])
    result7 = compute_forward_diff_table(x7, y7)
    display_diff_table(result7)

    print("\nTask 8: Polynomial of Degree 4")
    x8 = np.array([0, 1, 2, 3, 4])
    y8 = np.array([1, -1, 1, -1, 1])
    result8 = compute_forward_diff_table(x8, y8)
    y_5 = newton_forward_interpolate(5, result8)
    x8 = np.append(x8, 5)
    y8 = np.append(y8, y_5)
    y_6 = lagrange_interpolate(6, x8, y8)
    x8 = np.append(x8, 6)
    y8 = np.append(y8, y_6)
    y_7 = lagrange_interpolate(7, x8, y8)
    x8 = np.append(x8, 7)
    y8 = np.append(y8, y_7)
    print("After adding x=5, 6, 7:", x8, y8)

    print("\nTask 9: Another Product Function")
    def f9(x): return product_func(x, range(1, 20, 2))
    x9 = np.array([0, 1, 2, 3, 4])
    y9 = np.array([f9(x) for x in x9])
    result9 = compute_forward_diff_table(x9, y9)
    display_diff_table(result9)


if __name__ == "__main__":
    execute_tasks()
