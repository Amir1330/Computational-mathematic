import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

@dataclass
class FitResult:
    name: str
    equation: str
    coefficients: np.ndarray
    error: float
    function: Callable

class CurveFitter:
    def __init__(self):
        self.fits = {
            'linear': self._fit_linear,
            'quadratic': self._fit_quadratic,
            'exponential': self._fit_exponential,
            'quad_origin': self._fit_quad_origin,
            'rational': self._fit_rational
        }

    def _exp_function(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * np.exp(b * x)

    def _quad_origin_function(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x + b * x ** 2

    def _rational_function(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        with np.errstate(divide='raise', invalid='raise'):
            try:
                safe_x = np.where(x != 0, x, np.inf)
                return a * x + b / safe_x
            except FloatingPointError:
                return np.full_like(x, np.inf)

    def _calculate_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(np.sum((y_true - y_pred) ** 2))

    def _fit_linear(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        coeffs = np.polyfit(x, y, 1)
        poly = np.poly1d(coeffs)
        error = self._calculate_error(y, poly(x))
        equation = f"{coeffs[0]:.2f}x + {coeffs[1]:.2f}"
        return FitResult('Linear', equation, coeffs, error, poly)

    def _fit_quadratic(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        coeffs = np.polyfit(x, y, 2)
        poly = np.poly1d(coeffs)
        error = self._calculate_error(y, poly(x))
        equation = f"{coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}"
        return FitResult('Quadratic', equation, coeffs, error, poly)

    def _fit_exponential(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        try:
            coeffs, _ = curve_fit(self._exp_function, x, y)
            y_fit = self._exp_function(x, *coeffs)
            error = self._calculate_error(y, y_fit)
            equation = f"{coeffs[0]:.2f}e^({coeffs[1]:.2f}x)"
            return FitResult('Exponential', equation, coeffs, error,
                           lambda x: self._exp_function(x, *coeffs))
        except:
            return FitResult('Exponential', 'Failed to fit', np.array([]), float('inf'),
                           lambda x: np.zeros_like(x))

    def _fit_quad_origin(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        try:
            coeffs, _ = curve_fit(self._quad_origin_function, x, y)
            y_fit = self._quad_origin_function(x, *coeffs)
            error = self._calculate_error(y, y_fit)
            equation = f"{coeffs[0]:.2f}x + {coeffs[1]:.2f}x²"
            return FitResult('Quadratic through origin', equation, coeffs, error,
                           lambda x: self._quad_origin_function(x, *coeffs))
        except:
            return FitResult('Quadratic through origin', 'Failed to fit', np.array([]), float('inf'),
                           lambda x: np.zeros_like(x))

    def _fit_rational(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        if np.any(x == 0):
            return FitResult('Rational', 'Cannot fit - contains x=0', np.array([]), float('inf'),
                           lambda x: np.zeros_like(x))
        try:
            coeffs, _ = curve_fit(self._rational_function, x, y,
                                bounds=([-np.inf, -1e6], [np.inf, 1e6]))
            y_fit = self._rational_function(x, *coeffs)
            error = self._calculate_error(y, y_fit)
            equation = f"{coeffs[0]:.2f}x + {coeffs[1]:.2f}/x"
            return FitResult('Rational', equation, coeffs, error,
                           lambda x: self._rational_function(x, *coeffs))
        except:
            return FitResult('Rational', 'Failed to fit', np.array([]), float('inf'),
                           lambda x: np.zeros_like(x))

    def fit_all(self, x: np.ndarray, y: np.ndarray) -> Dict[str, FitResult]:
        return {name: method(x, y) for name, method in self.fits.items()}

    def find_best_fit(self, results: Dict[str, FitResult]) -> Tuple[str, FitResult]:
        valid_fits = {k: v for k, v in results.items() if v.error != float('inf')}
        if not valid_fits:
            return "None", FitResult("None", "No valid fits", np.array([]), float('inf'),
                                   lambda x: np.zeros_like(x))
        return min(valid_fits.items(), key=lambda x: x[1].error)

    def plot_all_fits(self, test_cases: list, results_list: list):
        fig, axs = plt.subplots(5, 1, figsize=(12, 25))
        fig.suptitle('All Fitting Methods Comparison', fontsize=16, y=0.95)

        for idx, ((x, y), results) in enumerate(zip(test_cases, results_list)):
            ax = axs[idx]
            ax.scatter(x, y, color='red', label='Data', zorder=5)

            x_smooth = np.linspace(min(x), max(x), 100)
            for result in results.values():
                if result.error != float('inf'):
                    try:
                        y_smooth = result.function(x_smooth)
                        if np.all(np.isfinite(y_smooth)):
                            ax.plot(x_smooth, y_smooth,
                                  label=f'{result.name}: {result.equation}')
                    except:
                        continue

            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)
            ax.set_title(f'Test Case {idx + 1}')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    fitter = CurveFitter()

    # Test cases
    test_cases = [
        (np.array([1, 2, 3, 4, 5]), np.array([2.1, 4.2, 5.8, 8.1, 9.9])),

        # (np.array([0, 1, 2, 3, 4, 5]), np.array([1, 2.1, 5.3, 10.2, 17.1, 25.8])),

        # (np.array([0, 1, 2, 3, 4]), np.array([1.1, 2.9, 7.8, 20.2, 54.6])),

        # (np.array([1, 2, 3, 4, 5]), np.array([10, 5.2, 3.8, 3.1, 2.7])),

        # (np.array([0, 1, 2, 3, 4]), np.array([0, 1.1, 4.2, 9.1, 16.2]))
    ]

    # Store all results
    all_results = []

    for i, (x, y) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}")
        print("=" * 40)

        results = fitter.fit_all(x, y)
        all_results.append(results)
        best_method, best_result = fitter.find_best_fit(results)

        print("Fitting errors:")
        for name, result in results.items():
            if result.error == float('inf'):
                print(f"{result.name}: Failed to fit")
            else:
                print(f"{result.name}: Error = {result.error:.4f}")

        print(f"\nBest fit: {best_result.name}")
        print(f"Equation: y = {best_result.equation}")
        print(f"Error: {best_result.error:.4f}")

    # Plot all test cases in one figure
    fitter.plot_all_fits(test_cases, all_results)
