import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x = np.array([1.23, 1.79, 2.24, 2.76, 3.20, 3.68, 4.16, 4.64, 5.22])
y = np.array([2.21, 2.84, 3.21, 3.96, 4.86, 6.06, 7.47, 9.25, 12.3])

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

popt_quadratic, _ = curve_fit(quadratic, x, y)

y_pred = quadratic(x, *popt_quadratic)

sse_quadratic = np.sum((y - y_pred) ** 2)

plt.scatter(x, y, color='black', label="Data Points")
plt.plot(x, y_pred, color='blue', label="Quadratic Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Quadratic Regression")
plt.show()

print(f"Quadratic Regression: y = {popt_quadratic[0]:.4f}x² + {popt_quadratic[1]:.4f}x + {popt_quadratic[2]:.4f}")
print(f"SSE = {sse_quadratic:.4f}")
