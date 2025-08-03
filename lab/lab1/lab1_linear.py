import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x = np.array([1.23, 1.79, 2.24, 2.76, 3.20, 3.68, 4.16, 4.64, 5.22])
y = np.array([2.21, 2.84, 3.21, 3.96, 4.86, 6.06, 7.47, 9.25, 12.3])

def linear(x, a, b):
    return a * x + b

popt_linear, _ = curve_fit(linear, x, y)

y_pred = linear(x, *popt_linear)

sse_linear = np.sum((y - y_pred) ** 2)

plt.scatter(x, y, color='black', label="Data Points")
plt.plot(x, y_pred, color='red', label="Linear Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression")
plt.show()

print(f"Linear Regression: y = {popt_linear[0]:.4f}x + {popt_linear[1]:.4f}")
print(f"SSE = {sse_linear:.4f}")
