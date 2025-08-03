import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x = np.array([1.23, 1.79, 2.24, 2.76, 3.20, 3.68, 4.16, 4.64, 5.22])
y = np.array([2.21, 2.84, 3.21, 3.96, 4.86, 6.06, 7.47, 9.25, 12.3])

log_y = np.log(y)

def exponential(x, a, b):
    return a * np.exp(b * x)

popt_exponential, _ = curve_fit(exponential, x, y)

y_pred = exponential(x, *popt_exponential)

sse_exponential = np.sum((y - y_pred) ** 2)

plt.scatter(x, y, color='black', label="Data Points")
plt.plot(x, y_pred, color='orange', label="Exponential Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Exponential Regression")
plt.show()

print(f"Exponential Regression: y = {popt_exponential[0]:.4f} * exp({popt_exponential[1]:.4f} * x)")
print(f"SSE = {sse_exponential:.4f}")
