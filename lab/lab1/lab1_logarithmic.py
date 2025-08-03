import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x = np.array([1.23, 1.79, 2.24, 2.76, 3.20, 3.68, 4.16, 4.64, 5.22])
y = np.array([2.21, 2.84, 3.21, 3.96, 4.86, 6.06, 7.47, 9.25, 12.3])

log_x = np.log(x)

def logarithmic(x, a, b):
    return a * np.log(x) + b

popt_logarithmic, _ = curve_fit(logarithmic, x, y)

y_pred = logarithmic(x, *popt_logarithmic)

sse_logarithmic = np.sum((y - y_pred) ** 2)

plt.scatter(x, y, color='black', label="Data Points")
plt.plot(x, y_pred, color='brown', label="Logarithmic Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Logarithmic Regression")
plt.show()

print(f"Logarithmic Regression: y = {popt_logarithmic[0]:.4f} * log(x) + {popt_logarithmic[1]:.4f}")
print(f"SSE = {sse_logarithmic:.4f}")
