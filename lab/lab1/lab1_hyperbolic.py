import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x = np.array([1.23, 1.79, 2.24, 2.76, 3.20, 3.68, 4.16, 4.64, 5.22])
y = np.array([2.21, 2.84, 3.21, 3.96, 4.86, 6.06, 7.47, 9.25, 12.3])

def hyperbolic(x, a, b):
    return a / x + b

popt_hyperbolic, _ = curve_fit(hyperbolic, x, y)

y_pred = hyperbolic(x, *popt_hyperbolic)

sse_hyperbolic = np.sum((y - y_pred) ** 2)

plt.scatter(x, y, color='black', label="Data Points")
plt.plot(x, y_pred, color='green', label="Hyperbolic Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Hyperbolic Regression")
plt.show()

print(f"Hyperbolic Regression: y = {popt_hyperbolic[0]:.4f}/x + {popt_hyperbolic[1]:.4f}")
print(f"SSE = {sse_hyperbolic:.4f}")