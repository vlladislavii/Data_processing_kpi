import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x = np.array([1.23, 1.79, 2.24, 2.76, 3.20, 3.68, 4.16, 4.64, 5.22])
y = np.array([2.21, 2.84, 3.21, 3.96, 4.86, 6.06, 7.47, 9.25, 12.3])

def linear(x, a, b):
    return a * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def hyperbolic(x, a, b):
    return a / x + b

def power_law(x, a, b):
    return a * x**b

def exponential(x, a, b):
    return a * np.exp(b * x)

def logarithmic(x, a, b):
    return a * np.log(x) + b

popt_linear, _ = curve_fit(linear, x, y)
popt_quadratic, _ = curve_fit(quadratic, x, y)
popt_hyperbolic, _ = curve_fit(hyperbolic, x, y)
popt_power, _ = curve_fit(power_law, x, y)
popt_exponential, _ = curve_fit(exponential, x, y)
popt_logarithmic, _ = curve_fit(logarithmic, x, y)

y_linear = linear(x, *popt_linear)
y_quadratic = quadratic(x, *popt_quadratic)
y_hyperbolic = hyperbolic(x, *popt_hyperbolic)
y_power = power_law(x, *popt_power)
y_exponential = exponential(x, *popt_exponential)
y_logarithmic = logarithmic(x, *popt_logarithmic)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Data Points')

x_fine = np.linspace(min(x), max(x), 100)
plt.plot(x_fine, linear(x_fine, *popt_linear), linestyle='dashed', label="Linear")
plt.plot(x_fine, quadratic(x_fine, *popt_quadratic), linestyle='dotted', label="Quadratic")
plt.plot(x_fine, hyperbolic(x_fine, *popt_hyperbolic), linestyle='dashdot', label="Hyperbolic")
plt.plot(x_fine, power_law(x_fine, *popt_power), label="Power")
plt.plot(x_fine, exponential(x_fine, *popt_exponential), label="Exponential")
plt.plot(x_fine, logarithmic(x_fine, *popt_logarithmic), label="Logarithmic")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Regression Models with Standard Transformations")
plt.grid(True)
plt.show()
