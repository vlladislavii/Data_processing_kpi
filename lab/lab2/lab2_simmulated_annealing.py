import numpy as np
import time

def objective_function(x):
    A, B = 1, 6
    return (x[0] ** 2 + x[1] ** 2) / A - B * (np.cos(x[0]) + np.cos(x[1]))


def perturb_solution(x, T):
    step_size = max(0.1, T)
    new_x = x + np.random.uniform(-step_size, step_size, size=2)
    return np.clip(new_x, -8, 8)


# Simulated Annealing Algorithm with different cooling schedules
def simulated_annealing(cooling_type, max_iters=50000):
    np.random.seed(42)
    x = np.random.uniform(-8, 8, size=2)  # Initial random solution
    T = 50  # Initial temperature
    alpha = 0.999  # Exponential decay factor
    beta = 1e-6  # Quadratic decay factor
    best_x, best_f = x, objective_function(x)

    start_time = time.time()

    for iteration in range(max_iters):
        new_x = perturb_solution(x, T)
        new_f = objective_function(new_x)

        if new_f < best_f or np.exp((best_f - new_f) / T) > np.random.rand():
            x, best_f = new_x, new_f
            best_x = new_x

        # Cooling schedules
        if cooling_type == "exponential":
            T *= alpha
        elif cooling_type == "linear":
            T -= 0.001
        elif cooling_type == "quadratic":
            T = 50 / (1 + beta * iteration**2)

        if T < 1e-6:
            break

    end_time = time.time()

    print(f"{cooling_type.capitalize()} Cooling:")
    print(f"Best solution found: F({best_x[0]:.6f}, {best_x[1]:.6f}) = {best_f:.6f}")
    print(f"Time taken: {end_time - start_time:.4f} seconds, Iterations: {iteration + 1}\n")


simulated_annealing("exponential", max_iters=50000)
simulated_annealing("linear", max_iters=50000)
simulated_annealing("quadratic", max_iters=50000)
