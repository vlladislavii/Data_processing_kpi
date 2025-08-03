import numpy as np
import time

def objective_function(x):
    A, B = 1, 6
    return (x[0] ** 2 + x[1] ** 2) / A - B * (np.cos(x[0]) + np.cos(x[1]))

def stochastic_search(num_points=10000, bounds=(-8, 8)):
    start_time = time.time()
    iterations = 0

    # Generate random points within the search bounds (neighbours)
    random_points = np.random.uniform(bounds[0], bounds[1], (num_points, 2))
    function_values = np.apply_along_axis(objective_function, 1, random_points)

    # Find the minimum value and corresponding point
    min_index = np.argmin(function_values)
    best_x, best_y = random_points[min_index]
    best_value = function_values[min_index]

    iterations = num_points  # Each random sample is considered an iteration
    execution_time = time.time() - start_time

    print(f'Stochastic Search Minimum: F({best_x}, {best_y}) = {best_value}')
    print(f'Total Iterations: {iterations}')
    print(f'Execution Time: {execution_time:.6f} seconds')


stochastic_search()
