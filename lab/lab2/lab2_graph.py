import numpy as np
import matplotlib.pyplot as plt

def plot_function():
    x = np.linspace(-8, 8, 100)
    y = np.linspace(-8, 8, 100)
    X, Y = np.meshgrid(x, y)
    Z = (X**2 + Y**2) / 1 - 6 * (np.cos(X) + np.cos(Y))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('F(X, Y)')
    plt.title('3D Plot of the Function')
    plt.show()

plot_function()