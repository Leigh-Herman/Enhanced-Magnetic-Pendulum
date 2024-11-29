import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, cos, sin, simplify

# Define symbolic variables
r, theta, r_dot, theta_dot = symbols('r theta r_dot theta_dot')

# System parameters
m = 1.0  # Mass
k = 50.0  # Spring constant
l0 = 1.0  # Natural spring length
g = 9.81  # Gravitational acceleration

# Define the Lyapunov function (Total mechanical energy)
V = (1/2) * m * r_dot**2 + (1/2) * m * (r**2) * theta_dot**2 + (1/2) * k * (r - l0)**2 - m * g * r * cos(theta)

# Compute the time derivative of the Lyapunov function
V_dot = diff(V, r) * r_dot + diff(V, theta) * theta_dot + diff(V, r_dot) * (
    r * theta_dot**2 - (k / m) * (r - l0) + g * cos(theta)
) + diff(V, theta_dot) * (
    - (2 * r_dot * theta_dot) / r - (g / r) * sin(theta)
)

# Simplify the expressions for readability
V = simplify(V)
V_dot = simplify(V_dot)

# Create numpy functions for numerical evaluation
def evaluate_V(r_vals, theta_vals, r_dot_fixed, theta_dot_fixed):
    return (0.5 * m * r_dot_fixed**2
            + 0.5 * m * r_vals**2 * theta_dot_fixed**2
            + 0.5 * k * (r_vals - l0)**2
            - m * g * r_vals * np.cos(theta_vals))

def evaluate_V_dot(r_vals, theta_vals, r_dot_fixed, theta_dot_fixed):
    r_ddot = r_vals * theta_dot_fixed**2 - (k / m) * (r_vals - l0) + g * np.cos(theta_vals)
    theta_ddot = - (2 * r_dot_fixed * theta_dot_fixed) / r_vals - (g / r_vals) * np.sin(theta_vals)
    return (m * r_dot_fixed * r_ddot
            + m * r_vals**2 * theta_dot_fixed * theta_ddot
            + k * (r_vals - l0) * r_dot_fixed
            + m * g * np.sin(theta_vals) * theta_dot_fixed)

# Generate a grid for (r, theta)
r_vals = np.linspace(0.5, 1.5, 100)  # Radial displacement range
theta_vals = np.linspace(-np.pi, np.pi, 100)  # Angular displacement range
r_grid, theta_grid = np.meshgrid(r_vals, theta_vals)

# Evaluate V and V_dot over the grid
r_dot_fixed = 0.1
theta_dot_fixed = 0.1
V_values = evaluate_V(r_grid, theta_grid, r_dot_fixed, theta_dot_fixed)
V_dot_values = evaluate_V_dot(r_grid, theta_grid, r_dot_fixed, theta_dot_fixed)

# Plot V(r, θ)
plt.figure(figsize=(8, 6))
plt.contourf(r_grid, theta_grid, V_values, levels=50, cmap="viridis")
plt.colorbar(label="V(r, θ)")
plt.title("Lyapunov Function V(r, θ)")
plt.xlabel("Radial Displacement (r)")
plt.ylabel("Angular Displacement (θ)")
plt.grid()
plt.savefig("lyapunov_function.png")
plt.show()

# Plot V̇(r, θ)
plt.figure(figsize=(8, 6))
plt.contourf(r_grid, theta_grid, V_dot_values, levels=50, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="V̇(r, θ)")
plt.title("Time Derivative of Lyapunov Function (V̇)")
plt.xlabel("Radial Displacement (r)")
plt.ylabel("Angular Displacement (θ)")
plt.grid()
plt.savefig("lyapunov_derivative.png")
plt.show()
