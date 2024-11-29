import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Matrix, diff, cos, sin, simplify

# Define symbolic variables
r, theta, r_dot, theta_dot = symbols('r theta r_dot theta_dot')
variables = [r, theta, r_dot, theta_dot]

# System parameters
m = 1.0  # Mass
k = 50.0  # Spring constant
l0 = 1.0  # Natural spring length
g = 9.81  # Gravitational acceleration

# Define equations of motion (using sympy.cos and sympy.sin)
r_ddot = r * theta_dot**2 - (k / m) * (r - l0) + g * cos(theta)
theta_ddot = (-2 * r_dot * theta_dot) / r - (g / r) * sin(theta)

# Equations for Jacobian
equations = [r_dot, theta_dot, r_ddot, theta_ddot]

# Compute the Jacobian matrix
jacobian = Matrix(equations).jacobian(variables)

# Substitute equilibrium point: r = l0, theta = 0, r_dot = 0, theta_dot = 0
equilibrium = {r: l0, theta: 0, r_dot: 0, theta_dot: 0}
jacobian_numeric = jacobian.subs(equilibrium)
print("Jacobian Matrix at Equilibrium:")
print(jacobian_numeric)

# Compute characteristic polynomial
char_poly = jacobian_numeric.charpoly()
print("\nCharacteristic Polynomial:")
print(char_poly)

# Compute eigenvalues
eigenvalues = jacobian_numeric.eigenvals()
print("\nEigenvalues:")
for val in eigenvalues:
    print(val)

# Visualize eigenvalues in the complex plane
eigenvalues_numeric = [complex(val) for val in eigenvalues.keys()]

# Plot the eigenvalues
plt.figure(figsize=(8, 6))
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.scatter([np.real(ev) for ev in eigenvalues_numeric], [np.imag(ev) for ev in eigenvalues_numeric],
            color='red', s=100, label='Eigenvalues')
plt.title("Eigenvalues in the Complex Plane")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid()
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.legend()
plt.savefig('eigenvalues_complex_plane.png')
plt.show()
