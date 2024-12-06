import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter

# Constants
m = 1.0  # Mass of the pendulum bob (kg)
k = 50.0  # Spring constant (N/m)
l0 = 1.0  # Natural length of the spring (m)
g = 9.81  # Gravitational acceleration (m/s^2)
mu = 0.5  # Magnetic moment (arbitrary units)
mu_max = 1.0  # Saturation magnetic moment
B0 = 0.2  # Magnetic field amplitude (T)
B_s = 0.1  # Saturation field (arbitrary units)
omega = 3.0  # Angular frequency of the magnetic field (rad/s)

# Initial conditions
r0 = 1.2  # Initial spring length (m)
theta0 = np.pi / 4  # Initial angle (rad)
r_dot0 = 0.0  # Initial radial velocity (m/s)
theta_dot0 = 0.0  # Initial angular velocity (rad/s)

# Time span
t_start = 0.0
t_end = 20.0
num_points = 10000  # Number of time points
t_eval = np.linspace(t_start, t_end, num_points)

# Magnetic field model
def magnetic_field(t):
    """Oscillating magnetic field."""
    return B0 * np.cos(omega * t)

def magnetic_moment(B):
    """Saturation magnetic moment."""
    return mu_max * B / (B + B_s)

# Equations of motion
def equations(t, y):
    """Equations of motion with refined magnetic forces."""
    r, theta, r_dot, theta_dot = y
    B_t = magnetic_field(t)
    mu_dynamic = magnetic_moment(np.abs(B_t))

    # Radial acceleration
    r_ddot = (
        r * theta_dot**2
        - (k / m) * (r - l0)
        + g * np.cos(theta)
        + (mu_dynamic / m) * (-np.sin(theta) * omega * B0 * np.sin(omega * t))
    )

    # Angular acceleration
    theta_ddot = (
        - (2 * r_dot * theta_dot) / r
        - (g / r) * np.sin(theta)
        + (mu_dynamic / (m * r)) * omega * B0 * np.cos(omega * t) * np.sin(theta)
    )

    return [r_dot, theta_dot, r_ddot, theta_ddot]

# Solve system
y0 = [r0, theta0, r_dot0, theta_dot0]
solution = solve_ivp(equations, [t_start, t_end], y0, method='RK45', t_eval=t_eval)

# Extract results
r_vals = solution.y[0]
theta_vals = solution.y[1]
r_dot_vals = solution.y[2]
theta_dot_vals = solution.y[3]
x_vals = r_vals * np.sin(theta_vals)
y_vals = -r_vals * np.cos(theta_vals)

# Magnetic field lines (dipole)
def dipole_field(x, y):
    """2D dipole field."""
    r = np.sqrt(x**2 + y**2) + 1e-5  # Avoid division by zero
    r_hat = np.stack([x / r, y / r], axis=-1)
    m_dot_r = mu * r_hat[..., 1]  # Assume dipole moment aligned along z-axis
    B_r = 1 / r**3
    B_theta = 1 / r**3
    Bx = B_r * r_hat[..., 0] - B_theta * r_hat[..., 1]
    By = B_r * r_hat[..., 1] + B_theta * r_hat[..., 0]
    return Bx, By

# Generate grid for field lines
x_grid = np.linspace(-2, 2, 100)
y_grid = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Bx, By = dipole_field(X, Y)

# Normalize field vectors for visualization
B_magnitude = np.sqrt(Bx**2 + By**2)
Bx_normalized = Bx / (B_magnitude + 1e-5)
By_normalized = By / (B_magnitude + 1e-5)

# Radial Phase Portrait with Magnetic Field Lines
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, Bx_normalized, By_normalized, color=B_magnitude, linewidth=1, cmap='viridis')
plt.plot(r_vals * np.sin(theta_vals), r_dot_vals, label="Radial Phase Portrait", color='red', lw=1.5)
plt.xlabel("$r \sin(\\theta)$ (m)")
plt.ylabel("$\dot{r}$ (m/s)")
plt.title("Radial Phase Portrait with Magnetic Field Lines")
plt.colorbar(label="Magnetic Field Strength")
plt.grid()
plt.legend()
plt.savefig('radial_phase_portrait_fieldLines.png')

# Angular Phase Portrait with Magnetic Field Lines
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, Bx_normalized, By_normalized, color=B_magnitude, linewidth=1, cmap='viridis')
plt.plot(theta_vals, theta_dot_vals, label="Angular Phase Portrait", color='orange', lw=1.5)
plt.xlabel(r"$\theta$ (rad)")
plt.ylabel(r"$\dot{\theta}$ (rad/s)")
plt.title("Angular Phase Portrait with Magnetic Field Lines")
plt.colorbar(label="Magnetic Field Strength")
plt.grid()
plt.legend()
plt.savefig('angular_phase_portrait_fieldLines.png')

# Poincaré Map
period = 2 * np.pi / omega
poincare_times = np.arange(t_start, t_end, period)
poincare_indices = [np.abs(t_eval - t).argmin() for t in poincare_times]

# Poincaré Map with Magnetic Field Lines and Improved Visibility
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, Bx_normalized, By_normalized, color=B_magnitude, linewidth=1, cmap='viridis')
plt.scatter(theta_vals[poincare_indices], theta_dot_vals[poincare_indices], color='red', label="Poincaré Points", s=30, edgecolor='black', zorder=5)
plt.xlabel(r"$\theta$ (rad)")
plt.ylabel(r"$\dot{\theta}$ (rad/s)")
plt.title("Poincaré Map with Magnetic Field Lines")
plt.colorbar(label="Magnetic Field Strength")
plt.grid()
plt.legend()
plt.savefig('poincare_map_fieldLines.png')

# Field lines with dynamics
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Magnetic Field Lines with Dynamics")
ax.streamplot(X, Y, Bx_normalized, By_normalized, color=B_magnitude, linewidth=1.5, cmap='viridis')
line, = ax.plot([], [], 'o-', lw=2, color="red")
spring, = ax.plot([], [], 'k-', lw=1)

def init():
    line.set_data([], [])
    spring.set_data([], [])
    return line, spring

def update(frame):
    x = [0, x_vals[frame]]
    y = [0, y_vals[frame]]
    line.set_data(x, y)
    spring.set_data(x, y)
    return line, spring

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True)
ani.save('magnetic_field_dynamics_fieldLines.gif', writer=PillowWriter(fps=30))
