import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
m = 1.0  # Mass of the pendulum bob (kg)
k = 50.0  # Spring constant (N/m)
l0 = 1.0  # Natural length of the spring (m)
g = 9.81  # Gravitational acceleration (m/s^2)

# Adjust these parameters for more chaotic behavior
omega = 3.0         # Driving frequency close to system's natural frequency
mu = 0.5            # Larger magnetic moment
B0 = 0.2            # Stronger magnetic field amplitude
gamma_theta = 0.05  # Moderate angular damping
r0 = 1.5            # Slightly different initial radial displacement
theta0 = np.pi / 4  # Larger initial angle

# Initial conditions
r_dot0 = 0.0  # Initial radial velocity (m/s)
theta_dot0 = 0.0  # Initial angular velocity (rad/s)

# Time span
t_start = 0.0
t_end = 20.0
num_points = 10000  # Number of time points for fine resolution
t_eval = np.linspace(t_start, t_end, num_points)


# Equations of motion
def equations(t, y):
    r, theta, r_dot, theta_dot = y

    # Magnetic field at time t
    B_t = B0 * np.cos(omega * t)

    # Radial acceleration
    r_ddot = (
            r * theta_dot ** 2
            - (k / m) * (r - l0)
            + g * np.cos(theta)
    )

    # Angular acceleration
    theta_ddot = (
            - (2 * r_dot * theta_dot) / r
            - (g / r) * np.sin(theta)
            + (mu * B_t * np.sin(theta)) / (m * r)
    )

    return [r_dot, theta_dot, r_ddot, theta_ddot]


# Solve the system using RK45
y0 = [r0, theta0, r_dot0, theta_dot0]
solution = solve_ivp(
    equations, [t_start, t_end], y0, method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-9
)

# Extract results
r_vals = solution.y[0]
theta_vals = solution.y[1]
r_dot_vals = solution.y[2]
theta_dot_vals = solution.y[3]

# Compute x and y coordinates
x_vals = r_vals * np.sin(theta_vals)
y_vals = -r_vals * np.cos(theta_vals)

# Create a phase portrait
plt.figure(figsize=(8, 6))
plt.plot(r_vals, r_dot_vals, label="Radial Phase Portrait: Chaos")
plt.xlabel("$r$ (m)")
plt.ylabel("$\dot{r}$ (m/s)")
plt.title("Radial Phase Portrait")
plt.grid()
plt.legend()
plt.savefig('radial_phase_portrait_Chaos.png')

plt.figure(figsize=(8, 6))
plt.plot(theta_vals, theta_dot_vals, label="Angular Phase Portrait: Chaos", color='orange')
plt.xlabel(r"$\theta$ (rad)")
plt.ylabel(r"$\dot{\theta}$ (rad/s)")
plt.title("Angular Phase Portrait")
plt.grid()
plt.legend()
plt.savefig('angular_phase_portrait_Chaos.png')

# Generate a Poincaré map
# Choose points where the magnetic field completes a period (T = 2π/ω)
period = 2 * np.pi / omega
poincare_times = np.arange(t_start, t_end, period)
poincare_indices = [np.abs(t_eval - t).argmin() for t in poincare_times]

# Plot Poincaré map
plt.figure(figsize=(8, 6))
plt.scatter(theta_vals[poincare_indices], theta_dot_vals[poincare_indices], color='purple')
plt.xlabel(r"$\theta$ (rad)")
plt.ylabel(r"$\dot{\theta}$ (rad/s)")
plt.title("Poincaré Map")
plt.grid()
plt.savefig('poincare_map_Chaos.png')

# Animation setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Spring Pendulum with Magnetic Field Chaos")
line, = ax.plot([], [], 'o-', lw=2)
spring, = ax.plot([], [], 'k-', lw=1)
time_text = ax.text(-1.8, 1.8, '', fontsize=10)


# Initialize animation
def init():
    line.set_data([], [])
    spring.set_data([], [])
    time_text.set_text('')
    return line, spring, time_text


# Update animation frame
def update(frame):
    x = [0, x_vals[frame]]
    y = [0, y_vals[frame]]
    line.set_data(x, y)
    spring.set_data(x, y)
    time_text.set_text(f"Time: {t_eval[frame]:.2f} s")
    return line, spring, time_text


# Create animation
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True)

# Save animation as a GIF
ani.save('spring_pendulum_Chaos.gif', writer=PillowWriter(fps=30))

# Show the animation
plt.show()
