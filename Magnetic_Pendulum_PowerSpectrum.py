import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq

# Constants
m = 1.0  # Mass
k = 50.0  # Spring constant
l0 = 1.0  # Natural spring length
g = 9.81  # Gravitational acceleration
B0 = 0.2  # Magnetic field amplitude
omega = 3.0  # Angular frequency of the magnetic field

# Initial conditions
r0 = 1.2  # Initial spring length
theta0 = np.pi / 4  # Initial angular displacement
r_dot0 = 0.0  # Initial radial velocity
theta_dot0 = 0.0  # Initial angular velocity
y0 = [r0, theta0, r_dot0, theta_dot0]  # Initial state vector

# Time span
t_start = 0.0
t_end = 20.0
num_points = 10000
t_eval = np.linspace(t_start, t_end, num_points)


# Magnetic field function
def magnetic_field(t):
    return B0 * np.cos(omega * t)


# Equations of motion
def equations(t, y):
    r, theta, r_dot, theta_dot = y
    B_t = magnetic_field(t)

    # Radial acceleration
    r_ddot = r * theta_dot ** 2 - (k / m) * (r - l0) + g * np.cos(theta)

    # Angular acceleration
    theta_ddot = - (2 * r_dot * theta_dot) / r - (g / r) * np.sin(theta)

    return [r_dot, theta_dot, r_ddot, theta_ddot]


# Solve the system using solve_ivp
solution = solve_ivp(equations, [t_start, t_end], y0, t_eval=t_eval, method='RK45')

# Extract time-series data
r_vals = solution.y[0]  # Radial displacement
theta_vals = solution.y[1]  # Angular displacement
r_dot_vals = solution.y[2]  # Radial velocity
theta_dot_vals = solution.y[3]  # Angular velocity

# Power Spectrum Analysis for radial displacement (r)
r_fft = fft(r_vals)  # Fourier Transform of r
frequencies_r = fftfreq(len(r_vals), d=(t_eval[1] - t_eval[0]))  # Frequency bins
pos_mask_r = frequencies_r > 0  # Positive frequencies only

# Plot power spectrum for radial displacement
plt.figure(figsize=(8, 6))
plt.plot(frequencies_r[pos_mask_r], np.abs(r_fft[pos_mask_r]) ** 2, label="Radial Displacement", color="blue")
plt.title("Power Spectrum of Radial Displacement")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.grid()
plt.legend()
plt.savefig("power_spectrum_radial.png")
plt.show()

# Power Spectrum Analysis for angular displacement (theta)
theta_fft = fft(theta_vals)  # Fourier Transform of theta
frequencies_theta = fftfreq(len(theta_vals), d=(t_eval[1] - t_eval[0]))  # Frequency bins
pos_mask_theta = frequencies_theta > 0  # Positive frequencies only

# Plot power spectrum for angular displacement
plt.figure(figsize=(8, 6))
plt.plot(frequencies_theta[pos_mask_theta], np.abs(theta_fft[pos_mask_theta]) ** 2, label="Angular Displacement",
         color="orange")
plt.title("Power Spectrum of Angular Displacement")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.grid()
plt.legend()
plt.savefig("power_spectrum_angular.png")
plt.show()
