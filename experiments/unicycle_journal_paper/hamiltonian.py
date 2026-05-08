import sympy as sp
from math import sin, cos, pi, sqrt, asin, atan2
import numpy as np
from arena import compute_motion_time_with_orientation, compute_three_maneuvers_no_collision_avoidance, circle_intersection, CurvilinearArcUnicycle, wrapPositiveAngle, CorridorWorld, Unicycle, get_corridor_from_vector, plot_corridors, compute_turn_direction, get_intersection, check_point_inside_corridor, compute_two_maneuvers
from math import sin, cos, pi, sqrt, asin, atan2
import matplotlib.pylab as plt
'''
This script computes the Hamiltonian associated with the free space motion planning problem for a unicycle model velocity-controlled.
'''
v_min = -0.5
v_max = 0.5

omega_min = -0.5
omega_max = 0.5

# Generate a grid of v and omega values
v_vals = np.linspace(v_min, v_max, 100)
omega_vals = np.linspace(omega_min, omega_max, 100)
V, Omega = np.meshgrid(v_vals, omega_vals)

# Sympy part
lambda1, lambda2, lambda3, v, omega, theta = sp.symbols('lambda1 lambda2 lambda3 v omega theta')

H = lambda1*v*sp.cos(theta) + lambda2*v*sp.sin(theta) + lambda3*omega

theta_val = pi/4
H = H.subs(theta, theta_val)

## Case 1: lambda1 > 0 and lambda2 > 0 and lambda3 > 0
lambda1_val = 1
lambda2_val = 1
lambda3_val = 1

H_positive_costate_var = H.subs({lambda1: lambda1_val, lambda2: lambda2_val, lambda3: lambda3_val})

H_func_positive_costate_var = sp.lambdify([v, omega], H_positive_costate_var, "numpy")

# Compute the Hamiltonian over the grid
H_vals_positive_costate_var= H_func_positive_costate_var(V, Omega)

# Find the minimum value of the Hamiltonian and its corresponding indices in the grid
H_min_positive_costate_var = np.min(H_vals_positive_costate_var)
min_index = np.unravel_index(np.argmin(H_vals_positive_costate_var), H_vals_positive_costate_var.shape)

# Get the corresponding v and omega values at the minimum
v_min_positive_costate_var = V[min_index]
omega_min_positive_costate_var = Omega[min_index]

## Case 2: lambda1 = 0 and lambda2 = 0 and lambda3 = 0
# lambda1_val = 0
# lambda2_val = 0
# lambda3_val = 0

# H_zero_costate_var = H.subs({lambda1: lambda1_val, lambda2: lambda2_val, lambda3: lambda3_val})

# H_func_zero_costate_var = sp.lambdify([v, omega], H_zero_costate_var, "numpy")

# # Compute the Hamiltonian over the grid
# H_vals_zero_costate_var= H_func_zero_costate_var(V, Omega)

# # Find the minimum value of the Hamiltonian and its corresponding indices in the grid
# H_min_zero_costate_var = np.min(H_vals_zero_costate_var)
# min_index = np.unravel_index(np.argmin(H_vals_zero_costate_var), H_vals_zero_costate_var.shape)

# # Get the corresponding v and omega values at the minimum
# v_min_zero_costate_var = V[min_index]
# omega_min_zero_costate_var = Omega[min_index]

## Case 3: lambda1 < 0 and lambda2 < 0 and lambda3 < 0
lambda1_val = -1
lambda2_val = -1
lambda3_val = -1

H_negative_costate_var = H.subs({lambda1: lambda1_val, lambda2: lambda2_val, lambda3: lambda3_val})

H_func_negative_costate_var = sp.lambdify([v, omega], H_negative_costate_var, "numpy")

# Compute the Hamiltonian over the grid
H_vals_negative_costate_var= H_func_negative_costate_var(V, Omega)

# Find the minimum value of the Hamiltonian and its corresponding indices in the grid
H_min_negative_costate_var = np.min(H_vals_negative_costate_var)
min_index = np.unravel_index(np.argmin(H_vals_negative_costate_var), H_vals_negative_costate_var.shape)

# Get the corresponding v and omega values at the minimum
v_min_negative_costate_var = V[min_index]
omega_min_negative_costate_var = Omega[min_index]

# Study on behavior of costate variables
theta_array = np.linspace(0, 2*pi, 1000)
cos_theta_array = np.cos(theta_array)
sin_theta_array = np.sin(theta_array)   


plt.figure()

# Set x-axis ticks at specific positions
plt.xticks([0, pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 7*pi/4, 2*pi], ['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4', '2π'])
# Set y-axis ticks at specific positions
plt.yticks([0, 1, -1], ['0', '1', '-1'])

# Vertical lines
plt.axvline(x=pi/4, color='red', linestyle='--', linewidth=1)
plt.axvline(x=5*pi/4, color='red', linestyle='--', linewidth=1)

# Horizontal line
plt.axhline(y=0, color='blue', linestyle=':', linewidth=1)

plt.plot(theta_array, cos_theta_array, label='cos(θ)', color='blue')
plt.plot(theta_array, sin_theta_array, label='sin(θ)', color='orange')
plt.title('Behavior of Costate Variables')
plt.legend()

plt.figure()
# Set x-axis ticks at specific positions
plt.xticks([0, pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 7*pi/4, 2*pi], ['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4', '2π'])
# Set y-axis ticks at specific positions
plt.yticks([0, 1, -1], ['0', '1', '-1'])
# Horizontal line
plt.axhline(y=0, color='blue', linestyle=':', linewidth=1)
# Vertical lines
plt.axvline(x=pi/4, color='red', linestyle='--', linewidth=1)
plt.axvline(x=5*pi/4, color='red', linestyle='--', linewidth=1)

plt.plot(theta_array, cos_theta_array + sin_theta_array, label='cos(θ) + sin(θ)', color='blue')
plt.legend()

# Plot the Hamiltonian
plt.figure(figsize=(8, 6))
plt.contourf(V, Omega, H_vals_positive_costate_var, levels=50, cmap='viridis')
plt.colorbar(label="Hamiltonian Value")
plt.plot(v_min_positive_costate_var, omega_min_positive_costate_var, 'ro', markersize=10, label='Minimum Point')
plt.title('Hamiltonian as a function of $v$ and $\\omega$ for positive costate variables')
plt.xlabel('Linear velocity $v$')
plt.ylabel('Angular velocity $\\omega$')
plt.legend()
plt.grid(True)
## Case 2: lambda1 = 0 and lambda2 = 0 and lambda3 = 0

# plt.figure2(figsize=(8, 6))
# plt.contourf(V, Omega, H_vals_zero_costate_var, levels=50, cmap='viridis')
# plt.colorbar(label="Hamiltonian Value")
# plt.plot(v_min_zero_costate_var, omega_min_zero_costate_var, 'ro', markersize=10, label='Minimum Point')
# plt.title('Hamiltonian as a function of $v$ and $\\omega$')
# plt.xlabel('Linear velocity $v$')
# plt.ylabel('Angular velocity $\\omega$')

## Case 3: lambda1 < 0 and lambda2 < 0 and lambda3 < 0

plt.figure(figsize=(8, 6))
plt.contourf(V, Omega, H_vals_negative_costate_var, levels=50, cmap='viridis')
plt.colorbar(label="Hamiltonian Value")
plt.plot(v_min_negative_costate_var, omega_min_negative_costate_var, 'ro', markersize=10, label='Minimum Point')
plt.title('Hamiltonian as a function of $v$ and $\\omega$ for negative costate variables')
plt.xlabel('Linear velocity $v$')
plt.ylabel('Angular velocity $\\omega$')

plt.legend()
plt.grid(True)
plt.show()
