from arena import CorridorWorld, Unicycle, get_corridor_from_vector, plot_corridors, compute_turn_direction, get_intersection, compute_three_maneuvers_compact
from math import sin, cos, pi
import matplotlib.pylab as plt
import numpy as np

# Define parameters
x0, y0, theta0 = 0, 0, 3 * pi/2
R = 1
xc2, yc2 = 5, 5
tau2 = 1 

#Create a corridor
corridor1 = CorridorWorld(width = 15, height = 15, center = [0 , 5 ], tilt = pi/2)

#Create a vehicle
unicycle = Unicycle([x0, y0, theta0], 0.430, length = 0.430, v_max = 1, v_min = 0, omega_max = 1, omega_min = -1)

# Compute the three maneuvers turning to the right 
T1_right, C1_right, S1_right = compute_three_maneuvers_compact(corridor1, corridor1, [x0, y0, theta0], unicycle, xc2, yc2, tau2, t0 = 0, turn1 = -1)
time_right = T1_right.maneuver_time + C1_right.maneuver_time + S1_right.maneuver_time
# Compute the three maneuvers turning to the left 
T1_left, C1_left, S1_left = compute_three_maneuvers_compact(corridor1, corridor1, [x0, y0, theta0], unicycle, xc2, yc2, tau2, t0 = 0, turn1 = 1)
time_left = T1_left.maneuver_time + C1_left.maneuver_time + S1_left.maneuver_time

#Print the results
print(f'The maneuvr time right-left is: {time_right}s. \nThe maneuver time left-left is: {time_left}s')

#Plot the results
#Plot right left----------------
figure1 = plot_corridors(corridor_list = [corridor1], figure = None, plot_vectors = False)
T1_right.plot_path(figure1)
C1_right.plot_path(figure1)
S1_right.plot_path(figure1)
plt.plot(x0, y0, 'bo')
plt.arrow(x0, y0, cos(theta0), sin(theta0), head_width = 0.1, color = 'r')
plt.plot(xc2 + R * np.cos(np.linspace(0, 2*pi, 50)), yc2 + R * np.sin(np.linspace(0, 2*pi, 50)), 'k--', linewidth = 0.8)
#-------------------------------
#Plot left left-----------------
figure2 = plot_corridors(corridor_list = [corridor1], figure = None, plot_vectors = False)
T1_left.plot_path(figure2)
C1_left.plot_path(figure2)
S1_left.plot_path(figure2)
plt.plot(x0, y0, 'bo')
plt.arrow(x0, y0, cos(theta0), sin(theta0), head_width = 0.1, color = 'r')
plt.plot(xc2 + R * np.cos(np.linspace(0, 2*pi, 50)), yc2 + R * np.sin(np.linspace(0, 2*pi, 50)), 'k--', linewidth = 0.8)
#-------------------------------

# Compute the maneuver time with varying theta0
N = 50
time_right_array = np.zeros(N)
time_left_array = np.zeros(N)
ind = 0
theta0_array = np.linspace(0, 2*pi, N)
for theta0 in theta0_array:
    # Compute the three maneuvers turning to the right 
    T1_right, C1_right, S1_right = compute_three_maneuvers_compact(corridor1, corridor1, [x0, y0, theta0], unicycle, xc2, yc2, tau2, t0 = 0, turn1 = -1)
    time_right_array[ind] = T1_right.maneuver_time + C1_right.maneuver_time + S1_right.maneuver_time
    # Compute the three maneuvers turning to the left 
    T1_left, C1_left, S1_left = compute_three_maneuvers_compact(corridor1, corridor1, [x0, y0, theta0], unicycle, xc2, yc2, tau2, t0 = 0, turn1 = 1)
    time_left_array[ind] = T1_left.maneuver_time + C1_left.maneuver_time + S1_left.maneuver_time
    ind += 1

#Print the results
print(f'RIGHT-LEFT\nMaximum time: {max(time_right_array)}\nMinimum time: {min(time_right_array)}')
print(f'LEFT-LEFT\nMaximum time: {max(time_left_array)}\nMinimum time: {min(time_left_array)}')

#Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Maneuver time with the initial orientation')

ax1.plot(theta0_array, time_right_array, '.-')
ax1.set_ylabel('Time Right-Left [s]')

ax2.plot(theta0_array, time_left_array, '.-')
ax2.set_xlabel('Initial orientation theta0 [rad]')
ax2.set_ylabel('Time Left-Left [s]')

plt.show(block = True)
