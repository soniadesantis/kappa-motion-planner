import sympy as sp
from math import sin, cos, pi, sqrt, asin, atan2
import numpy as np
from arena import get_corner_point, shrink_corridor_list, compute_end_pose, compute_start_pose, MotionPlanner, CurvilinearArcUnicycle, CorridorWorld, Unicycle, get_corridor_from_vector, plot_corridors
from math import sin, cos, pi, sqrt, asin, atan2
import matplotlib.pylab as plt

##### Description #####
# This script generates a sequence of corridors and computes the trajectory of a unicycle vehicle moving through them.
# 3 figure are generated:
# 1. Sequence of corridors + intermediate circumferences
# 2. Trajectory pieces without intermediate arcs
# 3. Full trajectory

# Define vehicle
vehicle = Unicycle(model = 'Rosbot circular')
vehicle.width = 0.4
r = vehicle.width*0.5
v_max = 0.8
vehicle.update(v_max = v_max )
vehicle.update(omega_max = 1)

# Define corridors parameters
width1 = 1.1
height1 = 4
phi1 = pi/6

width2 = 1.4
height2 = 2
phi2 = pi/2
add_height2 = 2

width3 = width1
height3 = 3.5
phi3 = -pi/6
add_height3 = 1.5

# Define corridor1 
corridor1 = CorridorWorld(width = width1, height = height1, center = [0, 0], tilt = phi1)

# Define corridor2
tail_vector2 = corridor1.head
head_vector2 = [corridor1.head[0] + height2 * cos(phi2), corridor1.head[1] + height2 * sin(phi2)]
corridor2 = get_corridor_from_vector(tail_vector2, head_vector2, width2, add_height = add_height2)

# Define corridor3 
tail_vector3 = corridor2.head
head_vector3 = [corridor2.head[0] + height3 * cos(phi3), corridor2.head[1] + height3 * sin(phi3)]
corridor3 = get_corridor_from_vector(tail_vector3, head_vector3, width3, add_height = add_height3)

corridor_list = [corridor1, corridor2, corridor3]
shrunk_corridor_list = shrink_corridor_list(corridor_list, r)
x0 = (shrunk_corridor_list[0].center[0] - 1*cos(phi1)) + shrunk_corridor_list[0].width/2 * cos(shrunk_corridor_list[0].tilt - pi/2)
y0 = (shrunk_corridor_list[0].center[1] - 1*sin(phi1)) + shrunk_corridor_list[0].width/2 * sin(shrunk_corridor_list[0].tilt - pi/2)
initial_pose = compute_start_pose(corridor1, vehicle, 0)
initial_pose[0] = x0
initial_pose[1] = y0
initial_pose[2] = phi1 + pi/6
final_pose = compute_end_pose(corridor3, vehicle, 0.5)

mp = MotionPlanner(vehicle, corridor_list, start_pose=initial_pose, end_pose=final_pose)

corner_point1 = get_corner_point(corridor1, corridor2, 1)
corner_point2 = get_corner_point(corridor2, corridor3, -1)

## Figure 1: sequence of corridors
l = r + 0.2
angle_array = np.linspace(0, 2 * np.pi, 100)

figure = plot_corridors(corridor_list = [corridor_list[1]], color = 'k', linestyle = '-', linewidth = 1, figure = None, plot_vectors = False)
plot_corridors(corridor_list = [shrunk_corridor_list[1]], color = 'k', linestyle = '--', linewidth = 1, figure = figure, plot_vectors = False)
# x0 = (shrunk_corridor_list[1].center[0] - 1*cos(phi2)) + shrunk_corridor_list[1].width/2 * cos(shrunk_corridor_list[1].tilt - pi/2)
# y0 = (shrunk_corridor_list[1].center[1] - 1*sin(phi2)) + shrunk_corridor_list[1].width/2 * sin(shrunk_corridor_list[1].tilt - pi/2)
# initial_pose[0] = x0
# initial_pose[1] = y0
# initial_pose[2] = phi2 + pi/6
plt.plot(corridor2.center[0], corridor2.center[1], 'ko', markersize=5)
plt.arrow(corridor2.center[0], corridor2.center[1], l * cos(phi2), l * sin(phi2), head_width = 0.05, color = 'k')
plt.plot(initial_pose[0] + r * np.cos(angle_array), initial_pose[1] + r * np.sin(angle_array), 'k-')
plt.arrow(initial_pose[0], initial_pose[1], l * cos(initial_pose[2]), l * sin(initial_pose[2]), head_width = 0.05, color = 'k')
plt.axis('off')
plt.tight_layout()

## Uncomment for three-corridor sequence figure in the environment section of the journal paper
figure1 = plot_corridors(corridor_list = corridor_list, color = 'k', linestyle = '-', linewidth = 1, figure = None, plot_vectors = False)
plot_corridors(corridor_list = shrunk_corridor_list, color = 'k', linestyle = '--', linewidth = 1, figure = figure1, plot_vectors = False)
plt.plot(initial_pose[0] + r * np.cos(angle_array), initial_pose[1] + r * np.sin(angle_array), 'k-')
plt.arrow(initial_pose[0], initial_pose[1], l * cos(initial_pose[2]), l * sin(initial_pose[2]), head_width = 0.05, color = 'k')
plt.plot(corner_point1[0], corner_point1[1], 'ro', markersize=2)
plt.plot(corner_point2[0], corner_point2[1], 'ro', markersize=2)
# plt.plot(corner_point2[0] + r * np.cos(angle_array), corner_point2[1] + r * np.sin(angle_array), 'r-', linewidth =1)
plt.plot(corridor1.center[0], corridor1.center[1], 'ko', markersize=5)
plt.plot(corridor2.center[0], corridor2.center[1], 'ko', markersize=5)
plt.plot(corridor3.center[0], corridor3.center[1], 'ko', markersize=5)
plt.arrow(corridor1.center[0], corridor1.center[1], l * cos(phi1), l * sin(phi1), head_width = 0.05, color = 'k')
plt.arrow(corridor2.center[0], corridor2.center[1], l * cos(phi2), l * sin(phi2), head_width = 0.05, color = 'k')
plt.arrow(corridor3.center[0], corridor3.center[1], l * cos(phi3), l * sin(phi3), head_width = 0.05, color = 'k')
plt.axis('off')
plt.tight_layout()

# min_widths_corridors = mp.min_corridor_widths
# corridor1_min_width = CorridorWorld(min_widths_corridors[0], corridor1.height, corridor1.center, corridor1.tilt)
# corridor2_min_width = CorridorWorld(min_widths_corridors[1], corridor2.height, corridor2.center, corridor2.tilt)
# corridor3_min_width = CorridorWorld(min_widths_corridors[2], corridor3.height, corridor3.center, corridor3.tilt)
# min_width_corridor_list = [corridor1_min_width, corridor2_min_width, corridor3_min_width]
# mp_min_widths_corridors = MotionPlanner(vehicle, min_width_corridor_list)

# mp1 = MotionPlanner(vehicle, [corridor1, corridor2])
# mp1_min_widths_corridors = mp1.min_corridor_widths 

# figure2 = plot_corridors(min_width_corridor_list, linestyle= 'solid', linewidth = 1,)
# for center in mp_min_widths_corridors.intermediate_circle_centers:
#     plt.plot(center[0], center[1], 'ko', markersize=2)
#     circle = plt.Circle((center[0], center[1]), vehicle.max_radius, color='red', fill= False, linestyle= 'solid')
#     figure2.gca().add_artist(circle)
#     circle_outer = plt.Circle((center[0], center[1]), vehicle.max_radius + r, color='orange', fill= False, linestyle= 'solid')
#     figure2.gca().add_artist(circle_outer)
# plt.tight_layout()
# plt.axis('off')


## Uncomment to save
plt.savefig("fig_free_region.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
plt.show(block = True)