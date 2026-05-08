import sympy as sp
from math import sin, cos, pi, sqrt, asin, atan2
import numpy as np
from arena import shrink_corridor_list, compute_end_pose, compute_start_pose, MotionPlanner, CurvilinearArcUnicycle, CorridorWorld, Unicycle, get_corridor_from_vector, plot_corridors
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
r = vehicle.width*0.5
v_max = 0.6
vehicle.update(v_max = v_max )
vehicle.update(omega_max = 1)

# Define corridors parameters
width1 =1.3
height1 = 3
phi1 = 0

width2 = 1
height2 = 3
phi2 = pi/2
add_height2 = 0.7

width3 = width2 
height3 = 3
phi3 = -pi/6
add_height3 = add_height2

width4 = 1.2
height4 = 2.3
phi4 = -pi/2 - 0.2
add_height4 = add_height3

width5 = width4
height5 = height1
phi5 = 0
add_height5 = add_height4

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

# Define corridor4
tail_vector4 = corridor3.head
head_vector4 = [corridor3.head[0] + height4 * cos(phi4), corridor3.head[1] + height4 * sin(phi4)]
corridor4 = get_corridor_from_vector(tail_vector4, head_vector4, width4, add_height = add_height4)

# Define corridor5
tail_vector5 = corridor4.head
head_vector5 = [corridor4.head[0] + height5 * cos(phi5), corridor4.head[1] + height5 * sin(phi5)]
corridor5 = get_corridor_from_vector(tail_vector5, head_vector5, width5, add_height = add_height5)

corridor_list = [corridor1, corridor2, corridor3, corridor4]
shrunk_corridor_list = shrink_corridor_list(corridor_list, r)

### Define initial pose and final pose ###
# Define relative poses wrt the center of shrunken corridors
x_percentage_initial = 0.8
y_percentage_initial = -0.75
theta0_relative = pi + 0.7
x_percentage_final = -0.2
y_percentage_final = 0.8
thetaf_relative = 0

relative_start_pose = [x_percentage_initial,
                         y_percentage_initial,
                         theta0_relative]

relative_end_pose = [x_percentage_final,
                       y_percentage_final,
                       thetaf_relative]

mp = MotionPlanner(
    vehicle,
    corridor_list,
    relative_start_pose=relative_start_pose,
    relative_end_pose=relative_end_pose)

trajectory = mp.compute_trajectory_analytical()
initial_pose = mp.start_pose
final_pose = mp.end_pose

## Figure 1: sequence of corridors + intermediate circumferences
figure1 = plot_corridors(corridor_list = corridor_list, color = 'k', linestyle = '-', linewidth = 1, figure = None, plot_vectors = False)
plot_corridors(corridor_list = shrunk_corridor_list, color = 'k', linestyle = '--', linewidth = 1, figure = figure1, plot_vectors = False)

angle_array = np.linspace(0, 2 * np.pi, 100)

# Plot intermediate circumferences
for trajectory_piece in trajectory[3:-3]:
    if isinstance(trajectory_piece, CurvilinearArcUnicycle):
        trajectory_piece.plot_circle(figure1, color = 'k')
    # for point in trajectory_piece.path_coordinates:
    #     plt.plot(point[0] + r * np.cos(angle_array), point[1] + r * np.sin(angle_array), 'b-')
    # trajectory_piece.plot_path(figure1, linewidth = 2.5)
    
# plt.plot(initial_pose[0], initial_pose[1], 'ro')
plt.plot(initial_pose[0] + r * np.cos(angle_array), initial_pose[1] + r * np.sin(angle_array), 'k-')
l = r + 0.1
plt.arrow(initial_pose[0], initial_pose[1], l * cos(initial_pose[2]), l * sin(initial_pose[2]), head_width = 0.05, color = 'k')

# plt.plot(final_pose[0], final_pose[1], 'ro')
plt.plot(final_pose[0] + r * np.cos(angle_array), final_pose[1] + r * np.sin(angle_array), 'k-')
plt.arrow(final_pose[0], final_pose[1], l * cos(final_pose[2]), l * sin(final_pose[2]), head_width = 0.05, color = 'k')
plt.axis('off')
plt.tight_layout()
# plt.savefig("main_example1.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)


# ## Figure 2: sequence of corridors + intermediate circumferences + trajectory pieces without intermediate arcs
# figure2 = plot_corridors(corridor_list = corridor_list, color = 'k', linestyle = '-', linewidth = 1, figure = None, plot_vectors = False)
# plot_corridors(corridor_list = shrunk_corridor_list, color = 'k', linestyle = '--', linewidth = 1, figure = figure2, plot_vectors = False)

# angle_array = np.linspace(0, 2 * np.pi, 100)

# # Plot intermediate circumferences
# for trajectory_piece in trajectory[3:-3]:
#     if isinstance(trajectory_piece, CurvilinearArcUnicycle):
#         trajectory_piece.plot_circle(figure2, color = 'k')
#     # for point in trajectory_piece.path_coordinates:
#     #     plt.plot(point[0] + r * np.cos(angle_array), point[1] + r * np.sin(angle_array), 'b-')
    
# first_circ = True 
# last_circ = False

# init_part = 'y'
# int_part = 'orange'
# final_part = 'r'
# arcs = 'g'

# colors = [init_part, init_part, init_part, arcs, int_part, arcs, int_part, arcs, int_part, arcs, final_part, final_part, final_part]
# for ind, trajectory_piece in enumerate(trajectory):
#     if not(isinstance(trajectory_piece, CurvilinearArcUnicycle)) or first_circ or last_circ:
#         trajectory_piece.plot_path(figure2, linewidth = 2.5, color = colors[ind])    
#     if isinstance(trajectory_piece, CurvilinearArcUnicycle):
#         first_circ = False 
#     if ind == len(trajectory) - 3:
#         last_circ = True
    
# # plt.plot(initial_pose[0], initial_pose[1], 'ro')
# plt.plot(initial_pose[0] + r * np.cos(angle_array), initial_pose[1] + r * np.sin(angle_array), 'k-')
# l = r + 0.1
# plt.arrow(initial_pose[0], initial_pose[1], l * cos(initial_pose[2]), l * sin(initial_pose[2]), head_width = 0.05, color = 'k')

# # plt.plot(final_pose[0], final_pose[1], 'ro')
# plt.plot(final_pose[0] + r * np.cos(angle_array), final_pose[1] + r * np.sin(angle_array), 'k-')
# plt.arrow(final_pose[0], final_pose[1], l * cos(final_pose[2]), l * sin(final_pose[2]), head_width = 0.05, color = 'k')
# plt.axis('off')
# plt.tight_layout()

## Figure 3: sequence of corridors + intermediate circumferences + trajectory
figure3 = plot_corridors(corridor_list = corridor_list, color = 'k', linestyle = '-', linewidth = 1, figure = None, plot_vectors = False)
plot_corridors(corridor_list = shrunk_corridor_list, color = 'k', linestyle = '--', linewidth = 1, figure = figure3, plot_vectors = False)

angle_array = np.linspace(0, 2 * np.pi, 100)
line_width = 3

# Plot intermediate circumferences
for ind, trajectory_piece in enumerate(trajectory):
    if isinstance(trajectory_piece, CurvilinearArcUnicycle) and ind>2 and ind < len(trajectory) -3:
        trajectory_piece.plot_circle(figure3, color = 'r')
    elif isinstance(trajectory_piece, CurvilinearArcUnicycle):
        trajectory_piece.plot_circle(figure3, color = 'k')

for trajectory_piece in trajectory[:4]:
    trajectory_piece.plot_path(figure3, linewidth = line_width, color = 'b')

for trajectory_piece in trajectory[4:7]:
    if isinstance(trajectory_piece, CurvilinearArcUnicycle):
        trajectory_piece.plot_path(figure3, linewidth = line_width, color = 'm')
    else:
        trajectory_piece.plot_path(figure3, linewidth = line_width, color = 'm')

for trajectory_piece in trajectory[7:]:
    trajectory_piece.plot_path(figure3, linewidth = line_width, color = 'g')

for trajectory_piece in trajectory:
    plt.plot(trajectory_piece.xf + r * np.cos(angle_array), trajectory_piece.yf + r * np.sin(angle_array), 'k-', zorder = 3)
    plt.arrow(trajectory_piece.xf, trajectory_piece.yf, (r + 0.1) * cos(trajectory_piece.thetaf), (r + 0.1) * sin(trajectory_piece.thetaf), head_width = 0.05, color = 'k', zorder = 3)
   


        # trajectory_piece.plot_path(figure3, linewidth = 2.5, color = arcs) 
    # for point in trajectory_piece.path_coordinates:
    #     plt.plot(point[0] + r * np.cos(angle_array), point[1] + r * np.sin(angle_array), 'b-')
    
# first_circ = True 
# last_circ = False
# for ind, trajectory_piece in enumerate(trajectory):
#     if not(isinstance(trajectory_piece, CurvilinearArcUnicycle)) or first_circ or last_circ:
#         trajectory_piece.plot_path(figure3, linewidth = 2.5, color = colors[ind])    
#     if isinstance(trajectory_piece, CurvilinearArcUnicycle):
#         first_circ = False 
#     if ind == len(trajectory) - 3:
#         last_circ = True
    
# plt.plot(initial_pose[0], initial_pose[1], 'ro')
plt.plot(initial_pose[0] + r * np.cos(angle_array), initial_pose[1] + r * np.sin(angle_array), 'k-')
l = r + 0.1
plt.arrow(initial_pose[0], initial_pose[1], l * cos(initial_pose[2]), l * sin(initial_pose[2]), head_width = 0.05, color = 'k')

# plt.plot(final_pose[0], final_pose[1], 'ro')
plt.plot(final_pose[0] + r * np.cos(angle_array), final_pose[1] + r * np.sin(angle_array), 'k-')
plt.arrow(final_pose[0], final_pose[1], l * cos(final_pose[2]), l * sin(final_pose[2]), head_width = 0.05, color = 'k')
plt.axis('off')
plt.tight_layout()
plt.savefig("main_example2_red.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
plt.show(block = True)