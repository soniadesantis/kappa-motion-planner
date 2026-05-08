import sympy as sp
from math import sin, cos, pi, sqrt, asin, atan2
import numpy as np
from arena import compute_end_pose, compute_start_pose, MotionPlanner, compute_three_maneuvers_no_collision_avoidance, CurvilinearArcUnicycle, wrapPositiveAngle, CorridorWorld, Unicycle, get_corridor_from_vector, plot_corridors, compute_turn_direction, get_intersection, compute_two_maneuvers
from math import sin, cos, pi, sqrt, asin, atan2
import matplotlib.pylab as plt


# Define parameters

vehicle = Unicycle(model = 'Rosbot circular')
vehicle.update(v_max = 0.6)
vehicle.update(omega_max = 1)
vehicle.update(length = 0.5)
vehicle.update(width = 0.5)

# vehicle.set_length(0.5)
# vehicle.set_width(0.5)
# corridor4 = CorridorWorld(width = 1.11, height = 6.38, center = [0, 0], tilt = 0)
# corridor3 = CorridorWorld(width = 1.95, height = 3.65, center = [-2.215, 1.27], tilt = 3 * pi/2)
# corridor2 = CorridorWorld(width = 1.12, height = 4.51, center = [-3.495, 2.535], tilt = 0)
# corridor1 = CorridorWorld(width = 1.45, height = 2.55, center = [-5.025, 3.25], tilt = 3 * pi/2)

# Define corridor1 
corridor1 = CorridorWorld(width = 1.2, height = 3.5, center = [0, 0], tilt = pi/2)

# Define corridor2
tail_vector2 = corridor1.head
height2 = 4
width2 = 1.2
phi2 = -pi/6
head_vector2 = [corridor1.head[0] + height2 * cos(phi2), corridor1.head[1] + height2 * sin(phi2)]
corridor2 = get_corridor_from_vector(tail_vector2, head_vector2, width2, add_height = 0)

# Define corridor3 
tail_vector3 = corridor2.head
height3 = 3.5
width3 = 1.2
phi3 = pi/2
head_vector3 = [tail_vector3[0] + height3 * cos(phi3), tail_vector3[1] + height3 * sin(phi3)]
corridor3 = get_corridor_from_vector(tail_vector3, head_vector3, width3, add_height = 0)
# corridor1 = CorridorWorld(width = 1.45, height = 2.55, center = [-5.025, 3.25], tilt = 3 * pi/2)
corridor_list = [corridor1, corridor2, corridor3]
shrunken_corridor_list = [corridor1.shrink(vehicle.length*0.3), corridor2.shrink(vehicle.length*0.3), corridor3.shrink(vehicle.length*0.3)]

start_pose = compute_start_pose(corridor_list[0], vehicle, vehicle.length*0.5)
end_pose = compute_end_pose(corridor_list[-1], vehicle, vehicle.length*0.5)
start_pose[2] = pi
end_pose[2] = 0
mp = MotionPlanner(vehicle, corridor_list, start_pose=start_pose, end_pose=end_pose)

trajectory = mp.compute_trajectory_analytical()

figure1 = plot_corridors(corridor_list = corridor_list, color = 'k', linestyle = '-', linewidth = 1, figure = None, plot_vectors = False)
# figure = plot_corridors(corridor_list = shrunken_corridor_list, linewidth = 1, figure = figure, plot_vectors = False)

# figure = plt.figure()
# for trajectory_piece in trajectory:
#     trajectory_piece.plot_path(figure1, linewidth = 2.5)
#     # Plot circumference
#     # if isinstance(trajectory_piece, CurvilinearArcUnicycle):
#     #     trajectory_piece.plot_circle(figure)
# for trajectory_piece in trajectory[2:-2]:
#     # trajectory_piece.plot_path(figure, linewidth = 2.5)
#     # Plot circumference
#     if isinstance(trajectory_piece, CurvilinearArcUnicycle):
#         trajectory_piece.plot_circle(figure1)
l = 0.3
plt.arrow(start_pose[0], start_pose[1], l * cos(start_pose[2]), l * sin(start_pose[2]), head_width = 0.1, color = 'r')
plt.arrow(end_pose[0], end_pose[1], l * cos(end_pose[2]), l * sin(end_pose[2]), head_width = 0.1, color = 'r')
plt.plot(start_pose[0] + vehicle.length * 0.5 * np.cos(np.linspace(0, 2*pi, 100)), start_pose[1] +  vehicle.length * 0.5 * np.sin(np.linspace(0, 2*pi, 100)), color = 'r', linestyle = '-')
plt.plot(end_pose[0] + vehicle.length * 0.5 * np.cos(np.linspace(0, 2*pi, 100)), end_pose[1] + vehicle.length * 0.5 * np.sin(np.linspace(0, 2*pi, 100)), color = 'r', linestyle = '-')
plt.axis('off')
figure2 = plot_corridors(corridor_list = corridor_list, color = 'k', linestyle = '-', linewidth = 1, figure = None, plot_vectors = False)
# figure = plot_corridors(corridor_list = shrunken_corridor_list, linewidth = 1, figure = figure, plot_vectors = False)

# Trajectory = [tots1, arc2, seg3, arc4, seg5, arc6, seg7, arc8, tots9]
# figure = plt.figure()
trajectory1 = [trajectory[0], trajectory[1], trajectory[2], trajectory[6], trajectory[7], trajectory[8]]
for trajectory_piece in trajectory1:
    trajectory_piece.plot_path(figure2, linewidth = 2.5)
    # Plot circumference
    # if isinstance(trajectory_piece, CurvilinearArcUnicycle):
    #     trajectory_piece.plot_circle(figure)
for trajectory_piece in trajectory[2:-2]:
    # trajectory_piece.plot_path(figure, linewidth = 2.5)
    # Plot circumference
    if isinstance(trajectory_piece, CurvilinearArcUnicycle):
        trajectory_piece.plot_circle(figure2)
l = 0.3
plt.arrow(start_pose[0], start_pose[1], l * cos(start_pose[2]), l * sin(start_pose[2]), head_width = 0.1, color = 'r')
plt.arrow(end_pose[0], end_pose[1], l * cos(end_pose[2]), l * sin(end_pose[2]), head_width = 0.1, color = 'r')
plt.plot(start_pose[0] + vehicle.length * 0.5 * np.cos(np.linspace(0, 2*pi, 100)), start_pose[1] +  vehicle.length * 0.5 * np.sin(np.linspace(0, 2*pi, 100)), color = 'r', linestyle = '-')
plt.plot(end_pose[0] + vehicle.length * 0.5 * np.cos(np.linspace(0, 2*pi, 100)), end_pose[1] + vehicle.length * 0.5 * np.sin(np.linspace(0, 2*pi, 100)), color = 'r', linestyle = '-')
plt.axis('off')
figure3 = plot_corridors(corridor_list = corridor_list, color = 'k', linestyle = '-', linewidth = 1, figure = None, plot_vectors = False)
# figure = plot_corridors(corridor_list = shrunken_corridor_list, linewidth = 1, figure = figure, plot_vectors = False)
trajectory2 = [trajectory[4]]
# figure = plt.figure()
for trajectory_piece in trajectory1:
    trajectory_piece.plot_path(figure3, linewidth = 2.5)
for trajectory_piece in trajectory2:
    trajectory_piece.plot_path(figure3, linewidth = 2.5)
    # Plot circumference
    # if isinstance(trajectory_piece, CurvilinearArcUnicycle):
    #     trajectory_piece.plot_circle(figure)
for trajectory_piece in trajectory[2:-2]:
    # trajectory_piece.plot_path(figure, linewidth = 2.5)
    # Plot circumference
    if isinstance(trajectory_piece, CurvilinearArcUnicycle):
        trajectory_piece.plot_circle(figure3)
l = 0.3
plt.arrow(start_pose[0], start_pose[1], l * cos(start_pose[2]), l * sin(start_pose[2]), head_width = 0.1, color = 'r')
plt.arrow(end_pose[0], end_pose[1], l * cos(end_pose[2]), l * sin(end_pose[2]), head_width = 0.1, color = 'r')
plt.plot(start_pose[0] + vehicle.length * 0.5 * np.cos(np.linspace(0, 2*pi, 100)), start_pose[1] +  vehicle.length * 0.5 * np.sin(np.linspace(0, 2*pi, 100)), color = 'r', linestyle = '-')
plt.plot(end_pose[0] + vehicle.length * 0.5 * np.cos(np.linspace(0, 2*pi, 100)), end_pose[1] + vehicle.length * 0.5 * np.sin(np.linspace(0, 2*pi, 100)), color = 'r', linestyle = '-')
plt.axis('off')
figure4 = plot_corridors(corridor_list = corridor_list, color = 'k', linestyle = '-', linewidth = 1, figure = None, plot_vectors = False)
# figure = plot_corridors(corridor_list = shrunken_corridor_list, linewidth = 1, figure = figure, plot_vectors = False)
trajectory2 = [trajectory[4]]
trajectory3 = [trajectory[3], trajectory[5]]
# figure = plt.figure()
for trajectory_piece in trajectory1:
    trajectory_piece.plot_path(figure4, linewidth = 2.5)
for trajectory_piece in trajectory2:
    trajectory_piece.plot_path(figure4, linewidth = 2.5)
for trajectory_piece in trajectory3:
    trajectory_piece.plot_path(figure4, linewidth = 2.5, color = 'r')
# for trajectory_piece in trajectory2:
#     trajectory_piece.plot_path(figure4, linewidth = 2.5)
    # Plot circumference
    # if isinstance(trajectory_piece, CurvilinearArcUnicycle):
    #     trajectory_piece.plot_circle(figure)
for trajectory_piece in trajectory[2:-2]:
    # trajectory_piece.plot_path(figure, linewidth = 2.5)
    # Plot circumference
    if isinstance(trajectory_piece, CurvilinearArcUnicycle):
        trajectory_piece.plot_circle(figure4)
l = 0.3
plt.arrow(start_pose[0], start_pose[1], l * cos(start_pose[2]), l * sin(start_pose[2]), head_width = 0.1, color = 'r')
plt.arrow(end_pose[0], end_pose[1], l * cos(end_pose[2]), l * sin(end_pose[2]), head_width = 0.1, color = 'r')
plt.plot(start_pose[0] + vehicle.length * 0.5 * np.cos(np.linspace(0, 2*pi, 100)), start_pose[1] +  vehicle.length * 0.5 * np.sin(np.linspace(0, 2*pi, 100)), color = 'r', linestyle = '-')
plt.plot(end_pose[0] + vehicle.length * 0.5 * np.cos(np.linspace(0, 2*pi, 100)), end_pose[1] + vehicle.length * 0.5 * np.sin(np.linspace(0, 2*pi, 100)), color = 'r', linestyle = '-')
plt.axis('off')
figure5 = plot_corridors(corridor_list = corridor_list, color = 'k', linestyle = '-', linewidth = 1, figure = None, plot_vectors = False)
# figure = plot_corridors(corridor_list = shrunken_corridor_list, linewidth = 1, figure = figure, plot_vectors = False)
trajectory2 = [trajectory[4]]
trajectory3 = [trajectory[3], trajectory[5]]
# figure = plt.figure()
for trajectory_piece in trajectory1:
    trajectory_piece.plot_path(figure4, linewidth = 2.5)
for trajectory_piece in trajectory2:
    trajectory_piece.plot_path(figure4, linewidth = 2.5)
for trajectory_piece in trajectory3:
    trajectory_piece.plot_path(figure4, linewidth = 2.5)
# for trajectory_piece in trajectory2:
#     trajectory_piece.plot_path(figure4, linewidth = 2.5)
    # Plot circumference
    # if isinstance(trajectory_piece, CurvilinearArcUnicycle):
    #     trajectory_piece.plot_circle(figure)
# for trajectory_piece in trajectory[2:-2]:
#     # trajectory_piece.plot_path(figure, linewidth = 2.5)
#     # Plot circumference
#     if isinstance(trajectory_piece, CurvilinearArcUnicycle):
#         trajectory_piece.plot_circle(figure4)
l = 0.3
plt.arrow(start_pose[0], start_pose[1], l * cos(start_pose[2]), l * sin(start_pose[2]), head_width = 0.1, color = 'r')
plt.arrow(end_pose[0], end_pose[1], l * cos(end_pose[2]), l * sin(end_pose[2]), head_width = 0.1, color = 'r')
plt.plot(start_pose[0] + vehicle.length * 0.5 * np.cos(np.linspace(0, 2*pi, 100)), start_pose[1] +  vehicle.length * 0.5 * np.sin(np.linspace(0, 2*pi, 100)), color = 'r', linestyle = '-')
plt.plot(end_pose[0] + vehicle.length * 0.5 * np.cos(np.linspace(0, 2*pi, 100)), end_pose[1] + vehicle.length * 0.5 * np.sin(np.linspace(0, 2*pi, 100)), color = 'r', linestyle = '-')

plt.axis('off')


figure6 = plot_corridors(corridor_list = corridor_list, color = 'k', linestyle = '-', linewidth = 1, figure = None, plot_vectors = False)
# figure = plot_corridors(corridor_list = shrunken_corridor_list, linewidth = 1, figure = figure, plot_vectors = False)
trajectory2 = [trajectory[4]]
trajectory3 = [trajectory[3], trajectory[5]]
# # figure = plt.figure()
# for trajectory_piece in trajectory1:
#     trajectory_piece.plot_path(figure4, linewidth = 2.5)
# for trajectory_piece in trajectory2:
#     trajectory_piece.plot_path(figure4, linewidth = 2.5)
# for trajectory_piece in trajectory3:
#     trajectory_piece.plot_path(figure4, linewidth = 2.5)
# for trajectory_piece in trajectory2:
#     trajectory_piece.plot_path(figure4, linewidth = 2.5)
    # Plot circumference
    # if isinstance(trajectory_piece, CurvilinearArcUnicycle):
    #     trajectory_piece.plot_circle(figure)
for trajectory_piece in trajectory[2:-2]:
    # trajectory_piece.plot_path(figure, linewidth = 2.5)
    # Plot circumference
    if isinstance(trajectory_piece, CurvilinearArcUnicycle):
        trajectory_piece.plot_circle(figure6)
l = 0.3
plt.arrow(start_pose[0], start_pose[1], l * cos(start_pose[2]), l * sin(start_pose[2]), head_width = 0.1, color = 'r')
plt.arrow(end_pose[0], end_pose[1], l * cos(end_pose[2]), l * sin(end_pose[2]), head_width = 0.1, color = 'r')
plt.plot(start_pose[0] + vehicle.length * 0.5 * np.cos(np.linspace(0, 2*pi, 100)), start_pose[1] +  vehicle.length * 0.5 * np.sin(np.linspace(0, 2*pi, 100)), color = 'r', linestyle = '-')
plt.plot(end_pose[0] + vehicle.length * 0.5 * np.cos(np.linspace(0, 2*pi, 100)), end_pose[1] + vehicle.length * 0.5 * np.sin(np.linspace(0, 2*pi, 100)), color = 'r', linestyle = '-')

plt.axis('off')

plt.show(block = True)
