from math import sin, cos, pi, atan2
import numpy as np
from arena import (
    MotionPlanner,
    CorridorWorld,
    Unicycle,
) 
import arena
from math import sin, cos, pi, sqrt, asin, atan2
import matplotlib.pylab as plt

# # Define vehicle
# vehicle = Unicycle(model = 'Rosbot circular')
# r = vehicle.width*0.5
# R = r + 0.1
# vehicle.update(v_max = R )
# vehicle.update(omega_max = 1)
# # Define corridors parameters
# width1 =2*r + 0.5
# height1 = 4
# phi1 = pi/2

# width2 = width1 
# height2 = height1
# phi2 = phi1 + pi/3
# add_height = 0

# # Define corridor1 
# corridor1 = CorridorWorld(width = width1, height = height1, center = [0, 0], tilt = phi1)

# # Define corridor2
# tail_vector2 = corridor1.head
# head_vector2 = [corridor1.head[0] + height2 * cos(phi2), corridor1.head[1] + height2 * sin(phi2)]
# corridor2 = get_corridor_from_vector(tail_vector2, head_vector2, width2, add_height = add_height)

# corridor_list = [corridor1, corridor2]

# initial_pose = compute_start_pose(corridor1, vehicle, 0.5)
# final_pose = compute_end_pose(corridor2, vehicle, 0.5)

# turn_direction = compute_turn_direction(corridor1.vector, corridor2.vector)
# corner_point = get_corner_point(corridor1, corridor2, turn_direction)
# xc2, yc2 = second_circle(corridor1, corridor2, turn_direction, corner_point, vehicle, initial_pose)
# corner_point_direction = atan2(corner_point[1] - yc2, corner_point[0] - xc2)
# # beta = abs((phi1 - turn_direction * pi * 0.5) - atan2(corner_point[1] - yc2, corner_point[0] - xc2))
# beta = abs((phi1 - phi2)*0.5)
# c = (R-r) * cos(beta)
# minimum_width = r + R - c

# # minimum_width = 2 * r
# center1 = [corridor1.center[0] + (width1 * 0.5 - minimum_width * 0.5) * cos(phi1 + turn_direction * pi * 0.5),
#            corridor1.center[1] + (width1 * 0.5 - minimum_width * 0.5) * sin(phi1 + turn_direction * pi * 0.5)]
# center2 = [corridor2.center[0] + (width2 * 0.5 - minimum_width * 0.5) * cos(phi2 + turn_direction * pi * 0.5),
#            corridor2.center[1] + (width2 * 0.5 - minimum_width * 0.5) * sin(phi2 + turn_direction * pi * 0.5)]
# quant = 0.5
# min_width_corridor1 = CorridorWorld(width = minimum_width, height = height1, center = center1, tilt = phi1)
# min_width_corridor2 = CorridorWorld(width = minimum_width, height = corridor2.height, center = center2, tilt = phi2)

# min_width_corridor_list = [min_width_corridor1, min_width_corridor2]

# print(f'Minimum width: {minimum_width}')
# print(f'R = {R}')
# print(f'r = {r}')
# print(f'2r = {2*r}')
# print(f'beta = {beta}')
# print(f'beta degrees = {beta* 180/pi}')
# print(f'c = {c}')

# print(f'cos(beta) = {cos(beta)}')
# initial_pose = compute_start_pose(min_width_corridor1, vehicle, 0.5)
# final_pose = compute_end_pose(min_width_corridor2, vehicle, 0.5)

# mp = MotionPlanner(vehicle, corridor_list, start_pose=initial_pose, end_pose=final_pose)

# trajectory = mp.compute_trajectory_analytical()

# if turn_direction == -1:
#     w1 = min_width_corridor1.W[:,3]
#     w2 = min_width_corridor2.W[:,3]
# else:
#     w1 = min_width_corridor1.W[:,1]
#     w2 = min_width_corridor2.W[:,1]
    
# int_point = get_intersection(w1, w2)

# # compute_distance_two_points(int_point, point2)
# figure1 = plot_corridors(corridor_list = min_width_corridor_list, color = 'k', linestyle = '--', linewidth = 1, figure = None, plot_vectors = False)
# plot_corridors(corridor_list = corridor_list, color = 'k', linestyle = '-', linewidth = 1.5, figure = figure1, plot_vectors = False)
# plt.plot(int_point[0], int_point[1], 'ro')
# angle_array = np.linspace(0, 2 * np.pi, 10000)

# for trajectory_piece in trajectory:

#     trajectory_piece.plot_path(figure1, linewidth = 2.5)
# plt.plot(initial_pose[0], initial_pose[1], 'ro')
# l = r + 0.3
# plt.arrow(initial_pose[0], initial_pose[1], l * cos(initial_pose[2]), l * sin(initial_pose[2]), head_width = 0.1, color = 'r')

# plt.plot(final_pose[0], final_pose[1], 'ro')
# plt.arrow(final_pose[0], final_pose[1], l * cos(final_pose[2]), l * sin(final_pose[2]), head_width = 0.1, color = 'r')

# angle_array = np.linspace(0, 2 * np.pi, 10000)
# plt.plot(xc2 + R * np.cos(angle_array), yc2 + R * np.sin(angle_array), 'r-', linewidth = 0.5)

# plt.plot(xc2, yc2, 'ro')

# plt.plot([xc2 + (R + r)* cos(corner_point_direction), xc2], [yc2 + (R + r )* sin(corner_point_direction), yc2], 'k-', linewidth = 1)
# plt.plot(xc2 + (R + r) * np.cos(angle_array), yc2 + (R + r) * np.sin(angle_array), 'r-', linewidth = 0.5)


# plt.show(block = True)



# Define vehicle
vehicle = arena.Unicycle(model = 'Rosbot circular')

vehicle.update(v_max = 0.8)
vehicle.update(omega_max = 1)

r = vehicle.width*0.5
R = vehicle.max_radius
# Define corridors parameters
width1 =1
height1 = 4
phi1 = pi/6

width2 = width1 
height2 = 3
phi2 = pi/2
add_height2 = 0.7

width3 = width2 
height3 = height1
phi3 = -pi/6
add_height3 = add_height2

width4 = width3 
height4 = 3
phi4 = -pi/2 - 0.2
add_height4 = add_height3

width5 = width4
height5 = height1
phi5 = 0
add_height5 = add_height4

# Define corridor1 
corridor1 = arena.CorridorWorld(width = width1, height = height1, center = [0, 0], tilt = phi1)

# Define corridor2
tail_vector2 = corridor1.head
head_vector2 = [corridor1.head[0] + height2 * cos(phi2), corridor1.head[1] + height2 * sin(phi2)]
corridor2 = arena.get_corridor_from_vector(tail_vector2, head_vector2, width2, add_height = add_height2)

# Define corridor3 
tail_vector3 = corridor2.head
head_vector3 = [corridor2.head[0] + height3 * cos(phi3), corridor2.head[1] + height3 * sin(phi3)]
corridor3 = arena.get_corridor_from_vector(tail_vector3, head_vector3, width3, add_height = add_height3)

# Define corridor4
tail_vector4 = corridor3.head
head_vector4 = [corridor3.head[0] + height4 * cos(phi4), corridor3.head[1] + height4 * sin(phi4)]
corridor4 = arena.get_corridor_from_vector(tail_vector4, head_vector4, width4, add_height = add_height4)

# Define corridor5
tail_vector5 = corridor4.head
head_vector5 = [corridor4.head[0] + height5 * cos(phi5), corridor4.head[1] + height5 * sin(phi5)]
corridor5 = arena.get_corridor_from_vector(tail_vector5, head_vector5, width5, add_height = add_height5)

corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5]

initial_pose = arena.compute_start_pose(corridor1, vehicle, 0.5)
final_pose = arena.compute_end_pose(corridor5, vehicle, 0.5)

mp = arena.MotionPlanner(vehicle, corridor_list, start_pose=initial_pose, end_pose=final_pose)

analytical_trajectory = mp.compute_trajectory_analytical()

phi1 = corridor1.tilt
phi2 = corridor2.tilt
phi3 = corridor3.tilt
phi4 = corridor4.tilt
phi5 = corridor5.tilt

beta1 = abs(arena.compute_angular_difference(phi1, phi2))*0.5
beta2 = abs(arena.compute_angular_difference(phi2, phi3))*0.5
beta3 = abs(arena.compute_angular_difference(phi3, phi4))*0.5
beta4 = abs(arena.compute_angular_difference(phi4, phi5))*0.5

q1 = (R - r) * cos(beta1)
q2 = (R - r) * cos(beta2)
q3 = (R - r) * cos(beta3)
q4 = (R - r) * cos(beta4)


minimum_width1 = r + R - q1

minimum_width21 = r + R - q1
minimum_width2 = r + R - q2

minimum_width32 = r + R - q2
minimum_width3 = r + R - q3

minimum_width43 = r + R - q3
minimum_width4 = r + R - q4

minimum_width54 = r + R - q4

min_width2 = max(minimum_width21, minimum_width2)
min_width3 = max(minimum_width32, minimum_width3)
min_width4 = max(minimum_width43, minimum_width4)


tau1 = corridor1.compute_relative_turn_direction(corridor2)
tau2 = corridor2.compute_relative_turn_direction(corridor3)
tau3 = corridor3.compute_relative_turn_direction(corridor4)
tau4 = corridor4.compute_relative_turn_direction(corridor5)

center1 = [corridor1.center[0] + (width1 * 0.5 - minimum_width1 * 0.5) * cos(phi1 + tau1 * pi * 0.5),
           corridor1.center[1] + (width1 * 0.5 - minimum_width1 * 0.5) * sin(phi1 + tau1 * pi * 0.5)]

center21 = [corridor2.center[0] + (width2 * 0.5 - minimum_width21 * 0.5) * cos(phi2 + tau1 * pi * 0.5),
            corridor2.center[1] + (width2 * 0.5 - minimum_width21 * 0.5) * sin(phi2 + tau1 * pi * 0.5)]

center2 = [corridor2.center[0] + (width2 * 0.5 - minimum_width2 * 0.5) * cos(phi2 + tau2 * pi * 0.5),
           corridor2.center[1] + (width2 * 0.5 - minimum_width2 * 0.5) * sin(phi2 + tau2 * pi * 0.5)]

center32 = [corridor3.center[0] + (width3 * 0.5 - minimum_width32 * 0.5) * cos(phi3 + tau2 * pi * 0.5),
            corridor3.center[1] + (width3 * 0.5 - minimum_width32 * 0.5) * sin(phi3 + tau2 * pi * 0.5)]

center3 = [corridor3.center[0] + (width3 * 0.5 - minimum_width3 * 0.5) * cos(phi3 + tau3 * pi * 0.5),
            corridor3.center[1] + (width3 * 0.5 - minimum_width3 * 0.5) * sin(phi3 + tau3 * pi * 0.5)]

center43 = [corridor4.center[0] + (width4 * 0.5 - minimum_width43 * 0.5) * cos(phi4 + tau3 * pi * 0.5),
            corridor4.center[1] + (width4 * 0.5 - minimum_width43 * 0.5) * sin(phi4 + tau3 * pi * 0.5)]

center4 = [corridor4.center[0] + (width4 * 0.5 - minimum_width4 * 0.5) * cos(phi4 + tau4 * pi * 0.5),
           corridor4.center[1] + (width4 * 0.5 - minimum_width4 * 0.5) * sin(phi4 + tau4 * pi * 0.5)]
center54 = [corridor5.center[0] + (width5 * 0.5 - minimum_width54 * 0.5) * cos(phi5 + tau4 * pi * 0.5),
            corridor5.center[1] + (width5 * 0.5 - minimum_width54 * 0.5) * sin(phi5 + tau4 * pi * 0.5)]


min_width_corridor1 = CorridorWorld(width = minimum_width1, height = corridor1.height, center = center1, tilt = phi1)
min_width_corridor21 = CorridorWorld(width = minimum_width21, height = corridor2.height, center = center21, tilt = phi2)

min_width_corridor2 = CorridorWorld(width = minimum_width2, height = corridor2.height, center = center2, tilt = phi2)
min_width_corridor32 = CorridorWorld(width = minimum_width32, height = corridor3.height, center = center32, tilt = phi3)

min_width_corridor3 = CorridorWorld(width = minimum_width3, height = corridor3.height, center = center3, tilt = phi3)
min_width_corridor43 = CorridorWorld(width = minimum_width43, height = corridor4.height, center = center43, tilt = phi4)

min_width_corridor4 = CorridorWorld(width = minimum_width4, height = corridor4.height, center = center4, tilt = phi4)
min_width_corridor54 = CorridorWorld(width = minimum_width54, height = corridor5.height, center = center54, tilt = phi5)

min_width_corridor_list = [min_width_corridor1,
                           min_width_corridor21,
                           min_width_corridor2,
                           min_width_corridor32,
                           min_width_corridor3,
                           min_width_corridor43,
                           min_width_corridor4,
                           min_width_corridor54
                           ]


corridor1_min = CorridorWorld(width = minimum_width1, height = corridor1.height, center = corridor1.center, tilt = phi1)
corridor2_min = CorridorWorld(width = min_width2, height = corridor2.height, center = corridor2.center, tilt = phi2)
corridor3_min = CorridorWorld(width = min_width3, height = corridor3.height, center = corridor3.center, tilt = phi3)
corridor4_min = CorridorWorld(width = min_width4, height = corridor4.height, center = corridor4.center, tilt = phi4)
corridor5_min = CorridorWorld(width = minimum_width54, height = corridor5.height, center = corridor5.center, tilt = phi5)

min_width_corridor_list2 = [corridor1_min,
                           corridor2_min,
                           corridor3_min,
                           corridor4_min,
                           corridor5_min]

initial_pose = arena.compute_start_pose(corridor1_min, vehicle, 0.5)
final_pose = arena.compute_end_pose(corridor5_min, vehicle, 0.5)

mp = arena.MotionPlanner(vehicle, min_width_corridor_list2, start_pose=initial_pose, end_pose=final_pose)

analytical_trajectory = mp.compute_trajectory_analytical()

figure = arena.plot_corridors(corridor_list = min_width_corridor_list2, color = 'k', linestyle = '--', linewidth = 1, figure = None, plot_vectors = False)
# arena.plot_corridors(corridor_list = corridor_list, color = 'k', linestyle = '-', linewidth = 1.5, figure = figure, plot_vectors = False)
arena.plot_analytical_trajectory(analytical_trajectory, figure, plot_circles = True)

for center in mp.intermediate_circle_centers:
    xc = center[0]
    yc = center[1]
    angle_array = np.linspace(0, 2 * np.pi, 10000)
    plt.plot(xc + (R+r) * np.cos(angle_array), yc + (R + r) * np.sin(angle_array), 'r-', linewidth = 0.5)
plt.show(block = True)


def check_minimum_width_corridors(planner):
    """
    Check if the corridors have sufficient width for the vehicle to pass through.
    """
    corridor_list = planner.corridor_list
    min_widths = [0] * len(corridor_list)
    R = planner.vehicle.max_radius
    r = planner.vehicle.width * 0.5

    phi1 = corridor_list[0].tilt
    phi2 = corridor_list[1].tilt

    beta1 = abs(arena.compute_angular_difference(phi1, phi2))*0.5
    q1 = (R-r) * cos(beta1)
    min_widths[0] = r + R - q1
    
    for j in range(1, len(corridor_list) - 1):
        phi_j = corridor_list[j].tilt
        phi_j_plus_1 = corridor_list[j+1].tilt

        betaj = abs(arena.compute_angular_difference(phi_j, phi_j_plus_1))*0.5
        width_j = r + R - (R - r) * cos(betaj)
        min_widths[j] = max(min_widths[j-1], width_j)

    arena.plot_corridors(corridor_list = corridor_list, color = 'k', linestyle = '-', linewidth = 1.5, figure = None, plot_vectors = False)
    turn = corridor_list[0].compute_relative_turn_direction(corridor_list[1])
    corner_point = arena.get_corner_point(
    corridor_list[0],
    corridor_list[0+1],
    turn)
    xc, yc = arena.compute_center_coordinates_second_circle(corner_point, turn, R, r*2, 0, corridor_list[0].tilt, corridor_list[0+1].tilt, corridor_list[0], corridor_list[0+1])
    angle = atan2(corner_point[1] - yc, corner_point[0] - xc)
    plt.plot([xc, (xc + (R - r) * cos(beta1) * cos(angle +beta1))], [yc, (yc + (R - r) * cos(beta1) * sin(angle + beta1))], 'k-', linewidth = 1)
    plt.plot([xc, (xc + (R - r) * cos(beta1) * cos(corridor_list[0].tilt - pi/2))], [yc, (yc + (R - r) * cos(beta1) * sin(corridor_list[0].tilt - pi/2))], 'k-', linewidth = 1)
    plt.arrow(xc, yc, (R-r) * cos(beta1), 0, 
          head_width=0, 
          head_length=0, 
          fc='black', 
          ec='black')
    plt.arrow(xc, yc, ((R-r) * cos(beta1))* cos(corridor_list[0].tilt - pi/2), ((R-r) * cos(beta1)) * sin(corridor_list[0].tilt - pi/2), 
          head_width=0, 
          head_length=0, 
          fc='black', 
          ec='black')
    plt.show(block = True   )

    return min_widths

def compute_minimum_widths(planner):
    corridor_list = planner.corridor_list
    R = planner.vehicle.max_radius
    r = planner.vehicle.width * 0.5

    # Extract tilt angles
    phis = [c.tilt for c in corridor_list]
    
    # Compute betas
    betas = [0.5 * abs(arena.compute_angular_difference(phis[i], phis[i+1]))
             for i in range(len(phis)-1)]
    
    # Compute q values
    q = [(R - r) * cos(beta) for beta in betas]
    
    min_widths = []
    
    # First corridor:
    min_widths.append(r + R - q[0])
    
    # Middle corridors:
    for i in range(1, len(phis)-1):
        mw = max(r + R - q[i-1], r + R - q[i])
        min_widths.append(mw)
    
    # Last corridor:
    min_widths.append(r + R - q[-1])
    
    return min_widths

mp = arena.MotionPlanner(vehicle, corridor_list)
min_widths = compute_minimum_widths(mp)
m_w = check_minimum_width_corridors(mp)

print(min_widths)

print(minimum_width1)
print(min_width2)
print(min_width3)
print(min_width4)
print(minimum_width54)


