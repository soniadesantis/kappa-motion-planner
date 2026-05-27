from math import sin, cos, pi, atan2
import numpy as np
from arena import (
    MotionPlanner,
    CorridorWorld,
    Unicycle,
    get_corner_point,
    compute_center_coordinates_second_circle,
    get_bisector_direction,

) 
import arena
from math import sin, cos, pi, sqrt, asin, atan2
import matplotlib.pylab as plt

# Define vehicle
vehicle = Unicycle(model = 'Rosbot circular')
vehicle.width = 0.4
r = vehicle.width*0.5
v_max = 0.8
vehicle.update(v_max = v_max )
vehicle.update(omega_max = 1)

# Define corridors parameters
width1 = 1.1
height1 = 3
phi1 = pi/6

width2 = 1.3
height2 = 2
phi2 = pi/2
add_height2 = 2

width3 = 1.5
height3 = 3
phi3 = -pi/6
add_height3 = 1.5

# Define corridor1 
corridor1 = CorridorWorld(width = width1, height = height1, center = [0, 0], tilt = phi1)

# Define corridor2
tail_vector2 = corridor1.head
head_vector2 = [corridor1.head[0] + height2 * cos(phi2), corridor1.head[1] + height2 * sin(phi2)]
corridor2 = arena.get_corridor_from_vector(tail_vector2, head_vector2, width2, add_height = add_height2)

# Define corridor3 
tail_vector3 = corridor2.head
head_vector3 = [corridor2.head[0] + height3 * cos(phi3), corridor2.head[1] + height3 * sin(phi3)]
corridor3 = arena.get_corridor_from_vector(tail_vector3, head_vector3, width3, add_height = add_height3)

corridor_list = [corridor1, corridor2, corridor3]

mp = MotionPlanner(vehicle, corridor_list)

turn1 = corridor1.compute_relative_turn_direction(corridor2)
corner_point1 = get_corner_point(corridor1, corridor2, turn1)
xc2, yc2 = compute_center_coordinates_second_circle(corner_point1, turn1, vehicle.max_radius, vehicle.width, 2, phi1, phi2, corridor1 = None, corridor2 = None)
angle_bisector = get_bisector_direction(corridor1, corridor2)
# Extract tilt angles
phis = [c.tilt for c in corridor_list]

# Compute betas
betas = [0.5 * abs(arena.compute_angular_difference(phis[i], phis[i+1]))
        for i in range(len(phis)-1)]
    
s = (width2 - mp.min_corridor_widths[0])/cos(betas[0])
xc2_max = corner_point1[0] + (vehicle.max_radius - s - 0.5 * vehicle.width) * cos(angle_bisector)
yc2_max = corner_point1[1] + (vehicle.max_radius - s - 0.5 * vehicle.width) * sin(angle_bisector)

for c in mp.intermediate_circles:
    c.update_s(s = c.s_max)
    
figure = arena.plot_corridors(corridor_list, linestyle= 'solid')
for c in mp.intermediate_circles:
    circle_artist1 = plt.Circle((c.canonical_center[0], c.canonical_center[1]), vehicle.max_radius, color='grey', fill= False, linestyle= 'dashed')
    circle_artist2 = plt.Circle((c.canonical_center[0], c.canonical_center[1]), vehicle.max_radius + vehicle.width/2, color='grey', fill= False, linestyle= 'dashed')
    circle_artist3 = plt.Circle((c.center[0], c.center[1]), vehicle.max_radius, color='red', fill= False, linestyle= 'solid')
    circle_artist4 = plt.Circle((c.center[0], c.center[1]), vehicle.max_radius + vehicle.width/2, color='orange', fill= False, linestyle= 'solid')
    figure.gca().add_artist(circle_artist1)
    figure.gca().add_artist(circle_artist2)
    figure.gca().add_artist(circle_artist3)
    figure.gca().add_artist(circle_artist4)
    figure.gca().plot(*c.canonical_center, color = 'grey', marker='o', markersize=3)
    figure.gca().plot(*c.center, 'ko', markersize=3)
ax = figure.gca()
ax.axis('off')
# for c in mp.intermediate_circles:
#     c.update_s(s = c.s_max/2)
#     circle_artist1 = plt.Circle((c.center[0], c.center[1]), vehicle.max_radius, color='blue', fill= False, linestyle= 'dashed')
#     circle_artist2 = plt.Circle((c.center[0], c.center[1]), vehicle.max_radius + vehicle.width/2, color='blue', fill= False, linestyle= 'dashed')
#     figure.gca().add_artist(circle_artist1)
#     figure.gca().add_artist(circle_artist2)    

# figure.savefig(
#     "max_moving_circle.svg",
#     bbox_inches='tight',
#     pad_inches=0,
#     transparent=True
# )
plt.show(block = True)

