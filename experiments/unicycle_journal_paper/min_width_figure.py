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

width2 = 1.4
height2 = 2
phi2 = pi/2
add_height2 = 2

width3 = width1
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
shrunk_corridor_list = arena.shrink_corridor_list(corridor_list, r)
x0 = (shrunk_corridor_list[0].center[0] - 1*cos(phi1)) + shrunk_corridor_list[0].width/2 * cos(shrunk_corridor_list[0].tilt - pi/2)
y0 = (shrunk_corridor_list[0].center[1] - 1*sin(phi1)) + shrunk_corridor_list[0].width/2 * sin(shrunk_corridor_list[0].tilt - pi/2)
initial_pose = arena.compute_start_pose(corridor1, vehicle, 0)
initial_pose[0] = x0
initial_pose[1] = y0
initial_pose[2] = phi1 + pi/6
final_pose = arena.compute_end_pose(corridor3, vehicle, 0.5)

mp = MotionPlanner(vehicle, corridor_list)
mp_min_widths_corridors = mp.min_corridor_widths
corridor1_min_width = CorridorWorld(mp_min_widths_corridors[0], corridor1.height, corridor1.center, corridor1.tilt)
corridor2_min_width = CorridorWorld(mp_min_widths_corridors[1], corridor2.height, corridor2.center, corridor2.tilt)
corridor3_min_width = CorridorWorld(mp_min_widths_corridors[2], corridor3.height, corridor3.center, corridor3.tilt)
min_width_corridor_list = [corridor1_min_width, corridor2_min_width, corridor3_min_width]
mp_min_widths_corridors = MotionPlanner(vehicle, min_width_corridor_list)

mp1 = MotionPlanner(vehicle, [corridor1, corridor2])
mp1_min_widths_corridors = mp1.min_corridor_widths 

# corridor1_min_width_mp1 = CorridorWorld(mp1_min_widths_corridors[0], corridor1.height, corridor1.center, corridor1.tilt)
# corridor2_min_width_mp1 = CorridorWorld(mp1_min_widths_corridors[1], corridor2.height, corridor2.center, corridor2.tilt)

# mp2 = MotionPlanner(vehicle, [corridor2, corridor3])
# mp2_min_widths_corridors = mp2.min_corridor_widths
# corridor2_min_width_mp2 = CorridorWorld(mp2_min_widths_corridors[0], corridor2.height, corridor2.center, corridor2.tilt)
# corridor3_min_width_mp2 = CorridorWorld(mp2_min_widths_corridors[1], corridor3.height, corridor3.center, corridor3.tilt)

# figure = mp.plot_planner_inputs()
figure = arena.plot_corridors(min_width_corridor_list, linestyle= 'solid')
for center in mp_min_widths_corridors.intermediate_circle_centers:
    circle = plt.Circle((center[0], center[1]), vehicle.max_radius, color='red', fill= False, linestyle= 'solid')
    figure.gca().add_artist(circle)
    circle_outer = plt.Circle((center[0], center[1]), vehicle.max_radius + r, color='orange', fill= False, linestyle= 'solid')
    figure.gca().add_artist(circle_outer)
    figure.gca().plot(center[0], center[1], 'ko', markersize=3)
    figure.gca().plot(center[0], center[1], 'ko', markersize=3)
ax = figure.gca()
ax.axis('off')
figure.savefig(
    "min_width_corridors.svg",
    bbox_inches='tight',
    pad_inches=0,
    transparent=True
)
# arena.plot_corridors([corridor1_min_width_mp1, corridor2_min_width_mp1], figure = figure)
# arena.plot_corridors([corridor2_min_width_mp2, corridor3_min_width_mp2], figure = figure)
plt.show()