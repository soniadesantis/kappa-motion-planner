import sympy as sp
from math import sin, cos, pi, sqrt, asin, atan2
import numpy as np
from arena import MotionPlanner, compute_motion_time_with_orientation, compute_three_maneuvers_no_collision_avoidance, circle_intersection, CurvilinearArcUnicycle, wrapPositiveAngle, CorridorWorld, Unicycle, get_corridor_from_vector, plot_corridors, compute_turn_direction, get_intersection, check_point_inside_corridor, compute_three_maneuvers_compact, compute_two_maneuvers
from math import sin, cos, pi, sqrt, asin, atan2
import matplotlib.pylab as plt
'''
This script computes the trajectory within a few corridors (example that I show at the beginning of every presentation) and plots it.
'''

# Define parameters

vehicle = Unicycle(model = 'Rosbot circular')
vehicle.set_maximum_forward_velocity(v_max = 1/2)
vehicle.set_maximum_angular_velocity(omega_max = 1/2)
# vehicle.set_length(0.5)
# vehicle.set_width(0.5)
corridor4 = CorridorWorld(width = 1.11, height = 6.38, center = [0, 0], tilt = 0)
corridor3 = CorridorWorld(width = 1.95, height = 3.65, center = [-2.215, 1.27], tilt = 3 * pi/2)
corridor2 = CorridorWorld(width = 1.12, height = 4.51, center = [-3.495, 2.535], tilt = 0)
corridor1 = CorridorWorld(width = 1.45, height = 2.55, center = [-5.025, 3.25], tilt = 3 * pi/2)

corridor_list = [corridor1, corridor2, corridor3, corridor4]
mp = MotionPlanner(vehicle, corridor_list)

trajectory = mp.compute_trajectory_analytical()

figure = plot_corridors(corridor_list = corridor_list, color = 'r', figure = None, plot_vectors = False)
# figure = plt.figure()
for trajectory_piece in trajectory:
    trajectory_piece.plot_path(figure, linewidth = 2.5)
    # Plot circumference
    # if isinstance(trajectory_piece, CurvilinearArcUnicycle):
    #     trajectory_piece.plot_circle(figure)
for trajectory_piece in trajectory[2:-2]:
    # trajectory_piece.plot_path(figure, linewidth = 2.5)
    # Plot circumference
    if isinstance(trajectory_piece, CurvilinearArcUnicycle):
        trajectory_piece.plot_circle(figure)
plt.show(block = True)
