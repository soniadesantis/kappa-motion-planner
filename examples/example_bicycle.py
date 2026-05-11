from math import sin, cos, pi
import matplotlib.pyplot as plt

from kappa_planner.motion_planner import MotionPlanner
from kappa_planner.vehicle import Bicycle
from kappa_planner.helpers.corridor_geometry import get_corridor_from_vector
from kappa_planner.helpers.plot_helpers import plot_analytical_trajectory, plot_velocity_profiles
"""
Example: Analytical motion planning for a bicycle model in a two-corridor environment.

This example demonstrates:
  • How to construct corridors using `get_corridor_from_vector`.
  • How to specify start and end poses in the shrunken corridor reference frames.
  • How to instantiate and configure a bicycle vehicle model.
  • How to compute and visualize the resulting analytical trajectory.
"""

### Define corridors ###
width1, width2 = 3, 3
height1, height2 = 6, 4
phi1, phi2 = pi/2, pi/6

# Instantiate two corridors by defining two consecutive vectors. 
# Tail point corridor1
start_point1 = [0, 0] 
# Head point corridor1
end_point1   = [start_point1[0] + height1 * cos(phi1),
                start_point1[1] + height1 * sin(phi1)] # Head point
# Tail point corridor2
start_point2 = [end_point1[0],
                end_point1[1]]
# Head point corridor2
end_point2   = [start_point2[0] + height2 * cos(phi2),
                start_point2[1] + height2 * sin(phi2)]

# Instantiate corridors with get_corridor_from_vector method
corridor1 = get_corridor_from_vector(
    start_point1,
    end_point1,
    width1,
    add_height = 0.2 * height1)

corridor2 = get_corridor_from_vector(
    start_point2,
    end_point2,
    width2,
    add_height = 0.2 * height2)

corridor_list = [corridor1, corridor2]

### Define initial pose and final pose ###
# Define relative poses wrt the center of shrunken corridors
x_percentage_initial = 0.8
y_percentage_initial = -0.7
theta0_relative = -pi/2
x_percentage_final = -0.4
y_percentage_final = 0.8
thetaf_relative = 5 * pi/4

relative_start_pose = [x_percentage_initial,
                         y_percentage_initial,
                         theta0_relative]

relative_end_pose = [x_percentage_final,
                       y_percentage_final,
                       thetaf_relative]

### Define Bicycle vehicle ###
vehicle_width = 0.430 
vehicle_length = 0.508
vehicle_wheelbase = 0.4
vehicle_vmax = 1
vehicle_deltamax = 0.5

bicycle = Bicycle(
    [0,0,0],
    width = vehicle_width,
    length = vehicle_length,
    wheelbase = vehicle_wheelbase, 
    v_max = vehicle_vmax,
    v_min = -vehicle_vmax,
    delta_max = vehicle_deltamax, 
    delta_min = -vehicle_deltamax)

### Define Motion Planner ###
mp = MotionPlanner(bicycle,
                         corridor_list,
                         relative_start_pose=relative_start_pose,
                         relative_end_pose=relative_end_pose)

### Compute analytical trajectory ###
analytical_trajectory = mp.compute_trajectory_analytical()
print(f"Analytical trajectory computed in {mp.comp_time_analytical_sol} seconds.")
### Plot results ###
figure = mp.plot_planner_inputs()
plt.title('Analytical Motion Planner - Bicycle in Two Corridors')
plot_analytical_trajectory(analytical_trajectory, figure)
plt.savefig("example_bicycle_path.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)

plot_velocity_profiles(analytical_trajectory, bicycle)
plt.savefig("example_bicycle_velocity.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)

plt.show(block = True)

