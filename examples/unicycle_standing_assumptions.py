from math import sin, cos, pi

import matplotlib.pylab as plt

from kappa_planner.corridor import CorridorWorld
from kappa_planner.motion_planner import MotionPlanner
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.corridor_geometry import get_corridor_from_vector
from kappa_planner.helpers.plot_helpers import plot_analytical_trajectory, plot_velocity_profiles

"""
Example: Analytical motion planning for a unicycle model in a multi-corridor environment.

This example demonstrates:
  • How to construct multiple corridors using `get_corridor_from_vector`.
  • How to load a unicycle vehicle model from the `unicycle_library`.
  • How to modify vehicle parameters after instantiation.
  • How to specify offset start and end poses in normalized corridor frames.
  • How to generate and visualize the resulting analytical trajectory.
"""


### Define corridors ###
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

corridor_list = [corridor1, corridor2, corridor3, corridor4, corridor5]

### Define Unicycle vehicle ###
unicycle = Unicycle(model = 'Rosbot circular')
# Modify vehicle parameters
unicycle.update(v_max = 0.8)
unicycle.update(omega_max = 1)

### Define initial pose and final pose ###
# Positions are fractions of the shrunken corridor dimensions in [-1, 1].
# Both poses are off-center, and their headings differ from corridor traversal.
# Heading is measured from the corridor frame's x axis (across its width).
relative_start_pose = [0.7, -0.65, 5*pi/4]
relative_end_pose = [-0.7, 0.65, pi]

### Define Bicycle vehicle ###
vehicle_width = 0.430 
vehicle_length = 0.508
vehicle_wheelbase = 0.4
vehicle_vmax = 1
vehicle_deltamax = 0.5

### Define Motion Planner ###
mp = MotionPlanner(unicycle, corridor_list,
                   relative_start_pose=relative_start_pose,
                   relative_end_pose=relative_end_pose,
                   assumptions="standing")

### Compute analytical trajectory ###
analytical_trajectory = mp.compute_trajectory_analytical()
print(f"Analytical trajectory computed in {mp.comp_time_analytical_sol} seconds.")

### Plot results ###
figure, ax = plt.subplots(figsize=(12, 6))
mp.plot_planner_inputs(figure=figure, plot_shrunken_corridors=False,
                       plot_corridor_numbers=True)
plot_analytical_trajectory(analytical_trajectory, figure=ax, color="#2563eb", linewidth=2.5)
ax.set(title="Standing unicycle planner: five corridors", xlabel="x [m]", ylabel="y [m]")
ax.grid(alpha=0.15)
figure.tight_layout()
plot_velocity_profiles(analytical_trajectory, unicycle)

plt.show(block=True)
