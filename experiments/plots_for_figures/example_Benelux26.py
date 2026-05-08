from math import sin, cos, pi
import arena
import matplotlib.pylab as plt
"""
Example: Analytical motion planning for a unicycle model in a multi-corridor environment.

This example demonstrates:
  • How to construct multiple corridors using `get_corridor_from_vector`.
  • How to load a unicycle vehicle model from the `unicycle_library`.
  • How to modify vehicle parameters after instantiation.
  • How to compute start and end poses using `compute_start_pose` and `compute_end_pose`.
  • How to generate and visualize the resulting analytical trajectory.
"""


### Define corridors ###
width1 =0.6
height1 = 2
phi1 = pi/2

width2 = width1 
height2 = 1.5
phi2 = 0
add_height2 = 0.6

width3 = width2 
height3 = 1.5
phi3 = -pi/2
add_height3 = add_height2

width4 = width3 
height4 = 1.5
phi4 = 0
add_height4 = add_height3

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

corridor_list = [corridor1, corridor2, corridor3, corridor4]

### Define Unicycle vehicle ###
unicycle = arena.Unicycle(model = 'Rosbot circular')
# Modify vehicle parameters
# unicycle.update(v_max = 0.4)
# unicycle.update(omega_max = 1)

### Define initial pose and final pose ###
initial_pose = arena.compute_start_pose(corridor1, unicycle, 0.5)
final_pose = arena.compute_end_pose(corridor4, unicycle, 0.5)

x_percentage_initial = 0.8
y_percentage_initial = -0.7
theta0_relative = pi
x_percentage_final = -0.4
y_percentage_final = 0.8
thetaf_relative = 5 * pi/4

relative_start_pose = [x_percentage_initial,
                         y_percentage_initial,
                         theta0_relative]

relative_end_pose = [x_percentage_final,
                       y_percentage_final,
                       thetaf_relative]
### Define Motion Planner ###
mp = arena.MotionPlanner(unicycle, corridor_list, relative_start_pose=relative_start_pose, relative_end_pose=relative_end_pose)

### Compute analytical trajectory ###
analytical_trajectory = mp.compute_trajectory_analytical()

### Plot results ###
figure = mp.plot_planner_inputs(plot_shrunken_corridors=False)
arena.plot_analytical_trajectory(analytical_trajectory, figure = figure)
plt.axis('off')
plt.savefig("benelux26_path.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)


arena.plot_velocity_profiles(analytical_trajectory, unicycle)
# plt.savefig("example_multiple_velocity.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)

plt.show(block = True)