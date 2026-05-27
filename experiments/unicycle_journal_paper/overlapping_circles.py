from math import sin, cos, pi
import matplotlib.pyplot as plt
import arena
import numpy as np
"""
Example: Analytical motion planning for a bicycle model in a two-corridor environment.

This example demonstrates:
  • How to construct corridors using `get_corridor_from_vector`.
  • How to specify start and end poses in the shrunken corridor reference frames.
  • How to instantiate and configure a bicycle vehicle model.
  • How to compute and visualize the resulting analytical trajectory.
"""

### Define corridors ###
width1, width2 = 1.5, 1.5
height1, height2 = 3.5, 3.5
phi1, phi2 = pi/2, 0

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
corridor1 = arena.get_corridor_from_vector(
    start_point1,
    end_point1,
    width1,
    add_height = 0.2 * height1)

corridor2 = arena.get_corridor_from_vector(
    start_point2,
    end_point2,
    width2,
    add_height = 0.2 * height2)

corridor_list = [corridor1, corridor2]

### Define initial pose and final pose ###
# Define relative poses wrt the center of shrunken corridors
x_percentage_initial = 0.7
y_percentage_initial = -0.2
theta0_relative = -pi/6
x_percentage_final = 0.45
y_percentage_final = 0.3
thetaf_relative = 5 * pi/4

relative_start_pose = [x_percentage_initial,
                         y_percentage_initial,
                         theta0_relative]

relative_end_pose = [x_percentage_final,
                       y_percentage_final,
                       thetaf_relative]

### Define Unicycle vehicle ###
unicycle = arena.Unicycle(model = 'Rosbot circular')
unicycle.update(v_max = 0.8)
unicycle.update(omega_max = 1)

### Define Motion Planner ###
mp = arena.MotionPlanner(unicycle,
                         corridor_list,
                         relative_start_pose=relative_start_pose,
                         relative_end_pose=relative_end_pose)
initial_pose = mp.start_pose
final_pose = mp.end_pose
corridor1 = arena.CorridorWorld(width = 1.5, height = 3.5, center = [0, 2.5], tilt = pi/2)
corridor2 = arena.CorridorWorld(width = 1.5, height = 3.5, center = [1, 3.5], tilt = 0)
corridor_list = [corridor1, corridor2]

mp = arena.MotionPlanner(unicycle,
                         corridor_list,
                         start_pose=initial_pose,
                         end_pose=final_pose)
### Compute analytical trajectory ###
analytical_trajectory = mp.compute_trajectory_analytical()
analytical_trajectory[3].resample(50)
line_width = 3
r = unicycle.width*0.5
angle_array = np.linspace(0, 2*pi, 100)
### Plot results ###
figure = arena.plot_corridors(corridor_list = corridor_list, color = 'k', linestyle = '-', linewidth = 1, figure = None, plot_vectors = False)
arena.plot_corridors(corridor_list = mp.shrunken_corridor_list, color = 'k', linestyle = '--', linewidth = 1, figure = figure, plot_vectors = False)
# plt.title('Analytical Motion Planner - Unicycle in Two Corridors')
# Plot intermediate circumferences
for ind, trajectory_piece in enumerate(analytical_trajectory):
    if isinstance(trajectory_piece, arena.CurvilinearArcUnicycle) and ind>2 and ind < len(analytical_trajectory) -3:
        trajectory_piece.plot_circle(figure, color = 'r')
    elif isinstance(trajectory_piece, arena.CurvilinearArcUnicycle):
        trajectory_piece.plot_circle(figure, color = 'k')
    if ind <= 2:
        trajectory_piece.plot_path(figure, linewidth = line_width, color = 'k')
    elif ind <= 3:
        trajectory_piece.plot_path(figure, linewidth = line_width, color = 'k') # y
    else:
        trajectory_piece.plot_path(figure, linewidth = line_width, color = 'k') # g
for trajectory_piece in analytical_trajectory:
    plt.plot(trajectory_piece.xf + r * np.cos(angle_array), trajectory_piece.yf + r * np.sin(angle_array), 'k-', zorder = 3)
    plt.arrow(trajectory_piece.xf, trajectory_piece.yf, (r + 0.1) * cos(trajectory_piece.thetaf), (r + 0.1) * sin(trajectory_piece.thetaf), head_width = 0.05, color = 'k', zorder = 3)

plt.plot(initial_pose[0] + r * np.cos(angle_array), initial_pose[1] + r * np.sin(angle_array), 'k-')
l = r + 0.1
plt.arrow(initial_pose[0], initial_pose[1], l * cos(initial_pose[2]), l * sin(initial_pose[2]), head_width = 0.05, color = 'k')

# plt.plot(final_pose[0], final_pose[1], 'ro')
plt.plot(final_pose[0] + r * np.cos(angle_array), final_pose[1] + r * np.sin(angle_array), 'k-')
plt.arrow(final_pose[0], final_pose[1], l * cos(final_pose[2]), l * sin(final_pose[2]), head_width = 0.05, color = 'k')

# arena.plot_velocity_profiles(analytical_trajectory, unicycle)
plt.axis('off')
plt.tight_layout()

# plt.savefig("overlapping_circles_blue.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
# plt.savefig(
#     "overlapping_circles_black.pdf",
#     format="pdf",
#     bbox_inches="tight",
#     pad_inches=0,
#     transparent=True
# )
plt.savefig(
    "overlapping_circles_black.svg",
    format="svg",
    bbox_inches="tight",
    pad_inches=0,
    transparent=True
)
plt.show(block = True)

