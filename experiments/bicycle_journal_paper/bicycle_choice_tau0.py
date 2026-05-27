import sympy as sp
from math import sin, cos, pi, sqrt, asin, atan2
import numpy as np
from arena import compute_traj_to_circle_free_space_bicycle, Point, Pose, Circle, IntermediateCircle, Bicycle, plot_analytical_trajectory, BackwardArc, CurvilinearArcUnicycle
from math import sin, cos, pi, sqrt, asin, atan2
# import matplotlib.pylab as plt

import matplotlib.pyplot as plt
import matplotlib as mpl

from arena.helpers.helper_functions import compute_angular_difference_with_turn_direction, wrapPositiveAngle

# mpl.rcParams['text.usetex'] = True  # Enable LaTeX
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Default LaTeX serif font
# Set global font to serif (LaTeX-style)
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({
    'font.size': 10,          # base font size
    'axes.labelsize': 14,     # x- and y-labels
    'axes.titlesize': 18,     # axes titles
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

### Define Bicycle vehicle ###
bicycle = Bicycle(model = 'Bicycle circular')
R = bicycle.max_radius
## Define the circle to reach
xc, yc = 0, 5
x0, y0 = 0, 0
tau1 = 1
alpha0 = wrapPositiveAngle(atan2((yc - y0), (xc - x0)))
# Some computations
a = sqrt((xc - x0)**2 + (yc - y0)**2)
beta = asin(R/a)
angle_of_separation = alpha0 - tau1 * beta
cat1 = 2 * R # Radius of enlarged second circle for right-left case
cat2 = sqrt(a**2 - cat1**2) # Lenght of tangent to second circle for right-left case
beta_2r = asin(cat1/a)
gamma = alpha0 - tau1 * beta_2r # Direction of center first circle for right-left case
xc_ref = x0 + cat2 * cos(gamma) 
yc_ref = y0 + cat2 * sin(gamma)
angle_array = np.linspace(0, 2*pi, 100)
theta_array = angle_of_separation + angle_array
motion_time_array_ll = [0] * len(theta_array)
motion_time_array_rl = [0] * len(theta_array)
motion_time_backward_arc = [0] * len(theta_array)
motion_time_forward_arc = [0] * len(theta_array)
motion_time_segmnet = [0] * len(theta_array)
motion_time_final_arc = [0] * len(theta_array)

limit_theta0_rl = alpha0 - tau1*(beta_2r - pi/2)
limit_theta0_ll = alpha0 - tau1 * pi/2
optimal_circle_ll = Circle(center=Point(x0 + R * cos(alpha0), y0 + R * sin(alpha0)), radius=R)
optimal_circle_rl = Circle(center=Point(x0 + R * cos(alpha0 - tau1 * beta_2r), y0 + R * sin(alpha0 - tau1 * beta_2r)), radius=R)

circ1 = IntermediateCircle(center=Point(xc, yc), radius=R, turn_direction = tau1, corner_point = Point(0.5, 0.5))
target_angle = pi/3
xt, yt_val = xc + R * cos(target_angle), yc + R * sin(target_angle)
thetaf_val = target_angle +tau1 * pi/2 # Final orientation of the unicycle
end_pose = Pose(position=Point(xc + R * cos(pi/2), yc + R * sin(pi/2)), theta = pi)
limit_start_pose_ll = Pose(position=Point(x0, y0), theta = limit_theta0_ll)
limit_start_pose_rl = Pose(position=Point(x0, y0), theta = limit_theta0_rl)

limit_trajectory_ll = compute_traj_to_circle_free_space_bicycle(limit_start_pose_ll, bicycle, circ1, tau0 = 1)
limit_trajectory_rl = compute_traj_to_circle_free_space_bicycle(limit_start_pose_rl, bicycle, circ1, tau0 = -1)
half_angle = (theta_array[-1] - theta_array[0])/2
intersection_angle = angle_of_separation + half_angle
intersection_trajectory_ll = compute_traj_to_circle_free_space_bicycle(Pose(position=Point(x0, y0), theta = intersection_angle), bicycle, circ1, tau0 = 1)
intersection_trajectory_rl = compute_traj_to_circle_free_space_bicycle(Pose(position=Point(x0, y0), theta = intersection_angle), bicycle, circ1, tau0 = -1)
final_arc_ll = abs(compute_angular_difference_with_turn_direction(intersection_trajectory_ll[-1].thetaf, end_pose.theta, 1))
intersection_motion_time_array_ll = intersection_trajectory_ll[-1].tf + final_arc_ll * R / bicycle.v_max

final_arc_rl = abs(compute_angular_difference_with_turn_direction(intersection_trajectory_rl[-1].thetaf, end_pose.theta, 1))
intersection_motion_time_array_rl = intersection_trajectory_rl[-1].tf + final_arc_rl * R / bicycle.v_max
print(f"Motion time at intersection angle {intersection_angle:.8f} rad: LL = {intersection_motion_time_array_ll:.8f} s, RL = {intersection_motion_time_array_rl:.8f} s")
print(f"Orientation of segment ll at intersection: {intersection_trajectory_ll[-1].thetaf:.8f} rad, orientation of segment rl at intersection: {intersection_trajectory_rl[-1].thetaf:.8f} rad")
figure1, ax1 = plt.subplots(figsize=(10, 10))
ax1.set_aspect("equal", adjustable="box")
ax1.set_title("Bicycle example")
plt.plot(xc + R * np.cos(angle_array), yc + R * np.sin(angle_array), 'k--', label='Target Circle')
plt.plot(x0 + R * np.cos(angle_array), y0 + R * np.sin(angle_array), 'k--', label='Centers of backward arcs')
for angle in angle_array: 
    plt.plot(x0 + R * np.cos(angle) + R * np.cos(angle_array), y0 + R * np.sin(angle) + R * np.sin(angle_array), 'k--', alpha=0.1)
# plt.plot(xc_ref + R * np.cos(angle_array), yc_ref + R * np.sin(angle_array), 'r--', label='Reflected Circle')
plot_analytical_trajectory(intersection_trajectory_ll, figure=figure1)
plot_analytical_trajectory(intersection_trajectory_rl, figure=figure1)
ax1.arrow(x=x0, y=y0, dx=0.3 * cos(intersection_angle), dy=0.3 * sin(intersection_angle), head_width=0.1, head_length=0.2, fc='blue', ec='blue')
ax1.arrow(x=x0, y=y0, dx=0.3 * cos(intersection_angle), dy=0.3 * sin(intersection_angle), head_width=0.1, head_length=0.2, fc='orange', ec='orange')
plt.legend()
plt.show(block=True)
for i, theta0 in enumerate(theta_array): 
    start_pose = Pose(position=Point(x0, y0), theta = theta0)
    trajectory_ll = compute_traj_to_circle_free_space_bicycle(start_pose, bicycle, circ1, tau0 = 1)
    final_arc_ll = abs(compute_angular_difference_with_turn_direction(trajectory_ll[-1].thetaf, end_pose.theta, 1))
    motion_time_array_ll[i] = trajectory_ll[-1].tf + final_arc_ll * R / bicycle.v_max
    if isinstance(trajectory_ll[0], BackwardArc): 
        motion_time_backward_arc[i] = trajectory_ll[0].maneuver_time
    if isinstance(trajectory_ll[0], CurvilinearArcUnicycle):
        motion_time_forward_arc[i] = trajectory_ll[0].maneuver_time
    else: 
        motion_time_forward_arc[i] = trajectory_ll[1].maneuver_time

    motion_time_segmnet[i] = trajectory_ll[-1].maneuver_time

    motion_time_final_arc[i] = final_arc_ll * R / bicycle.v_max

    trajectory_rl = compute_traj_to_circle_free_space_bicycle(start_pose, bicycle, circ1, tau0 = -1)
    final_arc_rl = abs(compute_angular_difference_with_turn_direction(trajectory_rl[-1].thetaf, end_pose.theta, 1))
    motion_time_array_rl[i] = trajectory_rl[-1].tf + final_arc_rl * R / bicycle.v_max
    # motion_time_array_rl[i] = final_arc_rl * R / bicycle.v_max
    # stop = 2
    if abs(motion_time_array_rl[i] - motion_time_array_ll[i]) < 1e-3:
        print(f"Intersection at theta0 = {theta0:.4f} rad, motion time = {motion_time_array_ll[i]:.4f} s")
        # figure1, ax1 = plt.subplots(figsize=(10, 10))
        # ax1.set_aspect("equal", adjustable="box")
        # ax1.set_title(f"Motion time ll = {motion_time_array_ll[i]:.2f} s, motion time rl = {motion_time_array_rl[i]:.2f} s")
        # plt.plot(optimal_circle_ll.xc + optimal_circle_ll.radius * np.cos(angle_array), optimal_circle_ll.yc + optimal_circle_ll.radius * np.sin(angle_array), 'g--', label='Optimal Circle LL')
        # plt.plot(optimal_circle_rl.xc + optimal_circle_rl.radius * np.cos(angle_array), optimal_circle_rl.yc + optimal_circle_rl.radius * np.sin(angle_array), 'b--', label='Optimal Circle RL')
        # plt.plot(xc + R * np.cos(angle_array), yc + R * np.sin(angle_array), 'k--', label='Target Circle')
        # plt.plot(xc_ref + R * np.cos(angle_array), yc_ref + R * np.sin(angle_array), 'r--', label='Reflected Circle')
        # plot_analytical_trajectory(trajectory_ll, figure=figure1)
        # plot_analytical_trajectory(trajectory_rl, figure=figure1)
        # ax1.arrow(x=x0, y=y0, dx=0.3 * cos(theta0), dy=0.3 * sin(theta0), head_width=0.1, head_length=0.2, fc='blue', ec='blue')
        # plt.show(block=True)
        # stop = 2

figure1, ax1 = plt.subplots(figsize=(10, 10))
ax1.set_aspect("equal", adjustable="box")
ax1.set_title("Analytical Motion Planner - Unicycle Within Multiple Corridors")
plt.plot(xc + R * np.cos(angle_array), yc + R * np.sin(angle_array), 'k--', label='Target Circle')
plt.plot(xc_ref + R * np.cos(angle_array), yc_ref + R * np.sin(angle_array), 'r--', label='Reflected Circle')
plot_analytical_trajectory(limit_trajectory_ll, figure=figure1)
plot_analytical_trajectory(limit_trajectory_rl, figure=figure1)
ax1.arrow(x=x0, y=y0, dx=0.3 * cos(limit_theta0_ll), dy=0.3 * sin(limit_theta0_ll), head_width=0.1, head_length=0.2, fc='blue', ec='blue')
ax1.arrow(x=x0, y=y0, dx=0.3 * cos(limit_theta0_rl), dy=0.3 * sin(limit_theta0_rl), head_width=0.1, head_length=0.2, fc='orange', ec='orange')
plt.legend()
# plt.savefig("example_multiple_path.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)

# Figure 2: motion time comparison (styled like Figure 3)
fig2, ax2 = plt.subplots(figsize=(8, 3.2))  # same aspect ratio as Figure 3
ax2.set_aspect("auto")  # time plots usually shouldn't be forced "equal"

linewidth = 0.8
color_ll = "#008080"  # teal (same as Fig 3)
color_rl = "#8E44AD"  # muted magenta (same as Fig 3)

ax2.set_title("Motion time comparison")

# Vertical reference line (style like Fig 3)
x_intersection = angle_of_separation + half_angle
print(f"Expected intersection at theta0 = {x_intersection:.4f} rad")
ax2.axvline(
    x=x_intersection,
    color="0.5",
    linewidth=0.5,
    linestyle="--",
    label=r"Intersection angle",
)

# Curves (use ax2, not plt)
ax2.plot(
    theta_array,
    motion_time_array_ll,
    linewidth=linewidth,
    color=color_ll,
    label=r"$\boldsymbol{T}^{\mathrm{eq}}$ (Left-Left)",
)
ax2.plot(
    theta_array,
    motion_time_array_rl,
    linewidth=linewidth,
    color=color_rl,
    label=r"$\boldsymbol{T}^{\mathrm{neq}}$ (Right-Left)",
)

# Labels (optional but matches Fig 3 vibe)
ax2.set_xlabel(r"Initial orientation $\theta_0$")
ax2.set_ylabel(r"Motion time $\boldsymbol{T}$")

# Legend above plot, frameless, multi-column (like Fig 3)
ax2.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    columnspacing=1.2,
    handletextpad=0.6,
)

plt.tight_layout()

fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)

linewidth = 1.0
color = "#2C3E50"  # neutral dark blue/gray

# 1 — backward arc
axes[0].plot(theta_array, motion_time_backward_arc, color=color, linewidth=linewidth)
axes[0].set_ylabel(r"$T_{\mathrm{back}}$")

# 2 — forward arc
axes[1].plot(theta_array, motion_time_forward_arc, color=color, linewidth=linewidth)
axes[1].set_ylabel(r"$T_{\mathrm{fwd}}$")

# 3 — segment
axes[2].plot(theta_array, motion_time_segmnet, color=color, linewidth=linewidth)
axes[2].set_ylabel(r"$T_{\mathrm{seg}}$")

# 4 — final arc
axes[3].plot(theta_array, motion_time_final_arc, color=color, linewidth=linewidth)
axes[3].set_ylabel(r"$T_{\mathrm{final}}$")
axes[3].set_xlabel(r"Initial orientation $\theta_0$")

# Styling similar to your other figures
for ax in axes:
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

plt.tight_layout() 
plt.show(block=True)