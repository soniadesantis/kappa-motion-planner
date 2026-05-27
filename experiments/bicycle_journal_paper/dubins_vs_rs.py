from math import asin, atan2, cos, pi, sin, sqrt

import matplotlib.pyplot as plt
import numpy as np

from arena import (
    BackwardArc,
    Bicycle,
    Circle,
    CurvilinearArcUnicycle,
    IntermediateCircle,
    Point,
    Pose,
    compute_traj_to_circle_free_space_bicycle,
    compute_two_maneuvers,
    plot_analytical_trajectory,
)
from arena.helpers.helper_functions import (
    compute_angular_difference_with_turn_direction,
    wrapPositiveAngle,
)

## Set font style for all plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({
    'font.size': 10,          # base font size
    'axes.labelsize': 14,     # x- and y-labels
    'axes.titlesize': 18,     # axes titles
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

## Define Bicycle vehicle
bicycle = Bicycle(model = 'Bicycle circular')
R = bicycle.max_radius

## Define the circle to reach
xc, yc = 0, 5
x0, y0 = 0, 0
tau1 = 1

## Define circle to be reached (corner point is not important)
circ1 = IntermediateCircle(center=Point(xc, yc), radius=R, turn_direction = tau1, corner_point = Point(0.5, 0.5))
target_angle = pi/3
xt, yt = xc + R * cos(target_angle), yc + R * sin(target_angle)
thetat = target_angle +tau1 * pi/2 # Final orientation of the unicycle
end_pose = Pose(position=Point(xt, yt), theta = thetat)

## Some useful computations
alpha0 = wrapPositiveAngle(atan2((yc - y0), (xc - x0)))
a = sqrt((xc - x0)**2 + (yc - y0)**2)
beta = asin(R/a)
angle_of_separation = alpha0 - tau1 * beta
cat1 = 2 * R # Radius of enlarged second circle for right-left case
cat2 = sqrt(a**2 - cat1**2) # Lenght of tangent to second circle for right-left case
beta_2r = asin(cat1/a)
gamma = alpha0 - tau1 * beta_2r # Direction of center first circle for right-left case
xc_ref, yc_ref = x0 + cat2 * cos(gamma) , y0 + cat2 * sin(gamma)

## Define useful arrays of size N 
N = 100
angle_array = np.linspace(0, 2*pi, N)
theta_array = angle_of_separation + angle_array

motion_time_array_ll = [0] * N
motion_time_array_rl = [0] * N

dubins_motion_time_ll = [0] * N
dubins_motion_time_rl = [0] * N

motion_time_backward_arc = [0] * N
motion_time_forward_arc = [0] * N
motion_time_segment = [0] * N
motion_time_final_arc = [0] * N
rs_total_angular_displacement = [0] * N

dubins_first_arc_motion_time_ll = [0] * N
dubins_first_arc_motion_time_rl = [0] * N
dubins_total_angular_displacement = [0] * N

## Compute the limit angles for rl and ll that do not require a backward maneuver
limit_theta0_rl = alpha0 - tau1*(beta_2r - pi/2)
limit_theta0_ll = alpha0 - tau1 * pi/2
limit_start_pose_ll = Pose(position=Point(x0, y0), theta = limit_theta0_ll)
limit_start_pose_rl = Pose(position=Point(x0, y0), theta = limit_theta0_rl)

## Compute the limit trajectories for rl and ll
limit_trajectory_ll = compute_traj_to_circle_free_space_bicycle(limit_start_pose_ll, bicycle, circ1, tau0 = 1)
limit_trajectory_rl = compute_traj_to_circle_free_space_bicycle(limit_start_pose_rl, bicycle, circ1, tau0 = -1)

# Compute an example trajectory (modify if needed)
theta0_example_ll = -pi/6
trajectory_example_ll = compute_traj_to_circle_free_space_bicycle(
    Pose(
        position=Point(x0, y0),
        theta = theta0_example_ll 
    ),
    bicycle,
    circ1,
    tau0=1,
)

trajectory_example_ll.append(
    CurvilinearArcUnicycle(xc=circ1.xc, yc=circ1.yc, x0 = trajectory_example_ll[-1].xf, y0 = trajectory_example_ll[-1].yf,
                                        theta0 = trajectory_example_ll[-1].thetaf, xf = xt, yf = yt,
                                        thetaf = thetat, radius = circ1.radius,
                                        turn_direction = tau1, v = bicycle.max_radius,
                                        omega = bicycle.omega_max, unicycle = bicycle,
                                        t0 = trajectory_example_ll[-1].tf, samples_number = 100)
)

man1, man2 = compute_two_maneuvers([x0, y0, theta0_example_ll], bicycle, xc, yc, tau1, t0 = 0, turn1 = 1)
trajectory_example_ll_rs = [man1, man2]
trajectory_example_ll_rs.append(
        CurvilinearArcUnicycle(xc=circ1.xc, yc=circ1.yc, x0 = trajectory_example_ll_rs[-1].xf, y0 = trajectory_example_ll_rs[-1].yf,
                                        theta0 = trajectory_example_ll_rs[-1].thetaf, xf = xt, yf = yt,
                                        thetaf = thetat, radius = circ1.radius,
                                        turn_direction = tau1, v = bicycle.max_radius,
                                        omega = bicycle.omega_max, unicycle = bicycle,
                                        t0 = trajectory_example_ll_rs[-1].tf, samples_number = 100)
) 

## Compute an example trajectory for the rl case (modify if needed)
trajectory_example_rl = compute_traj_to_circle_free_space_bicycle(
    Pose(
        position=Point(x0, y0),
        theta=angle_of_separation + 0.3,
    ),
    bicycle,
    circ1,
    tau0=-1,
)

motion_time_trajectory_example_ll = (
    trajectory_example_ll[-1].tf
    + abs(
        compute_angular_difference_with_turn_direction(
            trajectory_example_ll[-1].thetaf,
            end_pose.theta,
            1,
        )
    )
    * R
    / bicycle.v_max
)

motion_time_trajectory_example_rl = (
    trajectory_example_rl[-1].tf
    + abs(
        compute_angular_difference_with_turn_direction(
            trajectory_example_rl[-1].thetaf,
            end_pose.theta,
            1,
        )
    )
    * R
    / bicycle.v_max
)

## Compute the expected intersection angle
half_angle = (theta_array[-1] - theta_array[0])/2
intersection_angle = angle_of_separation + half_angle

## Compute the trajectories at the expected intersection angle to verify that the motion times are equal
intersection_trajectory_ll = compute_traj_to_circle_free_space_bicycle(Pose(position=Point(x0, y0), theta = intersection_angle), bicycle, circ1, tau0 = 1)
intersection_trajectory_rl = compute_traj_to_circle_free_space_bicycle(Pose(position=Point(x0, y0), theta = intersection_angle), bicycle, circ1, tau0 = -1)
final_arc_ll = abs(compute_angular_difference_with_turn_direction(intersection_trajectory_ll[-1].thetaf, end_pose.theta, 1))
intersection_motion_time_ll = intersection_trajectory_ll[-1].tf + final_arc_ll * R / bicycle.v_max
final_arc_rl = abs(compute_angular_difference_with_turn_direction(intersection_trajectory_rl[-1].thetaf, end_pose.theta, 1))
intersection_motion_time_rl = intersection_trajectory_rl[-1].tf + final_arc_rl * R / bicycle.v_max
print(f"Motion time at intersection angle {intersection_angle:.8f} rad: LL = {intersection_motion_time_ll:.8f} s, RL = {intersection_motion_time_rl:.8f} s")
print(f"Orientation of segment ll at intersection: {intersection_trajectory_ll[-1].thetaf:.8f} rad, orientation of segment rl at intersection: {intersection_trajectory_rl[-1].thetaf:.8f} rad")

## First Plot: intersection trajectories
figure1, ax1 = plt.subplots(figsize=(10, 10))
ax1.set_aspect("equal", adjustable="box")
ax1.set_title("Bicycle intersection trajectories")
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

## Main for loop
for i, theta0 in enumerate(theta_array): 
    # Define start pose
    start_pose = Pose(position=Point(x0, y0), theta = theta0)

    # Compute RS trajectories and motion times
    trajectory_ll = compute_traj_to_circle_free_space_bicycle(start_pose, bicycle, circ1, tau0 = 1)
    final_arc_ll = abs(compute_angular_difference_with_turn_direction(trajectory_ll[-1].thetaf, end_pose.theta, 1))
    motion_time_array_ll[i] = trajectory_ll[-1].tf + final_arc_ll * R / bicycle.v_max
    
    trajectory_rl = compute_traj_to_circle_free_space_bicycle(start_pose, bicycle, circ1, tau0 = -1)
    final_arc_rl = abs(compute_angular_difference_with_turn_direction(trajectory_rl[-1].thetaf, end_pose.theta, 1))
    motion_time_array_rl[i] = trajectory_rl[-1].tf + final_arc_rl * R / bicycle.v_max

    # Store the motion times of the different motion primitives of the trajectory
    bw_arc = 0
    fw_arc = 0
    
    if isinstance(trajectory_ll[0], BackwardArc): 
        motion_time_backward_arc[i] = trajectory_ll[0].maneuver_time
        bw_arc = trajectory_ll[0].delta_angle
    if isinstance(trajectory_ll[0], CurvilinearArcUnicycle):
        motion_time_forward_arc[i] = trajectory_ll[0].maneuver_time
        fw_arc = trajectory_ll[0].delta_angle
    else: 
        motion_time_forward_arc[i] = trajectory_ll[1].maneuver_time
        fw_arc = trajectory_ll[1].delta_angle

    rs_total_angular_displacement[i] = final_arc_ll + fw_arc + bw_arc
    motion_time_segment[i] = trajectory_ll[-1].maneuver_time

    motion_time_final_arc[i] = final_arc_ll * R / bicycle.v_max

    # Compute Dubins trajectories and motion times
    C1_ll, S1_ll = compute_two_maneuvers([start_pose.x, start_pose.y, start_pose.theta], bicycle, xc, yc, tau1, t0 = 0, turn1 = 1)
    C2_ll = abs(compute_angular_difference_with_turn_direction(S1_ll.thetaf, end_pose.theta, 1))
    dubins_motion_time_ll[i] = C1_ll.maneuver_time + S1_ll.maneuver_time + C2_ll * R / bicycle.v_max
    C1_rl, S1_rl = compute_two_maneuvers([start_pose.x, start_pose.y, start_pose.theta], bicycle, xc, yc, tau1, t0 = 0, turn1 = -1)
    C2_rl = abs(compute_angular_difference_with_turn_direction(S1_rl.thetaf, end_pose.theta, 1))
    dubins_motion_time_rl[i] = C1_rl.maneuver_time + S1_rl.maneuver_time + C2_rl * R / bicycle.v_max
    dubins_first_arc_motion_time_ll[i] = C1_ll.maneuver_time
    dubins_total_angular_displacement[i] = C2_ll + abs(C1_ll.delta_angle)
    dubins_first_arc_motion_time_rl[i] = C1_rl.maneuver_time

    # Check for intersection
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

## Final Plots
# Figure 1: Example trajectories and/or limit trajectories
figure1, ax1 = plt.subplots(figsize=(10, 10))
ax1.set_aspect("equal", adjustable="box")
ax1.set_title(f"Motion times example trajectories = LL: {motion_time_trajectory_example_ll:.6f} s, RL: {motion_time_trajectory_example_rl:.6f} s")
plt.plot(xc + R * np.cos(angle_array), yc + R * np.sin(angle_array), 'k--', label='Target Circle')
# plt.plot(xc_ref + R * np.cos(angle_array), yc_ref + R * np.sin(angle_array), 'r--', label='Reflected Circle')
# plot_analytical_trajectory(limit_trajectory_ll, figure=figure1)
# plot_analytical_trajectory(limit_trajectory_rl, figure=figure1)
plot_analytical_trajectory(trajectory_example_ll, figure=figure1)
plot_analytical_trajectory(trajectory_example_ll_rs, figure=figure1)
ax1.arrow(x=x0, y=y0, dx=0.3 * cos(theta0_example_ll), dy=0.3 * sin(theta0_example_ll), head_width=0.1, head_length=0.2, fc='blue', ec='blue')
# plot_analytical_trajectory(trajectory_example_rl, figure=figure1)
ax1.arrow(x=x0, y=y0, dx=0.3 * cos(limit_theta0_ll), dy=0.3 * sin(limit_theta0_ll), head_width=0.1, head_length=0.2, fc='blue', ec='blue')
ax1.arrow(x=x0, y=y0, dx=0.3 * cos(limit_theta0_rl), dy=0.3 * sin(limit_theta0_rl), head_width=0.1, head_length=0.2, fc='orange', ec='orange')
plt.legend()
# plt.savefig("example_multiple_path.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)

# Figure 2: motion time comparison 
fig2, ax2 = plt.subplots(figsize=(8, 3.2))  # same aspect ratio as Figure 3
ax2.set_aspect("auto")  # time plots usually shouldn't be forced "equal"

linewidth = 0.8
color_ll = "#008080"  # teal 
color_rl = "#8E44AD"  # muted magenta 

ax2.set_title("Motion time comparison")

# Vertical reference line 
x_intersection = angle_of_separation + half_angle
x_switch_ll = limit_theta0_ll
x_switch_rl = limit_theta0_rl

print(f"Expected intersection at theta0 = {x_intersection:.4f} rad")
ax2.axvline(
    x=x_intersection,
    color="0.5",
    linewidth=0.5,
    linestyle="--",
    label=r"Intersection angle",
)

ax2.axvline(
    x=x_switch_ll,
    color=color_ll,
    linewidth=0.5,
    linestyle="--",
    label=r"LL switch angle",
)   

ax2.axvline(
    x=x_switch_rl,
    color=color_rl,
    linewidth=0.5,
    linestyle="--",
    label=r"RL switch angle",
)

# Curves 
ax2.plot(
    theta_array,
    motion_time_array_ll,
    linewidth=linewidth,
    color=color_ll,
    label=r"$\boldsymbol{T}^{\mathrm{eq}}$ (Left-Left)",
)
ax2.plot(
    theta_array,
    dubins_motion_time_ll,
    linewidth=linewidth,
    color='r',
    label=r"Dubins (Left-Left)",
)
ax2.plot(
    theta_array,
    dubins_motion_time_rl,
    linewidth=linewidth,
    color='b',
    label=r"Dubins (Right-Left)",
)
ax2.plot(
    theta_array,
    motion_time_array_rl,
    linewidth=linewidth,
    color=color_rl,
    label=r"$\boldsymbol{T}^{\mathrm{neq}}$ (Right-Left)",
)

# Labels 
ax2.set_xlabel(r"Initial orientation $\theta_0$")
ax2.set_ylabel(r"Motion time $\boldsymbol{T}$")

# Legend above plot, frameless, multi-column 
ax2.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    columnspacing=1.2,
    handletextpad=0.6,
)

plt.tight_layout()

## Figure 3: motion time of different primitives for the RS trajectory
fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)

linewidth = 1.0
color = "#2C3E50"  # neutral dark blue/gray

# 1 — backward arc
axes[0].plot(theta_array, motion_time_backward_arc, color=color, linewidth=linewidth)
axes[0].set_ylabel(r"Time backward arc")

# 2 — forward arc
axes[1].plot(theta_array, motion_time_forward_arc, color=color, linewidth=linewidth)
axes[1].set_ylabel(r"Time forward arc")

# 3 — segment
axes[2].plot(theta_array, motion_time_segment, color=color, linewidth=linewidth)
axes[2].set_ylabel(r"Time segment")
# 4 — final arc
axes[3].plot(theta_array, motion_time_final_arc, color=color, linewidth=linewidth)
axes[3].set_ylabel(r"Time final arc")
axes[3].set_xlabel(r"Initial orientation $\theta_0$")

for ax in axes:
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

plt.tight_layout() 

## Figure 4: comparison of the time of the backward arc + forward arc of the RS trajectory with the time of the first arc of the Dubins trajectory
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

linewidth = 1.0
color = "#2C3E50"  # neutral dark blue/gray

# 1 — backward arc
sum_list = [a + b for a, b in zip(motion_time_backward_arc, motion_time_forward_arc)]
axes[0].plot(theta_array, sum_list, color=color, linewidth=linewidth)
axes[0].plot(theta_array, dubins_first_arc_motion_time_ll, color='r', linewidth=linewidth)
axes[0].set_ylabel(r"One forward arc (Dubins) vs one backward + one forward arc (RS)")

# 2 — forward arc
axes[1].plot(theta_array, rs_total_angular_displacement, color=color, linewidth=linewidth)
axes[1].plot(theta_array, dubins_total_angular_displacement, color='r', linewidth=linewidth)
axes[1].set_ylabel(r"Total angular displacement of RS trajectory vs Dubins")
axes[1].axvline(
    x=x_intersection,
    color="0.5",
    linewidth=0.8,
    linestyle="--",
    label="Intersection",
)
for ax in axes:
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

plt.tight_layout() 



for maneuver in trajectory_ll:
    print(f"{maneuver.__class__.__name__} with time {maneuver.maneuver_time:.4f} s")


import numpy as np
import matplotlib.pyplot as plt

t_all = []
theta_all = []

prev_t_last = None

for i, prim in enumerate(trajectory_example_ll):
    t = np.asarray(prim.time_grid).ravel()
    theta = np.asarray(prim.theta_trajectory).ravel()

    if t.size != theta.size:
        raise ValueError(
            f"Primitive {i}: time_grid and theta_trajectory length mismatch "
            f"({t.size} vs {theta.size})"
        )

    # If time is continuous, this is all you need. Optionally drop duplicate boundary point:
    if prev_t_last is not None and np.isclose(t[0], prev_t_last):
        t = t[1:]
        theta = theta[1:]

    t_all.append(t)
    theta_all.append(theta)
    prev_t_last = t[-1]

t_plot = np.concatenate(t_all)
theta_plot = np.concatenate(theta_all)

t_all_rs = []
theta_all_rs = []

prev_t_last = None

for i, prim in enumerate(trajectory_example_ll_rs):
    t = np.asarray(prim.time_grid).ravel()
    theta = np.asarray(prim.theta_trajectory).ravel()

    if t.size != theta.size:
        raise ValueError(
            f"Primitive {i}: time_grid and theta_trajectory length mismatch "
            f"({t.size} vs {theta.size})"
        )

    # If time is continuous, this is all you need. Optionally drop duplicate boundary point:
    if prev_t_last is not None and np.isclose(t[0], prev_t_last):
        t = t[1:]
        theta = theta[1:]

    t_all_rs.append(t)
    theta_all_rs.append(theta)
    prev_t_last = t[-1]

plt.figure()
plt.plot(t_plot, theta_plot, label="Dubins trajectory")
plt.plot(np.concatenate(t_all_rs), np.concatenate(theta_all_rs), label="RS trajectory")
# vertical line at x_intersection
plt.axvline(
    x=x_intersection,
    color="0.5",
    linewidth=0.8,
    linestyle="--",
    label="Intersection",
)
  
plt.xlabel("time")
plt.ylabel("theta")
plt.title("Theta trajectory (stitched primitives)")
plt.grid(True)
plt.show()


plt.show(block=True)