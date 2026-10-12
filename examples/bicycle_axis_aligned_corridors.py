"""Example: motion planning for a bicycle in axis-aligned corridors."""

from math import pi

import matplotlib.pylab as plt

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.plot_helpers import plot_velocity_profiles
from kappa_planner.motion_planner import MotionPlanner
from kappa_planner.vehicle import Bicycle


### Define corridors ###
# At tilt = 0, the corridor height is parallel to the global x axis and the
# corridor width is parallel to the global y axis.

width1 = 4.032
height1 = 20.044
center1 = [10.380, 17.000]

width2 = 7.729
height2 = 4.076
center2 = [18.364, 15.7085]

width3 = 1.874
height3 = 7.578
center3 = [20.044, 12.781]

width4 = 5.979
height4 = 1.600
center4 = [23.033, 11.1135]

width5 = 1.532
height5 = 7.230
center5 = [24.188, 8.890]

width6 = 9.533
height6 = 2.278
center6 = [26.664, 5.9715]

width7 = 6.584
height7 = 5.778
center7 = [28.295, 4.497]

width8 = 3.562
height8 = 9.779
center8 = [34.8085, 4.497]

width9 = 8.696
height9 = 2.320
center9 = [38.538, 6.888]

width10 = 4.591
height10 = 9.449
center10 = [41.4525, 8.9405]

corridor1 = CorridorWorld(width=width1, height=height1, center=center1)
corridor2 = CorridorWorld(width=width2, height=height2, center=center2)
corridor3 = CorridorWorld(width=width3, height=height3, center=center3)
corridor4 = CorridorWorld(width=width4, height=height4, center=center4)
corridor5 = CorridorWorld(width=width5, height=height5, center=center5)
corridor6 = CorridorWorld(width=width6, height=height6, center=center6)
corridor7 = CorridorWorld(width=width7, height=height7, center=center7)
corridor8 = CorridorWorld(width=width8, height=height8, center=center8)
corridor9 = CorridorWorld(width=width9, height=height9, center=center9)
corridor10 = CorridorWorld(width=width10, height=height10, center=center10)

corridor_list = [
    corridor1,
    corridor2,
    corridor3,
    corridor4,
    corridor5,
    corridor6,
    corridor7,
    corridor8,
    corridor9,
    corridor10,
]


### Define Bicycle vehicle ###
footprint_radius = 0.5
vehicle_wheelbase = 1.0
vehicle_vmax = 0.5
vehicle_deltamax = pi / 4

bicycle = Bicycle(
    state=[0, 0, 0],
    footprint_radius=footprint_radius,
    wheelbase=vehicle_wheelbase,
    v_max=vehicle_vmax,
    v_min=-vehicle_vmax,
    delta_max=vehicle_deltamax,
    delta_min=-vehicle_deltamax,
)


### Define initial pose and final pose ###
initial_pose = [8.675, 17.053, pi]
final_pose = [45.351, 9.680, 1.553005]


### Define Motion Planner ###
mp = MotionPlanner(
    bicycle,
    corridor_list,
    start_pose=initial_pose,
    end_pose=final_pose,
    assumptions="axis-aligned",
)


### Compute analytical trajectory ###
analytical_trajectory = mp.compute_trajectory_analytical()

print(
    "Analytical trajectory computed in "
    f"{mp.comp_time_analytical_sol} seconds."
)
print(f"Solution returned by the planner: {mp.solution_source}.")
print(f"Trajectory traversal time: {mp.traversal_time:.2f} seconds.")

if mp.baseline.trajectory is not None:
    baseline_time = mp.baseline.traversal_time
    time_reduction = 100 * (
        baseline_time - mp.traversal_time
    ) / baseline_time
    print(f"Baseline traversal time: {baseline_time:.2f} seconds.")
    print(f"Traversal-time reduction: {time_reduction:.2f}%.")


### Plot results ###
figure, ax = plt.subplots(figsize=(12, 6))

mp.plot_planner_inputs(
    figure=figure,
    plot_shrunken_corridors=False,
    plot_corridor_numbers=True,
)

# Plot the complete baseline as a dashed grey curve.
if mp.baseline.trajectory is not None:
    for index, maneuver in enumerate(mp.baseline.trajectory):
        maneuver.plot_path(
            ax,
            color="#6b7280",
            linestyle="--",
            linewidth=2,
            label="Baseline" if index == 0 else None,
        )

# Plot the trajectory returned by the planner in blue.
for index, maneuver in enumerate(analytical_trajectory):
    maneuver.plot_path(
        ax,
        color="#2563eb",
        linewidth=2.5,
        label="Returned trajectory" if index == 0 else None,
    )

ax.set(
    title="Axis-aligned bicycle planner: ten corridors",
    xlabel="x [m]",
    ylabel="y [m]",
)
ax.grid(alpha=0.15)
ax.legend(loc="upper right")
figure.tight_layout()

plot_velocity_profiles(analytical_trajectory, bicycle)

plt.show(block=True)
