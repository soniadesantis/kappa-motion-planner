"""Axis-aligned bicycle planning with optional poses and baseline fallback."""
from math import pi

import matplotlib.pyplot as plt

from kappa_planner import Bicycle, CorridorWorld, MotionPlanner
from kappa_planner.helpers.plot_helpers import plot_analytical_trajectory, plot_velocity_profiles


corridors = [
    CorridorWorld(width=4, height=12, center=[0, 0], tilt=0),
    CorridorWorld(width=4, height=12, center=[4, 4], tilt=pi/2),
    CorridorWorld(width=4, height=12, center=[8, 8], tilt=0),
]
bicycle = Bicycle(width=1, length=1, wheelbase=1, delta_max=pi/4, delta_min=-pi/4)
planner = MotionPlanner(bicycle, corridors, assumptions="axis-aligned")
# Optionally pass start_pose/end_pose in world coordinates, or normalized
# relative_start_pose/relative_end_pose in the shrunken corridor frames.
trajectory = planner.compute_trajectory_analytical()
print(f"Solution: {planner.solution_source}")
print(f"Computation: {planner.comp_time_analytical_sol:.4f} s")
print(f"Traversal: {planner.traversal_time:.2f} s")

figure = planner.plot_planner_inputs()
plot_analytical_trajectory(trajectory, figure=figure)
figure.axes[0].set_title("Axis-aligned bicycle planner")
plot_velocity_profiles(trajectory, bicycle)
plt.show()
