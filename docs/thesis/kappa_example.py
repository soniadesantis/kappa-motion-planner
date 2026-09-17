"""Minimal two-corridor example used in the software-framework chapter."""
from math import pi
import matplotlib.pyplot as plt

from kappa_planner import CorridorWorld, MotionPlanner, Unicycle
from kappa_planner.helpers.corridor_geometry import get_corridor_from_vector
from kappa_planner.helpers.plot_helpers import plot_analytical_trajectory

vehicle = Unicycle(model="Rosbot circular")
vehicle.update(v_max=0.6, omega_max=1.0)

first = CorridorWorld(width=1.3, height=3.0,
                      center=[0.0, 0.0], tilt=0.0)
second = get_corridor_from_vector(
    first.head, [1.5, 3.0], width=1.0, add_height=0.7)

planner = MotionPlanner(
    vehicle=vehicle,
    corridor_list=[first, second],
    start_pose=[-1.0, -0.4, pi / 2],
    end_pose=[1.5, 2.7, pi / 2],
    assumptions="standing",
)
trajectory = planner.compute_trajectory_analytical()

duration = sum(part.maneuver_time for part in trajectory)
print(f"Traversal time: {duration:.3f} s")
print(f"Planning call: {planner.comp_time_analytical_sol:.6f} s")

axes = planner.plot_planner_inputs()
plot_analytical_trajectory(trajectory, figure=axes)
plt.show()
