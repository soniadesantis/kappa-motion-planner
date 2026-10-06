"""A fixed random corridor sequence: baseline, refinement, and boundary maneuvers.

Taken from the seed-7 short-corridor benchmark, ten-corridor case 1; rotated,
translated, and rounded for readability. No random generator is needed to run
this example. Rectangle dimensions and endpoint poses are in meters/radians.
"""
from argparse import ArgumentParser
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt

from kappa_planner import Bicycle, CorridorWorld, MotionPlanner


# Each row is (xmin, xmax, ymin, ymax). Nonconsecutive corridors do not overlap.
CORRIDOR_BOUNDS = [
    (0.358, 20.402, 14.984, 19.016),
    (16.326, 20.402, 11.844, 19.573),
    (16.255, 23.833, 11.844, 13.718),
    (22.233, 23.833, 8.124, 14.103),
    (20.573, 27.803, 8.124, 9.656),
    (25.525, 27.803, 1.205, 10.738),
    (25.406, 31.184, 1.205, 7.789),
    (29.919, 39.698, 2.716, 6.278),
    (37.378, 39.698, 2.540, 11.236),
    (36.728, 46.177, 6.645, 11.236),
]
START_POSE = [8.675, 17.053, 4.868272]
END_POSE = [45.351, 9.680, 1.553005]


def main(save_path=None):
    # At the default tilt=0, height spans x and width spans y.
    corridors = [
        CorridorWorld(width=ymax-ymin, height=xmax-xmin,
                      center=[(xmin+xmax)/2, (ymin+ymax)/2])
        for xmin, xmax, ymin, ymax in CORRIDOR_BOUNDS
    ]
    # Circular footprint radius 0.5 m; turning radius R = wheelbase/tan(delta) = 1 m.
    bicycle = Bicycle(width=1, length=1, wheelbase=1, delta_max=pi/4, delta_min=-pi/4)
    planner = MotionPlanner(bicycle, corridors, start_pose=START_POSE,
                            end_pose=END_POSE, assumptions="axis-aligned")
    trajectory = planner.compute_trajectory_analytical()
    baseline = planner.baseline

    print(f"Solution: {planner.solution_source}")
    print(f"Total planning computation: {1000 * planner.comp_time_analytical_sol:.2f} ms")
    print(f"Returned trajectory traversal: {planner.traversal_time:.2f} s")
    if baseline.trajectory is not None:
        improvement = 100 * (baseline.traversal_time - planner.traversal_time) / baseline.traversal_time
        print(f"Baseline traversal: {baseline.traversal_time:.2f} s")
        print(f"Traversal time reduction: {improvement:.2f}%")

    figure, ax = plt.subplots(figsize=(12, 6))
    planner.plot_planner_inputs(figure=figure, plot_shrunken_corridors=False)

    def plot_sequence(maneuvers, label, color, linestyle="solid", linewidth=2.5):
        for index, maneuver in enumerate(maneuvers):
            maneuver.plot_path(ax, color=color, linestyle=linestyle, linewidth=linewidth,
                               label=label if index == 0 else None)

    if baseline.trajectory is not None:
        plot_sequence(baseline.trajectory, "Baseline", "#6b7280", "--", 2)
    plot_sequence(trajectory, "Refined trajectory" if planner.solution_source == "refined"
                  else "Baseline fallback", "#2563eb")
    solution = planner.refinement_result if planner.solution_source == "refined" else baseline
    plot_sequence(solution.initial_maneuvers, "Start connection", "#16a34a", linewidth=3)
    plot_sequence(solution.final_maneuvers, "Goal connection", "#ea580c", linewidth=3)
    for index, corridor in enumerate(corridors, start=1):
        ax.text(*corridor.center, f"C{index}", ha="center", va="center", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    for name, pose in (("Start", START_POSE), ("Goal", END_POSE)):
        ax.annotate(name, pose[:2], xytext=(6, 8), textcoords="offset points", color="#b91c1c")
    ax.set(title="Axis-aligned bicycle planner: ten corridors", xlabel="x [m]", ylabel="y [m]")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right")
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, help="Optional output image path.")
    main(parser.parse_args().save)
