"""Thesis comparison: rejected standing assumptions and an axis-aligned solution."""
from argparse import ArgumentParser
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from kappa_planner import (
    Bicycle, CorridorWorld, CurvilinearArcBicycle, MotionPlanner, Unicycle,
    resample_trajectory,
)
from kappa_planner.helpers.corridor_geometry import (
    compute_corner_point_vector_and_intersecting_edges,
    compute_turn_direction_vector,
)
from kappa_planner.helpers.intermediate_circles_geometry import (
    compute_center_coordinates_vector_according_to_edges,
)


def main(save_path=None, show=True):
    separation = 1.2
    corridors = [
        # Trim only the distant ends; both junction intersections stay fixed.
        CorridorWorld(width=0.6, height=3, center=[-0.5, 0], tilt=0),
        CorridorWorld(width=0.42, height=separation + 1,
                      center=[0, separation / 2], tilt=pi / 2),
        CorridorWorld(width=0.6, height=3, center=[0.5, separation], tilt=0),
    ]
    # R = v_max / omega_max = 0.5 m. The narrower middle corridor keeps
    # the circles overlapping while satisfying the minimum-width check.
    unicycle = Unicycle(footprint_radius=0.15, v_max=0.5, omega_max=1)
    # Default endpoint poses remain available even though validation fails.
    planner = MotionPlanner(unicycle, corridors, assumptions="standing")
    try:
        planner.compute_trajectory_analytical()
    except ValueError as error:
        print(error)
    else:
        raise RuntimeError("This example should be rejected by the standing planner.")

    # Validation does not retain intermediate circles for rejected inputs.
    # Reconstruct their nominal centers with the SAME geometry used by its check.
    turns = compute_turn_direction_vector(corridors)
    corners, edges = compute_corner_point_vector_and_intersecting_edges(corridors, turns)
    centers = np.asarray(compute_center_coordinates_vector_according_to_edges(
        corridors, turns, corners, edges, unicycle,
    ))
    radius = unicycle.max_radius
    distance = np.linalg.norm(centers[1] - centers[0])
    print(f"Circle turn directions: {turns} (+1 left, -1 right)")
    print(f"Center distance: {distance:.3f} m < 2R = {2 * radius:.3f} m")

    # The current axis-aligned pipeline uses a bicycle. Match the unicycle's
    # footprint, speed, turning radius, and exact world-frame endpoint poses.
    bicycle = Bicycle(footprint_radius=unicycle.footprint_radius, v_max=unicycle.v_max,
                       wheelbase=radius, delta_max=pi / 4, delta_min=-pi / 4)
    axis_planner = MotionPlanner(bicycle, corridors, start_pose=planner.start_pose,
                                 end_pose=planner.end_pose, assumptions="axis-aligned")
    trajectory = axis_planner.compute_trajectory_analytical()
    print(f"Axis-aligned solution: {axis_planner.solution_source}, "
          f"traversal time {axis_planner.traversal_time:.3f} s")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax in axes:
        for corridor in corridors:
            outline = np.vstack([corridor.corners, corridor.corners[0]])
            ax.plot(*outline.T, color="black", linestyle="solid", linewidth=0.8)
    resample_trajectory(trajectory, samples_number=2001).plot_path(
        axes[1], color="#2563eb", linewidth=1.5,
    )
    # Plot the actual intermediate circles used by the returned solution,
    # excluding the small arcs that connect the boundary poses to them.
    solution = (axis_planner.refinement_result if axis_planner.solution_source == "refined"
                else axis_planner.baseline)
    boundary_ids = {id(primitive) for primitive in
                    solution.initial_maneuvers + solution.final_maneuvers}
    for primitive in trajectory:
        if isinstance(primitive, CurvilinearArcBicycle) and id(primitive) not in boundary_ids:
            center = (primitive.xc, primitive.yc)
            axes[1].add_patch(Circle(center, primitive.radius, fill=False,
                                     edgecolor="red", linestyle="--", linewidth=1))
            axes[1].plot(*center, marker="o", color="red", markersize=4)
    for ax in axes:
        for pose, color in [(planner.start_pose, "green"), (planner.end_pose, "red")]:
            x, y, heading = pose
            ax.plot(x, y, marker="o", color=color, markersize=5, zorder=5)
            ax.add_patch(Circle((x, y), unicycle.footprint_radius,
                                fill=False, edgecolor=color, linewidth=0.8, zorder=5))
            ax.arrow(x, y, 0.3 * np.cos(heading), 0.3 * np.sin(heading),
                     head_width=0.06, color=color, length_includes_head=True, zorder=5)

    ax = axes[0]
    for center, turn, corner in zip(centers, turns, corners):
        color = "red"
        ax.add_patch(Circle(center, radius, fill=False, edgecolor=color,
                            linestyle="--", linewidth=1))
        ax.plot(*center, marker="o", color=color, markersize=4)
        ax.plot([corner[0], center[0]], [corner[1], center[1]],
                color="black", linewidth=0.8)
        ax.plot(*corner, marker="o", color="black", markersize=2.5)
        # A curved arrow shows the circle's turn direction.
        angles = np.linspace(pi / 4, pi / 4 + turn * pi / 6, 30)
        points = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
        ax.plot(*points.T, color=color, linewidth=1)
        ax.annotate("", xy=points[-1], xytext=points[-3],
                    arrowprops=dict(arrowstyle="->", color=color, lw=1))
    bounds = np.vstack([corridor.corners for corridor in corridors])
    for ax, label in zip(axes, [r"$\mathrm{(a)}$", r"$\mathrm{(b)}$"]):
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(bounds[:, 0].min() - 0.2, bounds[:, 0].max() + 0.2)
        ax.set_ylim(bounds[:, 1].min() - 0.2, bounds[:, 1].max() + 0.2)
        ax.text(0.5, 1.04, label, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=18, math_fontfamily="cm")
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return planner, figure


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, help="Optional output image path.")
    main(parser.parse_args().save)
