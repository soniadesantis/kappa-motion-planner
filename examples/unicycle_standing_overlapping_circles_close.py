"""Three standing-assumption corridors rejected because opposite-turn circles overlap.

All widths are 0.6 m and tilts are (0, pi/2, 0). The outer corridors
are 6 m long; the middle corridor is 2.8 m long. The robot has a
circular footprint of radius 0.15 m and a turning radius R = 1 m.

Rejection means the standing planner's sufficient assumptions fail; it does
not establish that no other collision-free trajectory exists.
"""
from argparse import ArgumentParser
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from kappa_planner import CorridorWorld, MotionPlanner, Unicycle
from kappa_planner.helpers.corridor_geometry import (
    compute_corner_point_vector_and_intersecting_edges,
    compute_turn_direction_vector,
)
from kappa_planner.helpers.intermediate_circles_geometry import (
    compute_center_coordinates_vector_according_to_edges,
)


def main(save_path=None, show=True):
    separation = 1.8
    corridors = [
        CorridorWorld(width=0.6, height=6, center=[-2, 0], tilt=0),
        CorridorWorld(width=0.6, height=separation + 1,
                      center=[0, separation / 2], tilt=pi / 2),
        CorridorWorld(width=0.6, height=6, center=[2, separation], tilt=0),
    ]
    unicycle = Unicycle(footprint_radius=0.15, v_max=1, omega_max=1)
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

    figure, ax = plt.subplots(figsize=(11, 5))
    planner.plot_planner_inputs(figure=figure, plot_shrunken_corridors=False,
                               plot_corridor_numbers=True)
    for center, turn, color in zip(centers, turns, ["#2563eb", "#ea580c"]):
        direction = "Left / counterclockwise" if turn == 1 else "Right / clockwise"
        ax.add_patch(Circle(center, radius, facecolor=color, edgecolor="none", alpha=0.12))
        ax.add_patch(Circle(center, radius, fill=False, edgecolor=color,
                            linestyle="--", linewidth=2, label=direction))
        ax.plot(*center, marker="+", color=color, markersize=9)
        # A curved arrow shows the circle's turn direction.
        angles = np.linspace(pi / 4, pi / 4 + turn * pi / 3, 40)
        points = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
        ax.plot(*points.T, color=color, linewidth=2)
        ax.annotate("", xy=points[-1], xytext=points[-3],
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
    ax.plot(*centers.T, color="#6b7280", linestyle=":", linewidth=1)
    ax.set(title="Standing planner: overlapping opposite-turn circles (2.8 m middle corridor)",
           xlabel="x [m]", ylabel="y [m]")
    ax.text(0.5, 0.02, f"Rejected: center distance {distance:.2f} m < 2R = {2 * radius:.2f} m",
            transform=ax.transAxes, ha="center", color="#991b1b",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.12)
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left")
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
