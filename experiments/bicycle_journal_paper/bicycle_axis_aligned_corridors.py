"""A fixed random corridor sequence: baseline, refinement, and boundary maneuvers.

Eight staircase corridors with coincident edges at each right-angle junction.
A short middle step and a narrow straight continuation make reachable-set
reductions visible. Dimensions and poses are in meters/radians.
"""
from argparse import ArgumentParser
from math import cos, sin, pi
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Polygon, Rectangle, Patch, PathPatch, FancyArrowPatch
from matplotlib.path import Path as PlotPath
import numpy as np
from shapely.geometry import Polygon as GeometryPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from kappa_planner import Bicycle, CorridorWorld, BackwardArc, CurvilinearArcUnicycle
from kappa_planner.baseline_construction_new import (
    compute_bicycle_baseline, compute_trajectory_traversal_time,
    compute_region_bounds, compute_reachable_region_slice, ReachableWaypointRegion,
)
from kappa_planner.refinement_new import refine_bicycle_baseline


# Each row is (xmin, xmax, ymin, ymax). Nonconsecutive corridors do not overlap.
CORRIDOR_BOUNDS = [
    (0.0, 8.0, 0.0, 3.6),
    (6.4, 8.0, -3.4, 3.6),
    (6.4, 9.8, -3.4, -1.8),
    (8.2, 9.8, -5.8, -1.8),
    (8.2, 12.7, -5.8, -3.8),
    (11.3, 19.7, -6.5, -3.1),
    (18.1, 19.7, -6.5, 2.5),
    (11.7, 19.7, -0.5, 2.5),
]
START_POSE = [2.0, 1.8, pi]
END_POSE = [12.9, 1.35, pi]

DOOR = "#d7e8f5"
DOOR_EDGE = "#397ba4"
LOCAL = "#fbe1c3"
LOCAL_EDGE = "#ba8047"
REACHABLE = "#d5edcf"
REACHABLE_EDGE = "#3c794a"
PROPAGATION = "#b83131"
POLYLINE = "#a85521"
BOUNDARY = "#d97706"


def region_outline(region, samples=401):
    """Evaluate analytic slices for display; never feed plot samples to the solver."""
    rows = []
    for x in np.linspace(*region.x_bounds, samples):
        interval = compute_reachable_region_slice(region, 0, x)
        if interval is not None:
            rows.append((x, *interval))
    if not rows:
        raise RuntimeError("A computed waypoint set has no displayable slices.")
    rows = np.asarray(rows)
    return np.vstack((rows[:, :2], rows[::-1][:, [0, 2]]))


def draw_set(ax, outline, facecolor, edgecolor, zorder=2):
    ax.add_patch(Polygon(outline, closed=True, facecolor=facecolor,
                         edgecolor=edgecolor, linewidth=0.8, zorder=zorder))


def set_label(ax, region, text, color, side="left"):
    x = sum(region.x_bounds) / 2
    interval = compute_reachable_region_slice(region, 0, x)
    point = (region.x_bounds[0 if side == "left" else 1], sum(interval) / 2)
    ax.annotate(text, point, xytext=(-8 if side == "left" else 8, 0), textcoords="offset points",
                ha="right" if side == "left" else "left", va="center",
                fontsize=15, color=color, zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.4),
                arrowprops=dict(arrowstyle="-", color=color, linewidth=0.6))


def draw_sequence(ax, primitives, color, linewidth=1.6, linestyle="solid", zorder=5):
    for primitive in primitives:
        if isinstance(primitive, (CurvilinearArcUnicycle, BackwardArc)):
            direction = (-primitive.turn_direction if isinstance(primitive, BackwardArc)
                         else primitive.turn_direction)
            start_angle = np.arctan2(primitive.y0 - primitive.yc, primitive.x0 - primitive.xc)
            angles = start_angle + direction * primitive.iota * np.linspace(0, 1, 201)
            points = np.column_stack((primitive.xc + primitive.radius * np.cos(angles),
                                      primitive.yc + primitive.radius * np.sin(angles)))
        else:
            points = np.array([[primitive.x0, primitive.y0], [primitive.xf, primitive.yf]])
        ax.plot(*points.T, color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder)


def draw_propagation(ax, baseline, radius):
    """Draw the actual 2R spacing and transverse bounds used by this baseline."""
    for j, previous in enumerate(baseline.reachable_regions[:-1]):
        direction = baseline.corridor_directions[j + 1]
        axis = int(np.argmax(np.abs(direction)))
        transverse = 1 - axis
        sign = int(direction[axis])
        intervals = [previous.x_bounds, previous.y_bounds]
        extreme = intervals[axis][0 if sign > 0 else 1]
        interval = compute_reachable_region_slice(previous, axis, extreme)
        origin = np.empty(2)
        origin[axis], origin[transverse] = extreme, sum(interval) / 2
        shifted = origin + 2 * radius * direction
        # Clip the propagated half-strip boundaries to the following D_j.
        door = baseline.safe_overlaps[j + 1]
        if door[axis][0] < shifted[axis] < door[axis][1]:
            ends = np.zeros((2, 2))
            ends[:, axis] = shifted[axis]
            ends[:, transverse] = door[transverse]
            ax.plot(*ends.T, color=PROPAGATION, linewidth=0.8, linestyle="--", zorder=6)
        for bound in intervals[transverse]:
            if door[transverse][0] < bound < door[transverse][1]:
                ends = np.zeros((2, 2))
                ends[:, transverse] = bound
                ends[:, axis] = door[axis]
                ax.plot(*ends.T, color=PROPAGATION, linewidth=0.8, linestyle="--", zorder=6)


def create_progressive_figure(corridors, bicycle, baseline, solution, refined,
                              panels=(1, 2, 3, 4), start_pose=None, end_pose=None):
    start_pose = START_POSE if start_pose is None else start_pose
    end_pose = END_POSE if end_pose is None else end_pose
    local_regions = [ReachableWaypointRegion(region, *compute_region_bounds(region))
                     for region in baseline.admissible_regions]
    local_outlines = [region_outline(region) for region in local_regions]
    reachable_outlines = [region_outline(region) for region in baseline.reachable_regions]
    label_sides = ["left", "left", "right", "left", "right", "left", "left"]
    titles = [
        "(a)\nCorridors and\nboundary poses",
        "(b)\n" + r"Safe overlaps $\mathcal{D}_j$" + "\nand passage\ndirections",
        "(c)\nLocal fillet-\nadmissible sets\n" + r"$\mathcal{A}_j$",
        "(d)\nConstraint\npropagation and\nreachable sets\n" + r"$\mathcal{R}_j$",
        "(e)\nPolyline, fillets,\nand boundary\nconnections",
        "(f)\nBaseline and\nrefined trajectory" if refined else "(f)\nBaseline fallback",
    ]
    style = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12,
             "pdf.fonttype": 42, "savefig.facecolor": "white"}
    with plt.rc_context(style):
        figure = plt.figure(figsize=(10, 3.15 * len(panels)))
        grid = figure.add_gridspec(len(panels), 2, width_ratios=(1, 3.25),
                                  left=0.025, right=0.975, top=0.985, bottom=0.025,
                                  hspace=0.015, wspace=0)
        axes = {panel: figure.add_subplot(grid[row, 1])
                for row, panel in enumerate(panels)}
        label_axes = {}
        legend_positions = {}
        for row, panel in enumerate(panels):
            title = titles[panel].split("\n", 1)[1]
            if len(panels) > 1:
                title = f"({chr(ord('a') + row)})\n" + title
            label_ax = figure.add_subplot(grid[row, 0])
            label_ax.set_axis_off()
            label_axes[panel] = label_ax
            legend_positions[panel] = 0.9 - 0.08 * len(title.splitlines()) - 0.06
            label_ax.text(0, 0.9, title, ha="left", va="top", fontsize=14,
                          linespacing=1.3, transform=label_ax.transAxes)
        def legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            return FancyArrowPatch((-xdescent, height / 2 - ydescent),
                                   (width - xdescent, height / 2 - ydescent),
                                   arrowstyle="->", mutation_scale=fontsize,
                                   color="#334155", linewidth=1)

        def left_legend(row, handles):
            label_axes[row].legend(handles=handles, loc="upper left",
                                   bbox_to_anchor=(0, legend_positions[row]),
                                   borderaxespad=0, frameon=False, fontsize=14,
                                   handlelength=1.5, labelspacing=0.7,
                                   handler_map={FancyArrowPatch: HandlerPatch(patch_func=legend_arrow)})
        bounds = np.vstack([corridor.corners for corridor in corridors])
        corridor_union = unary_union([GeometryPolygon(corridor.corners) for corridor in corridors])
        polygons = [corridor_union] if corridor_union.geom_type == "Polygon" else corridor_union.geoms
        union_paths = []
        for polygon in polygons:
            polygon = orient(polygon, sign=1.0)
            for ring in [polygon.exterior, *polygon.interiors]:
                vertices = np.asarray(ring.coords)
                codes = np.full(len(vertices), PlotPath.LINETO, dtype=np.uint8)
                codes[0], codes[-1] = PlotPath.MOVETO, PlotPath.CLOSEPOLY
                union_paths.append(PlotPath(vertices, codes))
        union_outline = PlotPath.make_compound_path(*union_paths)
        for row, ax in axes.items():
            if row < 2:
                for corridor in corridors:
                    ax.add_patch(Polygon(corridor.corners, facecolor="#f6f8fa",
                                         edgecolor="#555555", linewidth=0.6, zorder=0))
                # Draw edges after all fills so overlaps cannot hide an edge.
                for corridor in corridors:
                    ax.add_patch(Polygon(corridor.corners, fill=False,
                                         edgecolor="#555555", linewidth=0.6, zorder=1))
            else:
                ax.add_patch(PathPatch(union_outline, facecolor="#f6f8fa",
                                       edgecolor="#555555", linewidth=0.6, zorder=0))
            ax.set_xlim(bounds[:, 0].min() - 0.25, bounds[:, 0].max() + 0.25)
            ax.set_ylim(bounds[:, 1].min() - 0.25, bounds[:, 1].max() + 0.25)
            ax.set_aspect("equal", adjustable="box", anchor="W")
            ax.set_axis_off()
            for pose, color in [(start_pose, "#16803a"), (end_pose, "#dc2626")]:
                x, y, heading = pose
                ax.plot(x, y, marker="o", color=color, markersize=4, zorder=10)
                ax.add_patch(Circle((x, y), bicycle.width / 2, fill=False,
                                    edgecolor=color, linewidth=0.8, zorder=9))
                ax.arrow(x, y, 1.1 * cos(heading), 1.1 * sin(heading), head_width=0.3,
                         color=color, length_includes_head=True, zorder=10)

        if 1 in axes:
            ax = axes[1]
            door_centers = []
            for j, (xb, yb) in enumerate(baseline.safe_overlaps, start=1):
                ax.add_patch(Rectangle((xb[0], yb[0]), xb[1] - xb[0], yb[1] - yb[0],
                                       facecolor=DOOR, edgecolor=DOOR_EDGE, linewidth=0.8, zorder=2))
                center = np.array([sum(xb) / 2, sum(yb) / 2])
                door_centers.append(center)
                side = label_sides[j - 1]
                point = (xb[0 if side == "left" else 1], center[1])
                ax.annotate(rf"$\mathcal{{D}}_{j}$", point,
                            xytext=(-8 if side == "left" else 8, 0),
                            textcoords="offset points", ha="right" if side == "left" else "left",
                            va="center", color=DOOR_EDGE, fontsize=15, zorder=8,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.4),
                            arrowprops=dict(arrowstyle="-", color=DOOR_EDGE, linewidth=0.6))
            for j, direction in enumerate(baseline.corridor_directions[1:-1]):
                center = (door_centers[j] + door_centers[j + 1]) / 2
                half_length = 0.45 if j in (1, 2) else 1.3
                ax.annotate("", center + half_length * direction, xytext=center - half_length * direction,
                            arrowprops=dict(arrowstyle="->", color="#334155", linewidth=1), zorder=5)
            for pose, door, direction in [
                (start_pose, door_centers[0], baseline.corridor_directions[0]),
                (end_pose, door_centers[-1], baseline.corridor_directions[-1]),
            ]:
                center = (np.asarray(pose[:2]) + door) / 2
                transverse = 1 - int(np.argmax(np.abs(direction)))
                center[transverse] -= 0.8
                ax.annotate("", center + 0.9 * direction, xytext=center - 0.9 * direction,
                            arrowprops=dict(arrowstyle="->", color="#334155", linewidth=1), zorder=5)
            left_legend(1, [Patch(facecolor=DOOR, edgecolor=DOOR_EDGE, label="Safe overlap\n" + r"$\mathcal{D}_j$"),
                               FancyArrowPatch((0, 0), (1, 0), arrowstyle="->",
                                               color="#334155", linewidth=1, label="Passage\ndirection")])

        for ax in [axes[panel] for panel in (2, 3) if panel in axes]:
            for outline in local_outlines:
                draw_set(ax, outline, LOCAL, LOCAL_EDGE)
        if 2 in axes:
            for j, region in enumerate(local_regions, start=1):
                set_label(axes[2], region, rf"$\mathcal{{A}}_{j}$", LOCAL_EDGE, label_sides[j - 1])
            left_legend(2, [Patch(facecolor=LOCAL, edgecolor=LOCAL_EDGE,
                                         label="Locally\nadmissible " + r"$\mathcal{A}_j$" + "\n(aligned: " + r"$\mathcal{A}_j=\mathcal{D}_j$" + ")")])
        if 3 in axes:
            for j, (region, outline) in enumerate(zip(baseline.reachable_regions, reachable_outlines), start=1):
                draw_set(axes[3], outline, REACHABLE, REACHABLE_EDGE, zorder=3)
                set_label(axes[3], region, rf"$\mathcal{{R}}_{j}$", REACHABLE_EDGE, label_sides[j - 1])
            left_legend(3, [Patch(facecolor=LOCAL, edgecolor=LOCAL_EDGE, label=r"$\mathcal{A}_j$"),
                                   Patch(facecolor=REACHABLE, edgecolor=REACHABLE_EDGE, label=r"$\mathcal{R}_j$")])

        if 4 in axes:
            waypoints = np.asarray(baseline.waypoints)
            entry, exit_point = np.array(start_pose[:2]), np.array(end_pose[:2])
            for point, waypoint, direction in [(entry, waypoints[0], baseline.corridor_directions[0]),
                                                (exit_point, waypoints[-1], baseline.corridor_directions[-1])]:
                transverse = 1 - int(np.argmax(np.abs(direction)))
                point[transverse] = waypoint[transverse]
            polyline = np.vstack([entry, waypoints, exit_point])
            axes[4].plot(*polyline.T, color=POLYLINE, linestyle="--", linewidth=1.1, zorder=4)
            axes[4].plot(*waypoints.T, linestyle="none", marker="o", markersize=4,
                         markerfacecolor="white", markeredgecolor=POLYLINE, zorder=8)
            draw_sequence(axes[4], baseline.trajectory, "#252525")
            draw_sequence(axes[4], baseline.initial_maneuvers, BOUNDARY, 2)
            draw_sequence(axes[4], baseline.final_maneuvers, BOUNDARY, 2)
            left_legend(4, [Line2D([], [], color=POLYLINE, linestyle="--", marker="o", markerfacecolor="white", label="Polyline"),
                                   Line2D([], [], color="#252525", label="Filleted\nbaseline"),
                                   Line2D([], [], color=BOUNDARY, label="Boundary\nconnections")])

        if 5 in axes:
            draw_sequence(axes[5], baseline.trajectory, "#9ca3af", 2, linestyle="--")
            draw_sequence(axes[5], solution.trajectory, "#2563eb", 1.8, zorder=6)
            draw_sequence(axes[5], solution.initial_maneuvers, "#16a34a", 2, zorder=7)
            draw_sequence(axes[5], solution.final_maneuvers, "#ea580c", 2, zorder=7)
            left_legend(5, [Line2D([], [], color="#9ca3af", linestyle="--", label="Baseline"),
                                   Line2D([], [], color="#2563eb", label="Refined\ntrajectory" if refined else "Baseline fallback"),
                                   Line2D([], [], color="#16a34a", label="Initial\nconnection"),
                                   Line2D([], [], color="#ea580c", label="Final\nconnection")])
        return figure


def main(save_path=None, show=True):
    # At the default tilt=0, height spans x and width spans y.
    corridors = [
        CorridorWorld(width=ymax-ymin, height=xmax-xmin,
                      center=[(xmin+xmax)/2, (ymin+ymax)/2])
        for xmin, xmax, ymin, ymax in CORRIDOR_BOUNDS
    ]
    # Circular footprint radius 0.5 m; turning radius R = wheelbase/tan(delta) = 1 m.
    # On develop, a circular footprint is specified by its diameter.
    bicycle = Bicycle(width=1, length=1, wheelbase=1, delta_max=pi/4, delta_min=-pi/4)
    started = perf_counter()
    baseline, failure = compute_bicycle_baseline(
        corridors, bicycle, initial_pose=START_POSE, final_pose=END_POSE,
        boundary_connections=True, return_failure=True,
    )
    if baseline is None:
        raise RuntimeError(f"Baseline failed: {failure.reason}")
    baseline_seconds = perf_counter() - started
    refinement_started = perf_counter()
    refinement = None
    try:
        refinement, failure = refine_bicycle_baseline(
            corridors, bicycle, baseline, initial_pose=START_POSE,
            final_pose=END_POSE, return_failure=True,
        )
        if refinement is None or refinement.trajectory is None:
            print(f"Refinement failed; retaining baseline: {failure}")
    except Exception as error:
        print(f"Refinement failed; retaining baseline: {error}")
    refinement_seconds = perf_counter() - refinement_started
    refined = refinement is not None and refinement.trajectory is not None
    solution = refinement if refined else baseline
    trajectory = solution.trajectory
    traversal_time = compute_trajectory_traversal_time(trajectory)

    print(f"Solution: {'refined' if refined else 'baseline fallback'}")
    print(f"Baseline computation: {1000 * baseline_seconds:.2f} ms")
    print(f"Refinement computation: {1000 * refinement_seconds:.2f} ms")
    print(f"Total planning computation: {1000 * (baseline_seconds + refinement_seconds):.2f} ms")
    print(f"Returned trajectory traversal: {traversal_time:.2f} s")
    if baseline.trajectory is not None:
        improvement = 100 * (baseline.traversal_time - traversal_time) / baseline.traversal_time
        print(f"Baseline traversal: {baseline.traversal_time:.2f} s")
        print(f"Traversal time reduction: {improvement:.2f}%")

    figure = create_progressive_figure(corridors, bicycle, baseline, solution, refined)
    if save_path is not None:
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
        comparison = create_progressive_figure(corridors, bicycle, baseline, solution,
                                              refined, panels=(5,))
        save_path = Path(save_path)
        comparison_path = save_path.with_name(save_path.stem + "_refinement" + save_path.suffix)
        comparison.savefig(comparison_path, dpi=180, bbox_inches="tight")
        comparison.savefig(comparison_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return baseline, refinement, figure


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, help="Optional output image path.")
    parser.add_argument("--no-show", action="store_true", help="Save without opening a window.")
    args = parser.parse_args()
    main(args.save, show=not args.no_show)
