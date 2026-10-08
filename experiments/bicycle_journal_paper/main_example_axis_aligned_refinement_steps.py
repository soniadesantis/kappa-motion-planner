"""Refinement stages for the fixed axis-aligned thesis example."""

from argparse import ArgumentParser
from copy import deepcopy
from math import cos, sin, pi
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as PlotPath
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from bicycle_axis_aligned_corridors import CORRIDOR_BOUNDS, START_POSE, END_POSE, draw_sequence
from kappa_planner import Bicycle, CorridorWorld
from kappa_planner.baseline_construction_new import compute_bicycle_baseline
from kappa_planner.refinement_new import (
    compute_straight_passage_groups,
    compute_baseline_turn_circle,
    compute_independent_turn_circle,
    get_active_circle_indices,
    get_single_circle,
    is_baseline_circle,
    rebuild_and_simplify_internal_chain,
    refine_bicycle_baseline,
)

OUTPUT = Path(__file__).parent / "figures"
BASELINE = "#9ca3af"
POSITIONED = "#7c3aed"
RETAINED = "#0f766e"
TANGENT = "#2563eb"
INITIAL = "#16a34a"
FINAL = "#ea580c"


def compute_stages(corridors=None, bicycle=None, start_pose=None, end_pose=None):
    if corridors is None:
        corridors = [CorridorWorld(width=d-c, height=b-a, center=[(a+b)/2, (c+d)/2])
                     for a, b, c, d in CORRIDOR_BOUNDS]
    if bicycle is None:
        bicycle = Bicycle(width=1, length=1, wheelbase=1, delta_max=pi/4, delta_min=-pi/4)
    start_pose = START_POSE if start_pose is None else start_pose
    end_pose = END_POSE if end_pose is None else end_pose
    baseline, failure = compute_bicycle_baseline(
        corridors, bicycle, initial_pose=start_pose, final_pose=end_pose,
        boundary_connections=True, return_failure=True,
    )
    if baseline is None:
        raise RuntimeError(f"Baseline failed: {failure}")

    groups = [{} for _ in baseline.turn_directions]
    fixed = set()
    for group in compute_straight_passage_groups(baseline.turn_directions):
        fixed.update(index for index in (group.left_circle_index, group.right_circle_index)
                     if index is not None)
    for j, tau in enumerate(baseline.turn_directions):
        if tau == 0:
            continue
        if j in fixed:
            circle = compute_baseline_turn_circle(
                baseline, j, bicycle.max_radius, placement_rule="baseline_straight_passage")
        else:
            circle = compute_independent_turn_circle(
                baseline, j, bicycle.max_radius, bicycle.width/2)
        if circle is None:
            raise RuntimeError(f"Circle placement failed at transition {j+1}.")
        groups[j][tau] = circle

    positioned_groups = deepcopy(groups)
    positioned_indices = get_active_circle_indices(groups)
    state = rebuild_and_simplify_internal_chain(
        circle_groups=groups, corridor_list=corridors, baseline=baseline,
        R=bicycle.max_radius, r=bicycle.width/2,
        active_circle_indices=positioned_indices, safe_union_cache={},
    )
    if state is None:
        raise RuntimeError("Internal tangent construction failed.")
    tangent_indices, tangents = state

    refined, failure = refine_bicycle_baseline(
        corridors, bicycle, baseline, initial_pose=start_pose,
        final_pose=end_pose, return_failure=True,
    )
    if refined is None or refined.trajectory is None:
        raise RuntimeError(f"Boundary-connected refinement failed: {failure}")
    print(f"Positioned circles: {[j+1 for j in positioned_indices]}")
    print(f"Circles after internal repair/skipping: {[j+1 for j in tangent_indices]}")
    print(f"Final retained circles: {[j+1 for j in refined.active_circle_indices]}")
    print(f"Traversal time: baseline {baseline.traversal_time:.2f} s; "
          f"refined {refined.traversal_time:.2f} s")
    return corridors, bicycle, baseline, positioned_groups, positioned_indices, groups, tangent_indices, tangents, refined, start_pose, end_pose


def union_outline(corridors):
    union = unary_union([Polygon(corridor.corners) for corridor in corridors])
    polygons = [union] if union.geom_type == "Polygon" else union.geoms
    paths = []
    for polygon in polygons:
        polygon = orient(polygon, sign=1)
        for ring in [polygon.exterior, *polygon.interiors]:
            vertices = np.asarray(ring.coords)
            codes = np.full(len(vertices), PlotPath.LINETO, dtype=np.uint8)
            codes[0], codes[-1] = PlotPath.MOVETO, PlotPath.CLOSEPOLY
            paths.append(PlotPath(vertices, codes))
    return PlotPath.make_compound_path(*paths)


def draw_circles(ax, groups, indices, positioning=False, label_fontsize=15,
                 label_offsets=None):
    for j in indices:
        circle = get_single_circle(groups[j])
        color = (RETAINED if is_baseline_circle(circle) else POSITIONED) if positioning else TANGENT
        ax.add_patch(Circle(circle.center, circle.radius, fill=False,
                            edgecolor=color, linewidth=1.1 if positioning else 0.7,
                            alpha=1 if positioning else 0.45, zorder=4))
        ax.plot(*circle.center, marker="o", markersize=3, color=color, zorder=5)
        if positioning:
            side = -1 if j in (0, 1, 2, 3, 5, 6) else 1
            point = circle.center + [side*circle.radius, 0]
            offset = (label_offsets or {}).get(j, (side*5, 0))
            ax.annotate(rf"$\mathbf{{o}}_{{{j+1}}}$", point,
                        xytext=offset, textcoords="offset points",
                        ha="right" if side < 0 else "left", va="center",
                        fontsize=label_fontsize, color=color, zorder=7,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.3))


def create_figure(corridors, bicycle, baseline, positioned_groups, positioned_indices,
                  tangent_groups, tangent_indices, tangents, refined, start_pose, end_pose):
    style = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 14,
             "pdf.fonttype": 42, "savefig.facecolor": "white"}
    with plt.rc_context(style):
        bounds = np.vstack([corridor.corners for corridor in corridors])
        span = np.ptp(bounds, axis=0) + .5
        row_height = max(3.15, 6.2 * span[1] / span[0])
        figure = plt.figure(figsize=(10, 3 * row_height))
        grid = figure.add_gridspec(3, 2, width_ratios=(1, 3.25),
                                  left=.025, right=.975, top=.985, bottom=.025,
                                  hspace=.015, wspace=0)
        titles = ["(a)\nCircle\npositioning", "(b)\nTangent\nconnections",
                  "(c)\nBoundary\nconnections"]
        handles = [
            [Line2D([], [], color=POSITIONED, marker="o", label="Repositioned\ncircles"),
             Line2D([], [], color=RETAINED, marker="o", label="Baseline\ncircles")],
            [Line2D([], [], color=TANGENT, label="Internal\ntangents")],
            [Line2D([], [], color=TANGENT, label="Internal\nrefined path"),
             Line2D([], [], color=INITIAL, label="Initial\nconnection"),
             Line2D([], [], color=FINAL, label="Final\nconnection")],
        ]
        if not any(is_baseline_circle(get_single_circle(positioned_groups[j]))
                   for j in positioned_indices):
            handles[0].pop()
        bounds = np.vstack([corridor.corners for corridor in corridors])
        outline = union_outline(corridors)
        axes = []
        for row, title in enumerate(titles):
            label_ax = figure.add_subplot(grid[row, 0])
            label_ax.set_axis_off()
            label_ax.text(0, .9, title, ha="left", va="top", fontsize=14,
                          linespacing=1.3, transform=label_ax.transAxes)
            label_ax.legend(handles=[Line2D([], [], color=BASELINE, linewidth=.9, linestyle="--", label="Baseline")]
                            + handles[row], loc="upper left", bbox_to_anchor=(0, .6),
                            borderaxespad=0, frameon=False, fontsize=14,
                            handlelength=1.5, labelspacing=.5)
            ax = figure.add_subplot(grid[row, 1])
            axes.append(ax)
            ax.add_patch(PathPatch(outline, facecolor="#f6f8fa", edgecolor="#555555",
                                   linewidth=.6, zorder=0))
            ax.set_xlim(bounds[:, 0].min()-.25, bounds[:, 0].max()+.25)
            ax.set_ylim(bounds[:, 1].min()-.25, bounds[:, 1].max()+.25)
            ax.set_aspect("equal", adjustable="box", anchor="W")
            ax.set_axis_off()
            draw_sequence(ax, baseline.trajectory, BASELINE, .9, linestyle="--", zorder=2)
            for pose, color in [(start_pose, "#16803a"), (end_pose, "#dc2626")]:
                x, y, heading = pose
                ax.add_patch(Circle((x, y), bicycle.width/2, fill=False,
                                    edgecolor=color, linewidth=.8, zorder=8))
                ax.plot(x, y, marker="o", color=color, markersize=4, zorder=9)
                ax.arrow(x, y, 1.1*cos(heading), 1.1*sin(heading), head_width=.3,
                         color=color, length_includes_head=True, zorder=9)

        draw_circles(axes[0], positioned_groups, positioned_indices, positioning=True)
        draw_circles(axes[1], tangent_groups, tangent_indices)
        for tangent in tangents:
            axes[1].plot(*np.array([tangent.start_point, tangent.end_point]).T,
                         color=TANGENT, linewidth=1.8, zorder=6)
            axes[1].plot(*np.array([tangent.start_point, tangent.end_point]).T,
                         linestyle="none", marker="o", markersize=3,
                         markerfacecolor="white", markeredgecolor=TANGENT, zorder=7)

        n_initial, n_final = len(refined.initial_maneuvers), len(refined.final_maneuvers)
        draw_sequence(axes[2], refined.trajectory[n_initial:-n_final], TANGENT, 1.8, zorder=5)
        draw_sequence(axes[2], refined.initial_maneuvers, INITIAL, 2, zorder=6)
        draw_sequence(axes[2], refined.final_maneuvers, FINAL, 2, zorder=6)
        return figure


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    figure = create_figure(*compute_stages())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = OUTPUT / f"main_example_axis_aligned_refinement_steps.{extension}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        print(path)
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
