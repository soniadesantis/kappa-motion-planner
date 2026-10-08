"""Baseline construction and refinement stages for EXAMPLE_THESIS map 1."""

from argparse import ArgumentParser
from copy import deepcopy
from math import cos, sin
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, PathPatch
import numpy as np

from kappa_planner import CorridorWorld
from examples_maps_polyline import example_corridor_sequence
from bicycle_axis_aligned_corridors import create_progressive_figure, draw_sequence
from main_example_axis_aligned_refinement_steps import (
    compute_stages, draw_circles, union_outline, BASELINE, POSITIONED, TANGENT, INITIAL, FINAL,
)
from kappa_planner.refinement_new import (
    build_and_repair_tangent_chain, find_first_problematic_tangent_intersection,
)

OUTPUT = Path(__file__).parent / "figures"


def trim_corridor_ends(corridors, start, end, bicycle):
    bounds = [np.array([corridor.corners.min(axis=0), corridor.corners.max(axis=0)])
              for corridor in corridors]
    margin = bicycle.max_radius + bicycle.width / 2
    shortened = []
    for j, corridor in enumerate(corridors):
        axis = int(np.argmax(np.abs(corridor.unit_vector)))
        required = []
        for neighbor in (j-1, j+1):
            if 0 <= neighbor < len(corridors):
                required.extend([max(bounds[j][0, axis], bounds[neighbor][0, axis]),
                                 min(bounds[j][1, axis], bounds[neighbor][1, axis])])
        if j == 0:
            required.extend([start[axis]-margin, start[axis]+margin])
        if j == len(corridors)-1:
            required.extend([end[axis]-margin, end[axis]+margin])
        low = max(bounds[j][0, axis], min(required))
        high = min(bounds[j][1, axis], max(required))
        center = np.array(corridor.center, dtype=float)
        center[axis] = (low+high)/2
        shortened.append(CorridorWorld(corridor.width, high-low, center, corridor.tilt))
    return shortened


def create_two_panel_refinement(stages):
    corridors, bicycle, baseline, positioned, indices, _, _, _, refined, start, end = stages
    groups = deepcopy(positioned)
    tangents = build_and_repair_tangent_chain(
        circle_groups=groups, corridor_list=corridors, baseline=baseline,
        R=bicycle.max_radius, r=bicycle.width/2, active_circle_indices=indices,
    )
    if tangents is None:
        raise RuntimeError("Cannot construct the tangent chain before circle skipping.")
    if find_first_problematic_tangent_intersection(tangents) is None:
        raise RuntimeError("The initial tangent chain no longer illustrates an intersection.")
    style = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 14,
             "pdf.fonttype": 42, "savefig.facecolor": "white"}
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(11, 5.9), sharex=True, sharey=True)
        figure.subplots_adjust(left=.01, right=.99, top=.90, bottom=.19, wspace=.035)
        outline = union_outline(corridors)
        bounds = np.vstack([corridor.corners for corridor in corridors])
        for ax, title in zip(axes, ["(a) Positioned circles and intersecting tangents",
                                   "(b) Refined trajectory and retained circles"]):
            ax.add_patch(PathPatch(outline, facecolor="#f6f8fa", edgecolor="#555555",
                                   linewidth=.6, zorder=0))
            ax.set_xlim(bounds[:, 0].min()-.25, bounds[:, 0].max()+.25)
            ax.set_ylim(bounds[:, 1].min()-.25, bounds[:, 1].max()+.25)
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_title(title, fontsize=13, pad=8)
            draw_sequence(ax, baseline.trajectory, BASELINE, .9, linestyle="--", zorder=2)
            for pose, color in [(start, "#16803a"), (end, "#dc2626")]:
                x, y, heading = pose
                ax.add_patch(Circle((x, y), bicycle.width/2, fill=False,
                                    edgecolor=color, linewidth=.8, zorder=8))
                ax.plot(x, y, marker="o", color=color, markersize=4, zorder=9)
                ax.arrow(x, y, 1.1*cos(heading), 1.1*sin(heading), head_width=.3,
                         color=color, length_includes_head=True, zorder=9)
        draw_circles(axes[0], groups, indices, positioning=True, label_fontsize=15,
                     label_offsets={2: (-5, -13)})
        for tangent in tangents:
            points = np.array([tangent.start_point, tangent.end_point])
            axes[0].plot(*points.T, color=TANGENT, linewidth=1.5, zorder=6)
        draw_circles(axes[1], refined.circle_groups, refined.active_circle_indices)
        initial_count = len(refined.initial_maneuvers)
        final_count = len(refined.final_maneuvers)
        draw_sequence(axes[1], refined.trajectory[initial_count:-final_count], TANGENT, 1.8, zorder=6)
        draw_sequence(axes[1], refined.initial_maneuvers, INITIAL, 2, zorder=6)
        draw_sequence(axes[1], refined.final_maneuvers, FINAL, 2, zorder=6)
        handles = [Line2D([], [], color=BASELINE, linewidth=.9, linestyle="--", label="Baseline"),
                   Line2D([], [], color=POSITIONED, marker="o", label="Positioned circles"),
                   Line2D([], [], color=TANGENT, label="Tangents / refined path"),
                   Line2D([], [], color=TANGENT, alpha=.45, marker="o", label="Retained circles"),
                   Line2D([], [], color=INITIAL, label="Initial connection"),
                   Line2D([], [], color=FINAL, label="Final connection")]
        figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015),
                      ncol=3, frameon=False, fontsize=12.5, columnspacing=1.1, handlelength=1.6)
        return figure


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    corridors, start, end, bicycle = example_corridor_sequence(1)
    start = list(start)
    start[1] += 4.0
    end = list(end)
    end[0] -= 5.0
    last = corridors[-1]
    center = np.array(last.center, dtype=float)
    center[0] -= 5.0
    corridors[-1] = CorridorWorld(last.width, last.height, center, last.tilt)
    corridors = trim_corridor_ends(corridors, start, end, bicycle)
    stages = compute_stages(corridors, bicycle, start, end)
    baseline, refined = stages[2], stages[8]
    construction = create_progressive_figure(
        corridors, bicycle, baseline, refined, True, start_pose=start, end_pose=end)
    refinement = create_two_panel_refinement(stages)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, figure in [("thesis_map_1_axis_aligned", construction),
                         ("thesis_map_1_axis_aligned_refinement_steps", refinement)]:
        for extension in ("pdf", "png"):
            path = OUTPUT / f"{name}.{extension}"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            print(path)
    if args.no_show:
        plt.close(construction)
        plt.close(refinement)
    else:
        plt.show()


if __name__ == "__main__":
    main()
