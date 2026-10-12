"""Diagnose a rejected refinement arc against an eroded local corridor union.

This is an experiment-side diagnostic only.  It does not alter the planner's
local arc-safety rule or its outcome.
"""

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as PlotPolygon
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union

from kappa_planner.helpers.collision_avoidance import (
    compute_safe_corridor_union,
    corridor_to_axis_aligned_polygon,
)
from kappa_planner.refinement_new import get_single_circle
from benchmark_new_bicycle_baseline import make_fixed_bicycle
from plot_refinement_failures import replay_failure


ROOT = Path(__file__).parent


def polygon_parts(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [part for part in geometry.geoms if part.geom_type == "Polygon"]


def add_geometry(ax, geometry, *, facecolor, edgecolor, alpha, linewidth, zorder):
    for polygon in polygon_parts(geometry):
        ax.add_patch(PlotPolygon(
            np.asarray(polygon.exterior.coords), closed=True,
            facecolor=facecolor, edgecolor=edgecolor, alpha=alpha,
            linewidth=linewidth, zorder=zorder,
        ))
        for interior in polygon.interiors:
            ax.add_patch(PlotPolygon(
                np.asarray(interior.coords), closed=True,
                facecolor="white", edgecolor=edgecolor, linewidth=linewidth,
                zorder=zorder + 0.1,
            ))


def arc_linestring(arc, samples=2001):
    angles = np.linspace(
        float(arc.epsilon),
        float(arc.epsilon) + float(arc.turn_direction) * float(arc.iota),
        samples,
    )
    points = np.column_stack((
        float(arc.xc) + float(arc.radius) * np.cos(angles),
        float(arc.yc) + float(arc.radius) * np.sin(angles),
    ))
    return LineString(points), points


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "results" / "new_bicycle_baseline"
        / "short_corridors_R1_500_assembly_diagnostics.json",
    )
    parser.add_argument("--corridors", type=int, default=10)
    parser.add_argument("--case", type=int, default=44)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "figures" / "new_bicycle_baseline"
        / "arc_collision_case_n10_044",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    group = next(g for g in report["groups"] if g["corridors"] == args.corridors)
    record = next(r for r in group["records"] if r["case"] == args.case)
    if record["refinement"].get("failure_reason") != "arc_collision_failed":
        raise RuntimeError("Selected record is not an arc_collision_failed case.")

    bicycle = make_fixed_bicycle()
    corridors, _, trace = replay_failure(record, bicycle)
    rejected = trace["failed_arcs"][-1]
    arc, circle = rejected["arc"], rejected["circle"]
    groups = trace["state"]["groups"]
    transition_index = next(
        index for index, group in enumerate(groups)
        if group and np.allclose(get_single_circle(group).center, circle.center)
    )

    # Mathematical neighborhood C_(j-1), C_j, C_(j+1), clipped at endpoints.
    local_indices = list(range(
        max(0, transition_index - 1),
        min(len(corridors), transition_index + 2),
    ))
    local_corridors = [corridors[index] for index in local_indices]
    original_union = unary_union([
        corridor_to_axis_aligned_polygon(corridor) for corridor in local_corridors
    ])
    safe_union = compute_safe_corridor_union(local_corridors, bicycle.width / 2, quad_segs=64)
    individually_eroded_union = unary_union([
        corridor_to_axis_aligned_polygon(corridor).buffer(
            -bicycle.width / 2, quad_segs=64,
        )
        for corridor in local_corridors
    ])
    individual_safe_regions = [
        corridor_to_axis_aligned_polygon(corridor).buffer(
            -bicycle.width / 2, quad_segs=64,
        )
        for corridor in local_corridors
    ]

    centerline, points = arc_linestring(arc)
    tolerance = 1e-9
    contained_in_eroded_union = safe_union.buffer(tolerance).covers(centerline)
    contained_in_individual_union = individually_eroded_union.buffer(tolerance).covers(centerline)
    outside = centerline.difference(safe_union.buffer(tolerance))
    outside_length = float(outside.length)
    footprint_sweep = centerline.buffer(bicycle.width / 2, quad_segs=64)
    footprint_contained = original_union.buffer(tolerance).covers(footprint_sweep)

    fig, ax = plt.subplots(figsize=(9, 8))
    add_geometry(
        ax, original_union, facecolor="#dbeafe", edgecolor="#64748b",
        alpha=0.45, linewidth=1.4, zorder=1,
    )
    add_geometry(
        ax, safe_union, facecolor="#bbf7d0", edgecolor="#15803d",
        alpha=0.55, linewidth=2.0, zorder=2,
    )
    add_geometry(
        ax, footprint_sweep, facecolor="#fecaca", edgecolor="#dc2626",
        alpha=0.28, linewidth=1.0, zorder=3,
    )
    ax.plot(points[:, 0], points[:, 1], color="#dc2626", linewidth=3, zorder=5)
    if not outside.is_empty:
        geometries = [outside] if outside.geom_type == "LineString" else list(outside.geoms)
        for geometry in geometries:
            xy = np.asarray(geometry.coords)
            ax.plot(xy[:, 0], xy[:, 1], color="#7e22ce", linewidth=5, zorder=6)
    for index in local_indices:
        ax.text(*corridors[index].center, f"C{index + 1}", ha="center", va="center",
                fontsize=11, color="#334155", zorder=7)
    ax.scatter([arc.x0, arc.xf], [arc.y0, arc.yf], color="#991b1b", s=35, zorder=8)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.margins(0.18)
    ax.grid(alpha=0.15)
    verdict = "CONTAINED" if contained_in_eroded_union else "NOT CONTAINED"
    ax.set_title(
        f"n={args.corridors}, case {args.case}, transition {transition_index + 1}\n"
        f"Rejected arc is {verdict} in the eroded local corridor union",
        fontsize=14,
    )
    ax.legend(handles=[
        Patch(facecolor="#dbeafe", edgecolor="#64748b", label="Original local corridor union"),
        Patch(facecolor="#bbf7d0", edgecolor="#15803d", label=r"Eroded union $(\cup C_k) \ominus B_r$"),
        Patch(facecolor="#fecaca", edgecolor="#dc2626", label="Swept circular footprint"),
        Line2D([], [], color="#dc2626", linewidth=3, label="Rejected arc centerline"),
        Line2D([], [], color="#7e22ce", linewidth=5, label="Centerline outside eroded union"),
    ], loc="best", framealpha=0.95)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output.with_suffix(f".{extension}"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    result = {
        "corridors": args.corridors,
        "case": args.case,
        "transition_index_zero_based": transition_index,
        "transition_number_one_based": transition_index + 1,
        "local_corridor_numbers_one_based": [index + 1 for index in local_indices],
        "footprint_radius": bicycle.width / 2,
        "arc_radius": float(arc.radius),
        "arc_amplitude_radians": float(arc.iota),
        "arc_length": float(centerline.length),
        "outside_centerline_length": outside_length,
        "arc_centerline_contained_in_eroded_union": contained_in_eroded_union,
        "swept_footprint_contained_in_original_union": footprint_contained,
        "arc_centerline_contained_in_union_of_individually_eroded_corridors": contained_in_individual_union,
        "outside_centerline_length_by_individually_eroded_corridor": {
            str(index + 1): float(centerline.difference(region.buffer(tolerance)).length)
            for index, region in zip(local_indices, individual_safe_regions)
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
