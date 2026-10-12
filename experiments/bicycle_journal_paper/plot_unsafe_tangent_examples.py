"""Plot successful experimental refinements containing unsafe tangents."""

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon as PlotPolygon
import numpy as np
from shapely.geometry import LineString

from benchmark_new_bicycle_baseline import (
    attach_boundary_connections,
    corridor_worlds_from_bounds,
    make_fixed_bicycle,
)
from diagnose_arc_collision_case import add_geometry
from kappa_planner.baseline_construction_new import compute_bicycle_baseline
from kappa_planner.helpers.collision_avoidance import compute_safe_corridor_union
from kappa_planner.refinement_new import get_single_circle, refine_bicycle_baseline


ROOT = Path(__file__).parent


def replay(record, bicycle):
    corridors = corridor_worlds_from_bounds(
        record["bounds"], record["sampled"]["headings"],
    )
    start = record["initial_pose"]["world"]
    end = record["final_pose"]["world"]
    baseline = compute_bicycle_baseline(
        corridors, bicycle, initial_pose=start, final_pose=end,
    )
    attach_boundary_connections(corridors, baseline, bicycle, start, end)
    result, failure = refine_bicycle_baseline(
        corridors, bicycle, baseline,
        initial_pose=start, final_pose=end, return_failure=True,
    )
    if result is None:
        raise RuntimeError(f"Could not replay direct refinement: {failure}")
    return corridors, result


def draw_example(ax, record, audit_record, bicycle):
    corridors, result = replay(record, bicycle)
    unsafe = audit_record["unsafe_tangents"][0]
    start_index = unsafe["start_transition"] - 1
    end_index = unsafe["end_transition"] - 1
    local_corridors = corridors[start_index:end_index + 2]
    safe_union = compute_safe_corridor_union(
        local_corridors, bicycle.width / 2, quad_segs=64,
    )

    position = next(
        position for position, (first, second) in enumerate(zip(
            result.active_circle_indices, result.active_circle_indices[1:],
        ))
        if first == start_index and second == end_index
    )
    tangent = result.tangents[position]
    tangent_points = np.asarray([tangent.start_point, tangent.end_point])
    tangent_line = LineString(tangent_points)
    swept_footprint = tangent_line.buffer(bicycle.width / 2, quad_segs=64)
    outside = tangent_line.difference(safe_union.buffer(1e-9))

    for index in range(start_index, end_index + 2):
        corridor = corridors[index]
        ax.add_patch(PlotPolygon(
            corridor.corners, closed=True, facecolor="#dbeafe",
            edgecolor="#64748b", alpha=0.35, linewidth=1.0, zorder=1,
        ))
        ax.text(*corridor.center, f"C{index + 1}", color="#475569",
                ha="center", va="center", fontsize=8, zorder=6)
    add_geometry(
        ax, safe_union, facecolor="#bbf7d0", edgecolor="#15803d",
        alpha=0.55, linewidth=1.5, zorder=2,
    )
    add_geometry(
        ax, swept_footprint, facecolor="#fecaca", edgecolor="#dc2626",
        alpha=0.30, linewidth=0.9, zorder=3,
    )
    for transition_index in (start_index, end_index):
        circle = get_single_circle(result.circle_groups[transition_index])
        ax.add_patch(Circle(
            circle.center, circle.radius, fill=False,
            edgecolor="#2563eb", linewidth=1.5, zorder=4,
        ))
        ax.scatter(*circle.center, color="#2563eb", s=12, zorder=5)
    ax.plot(*tangent_points.T, color="#dc2626", linewidth=2.5, zorder=7)
    if not outside.is_empty:
        pieces = [outside] if outside.geom_type == "LineString" else list(outside.geoms)
        for piece in pieces:
            points = np.asarray(piece.coords)
            ax.plot(*points.T, color="#7e22ce", linewidth=5, zorder=8)
    ax.scatter(*tangent_points.T, color="#991b1b", s=18, zorder=9)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.margins(0.14)
    ax.grid(alpha=0.12)
    ax.set_title(
        f"n={audit_record['corridors']}, case {audit_record['case']}\n"
        f"transitions {start_index + 1} → {end_index + 1}",
        fontsize=10,
    )
    return float(outside.length)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "results" / "new_bicycle_baseline"
        / "final_chapter5_no_nonoverlap_tangent_collision_500.json",
    )
    parser.add_argument(
        "--audit", type=Path,
        default=ROOT / "results" / "new_bicycle_baseline"
        / "final_chapter5_no_nonoverlap_tangent_collision_audit.json",
    )
    parser.add_argument("--per-count", type=int, default=2)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "figures" / "new_bicycle_baseline"
        / "unsafe_tangent_examples",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    audit = json.loads(args.audit.read_text())
    records_by_key = {
        (group["corridors"], record["case"]): record
        for group in report["groups"] for record in group["records"]
    }
    selected = []
    for corridor_count in sorted({r["corridors"] for r in audit["records"]}):
        matches = [
            record for record in audit["records"]
            if record["corridors"] == corridor_count and record["unsafe_tangents"]
        ]
        selected.extend(matches[:args.per_count])

    bicycle = make_fixed_bicycle()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = (len(selected) + 1) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(14, 5.5 * rows), squeeze=False)
    notes = []
    for ax, audit_record in zip(axes.flat, selected):
        record = records_by_key[(audit_record["corridors"], audit_record["case"])]
        outside_length = draw_example(ax, record, audit_record, bicycle)
        notes.append({
            "corridors": audit_record["corridors"],
            "case": audit_record["case"],
            "unsafe_tangent": audit_record["unsafe_tangents"][0],
            "centerline_length_outside_safe_union": outside_length,
        })
        individual, individual_ax = plt.subplots(figsize=(8, 7))
        draw_example(individual_ax, record, audit_record, bicycle)
        individual.legend(handles=[
            Patch(facecolor="#dbeafe", edgecolor="#64748b", label="Local corridors"),
            Patch(facecolor="#bbf7d0", edgecolor="#15803d", label="Eroded local union"),
            Line2D([], [], color="#2563eb", label="Connected circles"),
            Patch(facecolor="#fecaca", edgecolor="#dc2626", label="Swept footprint"),
            Line2D([], [], color="#7e22ce", linewidth=5, label="Unsafe tangent portion"),
        ], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.16))
        individual.tight_layout()
        stem = f"n{audit_record['corridors']}_case_{audit_record['case']:03d}"
        individual.savefig(args.output / f"{stem}.png", dpi=220, bbox_inches="tight")
        individual.savefig(args.output / f"{stem}.pdf", bbox_inches="tight")
        plt.close(individual)
    for ax in list(axes.flat)[len(selected):]:
        ax.set_axis_off()
    figure.suptitle(
        "Unsafe tangents accepted when non-overlapping tangent collision checks are disabled",
        fontsize=16,
    )
    figure.legend(handles=[
        Patch(facecolor="#dbeafe", edgecolor="#64748b", label="Local corridors"),
        Patch(facecolor="#bbf7d0", edgecolor="#15803d", label="Eroded local union"),
        Line2D([], [], color="#2563eb", label="Connected circles"),
        Patch(facecolor="#fecaca", edgecolor="#dc2626", label="Swept footprint"),
        Line2D([], [], color="#7e22ce", linewidth=5, label="Tangent outside safe union"),
    ], loc="lower center", ncol=5, frameon=False)
    figure.tight_layout(rect=(0, 0.035, 1, 0.97))
    figure.savefig(args.output / "overview.png", dpi=220, bbox_inches="tight")
    figure.savefig(args.output / "overview.pdf", bbox_inches="tight")
    plt.close(figure)
    (args.output / "examples.json").write_text(json.dumps(notes, indent=2) + "\n")
    print(json.dumps(notes, indent=2))


if __name__ == "__main__":
    main()
