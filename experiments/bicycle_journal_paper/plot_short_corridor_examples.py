"""Plot benchmark maps containing the shortest intermediate corridors."""

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np

from benchmark_new_bicycle_baseline import (
    attach_boundary_connections,
    corridor_worlds_from_bounds,
    make_fixed_bicycle,
    plot_maneuver_path,
)
from kappa_planner.baseline_construction_new import compute_bicycle_baseline
from kappa_planner.refinement_new import refine_bicycle_baseline


ROOT = Path(__file__).parent


def replay(record, bicycle):
    corridors = corridor_worlds_from_bounds(record["bounds"], record["sampled"]["headings"])
    start, end = record["initial_pose"]["world"], record["final_pose"]["world"]
    baseline = compute_bicycle_baseline(corridors, bicycle, initial_pose=start, final_pose=end)
    boundary = attach_boundary_connections(corridors, baseline, bicycle, start, end)
    refinement, failure = refine_bicycle_baseline(
        corridors, bicycle, baseline,
        initial_pose=start, final_pose=end, return_failure=True,
    )
    if boundary["status"] != "success" or refinement is None or refinement.trajectory is None:
        raise RuntimeError(f"Could not replay selected case: {boundary}, {failure}")
    return corridors, baseline, refinement


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "results" / "new_bicycle_baseline" / "final_chapter5_benchmark_500.json",
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "figures" / "new_bicycle_baseline" / "short_corridor_examples",
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    candidates = []
    for group in report["groups"]:
        for record in group["records"]:
            if record["boundary_success"] and record["refinement"].get("solution_source") == "refined":
                interior = record["sampled"]["walk_lengths"][1:-1]
                candidates.append((min(interior), group["corridors"], record))
    selected = sorted(candidates, key=lambda item: item[0])[:args.count]

    bicycle = make_fixed_bicycle()
    columns = 2
    rows = (len(selected) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(13, 5.8 * rows), squeeze=False)
    notes = []
    for ax, (minimum_length, corridor_count, record) in zip(axes.flat, selected):
        corridors, baseline, refinement = replay(record, bicycle)
        shortest_index = 1 + int(np.argmin(record["sampled"]["walk_lengths"][1:-1]))
        for index, corridor in enumerate(corridors):
            highlighted = index == shortest_index
            ax.add_patch(Polygon(
                corridor.corners, closed=True,
                facecolor="#fed7aa" if highlighted else "#dbeafe",
                edgecolor="#ea580c" if highlighted else "#64748b",
                linewidth=1.8 if highlighted else 0.7,
                alpha=0.72 if highlighted else 0.5,
            ))
        plot_maneuver_path(ax, baseline.trajectory, "#1f2937", linewidth=1.4)
        plot_maneuver_path(ax, refinement.trajectory, "#7c3aed", linewidth=2.0)
        for key, color in (("initial_pose", "#16a34a"), ("final_pose", "#dc2626")):
            x, y, heading = record[key]["world"]
            ax.plot(x, y, "o", color=color, markersize=4, zorder=8)
            ax.annotate("", (x + np.cos(heading), y + np.sin(heading)), xytext=(x, y),
                        arrowprops={"arrowstyle": "->", "color": color}, zorder=8)
        reduction = record["comparison"]["traversal_time_reduction_percent"]
        ax.set_title(
            f"$n={corridor_count}$, case {record['case']} · shortest corridor "
            f"$L={minimum_length:.3f}$ m\n$\\Delta T={reduction:.2f}\\%$",
            fontsize=11,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.autoscale_view()
        ax.margins(0.07)
        ax.set_xticks([])
        ax.set_yticks([])
        notes.append({
            "corridors": corridor_count,
            "case": record["case"],
            "shortest_corridor_number_one_based": shortest_index + 1,
            "shortest_walk_length": minimum_length,
            "traversal_time_reduction_percent": reduction,
        })
    for ax in list(axes.flat)[len(selected):]:
        ax.set_axis_off()
    fig.legend(handles=[
        Patch(facecolor="#dbeafe", edgecolor="#64748b", label="Corridors"),
        Patch(facecolor="#fed7aa", edgecolor="#ea580c", label="Shortest corridor"),
        Line2D([], [], color="#1f2937", label="Complete baseline"),
        Line2D([], [], color="#7c3aed", linewidth=2, label="Refined trajectory"),
        Line2D([], [], color="#16a34a", marker="o", linestyle="none", label="Initial pose"),
        Line2D([], [], color="#dc2626", marker="o", linestyle="none", label="Final pose"),
    ], loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "overview.pdf", bbox_inches="tight")
    fig.savefig(args.output / "overview.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    (args.output / "cases.json").write_text(json.dumps(notes, indent=2) + "\n")
    print(json.dumps(notes, indent=2))


if __name__ == "__main__":
    main()
