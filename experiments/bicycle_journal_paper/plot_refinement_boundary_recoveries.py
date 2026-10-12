"""Plot cases where baseline boundary attachment fails but refinement succeeds."""

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
    plot_baseline_panel,
    plot_maneuver_path,
)
from kappa_planner.baseline_construction_new import compute_bicycle_baseline
from kappa_planner.refinement_new import refine_bicycle_baseline


ROOT = Path(__file__).parent


def replay(record, bicycle):
    corridors = corridor_worlds_from_bounds(
        record["bounds"], record["sampled"]["headings"],
    )
    start, end = record["initial_pose"]["world"], record["final_pose"]["world"]
    baseline = compute_bicycle_baseline(
        corridors, bicycle, initial_pose=start, final_pose=end,
    )
    boundary = attach_boundary_connections(
        corridors, baseline, bicycle, start, end,
    )
    refinement, failure = refine_bicycle_baseline(
        corridors, bicycle, baseline,
        initial_pose=start, final_pose=end, return_failure=True,
    )
    if boundary["status"] == "success":
        raise RuntimeError("Baseline boundary attachment unexpectedly succeeded.")
    if refinement is None or refinement.trajectory is None:
        raise RuntimeError(f"Refinement recovery could not be replayed: {failure}")
    return corridors, baseline, boundary, refinement


def draw_refinement(ax, corridors, refinement, record, *, title=True):
    for corridor in corridors:
        ax.add_patch(Polygon(
            corridor.corners, closed=True, facecolor="#dbeafe",
            edgecolor="#64748b", linewidth=0.7, alpha=0.5,
        ))
    plot_maneuver_path(ax, refinement.trajectory, "#7c3aed", linewidth=2.1)
    for key, color in (("initial_pose", "#16a34a"), ("final_pose", "#dc2626")):
        x, y, heading = record[key]["world"]
        ax.plot(x, y, "o", color=color, markersize=4, zorder=8)
        ax.annotate(
            "", (x + np.cos(heading), y + np.sin(heading)), xytext=(x, y),
            arrowprops={"arrowstyle": "->", "color": color}, zorder=8,
        )
    if title:
        ax.set_title("Successful complete refinement", fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.margins(0.07)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_overlay(ax, corridors, baseline, refinement, record, boundary):
    plot_baseline_panel(ax, corridors, baseline, record)
    plot_maneuver_path(ax, refinement.trajectory, "#7c3aed", linewidth=2.0)
    ax.set_title(
        f"n={record['corridor_count']}, case {record['case']}\n"
        f"baseline: {boundary['failure_reason']} · refinement: success",
        fontsize=9,
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "results" / "new_bicycle_baseline"
        / "final_chapter5_benchmark_500.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "figures" / "new_bicycle_baseline"
        / "refinement_boundary_recoveries",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    records = []
    for group in report["groups"]:
        for record in group["records"]:
            if (
                record["baseline_success"]
                and not record["boundary_success"]
                and record["refinement"].get("solution_source") == "refined"
            ):
                record["corridor_count"] = group["corridors"]
                records.append(record)
    if not records:
        raise RuntimeError("No boundary-connection recoveries found.")

    bicycle = make_fixed_bicycle()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = (len(records) + 1) // 2
    overview, axes = plt.subplots(rows, 2, figsize=(14, 5.2 * rows), squeeze=False)
    notes = []
    for ax, record in zip(axes.flat, records):
        corridors, baseline, boundary, refinement = replay(record, bicycle)
        draw_overlay(ax, corridors, baseline, refinement, record, boundary)

        individual, pair = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
        plot_baseline_panel(pair[0], corridors, baseline, record)
        pair[0].set_title(
            f"Baseline boundary attachment failed\n{boundary['failure_reason']}",
            fontsize=11,
        )
        draw_refinement(pair[1], corridors, refinement, record)
        individual.suptitle(
            f"n={record['corridor_count']}, case {record['case']}", fontsize=15,
        )
        individual.legend(handles=[
            Patch(facecolor="#dbeafe", edgecolor="#64748b", label="Corridors"),
            Line2D([], [], color="#172033", label="Internal baseline"),
            Line2D([], [], color="#ea580c", label="Baseline fillets"),
            Line2D([], [], color="#16a34a", label="Initial connection / pose"),
            Line2D([], [], color="#dc2626", label="Final connection / pose"),
            Line2D([], [], color="#7c3aed", label="Complete refined trajectory"),
        ], loc="lower center", ncol=3, frameon=False)
        individual.tight_layout(rect=(0, 0.1, 1, 0.95))
        stem = f"n{record['corridor_count']}_case_{record['case']:03d}"
        individual.savefig(args.output / f"{stem}.png", dpi=200, bbox_inches="tight")
        individual.savefig(args.output / f"{stem}.pdf", bbox_inches="tight")
        plt.close(individual)
        notes.append({
            "corridors": record["corridor_count"],
            "case": record["case"],
            "generated_index": record["generated_index"],
            "baseline_boundary_failure": boundary["failure_reason"],
            "refinement_primitives": len(refinement.trajectory),
        })
    for ax in list(axes.flat)[len(records):]:
        ax.set_axis_off()
    overview.suptitle(
        "Complete refinements recovered from failed baseline boundary connections",
        fontsize=16,
    )
    overview.legend(handles=[
        Line2D([], [], color="#172033", label="Internal baseline"),
        Line2D([], [], color="#ea580c", label="Baseline fillets"),
        Line2D([], [], color="#7c3aed", label="Complete refined trajectory"),
        Line2D([], [], color="#16a34a", label="Initial pose"),
        Line2D([], [], color="#dc2626", label="Final pose"),
    ], loc="lower center", ncol=5, frameon=False)
    overview.tight_layout(rect=(0, 0.035, 1, 0.97))
    overview.savefig(args.output / "overview.png", dpi=200, bbox_inches="tight")
    overview.savefig(args.output / "overview.pdf", bbox_inches="tight")
    plt.close(overview)
    (args.output / "cases.json").write_text(json.dumps(notes, indent=2) + "\n")
    print(json.dumps(notes, indent=2))


if __name__ == "__main__":
    main()
