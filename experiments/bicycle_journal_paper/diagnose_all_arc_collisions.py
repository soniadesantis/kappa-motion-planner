"""Classify all benchmark arc-collision failures by local-union containment.

The diagnostic replays saved failures without changing planner acceptance.
For transition j it uses I_j = {j-1, j, j+1}, clipped to the sequence.
"""

from argparse import ArgumentParser
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.ops import unary_union

from benchmark_new_bicycle_baseline import make_fixed_bicycle
from diagnose_arc_collision_case import arc_linestring
from kappa_planner.helpers.collision_avoidance import (
    compute_safe_corridor_union,
    corridor_to_axis_aligned_polygon,
)
from kappa_planner.refinement_new import get_single_circle
from plot_refinement_failures import replay_failure


ROOT = Path(__file__).parent
CATEGORIES = (
    "safe_in_restricted_union",
    "safe_only_in_eroded_full_union",
    "unsafe_in_eroded_full_union",
)


def classify_record(record, corridors, trace, footprint_radius, tolerance=1e-9):
    rejected = trace["failed_arcs"][-1]
    arc, circle = rejected["arc"], rejected["circle"]
    groups = trace["state"]["groups"]
    transition_index = next(
        index for index, group in enumerate(groups)
        if group and np.allclose(get_single_circle(group).center, circle.center)
    )
    local_indices = list(range(
        max(0, transition_index - 1),
        min(len(corridors), transition_index + 2),
    ))
    local_corridors = [corridors[index] for index in local_indices]
    centerline, _ = arc_linestring(arc)

    individually_eroded = [
        corridor_to_axis_aligned_polygon(corridor).buffer(
            -footprint_radius, quad_segs=64,
        )
        for corridor in local_corridors
    ]
    restricted_union = unary_union(individually_eroded)
    full_union = compute_safe_corridor_union(
        local_corridors, footprint_radius, quad_segs=64,
    )
    restricted_outside = centerline.difference(
        restricted_union.buffer(tolerance)
    )
    full_outside = centerline.difference(full_union.buffer(tolerance))
    safe_restricted = restricted_outside.is_empty
    safe_full = full_outside.is_empty

    if safe_restricted:
        category = CATEGORIES[0]
    elif safe_full:
        category = CATEGORIES[1]
    else:
        category = CATEGORIES[2]

    return {
        "corridors": len(corridors),
        "case": record["case"],
        "transition_index_zero_based": transition_index,
        "transition_number_one_based": transition_index + 1,
        "local_corridor_numbers_one_based": [index + 1 for index in local_indices],
        "category": category,
        "safe_in_restricted_union": safe_restricted,
        "safe_in_eroded_full_union": safe_full,
        "arc_length": float(centerline.length),
        "outside_restricted_union_length": float(restricted_outside.length),
        "outside_eroded_full_union_length": float(full_outside.length),
        "arc_amplitude_radians": float(arc.iota),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "results" / "new_bicycle_baseline"
        / "short_corridors_R1_500_assembly_diagnostics.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "new_bicycle_baseline"
        / "arc_collision_union_diagnostics.json",
    )
    parser.add_argument(
        "--figure-output", type=Path,
        default=ROOT / "figures" / "new_bicycle_baseline"
        / "arc_collision_union_diagnostics.png",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    bicycle = make_fixed_bicycle()
    records = []
    for group in report["groups"]:
        failures = [
            record for record in group["records"]
            if record["refinement"].get("failure_reason") == "arc_collision_failed"
        ]
        for record in failures:
            corridors, _, trace = replay_failure(record, bicycle)
            result = classify_record(
                record, corridors, trace, bicycle.width / 2,
            )
            records.append(result)
            print(
                f"n={result['corridors']}, case={result['case']}, "
                f"transition={result['transition_number_one_based']}: "
                f"{result['category']}",
                flush=True,
            )

    by_count = {}
    for corridor_count in sorted({record["corridors"] for record in records}):
        subset = [record for record in records if record["corridors"] == corridor_count]
        counts = Counter(record["category"] for record in subset)
        by_count[str(corridor_count)] = {
            "total": len(subset),
            **{category: counts[category] for category in CATEGORIES},
        }
    totals = Counter(record["category"] for record in records)
    summary = {
        "admissible_index_rule": "I_j = {j-1, j, j+1}, clipped to sequence boundaries",
        "footprint_radius": bicycle.width / 2,
        "total_arc_collision_failures": len(records),
        "categories": {category: totals[category] for category in CATEGORIES},
        "by_corridor_count": by_count,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")

    csv_output = args.output.with_suffix(".csv")
    with csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            key for key in records[0] if key != "local_corridor_numbers_one_based"
        ] + ["local_corridor_numbers_one_based"])
        writer.writeheader()
        for record in records:
            writer.writerow({
                **record,
                "local_corridor_numbers_one_based": " ".join(
                    map(str, record["local_corridor_numbers_one_based"])
                ),
            })

    corridor_counts = sorted(int(value) for value in by_count)
    bottoms = np.zeros(len(corridor_counts), dtype=int)
    colors = ("#16a34a", "#f59e0b", "#dc2626")
    labels = (
        "Safe in restricted union",
        "Safe only in eroded full union",
        "Unsafe in eroded full union",
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for category, color, label in zip(CATEGORIES, colors, labels):
        values = np.array([
            by_count[str(count)][category] for count in corridor_counts
        ])
        ax.bar(corridor_counts, values, bottom=bottoms, color=color, label=label)
        bottoms += values
    ax.set_xlabel("Number of corridors")
    ax.set_ylabel("Arc-collision failures")
    ax.set_xticks(corridor_counts)
    ax.set_title("Independent local-union classification of rejected arcs")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    args.figure_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_output, dpi=200, bbox_inches="tight")
    fig.savefig(args.figure_output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({
        "total": len(records),
        "categories": summary["categories"],
        "by_corridor_count": by_count,
    }, indent=2))


if __name__ == "__main__":
    main()
