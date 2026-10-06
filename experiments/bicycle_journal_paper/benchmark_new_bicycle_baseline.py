"""Benchmark the new bicycle baseline on random corridor sequences.

The experiment collects exactly ``--cases`` geometrically valid sequences for
every requested corridor count.  Invalid generated sequences are counted and
skipped.  Only corridor sizes and the random walk geometry vary; the footprint
radius and fillet radius are fixed for the whole run.

Run from the repository root, for example::

    PYTHONPATH=src .venv/bin/python \
        experiments/bicycle_journal_paper/benchmark_new_bicycle_baseline.py

The default run tests 100 sequences each with 5, 10, and 15 corridors.  A JSON
report stores the sampled bounds, outcome, timing, and enough configuration to
replay every case.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import example_random_baseline_construction as generator
from kappa_planner.baseline_construction_new import (
    compute_bicycle_baseline,
    compute_overlap_two_axis_aligned_corridors,
    compute_safe_overlap,
    validate_baseline_corridor_sequence,
)
from kappa_planner.corridor import CorridorWorld
from kappa_planner.vehicle import Bicycle


SEED = 7
CORRIDOR_COUNTS = (5, 10, 15)
CASES_PER_COUNT = 100

# Keep the vehicle fixed in this first experiment.  The bicycle constructor
# derives max_radius = wheelbase / tan(delta_max).
FOOTPRINT_RADIUS = 0.5
ARC_RADIUS = 2.0
WIDTH_RANGE = (1.5, 10.0)
WALK_LENGTH_RANGE = (2.0, 20.0)
OVERHANG_RANGE = (0.0, 2.0)
ALIGNED_OFFSET_PROBABILITY = 0.25
ALIGNED_OFFSET_MARGIN = 0.05
MIN_ALIGNED_OFFSET = 0.10
MAX_CORRIDOR_ATTEMPTS = 500
MAX_SEQUENCE_RESTARTS = 100

OUTPUT = (
    Path(__file__).resolve().parent
    / "results"
    / "new_bicycle_baseline"
    / "random_size_sweep.json"
)
FIGURE_OUTPUT = (
    Path(__file__).resolve().parent
    / "figures"
    / "new_bicycle_baseline"
    / "representative_random_cases.png"
)


def corridor_worlds_from_bounds(bounds, headings=None):
    """Convert axis-aligned ``(xmin, xmax, ymin, ymax)`` bounds to corridors."""
    if headings is None:
        headings = [0] * len(bounds)
    corridors = []
    for number, ((xmin, xmax, ymin, ymax), heading) in enumerate(
        zip(bounds, headings)
    ):
        horizontal = heading % 2 == 0
        corridors.append(
            CorridorWorld(
                width=(ymax - ymin) if horizontal else (xmax - xmin),
                height=(xmax - xmin) if horizontal else (ymax - ymin),
                center=[0.5 * (xmin + xmax), 0.5 * (ymin + ymax)],
                tilt=heading * np.pi / 2.0,
                number=number,
            )
        )
    return corridors


def sample_validated_corridors(rng, corridor_count, bicycle):
    """Grow a random walk whose every accepted prefix passes validation."""
    axes = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    candidate_attempts = 0

    for restart in range(MAX_SEQUENCE_RESTARTS + 1):
        bounds = []
        widths = []
        lengths = []
        overhangs = []
        headings = []
        transition_types = []
        transverse_offsets = []
        heading = int(rng.integers(4))
        point = np.zeros(2)

        for j in range(corridor_count):
            accepted = False
            for _ in range(MAX_CORRIDOR_ATTEMPTS):
                candidate_attempts += 1
                width = float(rng.uniform(*WIDTH_RANGE))
                length = float(rng.uniform(*WALK_LENGTH_RANGE))
                overhang = rng.uniform(*OVERHANG_RANGE, 2)
                candidate_heading = heading
                candidate_point = point.copy()
                transition_type = None
                transverse_offset = 0.0

                if j:
                    heading_change = int(
                        rng.choice(
                            [0, 1, -1],
                            p=[
                                generator.STRAIGHT_PROBABILITY,
                                (1 - generator.STRAIGHT_PROBABILITY) / 2,
                                (1 - generator.STRAIGHT_PROBABILITY) / 2,
                            ],
                        )
                    )
                    candidate_heading = (heading + heading_change) % 4
                    if heading_change == 0:
                        max_offset = (
                            0.5 * (widths[-1] + width)
                            - 2.0 * FOOTPRINT_RADIUS
                            - ALIGNED_OFFSET_MARGIN
                        )
                        use_offset = (
                            max_offset >= MIN_ALIGNED_OFFSET
                            and rng.random() < ALIGNED_OFFSET_PROBABILITY
                        )
                        if use_offset:
                            magnitude = rng.uniform(MIN_ALIGNED_OFFSET, max_offset)
                            transverse_offset = float(
                                magnitude * rng.choice((-1.0, 1.0))
                            )
                            direction = axes[candidate_heading]
                            normal = np.array([-direction[1], direction[0]])
                            candidate_point += transverse_offset * normal
                            transition_type = "aligned_offset"
                        else:
                            transition_type = "aligned_collinear"
                    else:
                        transition_type = "turn"

                direction = axes[candidate_heading]
                endpoint = candidate_point + length * direction
                first = candidate_point - overhang[0] * direction
                last = endpoint + overhang[1] * direction
                transverse = np.abs(np.array([-direction[1], direction[0]]))
                low = np.minimum(first, last) - 0.5 * width * transverse
                high = np.maximum(first, last) + 0.5 * width * transverse
                candidate_bound = (
                    float(low[0]),
                    float(high[0]),
                    float(low[1]),
                    float(high[1]),
                )

                candidate_bounds = bounds + [candidate_bound]
                candidate_headings = headings + [candidate_heading]
                if j and not validate_baseline_corridor_sequence(
                    corridor_worlds_from_bounds(candidate_bounds, candidate_headings),
                    bicycle,
                ):
                    continue

                bounds = candidate_bounds
                widths.append(width)
                lengths.append(length)
                overhangs.append(overhang.tolist())
                headings.append(candidate_heading)
                if j:
                    transition_types.append(transition_type)
                    transverse_offsets.append(transverse_offset)
                heading = candidate_heading
                point = endpoint
                accepted = True
                break

            if not accepted:
                break
        else:
            spans = [(b - a, d - c) for a, b, c, d in bounds]
            sampled = {
                "widths": widths,
                "walk_lengths": lengths,
                "overhangs": overhangs,
                "headings": headings,
                "transition_types": transition_types,
                "transverse_offsets": transverse_offsets,
                "original_bounds": list(bounds),
                "edge_extensions": [],
                "final_widths": [
                    span[1 - h % 2] for span, h in zip(spans, headings)
                ],
                "final_lengths": [span[h % 2] for span, h in zip(spans, headings)],
                "candidate_attempts": candidate_attempts,
                "sequence_restarts": restart,
            }
            return bounds, sampled

    raise RuntimeError(
        f"Could not construct a validated {corridor_count}-corridor sequence."
    )


def make_fixed_bicycle():
    """Create a bicycle with the radii fixed by this experiment."""
    wheelbase = 1.0
    steering_limit = float(np.arctan(wheelbase / ARC_RADIUS))
    bicycle = Bicycle(
        width=2.0 * FOOTPRINT_RADIUS,
        length=2.0 * FOOTPRINT_RADIUS,
        wheelbase=wheelbase,
        delta_max=steering_limit,
        delta_min=-steering_limit,
    )
    if not np.isclose(bicycle.max_radius, ARC_RADIUS):
        raise RuntimeError("Failed to construct the requested fixed arc radius.")
    return bicycle


def validate_result(result, corridor_count):
    """Check the basic structural contract of a successful baseline result."""
    transition_count = corridor_count - 1
    fields = (
        "intersection_points",
        "corridor_overlaps",
        "safe_overlaps",
        "turn_directions",
        "candidate_corner_points",
        "admissible_regions",
        "reachable_regions",
        "waypoints",
        "fillets",
    )
    for field in fields:
        actual = len(getattr(result, field))
        if actual != transition_count:
            raise AssertionError(
                f"{field} has length {actual}; expected {transition_count}"
            )
    if len(result.corridor_directions) != corridor_count:
        raise AssertionError(
            "corridor_directions has length "
            f"{len(result.corridor_directions)}; expected {corridor_count}"
        )


def select_representative_successes(records, count=5):
    """Select timing-spanning cases that include both aligned geometries."""
    successful = sorted(
        (record for record in records if record["status"] == "success"),
        key=lambda record: record["baseline_ms"],
    )
    if len(successful) < count:
        raise RuntimeError(
            f"Need {count} successful cases, but only found {len(successful)}."
        )
    median_time = median(record["baseline_ms"] for record in successful)

    if count == 1:
        return [
            min(
                successful,
                key=lambda record: abs(record["baseline_ms"] - median_time),
            )
        ]

    def nearest_to_median(candidates):
        return min(
            candidates,
            key=lambda record: abs(record["baseline_ms"] - median_time),
        )

    selected = [successful[0], successful[-1]]
    for transition_type in ("aligned_collinear", "aligned_offset"):
        candidates = [
            record
            for record in successful
            if transition_type in record["sampled"]["transition_types"]
        ]
        if candidates:
            selected.append(nearest_to_median(candidates))

    quantile_indices = np.rint(
        np.linspace(0, len(successful) - 1, 2 * count + 1)
    ).astype(int)
    for index in quantile_indices:
        candidate = successful[index]
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) == count:
            break

    return sorted(selected[:count], key=lambda record: record["baseline_ms"])


def plot_baseline_panel(ax, corridors, baseline, record):
    """Plot the baseline layers used by EXAMPLE_THESIS on an existing axis."""
    for corridor in corridors:
        corners = np.asarray(corridor.corners, dtype=float)
        ax.add_patch(
            plt.Polygon(
                corners,
                closed=True,
                facecolor="#dbeafe",
                edgecolor="#64748b",
                linewidth=0.7,
                alpha=0.55,
            )
        )

    for (x_min, x_max), (y_min, y_max) in baseline.safe_overlaps:
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor="#7c3aed",
                linestyle=":",
                linewidth=1.0,
            )
        )

    waypoints = np.asarray(baseline.waypoints, dtype=float)
    ax.plot(
        waypoints[:, 0],
        waypoints[:, 1],
        "--o",
        color="#94a3b8",
        linewidth=0.8,
        markersize=2.5,
        zorder=4,
    )

    for j in range(len(waypoints) - 1):
        start = (
            waypoints[j]
            if baseline.fillets[j] is None
            else baseline.fillets[j].end_point
        )
        end = (
            waypoints[j + 1]
            if baseline.fillets[j + 1] is None
            else baseline.fillets[j + 1].start_point
        )
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#172033",
            linewidth=1.6,
            zorder=5,
        )

    for fillet in baseline.fillets:
        if fillet is None:
            continue
        radial = fillet.start_point - fillet.center
        start_angle = np.arctan2(radial[1], radial[0])
        angles = start_angle + fillet.turn_direction * np.linspace(0, np.pi / 2, 60)
        arc = fillet.center + np.linalg.norm(radial) * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        ax.plot(*arc.T, color="#ea580c", linewidth=1.8, zorder=6)

    ax.set_title(
        f"generated #{record['generated_index']} · {record['baseline_ms']:.3f} ms",
        fontsize=8,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.margins(0.06)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_failed_baseline_example(corridors, bicycle, failure, output, *, case_label=""):
    """Plot a validated sequence whose admissible or reachable set is empty."""
    fig, ax = plt.subplots(figsize=(8, 7))
    failing_index = failure.transition_index

    for j, corridor in enumerate(corridors):
        corners = np.asarray(corridor.corners, dtype=float)
        ax.add_patch(
            plt.Polygon(
                corners,
                closed=True,
                facecolor="#dbeafe",
                edgecolor="#64748b",
                linewidth=1.0,
                alpha=0.6,
            )
        )
        ax.text(*corridor.center, f"C{j + 1}", ha="center", va="center", fontsize=9)

    for j in range(len(corridors) - 1):
        overlap = compute_overlap_two_axis_aligned_corridors(
            corridors[j], corridors[j + 1]
        )
        safe_overlap = compute_safe_overlap(overlap, bicycle.width / 2)
        (x_min, x_max), (y_min, y_max) = safe_overlap
        failing = j == failing_index
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor="#fee2e2" if failing else "none",
                edgecolor="#dc2626" if failing else "#7c3aed",
                linestyle="-" if failing else ":",
                linewidth=2.5 if failing else 1.4,
                zorder=4,
            )
        )
        ax.text(
            0.5 * (x_min + x_max),
            0.5 * (y_min + y_max),
            f"D{j + 1}",
            color="#b91c1c" if failing else "#6d28d9",
            ha="center",
            va="center",
            fontsize=8,
            zorder=5,
        )

    suffix = f" · {case_label}" if case_label else ""
    ax.set_title(
        f"Validated 5-corridor sequence with no baseline{suffix}\n"
        f"{failure.reason} at transition j={failing_index} "
        f"(displayed as D{failing_index + 1})"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.margins(0.08)
    ax.grid(alpha=0.12)
    fig.legend(
        handles=[
            Patch(facecolor="#dbeafe", edgecolor="#64748b", label="Corridors"),
            Patch(fill=False, edgecolor="#7c3aed", linestyle=":", label="Safe overlaps"),
            Patch(facecolor="#fee2e2", edgecolor="#dc2626", label="Reported failure transition"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_failed_baseline_overview(cases, bicycle, output):
    """Plot several validated baseline failures in a compact overview."""
    columns = 3
    rows = (len(cases) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4.8 * rows), squeeze=False)

    for ax, case in zip(axes.flat, cases):
        corridors = case["corridors"]
        failure = case["failure"]
        failing_index = failure.transition_index
        for corridor in corridors:
            corners = np.asarray(corridor.corners, dtype=float)
            ax.add_patch(
                plt.Polygon(
                    corners,
                    closed=True,
                    facecolor="#dbeafe",
                    edgecolor="#64748b",
                    linewidth=0.8,
                    alpha=0.6,
                )
            )
        for j in range(len(corridors) - 1):
            overlap = compute_overlap_two_axis_aligned_corridors(
                corridors[j], corridors[j + 1]
            )
            safe_overlap = compute_safe_overlap(overlap, bicycle.width / 2)
            (x_min, x_max), (y_min, y_max) = safe_overlap
            failing = j == failing_index
            ax.add_patch(
                Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    facecolor="#fee2e2" if failing else "none",
                    edgecolor="#dc2626" if failing else "#7c3aed",
                    linestyle="-" if failing else ":",
                    linewidth=2.0 if failing else 1.0,
                    zorder=4,
                )
            )
        ax.set_title(
            f"generated #{case['case']} · {failure.reason}\n"
            f"transition j={failing_index} (D{failing_index + 1})",
            fontsize=10,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.autoscale_view()
        ax.margins(0.07)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in list(axes.flat)[len(cases):]:
        ax.set_axis_off()
    fig.suptitle("Validated 5-corridor sequences with no baseline", fontsize=15)
    fig.legend(
        handles=[
            Patch(facecolor="#dbeafe", edgecolor="#64748b", label="Corridors"),
            Patch(fill=False, edgecolor="#7c3aed", linestyle=":", label="Safe overlaps"),
            Patch(facecolor="#fee2e2", edgecolor="#dc2626", label="Reported failure transition"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_representative_examples(report, bicycle, output, columns=5):
    """Create an overview of successful representative baselines."""
    groups = report["groups"]
    fig, axes = plt.subplots(
        len(groups),
        columns,
        figsize=(18, 3.8 * len(groups)),
        squeeze=False,
    )
    selected_cases = {}

    for row, group in enumerate(groups):
        selected = select_representative_successes(group["records"], columns)
        selected_cases[str(group["corridors"])] = [
            record["generated_index"] for record in selected
        ]
        for ax, record in zip(axes[row], selected):
            corridors = corridor_worlds_from_bounds(
                record["bounds"], record["sampled"]["headings"]
            )
            baseline = compute_bicycle_baseline(corridors, bicycle)
            if baseline is None:
                raise RuntimeError(
                    f"Could not replay generated case {record['generated_index']}."
                )
            plot_baseline_panel(ax, corridors, baseline, record)
        axes[row, 0].set_ylabel(
            f"{group['corridors']} corridors",
            fontsize=11,
            labelpad=12,
        )

    subtitle = (
        "successful case nearest the median computation time"
        if columns == 1
        else "including aligned-collinear and aligned-offset transitions"
    )
    fig.suptitle(
        f"Representative successful bicycle baselines\n{subtitle}",
        fontsize=15,
    )
    fig.legend(
        handles=[
            Patch(facecolor="#dbeafe", edgecolor="#64748b", label="Corridors"),
            Patch(fill=False, edgecolor="#7c3aed", linestyle=":", label="Safe overlaps"),
            Line2D([], [], color="#94a3b8", linestyle="--", marker="o", label="Waypoints"),
            Line2D([], [], color="#172033", linewidth=2, label="Straight baseline"),
            Line2D([], [], color="#ea580c", linewidth=2, label="Radius-R fillets"),
        ],
        loc="lower center",
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return selected_cases


def run_group(corridor_count, cases, max_generated, rng, bicycle):
    generator.NUMBER_OF_CORRIDORS = corridor_count
    records = []
    validation_timings = []
    invalid_sequences = 0
    validation_errors = Counter()
    generated = 0
    candidate_attempts = 0
    sequence_restarts = 0

    while len(records) < cases and generated < max_generated:
        generated += 1
        bounds, sampled = sample_validated_corridors(rng, corridor_count, bicycle)
        candidate_attempts += sampled["candidate_attempts"]
        sequence_restarts += sampled["sequence_restarts"]
        corridors = corridor_worlds_from_bounds(bounds, sampled["headings"])

        validation_ms = None
        baseline_ms = None
        baseline_started = None
        failure_transition_index = None
        try:
            validation_started = perf_counter_ns()
            valid_sequence = validate_baseline_corridor_sequence(
                corridor_list=corridors,
                bicycle=bicycle,
            )
            validation_ms = (perf_counter_ns() - validation_started) / 1e6
            validation_timings.append(validation_ms)

            if not valid_sequence:
                invalid_sequences += 1
                continue

            case_number = len(records) + 1
            baseline_started = perf_counter_ns()
            result, failure = compute_bicycle_baseline(
                corridor_list=corridors,
                bicycle=bicycle,
                return_failure=True,
            )
            baseline_ms = (perf_counter_ns() - baseline_started) / 1e6
            if result is None:
                status = failure.reason
                error = None
                failure_transition_index = failure.transition_index
            else:
                validate_result(result, corridor_count)
                status = "success"
                error = None
                failure_transition_index = None
        except Exception as exception:  # Record crashes without losing later cases.
            if validation_ms is None:
                validation_ms = (perf_counter_ns() - validation_started) / 1e6
                validation_timings.append(validation_ms)
                validation_errors[type(exception).__name__] += 1
                continue
            status = "error"
            error = f"{type(exception).__name__}: {exception}"
            failure_transition_index = None
            if baseline_started is not None:
                baseline_ms = (perf_counter_ns() - baseline_started) / 1e6

        records.append(
            {
                "case": case_number,
                "generated_index": generated,
                "status": status,
                "error": error,
                "failure_transition_index": failure_transition_index,
                "validation_ms": validation_ms,
                "baseline_ms": baseline_ms,
                "bounds": bounds,
                "sampled": sampled,
            }
        )

    if len(records) < cases:
        raise RuntimeError(
            f"Generation cap reached for {corridor_count} corridors: "
            f"collected {len(records)}/{cases} valid sequences from "
            f"{generated} generated sequences."
        )

    baseline_timings = [
        record["baseline_ms"]
        for record in records
        if record["baseline_ms"] is not None
    ]
    successful_baseline_timings = [
        record["baseline_ms"]
        for record in records
        if record["status"] == "success"
    ]
    statuses = Counter(record["status"] for record in records)
    valid_transition_types = Counter(
        transition_type
        for record in records
        for transition_type in record["sampled"]["transition_types"]
    )
    successful_transition_types = Counter(
        transition_type
        for record in records
        if record["status"] == "success"
        for transition_type in record["sampled"]["transition_types"]
    )
    return {
        "corridors": corridor_count,
        "valid_cases": len(records),
        "generated_sequences": generated,
        "candidate_corridor_attempts": candidate_attempts,
        "sequence_restarts": sequence_restarts,
        "invalid_sequences_skipped": invalid_sequences,
        "validation_errors_skipped": dict(validation_errors),
        "statuses": dict(statuses),
        "transition_types_valid": dict(valid_transition_types),
        "transition_types_successful": dict(successful_transition_types),
        "validation_timing_ms": {
            "median": median(validation_timings) if validation_timings else None,
            "mean": float(np.mean(validation_timings)) if validation_timings else None,
            "p95": (
                float(np.percentile(validation_timings, 95))
                if validation_timings
                else None
            ),
            "maximum": max(validation_timings) if validation_timings else None,
        },
        "baseline_timing_ms": {
            "attempted_cases": len(baseline_timings),
            "median": median(baseline_timings) if baseline_timings else None,
            "mean": float(np.mean(baseline_timings)) if baseline_timings else None,
            "p95": (
                float(np.percentile(baseline_timings, 95))
                if baseline_timings
                else None
            ),
            "maximum": max(baseline_timings) if baseline_timings else None,
        },
        "successful_baseline_timing_ms": {
            "successful_cases": len(successful_baseline_timings),
            "median": (
                median(successful_baseline_timings)
                if successful_baseline_timings
                else None
            ),
            "mean": (
                float(np.mean(successful_baseline_timings))
                if successful_baseline_timings
                else None
            ),
            "p95": (
                float(np.percentile(successful_baseline_timings, 95))
                if successful_baseline_timings
                else None
            ),
            "minimum": (
                min(successful_baseline_timings)
                if successful_baseline_timings
                else None
            ),
            "maximum": (
                max(successful_baseline_timings)
                if successful_baseline_timings
                else None
            ),
        },
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corridors",
        type=int,
        nargs="+",
        default=CORRIDOR_COUNTS,
        help="Corridor counts to test (default: 5 10 15).",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=CASES_PER_COUNT,
        help="Valid sequences tested per corridor count (default: 100).",
    )
    parser.add_argument(
        "--max-generated",
        type=int,
        default=100000,
        help="Safety cap on generated sequences per corridor count.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--figure-output", type=Path, default=FIGURE_OUTPUT)
    parser.add_argument(
        "--representatives",
        type=int,
        default=5,
        help="Representative successful plots per corridor count (default: 5).",
    )
    args = parser.parse_args()

    if (
        args.cases < 1
        or args.max_generated < args.cases
        or args.representatives < 1
        or not args.corridors
        or min(args.corridors) < 2
    ):
        parser.error(
            "Use corridor counts >= 2, a positive case count, and a generation "
            "cap at least as large as the case count."
        )

    bicycle = make_fixed_bicycle()
    generator.WIDTH_RANGE = WIDTH_RANGE
    generator.LENGTH_RANGE = WALK_LENGTH_RANGE
    generator.OVERHANG_RANGE = OVERHANG_RANGE
    generator.ALIGNED_OFFSET_PROBABILITY = ALIGNED_OFFSET_PROBABILITY
    generator.ALIGNED_OFFSET_MARGIN = ALIGNED_OFFSET_MARGIN
    generator.MIN_ALIGNED_OFFSET = MIN_ALIGNED_OFFSET
    rng = np.random.default_rng(args.seed)
    report = {
        "seed": args.seed,
        "corridor_counts": args.corridors,
        "valid_cases_per_count": args.cases,
        "vehicle": {
            "footprint_radius": FOOTPRINT_RADIUS,
            "width": bicycle.width,
            "arc_radius": bicycle.max_radius,
        },
        "generator": {
            "width_range": generator.WIDTH_RANGE,
            "length_range": generator.LENGTH_RANGE,
            "overhang_range": generator.OVERHANG_RANGE,
            "straight_probability": generator.STRAIGHT_PROBABILITY,
            "aligned_offset_probability_given_straight": (
                generator.ALIGNED_OFFSET_PROBABILITY
            ),
            "aligned_offset_margin": generator.ALIGNED_OFFSET_MARGIN,
            "minimum_aligned_offset": generator.MIN_ALIGNED_OFFSET,
            "extend_perpendicular_corridors": (
                generator.EXTEND_PERPENDICULAR_CORRIDORS
            ),
        },
        "groups": [],
    }

    for corridor_count in args.corridors:
        group = run_group(
            corridor_count,
            args.cases,
            args.max_generated,
            rng,
            bicycle,
        )
        report["groups"].append(group)
        statuses = group["statuses"]
        timing = group["successful_baseline_timing_ms"]
        timing_text = (
            f"successful baseline median={timing['median']:.3f} ms, "
            f"p95={timing['p95']:.3f} ms"
            if timing["successful_cases"]
            else "no successful baselines"
        )
        print(
            f"{corridor_count} corridors: generated "
            f"{group['generated_sequences']} to collect "
            f"{group['valid_cases']} valid cases; {statuses}; {timing_text}",
            flush=True,
        )
        print(
            "  valid transition coverage: "
            f"{group['transition_types_valid']}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["representative_generated_indices"] = plot_representative_examples(
        report,
        bicycle,
        args.figure_output,
        columns=args.representatives,
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved replayable report to {args.output}")
    print(f"Saved representative figure to {args.figure_output}")


if __name__ == "__main__":
    main()
