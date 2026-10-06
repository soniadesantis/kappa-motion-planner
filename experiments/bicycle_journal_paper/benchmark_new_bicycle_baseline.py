"""Benchmark the new bicycle baseline on random corridor sequences.

The experiment collects exactly ``--cases`` geometrically valid sequences for
every requested corridor count.  Invalid generated sequences are counted and
skipped. Corridor sizes, random walk geometry, and endpoint poses vary; the
footprint radius and fillet radius are fixed for the whole run. Baseline and
boundary-connection outcomes are recorded independently. Every successful
internal baseline is also refined for the same endpoint poses; complete
baseline/refined solutions are compared on traversal and computation time.
If refinement fails, a complete baseline trajectory is retained and the
unsuccessful attempt is recorded separately from the final solution outcome.

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

from kappa_planner.baseline_construction_new import (
    compute_bicycle_baseline,
    compute_baseline_boundary_connections,
    assemble_baseline_trajectory,
    compute_trajectory_traversal_time,
    compute_overlap_two_axis_aligned_corridors,
    compute_safe_overlap,
    validate_baseline_corridor_sequence,
)
from kappa_planner.corridor import CorridorWorld
from kappa_planner.vehicle import Bicycle
from kappa_planner.refinement_new import refine_bicycle_baseline, is_baseline_circle
from kappa_planner.helpers.poses import (
    absolute_to_relative_pose,
    relative_to_absolute_pose,
)


SEED = 7
CORRIDOR_COUNTS = (5, 10, 15)
CASES_PER_COUNT = 100

# Keep the vehicle fixed in this first experiment.  The bicycle constructor
# derives max_radius = wheelbase / tan(delta_max).
FOOTPRINT_RADIUS = 0.5
ARC_RADIUS = 1.0
WIDTH_RANGE = (1.5, 10.0)
WALK_LENGTH_RANGE = (2.0, 20.0)
INTERIOR_WALK_LENGTH_RANGE = (2.0, 8.0)
OVERHANG_RANGE = (0.0, 2.0)
MAX_CORRIDOR_ATTEMPTS = 500
MAX_SEQUENCE_RESTARTS = 100
MAX_BACKTRACKS_PER_CORRIDOR = 100
POSE_OVERLAP_CLEARANCE_RADII = 2.0
ENDPOINT_SAMPLING_LENGTH = 1.0

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


CARDINAL_ROTATIONS = np.array([
    [[1, 0], [0, 1]],
    [[0, -1], [1, 0]],
    [[-1, 0], [0, -1]],
    [[0, 1], [-1, 0]],
])


def construct_connected_corridor(
    previous_bounds, previous_heading, heading_change, width, length,
    overhang, previous_forward_overhang=0.0,
):
    """Construct east-to-east/north/south geometry, then rotate to the world.

    At a turn, the previous complete forward edge lies on the outgoing
    rectangle's far longitudinal side edge in the incoming direction.
    Its full width extends back into the incoming corridor to maximize
    overlap. It continues beyond the contained edge in the new direction.
    Straight corridors share the same longitudinal centerline; their
    overlap consists only of the adjacent longitudinal overhangs.
    """
    if heading_change not in (-1, 0, 1):
        raise ValueError("Only straight and 90-degree turns are supported")
    xmin, xmax, ymin, ymax = previous_bounds
    rotation = CARDINAL_ROTATIONS[previous_heading % 4]
    direction = rotation[:, 0]
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
    previous_length = (xmax - xmin) if previous_heading % 2 == 0 else (ymax - ymin)
    previous_width = (ymax - ymin) if previous_heading % 2 == 0 else (xmax - xmin)
    forward_midpoint = center + 0.5 * previous_length * direction
    back, front = overhang
    if heading_change == 0:
        low_x = -previous_forward_overhang - back
        high_x = length - previous_forward_overhang + front
        low_y, high_y = -0.5 * width, 0.5 * width
    else:
        if length <= previous_width:
            raise ValueError("Outgoing length must exceed the incoming edge length")
        low_x, high_x = -width, 0.0
        low_y = -0.5 * previous_width - back
        high_y = -0.5 * previous_width + length + front
        if heading_change == -1:
            low_y, high_y = -high_y, -low_y
    corners = np.array([
        [low_x, low_y], [high_x, low_y],
        [high_x, high_y], [low_x, high_y],
    ]) @ rotation.T + forward_midpoint
    low, high = corners.min(axis=0), corners.max(axis=0)
    return tuple(float(value) for value in (low[0], high[0], low[1], high[1]))


def rectangles_overlap(first, second):
    """Positive-area overlap of axis-aligned rectangles; boundary touching is allowed."""
    return (
        first[0] < second[1] and second[0] < first[1]
        and first[2] < second[3] and second[2] < first[3]
    )


def sample_validated_corridors(rng, corridor_count, bicycle):
    """Construct structured connections; retain validation as a defensive check."""
    candidate_attempts = 0
    validation_rejections = 0
    nonconsecutive_overlap_rejections = 0
    total_backtracks = 0
    for restart in range(MAX_SEQUENCE_RESTARTS + 1):
        bounds, widths, lengths, overhangs, headings = [], [], [], [], []
        transition_types = []
        heading = int(rng.integers(4))
        initial_heading = heading
        j = 0
        backtracks = 0
        while j < corridor_count:
            # Keep the selected transition fixed when resampling dimensions.
            if j == 0:
                heading_change = 0
            elif j in (1, corridor_count - 1):
                # Both boundary attachments need a genuine turning circle.
                heading_change = int(rng.choice([1, -1]))
            else:
                heading_change = int(rng.choice([0, 1, -1], p=[0.15, 0.425, 0.425]))
            candidate_heading = (heading + heading_change) % 4
            length_range = (
                WALK_LENGTH_RANGE if j in (0, corridor_count - 1)
                else INTERIOR_WALK_LENGTH_RANGE
            )
            # No dimension draw can fit this incoming edge in the outgoing
            # length range. Backtrack immediately rather than retry 500 times.
            impossible_turn = (
                j and heading_change
                and widths[-1] + bicycle.width >= length_range[1]
            )
            for _ in range(0 if impossible_turn else MAX_CORRIDOR_ATTEMPTS):
                candidate_attempts += 1
                minimum_width = WIDTH_RANGE[0]
                if j in (0, corridor_count - 1):
                    # Reserve a turning diameter plus the circular footprint
                    # and additional room across the endpoint corridor.
                    minimum_width = max(
                        minimum_width,
                        2 * bicycle.max_radius + bicycle.width + ENDPOINT_SAMPLING_LENGTH,
                    )
                if minimum_width >= WIDTH_RANGE[1]:
                    raise ValueError("Endpoint maneuvering width exceeds WIDTH_RANGE")
                width = float(rng.uniform(minimum_width, WIDTH_RANGE[1]))
                overhang = rng.uniform(*OVERHANG_RANGE, 2)
                minimum_length = length_range[0]
                if j and heading_change:
                    # Contain the complete incoming edge and leave room beyond
                    # it for the footprint and the existing endpoint sampler.
                    minimum_length = max(
                        minimum_length, widths[-1] + bicycle.width + overhang[0]
                    )
                elif j:
                    # Longitudinal extensions alone must create a safe overlap.
                    minimum_back = max(OVERHANG_RANGE[0], bicycle.width + 1e-6 - overhangs[-1][1])
                    if minimum_back >= OVERHANG_RANGE[1]:
                        continue
                    overhang[0] = rng.uniform(minimum_back, OVERHANG_RANGE[1])
                    minimum_length = max(minimum_length, sum(overhangs[-1]) + bicycle.width)
                endpoint_clearance = POSE_OVERLAP_CLEARANCE_RADII * bicycle.max_radius
                if j == 0:
                    # Reserve space even if the next turning corridor has the
                    # largest permitted width and occupies this forward end.
                    minimum_length = max(
                        minimum_length,
                        WIDTH_RANGE[1] + bicycle.width + endpoint_clearance
                        + ENDPOINT_SAMPLING_LENGTH,
                    )
                elif j == corridor_count - 1:
                    minimum_length += endpoint_clearance + ENDPOINT_SAMPLING_LENGTH
                if minimum_length >= length_range[1]:
                    continue
                length = float(rng.uniform(minimum_length, length_range[1]))
                if j:
                    candidate_bound = construct_connected_corridor(
                        bounds[-1], heading, heading_change, width, length,
                        overhang, previous_forward_overhang=overhangs[-1][1],
                    )
                else:
                    corners = np.array([
                        [-overhang[0], -width / 2], [length + overhang[1], -width / 2],
                        [length + overhang[1], width / 2], [-overhang[0], width / 2],
                    ]) @ CARDINAL_ROTATIONS[heading].T
                    low, high = corners.min(axis=0), corners.max(axis=0)
                    candidate_bound = tuple(float(v) for v in (low[0], high[0], low[1], high[1]))
                # Only the immediately preceding rectangle may overlap. Test
                # raw bounds before constructing corridor objects/validating.
                if any(rectangles_overlap(candidate_bound, earlier) for earlier in bounds[:-1]):
                    nonconsecutive_overlap_rejections += 1
                    continue
                candidate_bounds = bounds + [candidate_bound]
                candidate_headings = headings + [candidate_heading]
                if j and not validate_baseline_corridor_sequence(
                    corridor_worlds_from_bounds(candidate_bounds, candidate_headings), bicycle
                ):
                    validation_rejections += 1
                    continue
                bounds, headings = candidate_bounds, candidate_headings
                widths.append(width)
                lengths.append(length)
                overhangs.append(overhang.tolist())
                if j:
                    transition_types.append("aligned_collinear" if heading_change == 0 else "turn")
                heading = candidate_heading
                break
            else:
                if j == 0 or backtracks >= MAX_BACKTRACKS_PER_CORRIDOR * corridor_count:
                    break
                # A blocked turn/overlap often depends on the previous size or
                # direction. Replace a short suffix, retaining a valid prefix.
                j -= min(j, int(rng.integers(1, 4)))
                del bounds[j:]
                del widths[j:]
                del lengths[j:]
                del overhangs[j:]
                del headings[j:]
                del transition_types[max(0, j - 1):]
                heading = headings[-1] if headings else initial_heading
                backtracks += 1
                total_backtracks += 1
                continue
            j += 1
        else:
            spans = [(b - a, d - c) for a, b, c, d in bounds]
            return bounds, {
                "widths": widths, "walk_lengths": lengths, "overhangs": overhangs,
                "headings": headings, "transition_types": transition_types,
                "transverse_offsets": [0.0] * (corridor_count - 1),
                "original_bounds": list(bounds), "edge_extensions": [],
                "final_widths": [span[1 - h % 2] for span, h in zip(spans, headings)],
                "final_lengths": [span[h % 2] for span, h in zip(spans, headings)],
                "candidate_attempts": candidate_attempts,
                "validation_rejections": validation_rejections,
                "nonconsecutive_overlap_rejections": nonconsecutive_overlap_rejections,
                "sequence_restarts": restart,
                "sequence_backtracks": total_backtracks,
            }
    raise RuntimeError(f"Could not construct a validated {corridor_count}-corridor sequence.")


def sample_endpoint_pose(rng, corridor, neighbor, bicycle, *, initial):
    """Sample a disk-safe local pose on the outer side of the adjacent overlap.

    Local x spans corridor width; local y spans its length. The heading is
    sampled uniformly in [0, pi] in this same frame, so headings face the
    forward half-plane. Require two turning radii of clearance beyond the
    footprint radius. An insufficiently long corridor is rejected rather
    than relaxing this requirement.
    """
    overlap = compute_overlap_two_axis_aligned_corridors(corridor, neighbor)
    if overlap is None:
        return None
    (xmin, xmax), (ymin, ymax) = overlap
    overlap_y = [
        absolute_to_relative_pose(corridor, [x, y, 0.0])[1]
        for x in (xmin, xmax)
        for y in (ymin, ymax)
    ]
    radius = 0.5 * bicycle.width
    half_width = 0.5 * corridor.width - radius
    low = -0.5 * corridor.height + radius
    high = 0.5 * corridor.height - radius
    if half_width <= 0.0 or high <= low:
        return None
    if initial:
        high = min(high, min(overlap_y) - radius)
    else:
        low = max(low, max(overlap_y) + radius)
    available = high - low
    clearance = POSE_OVERLAP_CLEARANCE_RADII * bicycle.max_radius
    if available <= clearance + 1e-9:
        return None
    if initial:
        high -= clearance
    else:
        low += clearance
    relative_pose = [
        float(rng.uniform(-half_width, half_width)),
        float(rng.uniform(low, high)),
        float(rng.uniform(0.0, np.pi)),
    ]
    overlap_distance = (
        min(overlap_y) - relative_pose[1]
        if initial else relative_pose[1] - max(overlap_y)
    )
    return {
        "relative": relative_pose,
        "world": relative_to_absolute_pose(corridor, relative_pose),
        "minimum_extra_clearance": clearance,
        "overlap_clearance": float(overlap_distance - radius),
    }


def attach_boundary_connections(corridors, baseline, bicycle, initial_pose, final_pose):
    """Record attachment failures without changing a successful baseline outcome."""
    outcome = {
        "status": "failed",
        "initial_status": "not_attempted",
        "final_status": "not_attempted",
        "failure_reason": None,
        "error": None,
        "connection_ms": None,
        "assembly_ms": None,
        "trajectory_primitives": None,
        "traversal_time": None,
    }
    if baseline.fillets[0] is None or baseline.fillets[-1] is None:
        for side, fillet in (("initial", baseline.fillets[0]), ("final", baseline.fillets[-1])):
            if fillet is None:
                outcome[f"{side}_status"] = "unsupported_straight_boundary"
        outcome["failure_reason"] = "unsupported_straight_boundary"
        return outcome

    phase = "connection_ms"
    started = perf_counter_ns()
    try:
        initial, final = compute_baseline_boundary_connections(
            corridors, baseline, bicycle, initial_pose, final_pose
        )
        outcome[phase] = (perf_counter_ns() - started) / 1e6
        outcome["initial_status"] = "success" if initial is not None else "failed"
        outcome["final_status"] = (
            "success" if final is not None
            else "not_attempted" if initial is None else "failed"
        )
        # Retain either successful attachment for plotting, even if the other
        # endpoint fails and a complete trajectory cannot be assembled.
        baseline.initial_maneuvers = initial
        baseline.final_maneuvers = final
        if initial is None or final is None:
            outcome["failure_reason"] = (
                "initial_boundary_connection_failed" if initial is None
                else "final_boundary_connection_failed"
            )
            return outcome
        phase = "assembly_ms"
        started = perf_counter_ns()
        trajectory = assemble_baseline_trajectory(corridors, baseline, bicycle)
        outcome[phase] = (perf_counter_ns() - started) / 1e6
        if trajectory is None:
            outcome["failure_reason"] = "baseline_trajectory_assembly_failed"
            return outcome
        baseline.trajectory = trajectory
        baseline.traversal_time = compute_trajectory_traversal_time(trajectory)
        outcome.update(
            status="success",
            trajectory_primitives=len(trajectory),
            traversal_time=baseline.traversal_time,
        )
    except Exception as exception:
        outcome[phase] = (perf_counter_ns() - started) / 1e6
        outcome.update(
            status="error",
            failure_reason=f"{phase[:-3]}_error",
            error=f"{type(exception).__name__}: {exception}",
        )
    return outcome


def evaluate_refinement(corridors, baseline, bicycle, initial_pose, final_pose):
    """Try refinement, retaining the complete baseline if the attempt fails."""
    outcome = {
        "status": "failed", "error": None, "failure_reason": None,
        "computation_ms": None, "traversal_time": None,
        "trajectory_primitives": None, "active_circles": None,
        "baseline_position_circles": None,
        "all_active_circles_at_baseline": None,
        "solution_source": None, "attempt_status": "failed",
    }
    started = perf_counter_ns()
    try:
        refinement_return = refine_bicycle_baseline(
            corridor_list=corridors, bicycle=bicycle, baseline=baseline,
            initial_pose=initial_pose, final_pose=final_pose,
            return_failure=True,
        )
        if isinstance(refinement_return, tuple) and len(refinement_return) == 2:
            result, failure = refinement_return
        else:
            # Preserve compatibility with patched/legacy implementations used
            # by downstream benchmark tests.
            result, failure = refinement_return, None
        outcome["computation_ms"] = (perf_counter_ns() - started) / 1e6
        if result is None or result.trajectory is None:
            outcome["failure_reason"] = (
                failure.reason if failure is not None else "complete_refinement_failed"
            )
        else:
            active_circles = [
                next(iter(result.circle_groups[index].values()))
                for index in result.active_circle_indices
            ]
            baseline_position_circles = sum(
                is_baseline_circle(circle) for circle in active_circles
            )
            outcome.update(
                status="success", attempt_status="success", solution_source="refined",
                traversal_time=compute_trajectory_traversal_time(result.trajectory),
                trajectory_primitives=len(result.trajectory),
                active_circles=len(result.active_circle_indices),
                baseline_position_circles=baseline_position_circles,
                all_active_circles_at_baseline=(
                    bool(active_circles)
                    and baseline_position_circles == len(active_circles)
                ),
            )
    except Exception as exception:
        outcome["computation_ms"] = (perf_counter_ns() - started) / 1e6
        outcome.update(
            status="error", attempt_status="error", failure_reason="refinement_error",
            error=f"{type(exception).__name__}: {exception}",
        )
    if outcome["status"] != "success" and getattr(baseline, "trajectory", None):
        outcome.update(
            status="success", solution_source="baseline_fallback",
            traversal_time=compute_trajectory_traversal_time(baseline.trajectory),
            trajectory_primitives=len(baseline.trajectory),
        )
    return outcome


def compare_solutions(baseline_ms, boundary_total_ms, boundary, refinement):
    """Compare complete solutions for one problem, counting shared baseline work once.

    Baseline computation = internal construction + baseline attachment/assembly.
    Refined computation = internal construction + refinement (including its own
    attachments/assembly). A fallback also counts baseline attachment/assembly.
    """
    baseline_complete_ms = baseline_ms + boundary_total_ms
    refined_complete_ms = baseline_ms + refinement["computation_ms"]
    if refinement.get("solution_source") == "baseline_fallback":
        refined_complete_ms += boundary_total_ms
    paired = boundary["status"] == "success" and refinement["status"] == "success"
    comparison = {
        "paired_success": paired,
        "baseline_computation_ms": baseline_complete_ms,
        "refined_computation_ms": refined_complete_ms,
        "traversal_time_saved": None,
        "traversal_time_reduction_percent": None,
        "refinement_computation_overhead_ms": refined_complete_ms - baseline_complete_ms,
    }
    if paired:
        saved = boundary["traversal_time"] - refinement["traversal_time"]
        comparison["traversal_time_saved"] = saved
        if boundary["traversal_time"] > 0:
            comparison["traversal_time_reduction_percent"] = (
                100 * saved / boundary["traversal_time"]
            )
    return comparison


def timing_summary(values):
    values = [value for value in values if value is not None]
    return {
        "cases": len(values),
        "median": median(values) if values else None,
        "mean": float(np.mean(values)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
    }


def summarize_comparison(records):
    """Use matched complete solutions for all baseline-versus-refinement summaries."""
    paired = [record for record in records if record.get("comparison", {}).get("paired_success")]
    tolerance = 1e-7
    return {
        "paired_cases": len(paired),
        "baseline_fallback_cases": sum(
            r["refinement"].get("solution_source") == "baseline_fallback" for r in paired
        ),
        "refined_shorter": sum(r["comparison"]["traversal_time_saved"] > tolerance for r in paired),
        "equal_traversal_time": sum(abs(r["comparison"]["traversal_time_saved"]) <= tolerance for r in paired),
        "refined_longer": sum(r["comparison"]["traversal_time_saved"] < -tolerance for r in paired),
        "baseline_traversal_time": timing_summary(r["boundary"]["traversal_time"] for r in paired),
        "refined_traversal_time": timing_summary(r["refinement"]["traversal_time"] for r in paired),
        "traversal_time_reduction_percent": timing_summary(r["comparison"]["traversal_time_reduction_percent"] for r in paired),
        "baseline_computation_ms": timing_summary(r["comparison"]["baseline_computation_ms"] for r in paired),
        "refinement_incremental_ms": timing_summary(r["refinement"]["computation_ms"] for r in paired),
        "refined_computation_ms": timing_summary(r["comparison"]["refined_computation_ms"] for r in paired),
    }


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
    """Select distinct timing-spanning cases covering straight and turning transitions."""
    successful = sorted(
        (record for record in records if record["status"] == "success"),
        key=lambda record: record["baseline_ms"],
    )
    if len(successful) <= count:
        return successful
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
    for transition_type in ("aligned_collinear", "turn"):
        candidates = [
            record
            for record in successful
            if transition_type in record["sampled"]["transition_types"]
        ]
        if candidates:
            candidate = nearest_to_median(candidates)
            if candidate not in selected:
                selected.append(candidate)

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


def plot_maneuver_path(ax, maneuvers, color, linewidth=2.0):
    """Plot primitive coordinates, as in EXAMPLE_THESIS's maneuver plotter."""
    for maneuver in maneuvers or []:
        coordinates = np.asarray(maneuver.path_coordinates, dtype=float)
        if coordinates.ndim == 2 and coordinates.shape[0] and coordinates.shape[1] >= 2:
            ax.plot(
                coordinates[:, 0], coordinates[:, 1],
                color=color, linewidth=linewidth, zorder=6,
            )


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

    if baseline.trajectory is None:
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

    else:
        # Use the assembled geometry, including the modified first/last arcs.
        initial_count = len(baseline.initial_maneuvers)
        final_count = len(baseline.final_maneuvers)
        middle = baseline.trajectory[initial_count:-final_count]
        for maneuver in middle:
            color = "#172033" if maneuver.label == "segment" else "#ea580c"
            plot_maneuver_path(ax, [maneuver], color)

    plot_maneuver_path(ax, baseline.initial_maneuvers, "#16a34a", linewidth=2.2)
    plot_maneuver_path(ax, baseline.final_maneuvers, "#dc2626", linewidth=2.2)

    for name, color in (("initial_pose", "#16a34a"), ("final_pose", "#dc2626")):
        if name in record:
            x, y, heading = record[name]["world"]
            ax.plot(x, y, "o", color=color, markersize=4, zorder=7)
            ax.annotate(
                "", xy=(x + np.cos(heading), y + np.sin(heading)), xytext=(x, y),
                arrowprops={"arrowstyle": "->", "color": color}, zorder=7,
            )

    ax.set_title(
        f"generated #{record['generated_index']} · {record['baseline_ms']:.3f} ms\n"
        f"boundary: {record.get('boundary', {}).get('status', 'not recorded')}",
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
            baseline = compute_bicycle_baseline(
                corridors, bicycle,
                initial_pose=record.get("initial_pose", {}).get("world"),
                final_pose=record.get("final_pose", {}).get("world"),
            )
            if baseline is None:
                raise RuntimeError(
                    f"Could not replay generated case {record['generated_index']}."
                )
            if "initial_pose" in record and "final_pose" in record:
                attach_boundary_connections(
                    corridors, baseline, bicycle,
                    record["initial_pose"]["world"], record["final_pose"]["world"],
                )
            plot_baseline_panel(ax, corridors, baseline, record)
            if record.get("refinement_success"):
                if record["refinement"].get("solution_source") == "baseline_fallback":
                    plot_maneuver_path(ax, baseline.trajectory, "#7c3aed", linewidth=1.8)
                    continue
                refined = refine_bicycle_baseline(
                    corridors, bicycle, baseline,
                    initial_pose=record["initial_pose"]["world"],
                    final_pose=record["final_pose"]["world"],
                )
                if refined is None or refined.trajectory is None:
                    raise RuntimeError(f"Could not replay refinement for case {record['case']}.")
                plot_maneuver_path(ax, refined.trajectory, "#7c3aed", linewidth=1.8)
        for ax in axes[row, len(selected):]:
            ax.text(0.5, 0.5, "No additional successful baseline", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(
            f"{group['corridors']} corridors",
            fontsize=11,
            labelpad=12,
        )

    subtitle = (
        "successful case nearest the median computation time"
        if columns == 1
        else "including straight continuations and orthogonal turns"
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
            Line2D([], [], color="#16a34a", linewidth=2, label="Initial connection"),
            Line2D([], [], color="#dc2626", linewidth=2, label="Final connection"),
            Line2D([], [], color="#7c3aed", linewidth=2, label="Refined trajectory / baseline fallback"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return selected_cases


def plot_solution_comparison(report, output):
    """Scatter paired traversal and total computation times against equality."""
    fig, axes = plt.subplots(len(report["groups"]), 2, figsize=(11, 4 * len(report["groups"])), squeeze=False)
    for row, group in enumerate(report["groups"]):
        paired = [r for r in group["records"] if r["comparison"]["paired_success"]]
        for column, (baseline_values, refined_values, unit) in enumerate((
            ([r["boundary"]["traversal_time"] for r in paired],
             [r["refinement"]["traversal_time"] for r in paired], "traversal time [s]"),
            ([r["comparison"]["baseline_computation_ms"] for r in paired],
             [r["comparison"]["refined_computation_ms"] for r in paired], "total computation time [ms]"),
        )):
            ax = axes[row, column]
            if paired:
                ax.scatter(baseline_values, refined_values, s=22, alpha=0.7, color="#7c3aed")
                maximum = 1.05 * max(baseline_values + refined_values)
                ax.plot([0, maximum], [0, maximum], "--", color="#64748b", label="Equal times")
                ax.set_xlim(0, maximum)
                ax.set_ylim(0, maximum)
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No paired complete solutions", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(f"Baseline {unit}")
            ax.set_ylabel(f"Refinement with fallback: {unit}")
            ax.set_title(f"{group['corridors']} corridors · {len(paired)} matched cases")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
    fig.suptitle("Baseline and refinement with fallback on identical scenarios\nPoints below the diagonal favor refinement", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_group(corridor_count, cases, max_generated, rng, bicycle):
    records = []
    validation_timings = []
    invalid_sequences = 0
    validation_errors = Counter()
    generated = 0
    candidate_attempts = 0
    sequence_restarts = 0
    endpoint_sampling_failures = 0

    while len(records) < cases and generated < max_generated:
        generated += 1
        bounds, sampled = sample_validated_corridors(rng, corridor_count, bicycle)
        candidate_attempts += sampled["candidate_attempts"]
        sequence_restarts += sampled["sequence_restarts"]
        corridors = corridor_worlds_from_bounds(bounds, sampled["headings"])
        initial_pose = sample_endpoint_pose(
            rng, corridors[0], corridors[1], bicycle, initial=True
        )
        final_pose = sample_endpoint_pose(
            rng, corridors[-1], corridors[-2], bicycle, initial=False
        )
        if initial_pose is None or final_pose is None:
            endpoint_sampling_failures += 1
            continue

        validation_ms = None
        baseline_ms = None
        baseline_started = None
        failure_transition_index = None
        try:
            validation_started = perf_counter_ns()
            valid_sequence = validate_baseline_corridor_sequence(
                corridor_list=corridors,
                bicycle=bicycle,
                initial_pose=initial_pose["world"],
                final_pose=final_pose["world"],
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
                initial_pose=initial_pose["world"],
                final_pose=final_pose["world"],
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

        boundary_started = perf_counter_ns()
        boundary = (
            attach_boundary_connections(
                corridors, result, bicycle, initial_pose["world"], final_pose["world"]
            )
            if status == "success"
            else {"status": "not_attempted", "failure_reason": "baseline_failed"}
        )
        boundary_total_ms = (
            (perf_counter_ns() - boundary_started) / 1e6 if status == "success" else None
        )
        refinement = (
            evaluate_refinement(
                corridors, result, bicycle, initial_pose["world"], final_pose["world"]
            )
            if status == "success"
            else {"status": "not_attempted", "failure_reason": "baseline_failed"}
        )
        comparison = (
            compare_solutions(baseline_ms, boundary_total_ms, boundary, refinement)
            if status == "success" else {"paired_success": False}
        )
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
                "initial_pose": initial_pose,
                "final_pose": final_pose,
                "baseline_success": status == "success",
                "boundary_success": boundary["status"] == "success",
                "baseline_success_boundary_failure": (
                    status == "success" and boundary["status"] != "success"
                ),
                "boundary": boundary,
                "boundary_total_ms": boundary_total_ms,
                "refinement": refinement,
                "refinement_success": refinement["status"] == "success",
                "comparison": comparison,
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
        "sequence_backtracks": sum(
            record["sampled"]["sequence_backtracks"] for record in records
        ),
        "defensive_validation_rejections": sum(
            record["sampled"]["validation_rejections"] for record in records
        ),
        "nonconsecutive_overlap_rejections": sum(
            record["sampled"]["nonconsecutive_overlap_rejections"] for record in records
        ),
        "invalid_sequences_skipped": invalid_sequences,
        "endpoint_sampling_failures_skipped": endpoint_sampling_failures,
        "validation_errors_skipped": dict(validation_errors),
        "statuses": dict(statuses),
        "boundary_statuses": dict(Counter(record["boundary"]["status"] for record in records)),
        "boundary_failure_reasons": dict(Counter(
            record["boundary"]["failure_reason"]
            for record in records
            if record["baseline_success_boundary_failure"]
        )),
        "baseline_success_boundary_failure_cases": sum(
            record["baseline_success_boundary_failure"] for record in records
        ),
        "refinement_statuses": dict(Counter(r["refinement"]["status"] for r in records)),
        "refinement_attempt_statuses": dict(Counter(
            r["refinement"].get("attempt_status", r["refinement"]["status"])
            for r in records
        )),
        "refinement_solution_sources": dict(Counter(
            r["refinement"].get("solution_source") for r in records
            if r["refinement_success"]
        )),
        "refinement_errors": dict(Counter(
            r["refinement"]["error"] for r in records if r["refinement"].get("error")
        )),
        "refinement_failure_reasons": dict(Counter(
            r["refinement"]["failure_reason"]
            for r in records
            if r["refinement"].get("attempt_status") in {"failed", "error"}
            and r["refinement"].get("failure_reason")
        )),
        "successful_refinements_all_circles_at_baseline": sum(
            r["refinement"].get("all_active_circles_at_baseline") is True
            for r in records
        ),
        "refinement_recovers_boundary_failure_cases": sum(
            r["baseline_success_boundary_failure"] and r["refinement_success"] for r in records
        ),
        "refinement_timing_ms": timing_summary(
            r["refinement"].get("computation_ms") for r in records
        ),
        "comparison": summarize_comparison(records),
        "successful_refinement_comparison": summarize_comparison([
            record for record in records
            if record["refinement"].get("solution_source") == "refined"
        ]),
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
    parser.add_argument("--comparison-figure-output", type=Path, default=None)
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
            "construction": "coincident_forward_and_far_side_edges_and_collinear_continuation",
            "width_range": WIDTH_RANGE,
            "length_range": WALK_LENGTH_RANGE,
            "interior_length_range": INTERIOR_WALK_LENGTH_RANGE,
            "endpoint_length_range": WALK_LENGTH_RANGE,
            "overhang_range": OVERHANG_RANGE,
            "straight_probability": 0.15,
            "left_probability": 0.425,
            "right_probability": 0.425,
            "endpoint_transitions": "mandatory 90-degree turns; left/right equally likely",
            "minimum_endpoint_width": (
                2 * bicycle.max_radius + bicycle.width + ENDPOINT_SAMPLING_LENGTH
            ),
            "dimension_sampling": "conditioned on edge containment and safe longitudinal overlap",
            "sideways_offsets": False,
            "nonconsecutive_intersections_allowed": False,
            "nonconsecutive_overlap_policy": "reject positive-area overlap before validation; boundary touching allowed",
            "blocked_sequence_policy": "replace 1-3 recent corridors; bounded backtracking before full restart",
        },
        "endpoint_sampling": {
            "frame": "corridor-centered; x across width, y along length",
            "heading_range_radians": [0.0, np.pi],
            "minimum_overlap_clearance_beyond_footprint": (
                POSE_OVERLAP_CLEARANCE_RADII * bicycle.max_radius
            ),
            "short_corridor_policy": "reserve endpoint length during construction; skip if insufficient",
        },
        "comparison_definition": {
            "traversal_time_unit": "seconds",
            "computation_time_unit": "milliseconds",
            "baseline_total": "baseline construction + baseline boundary attachment and assembly",
            "refined_total": "baseline construction + refinement; baseline fallback also counts baseline boundary attachment and assembly",
            "fallback_policy": "retain the complete baseline trajectory if refinement returns no complete trajectory or raises an error",
            "paired_statistics": "only scenarios with both complete trajectories",
            "excluded_from_computation": ["generation", "validation", "plotting"],
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
        print(
            f"  boundary outcomes: {group['boundary_statuses']}; "
            f"baseline succeeded but attachment failed: "
            f"{group['baseline_success_boundary_failure_cases']}; "
            f"reasons: {group['boundary_failure_reasons']}",
            flush=True,
        )
        comparison = group["comparison"]
        print(
            f"  refinement outcomes: {group['refinement_statuses']}; "
            f"sources: {group['refinement_solution_sources']}; "
            f"failure reasons: {group['refinement_failure_reasons']}; "
            f"all circles at baseline: "
            f"{group['successful_refinements_all_circles_at_baseline']}; "
            f"recovered boundary failures: {group['refinement_recovers_boundary_failure_cases']}; "
            f"paired cases: {comparison['paired_cases']}", flush=True,
        )
        if comparison["paired_cases"]:
            print(
                f"  paired median traversal: baseline={comparison['baseline_traversal_time']['median']:.3f} s, "
                f"refined={comparison['refined_traversal_time']['median']:.3f} s; "
                f"median per-case reduction={comparison['traversal_time_reduction_percent']['median']:.2f}%", flush=True,
            )
            print(
                f"  paired median computation: baseline={comparison['baseline_computation_ms']['median']:.3f} ms, "
                f"refined total={comparison['refined_computation_ms']['median']:.3f} ms, "
                f"incremental refinement={comparison['refinement_incremental_ms']['median']:.3f} ms", flush=True,
            )
        successful_refinement = group["successful_refinement_comparison"]
        if successful_refinement["paired_cases"]:
            print(
                f"  successful refinements only: {successful_refinement['paired_cases']} cases; "
                f"median per-case traversal reduction="
                f"{successful_refinement['traversal_time_reduction_percent']['median']:.2f}%",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Save the measured outcomes before producing the optional overview.
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["representative_generated_indices"] = plot_representative_examples(
        report,
        bicycle,
        args.figure_output,
        columns=args.representatives,
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    comparison_output = args.comparison_figure_output or args.figure_output.with_name(
        args.figure_output.stem + "_comparison.png"
    )
    plot_solution_comparison(report, comparison_output)
    print(f"Saved replayable report to {args.output}")
    print(f"Saved representative figure to {args.figure_output}")
    print(f"Saved paired comparison figure to {comparison_output}")


if __name__ == "__main__":
    main()
