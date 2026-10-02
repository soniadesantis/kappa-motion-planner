"""Compare the full graph, preserved half-plane rule, and first-chain local repair.

Run directly with VS Code. All methods receive identical repaired circles and
green intervals. Local repair checks green arcs and every straight footprint;
the graph is also run with those same checks and without overlap-order checks.
Timings cover connection only. Every call
starts with the same geometry cache produced by circle repair, so method order
cannot make another method appear faster by precomputing its tangent geometry.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np
from examples_maps_polyline import EXAMPLE_NUMBERS, example_corridor_sequence

from kappa_planner.baseline_construction import compute_filleted_baseline, _robot_radii
from kappa_planner.helpers.sequence_geometry import SequenceGeometry, sequence_geometry
from kappa_planner.refinement import (
    place_refinement_circles, restore_opposite_turn_overlaps, repair_same_turn_overlaps,
    compute_circle_safe_arcs, connect_safe_arc_circles, connect_simple_tangent_chain,
)

REPETITIONS = 20
OUTPUT = Path(__file__).parent / 'results/refinement_performance/simple_chain_checked_local_comparison.json'


def compare_case(number, repetitions):
    corridors, initial, final, robot = example_corridor_sequence(number)
    baseline = compute_filleted_baseline(corridors, robot, use_joint_solver=False,
                                         initial_position=initial[:2], final_position=final[:2])
    if not baseline.feasible:
        return dict(example=number, baseline_status=baseline.status)
    circles = place_refinement_circles(baseline, robot)
    circles = restore_opposite_turn_overlaps(circles, baseline)
    circles = repair_same_turn_overlaps(circles, baseline, robot)
    safe = compute_circle_safe_arcs(circles, baseline, robot)
    r, _ = _robot_radii(robot)
    initial_cache = sequence_geometry(baseline.feasibility, r, 1e-9)
    methods = dict(graph_full=(connect_safe_arc_circles, {}),
                   graph_footprint=(connect_safe_arc_circles, dict(check_overlap_order=False)),
                   legacy_halfplane=(connect_simple_tangent_chain, dict(strategy='legacy_halfplane')),
                   local_repair=(connect_simple_tangent_chain, {}))
    samples, results = {name: [] for name in methods}, {}
    for repeat in range(repetitions+1):
        names = list(methods)
        offset = repeat % len(names)
        names = names[offset:] + names[:offset]
        for name in names:
            geometry = SequenceGeometry(baseline.feasibility.corridor_bounds, r, 1e-9)
            geometry._tangents.update(initial_cache._tangents)
            fresh = replace(baseline, feasibility=replace(baseline.feasibility, _geometry=geometry))
            function, options = methods[name]
            started = perf_counter_ns()
            result = function(circles, fresh, robot, safe_arcs=safe, **options)
            elapsed = (perf_counter_ns()-started)/1e6
            if repeat:
                samples[name].append(elapsed)
            results[name] = result
    def record(name):
        result = results[name]
        return dict(status=result.status, connected=result.feasible, length=result.length,
                    selected_circles=list(result.selected_circles),
                    selected_waypoints=[circles.circles[i].waypoint_index for i in result.selected_circles],
                    attempted_pairs=result.attempted_pairs, states=len(result.tangent_states),
                    median_ms=median(samples[name]),
                    safe_arcs_checked=result.safe_arcs_checked,
                    footprint_checked=result.tangent_containment_checked,
                    footprint_scope=result.footprint_scope,
                    repair_rounds=list(getattr(result, 'repair_rounds', ())),
                    overlap_order_checked=result.ordered_overlap_crossings_checked)
    simple, footprint, full = (results[name] for name in ('local_repair', 'graph_footprint', 'graph_full'))
    same_length = (simple.feasible and footprint.feasible
                   and np.isclose(simple.length, footprint.length, rtol=1e-10, atol=1e-9))
    same_full_length = (simple.feasible and full.feasible
                        and np.isclose(simple.length, full.length, rtol=1e-10, atol=1e-9))
    return dict(example=number, baseline_status=baseline.status,
                methods={name: record(name) for name in methods},
                same_footprint_outcome=simple.feasible == footprint.feasible,
                same_footprint_chain=simple.feasible and footprint.feasible
                                     and simple.selected_circles == footprint.selected_circles,
                same_footprint_length=bool(same_length),
                same_full_chain=simple.feasible and full.feasible and simple.selected_circles == full.selected_circles,
                same_full_length=bool(same_full_length),
                proposed_shortcuts=list(simple.proposed_shortcuts), repair_rounds=list(simple.repair_rounds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=int, nargs='+', default=list(EXAMPLE_NUMBERS))
    parser.add_argument('--repetitions', type=int, default=REPETITIONS)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('repetitions must be positive')
    records = [compare_case(number, args.repetitions) for number in args.examples]
    eligible = [r for r in records if 'methods' in r]
    summary = dict(maps=len(records), eligible_maps=len(eligible), repetitions=args.repetitions,
                   timings='Connection only, with identical repair-stage geometry caches')
    for name in ('graph_full', 'graph_footprint', 'legacy_halfplane', 'local_repair'):
        rows = [r['methods'][name] for r in eligible]
        summary[name] = dict(connected=sum(r['connected'] for r in rows),
                            mean_median_ms=sum(r['median_ms'] for r in rows)/max(1, len(rows)))
    for key in ('same_footprint_outcome', 'same_footprint_chain', 'same_footprint_length', 'same_full_chain', 'same_full_length'):
        summary[key] = sum(bool(r[key]) for r in eligible)
    summary['simple_unresolved_maps'] = [r['example'] for r in eligible
                                         if not r['methods']['local_repair']['connected']]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, maps=records), indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
