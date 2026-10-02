"""Compare fresh baseline-to-refinement runs with the preserved working version.

Run directly in VS Code; imports, map setup and plotting are excluded. Each
timed run constructs a NEW baseline/context, so these are cold per-plan timings,
not repeated searches benefiting from an already-populated geometry cache.
All public geometric results are compared before benchmarking. Fine-grained
graph profiling is a separate pass to keep its clocks out of speed comparisons.
"""

import argparse
import importlib
import json
import sys
import tempfile
import zipfile
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from types import ModuleType

import numpy as np
from examples_maps_polyline import EXAMPLE_NUMBERS, example_corridor_sequence

from kappa_planner import baseline_construction, refinement

REPETITIONS = 20
HERE = Path(__file__).resolve().parent
REFERENCE = HERE / 'results/refinement_performance/reference_20261002.zip'
OUTPUT = HERE / 'results/refinement_performance/optimized_comparison.json'


def load_reference(archive, directory):
    with zipfile.ZipFile(archive) as source:
        source.extractall(directory)
    package = '_kappa_refinement_reference'
    module = ModuleType(package)
    module.__path__ = [str(Path(directory) / 'src/kappa_planner')]
    sys.modules[package] = module
    return (importlib.import_module(package + '.baseline_construction'),
            importlib.import_module(package + '.refinement'))


def pipeline(modules, case, *, profile=False):
    construction, refine = modules
    corridors, initial, final, robot = case
    times, stages = {}, {}
    started = perf_counter_ns()
    def measure(name, function, *args, **kwargs):
        before = perf_counter_ns()
        result = function(*args, **kwargs)
        times[name] = (perf_counter_ns()-before)/1e6
        stages[name] = result
        return result
    baseline = measure('baseline', construction.compute_filleted_baseline,
                       corridors, robot, use_joint_solver=False,
                       initial_position=initial[:2], final_position=final[:2])
    if baseline.feasible:
        circles = measure('placement', refine.place_refinement_circles, baseline, robot)
        circles = measure('opposite_repair', refine.restore_opposite_turn_overlaps, circles, baseline)
        circles = measure('matching_repair', refine.repair_same_turn_overlaps, circles, baseline, robot)
        arcs = measure('green_arcs', refine.compute_circle_safe_arcs, circles, baseline, robot)
        graph_times = {} if profile else None
        extra = {'timings': graph_times} if profile else {}
        measure('tangent_graph', refine.connect_safe_arc_circles, circles, baseline, robot,
                safe_arcs=arcs, **extra)
        if profile:
            times.update(graph_times)
    times['total'] = (perf_counter_ns()-started)/1e6
    return stages, times


def public_snapshot(value):
    if is_dataclass(value):
        return {f.name: public_snapshot(getattr(value, f.name)) for f in fields(value)
                if not f.name.startswith('_') and f.name != 'elapsed_ms'}
    if isinstance(value, Mapping):
        return {k: public_snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [public_snapshot(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def assert_equivalent(first, second, path='result'):
    if isinstance(first, dict):
        if first.keys() != second.keys():
            raise AssertionError(f'{path}: different fields')
        for key in first:
            assert_equivalent(first[key], second[key], f'{path}.{key}')
    elif isinstance(first, list):
        if len(first) != len(second):
            raise AssertionError(f'{path}: different lengths')
        for i, (a, b) in enumerate(zip(first, second)):
            assert_equivalent(a, b, f'{path}[{i}]')
    elif isinstance(first, float):
        if not np.isclose(first, second, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise AssertionError(f'{path}: {first} != {second}')
    elif first != second:
        raise AssertionError(f'{path}: {first!r} != {second!r}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=int, nargs='+', default=list(EXAMPLE_NUMBERS))
    parser.add_argument('--repetitions', type=int, default=REPETITIONS)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.repetitions < 1 or not args.reference.is_file():
        parser.error('Require positive repetitions and the preserved reference archive.')
    current = baseline_construction, refinement
    cases = {number: example_corridor_sequence(number) for number in args.examples}
    records = []
    with tempfile.TemporaryDirectory(prefix='kappa-reference-') as directory:
        reference = load_reference(args.reference, directory)
        for number, case in cases.items():
            before, _ = pipeline(reference, case)
            after, _ = pipeline(current, case)
            assert_equivalent(public_snapshot(before), public_snapshot(after), f'map {number}')
            # Alternate order to limit warmup/thermal bias. Every call is fresh.
            samples = [[], []]
            for repeat in range(args.repetitions):
                for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    _, measured = pipeline((reference, current)[index], case)
                    samples[index].append(measured)
            def medians(measured):
                return {key: median(sample[key] for sample in measured) for key in measured[0]}
            old, new = map(medians, samples)
            details = medians([pipeline(current, case, profile=True)[1]
                               for _ in range(min(7, args.repetitions))])
            connection = after.get('tangent_graph')
            record = dict(example=number, baseline_status=after['baseline'].status,
                          refinement_status=None if connection is None else connection.status,
                          length=None if connection is None else connection.length,
                          selected_circles=[] if connection is None else connection.selected_circles,
                          identical_results=True, reference_ms=old, optimized_ms=new,
                          profiled_ms=details, speedup=old['total']/new['total'])
            records.append(record)
            print(f'Map {number:2}: {old["total"]:7.3f} -> {new["total"]:7.3f} ms; '
                  f'{record["speedup"]:.2f}x; identical', flush=True)
    eligible = [r for r in records if r['baseline_status'] == 'feasible']
    old_total = sum(r['reference_ms']['total'] for r in eligible)
    new_total = sum(r['optimized_ms']['total'] for r in eligible)
    summary = dict(maps=len(records), eligible_maps=len(eligible), repetitions=args.repetitions,
                   all_results_identical=True, reference_mean_ms=old_total/max(1, len(eligible)),
                   optimized_mean_ms=new_total/max(1, len(eligible)),
                   aggregate_speedup=old_total/new_total if new_total else None,
                   fresh_context_each_run=True, boundary_directions=True, joint_solver=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, maps=records), indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
