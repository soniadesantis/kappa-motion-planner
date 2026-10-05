"""Time exact-baseline validation separately on the saved five/eight-corridor cases.

Run directly in VS Code. Total timings include the normal final validator.
Validation/build and build-only timings use already reconstructed successful
chains. Build-only is a benchmark prototype, never an accepted planner result.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np

from example_random_baseline_construction import corridors_from_bounds
from kappa_planner.baseline_construction import (
    QuarterCircleFillet, _validate_and_build_fillets, compute_filleted_baseline_exact,
)


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / 'results' / 'random_baseline' / 'exact_propagation'


def build_only(points, regions, R):
    """Materialize identical arc data, deliberately without acceptance checks."""
    fillets = [None] * len(points)
    trims = np.zeros(len(points))
    for j, region in enumerate(regions):
        if region is None:
            continue
        incoming, outgoing = region['incoming'], region['outgoing']
        before, after = points[j]-R*incoming, points[j]+R*outgoing
        turn = incoming[0]*outgoing[1]-incoming[1]*outgoing[0]
        fillets[j] = QuarterCircleFillet(j, points[j]-R*incoming+R*outgoing,
                                       before, after, R, float(turn*np.pi/2))
        trims[j] = R
    remaining = np.linalg.norm(np.diff(points, axis=0), axis=1)-trims[:-1]-trims[1:]
    return tuple(fillets), remaining


def measure(function, batch):
    start = perf_counter_ns()
    for _ in range(batch):
        function()
    return (perf_counter_ns()-start)/batch/1e6


def summary(values):
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(np.max(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=7)
    parser.add_argument('--micro-batch', type=int, default=10)
    args = parser.parse_args()
    if args.repetitions < 1 or args.micro_batch < 1:
        parser.error('Repetitions and micro-batch must be positive.')
    report = dict(repetitions=args.repetitions, micro_batch=args.micro_batch,
                  note='Per-case warmed medians; original successful exact baselines only. '
                       'Total includes validation; microtimings reuse reconstructed chains. '
                       'No generation, plotting, refinement or boundary arcs.', groups=[])
    for n in (5, 8):
        source = OUTPUT_DIRECTORY / f'n{n}' / 'random_baseline_seed7_exact.json'
        experiment = json.loads(source.read_text())
        config = experiment['configuration']
        robot = SimpleNamespace(r=config['r'], R=config['R'])
        cases = []
        for case in experiment['cases']:
            if not case['fillet_success']:
                continue
            corridors = corridors_from_bounds(case['bounds'])
            result = compute_filleted_baseline_exact(corridors, robot)
            if not result.feasible:
                raise RuntimeError(f'Replay failed: n={n}, case {case["case"]}.')
            full = lambda: compute_filleted_baseline_exact(corridors, robot)
            validation = lambda: _validate_and_build_fillets(
                result.polyline, result.feasibility, result.fillet_regions, robot.r, robot.R, 1e-9)
            construction = lambda: build_only(result.polyline, result.fillet_regions, robot.R)
            built, remaining = construction()
            np.testing.assert_allclose(remaining, result.remaining_lengths)
            for actual, expected in zip(built, result.fillets):
                if actual is None:
                    assert expected is None
                else:
                    np.testing.assert_allclose(actual.center, expected.center)
                    np.testing.assert_allclose(actual.incoming_tangent, expected.incoming_tangent)
                    np.testing.assert_allclose(actual.outgoing_tangent, expected.outgoing_tangent)
            validation()
            samples = {'total_ms': [], 'validation_and_build_ms': [], 'build_only_ms': []}
            jobs = [('total_ms', full, 1), ('validation_and_build_ms', validation, args.micro_batch),
                    ('build_only_ms', construction, args.micro_batch)]
            for repetition in range(args.repetitions):
                for key, function, batch in jobs if repetition % 2 == 0 else reversed(jobs):
                    samples[key].append(measure(function, batch))
            record = dict(case=case['case'], fillets=sum(f is not None for f in result.fillets),
                          **{key: float(np.median(values)) for key, values in samples.items()})
            record['validation_share_percent'] = 100*record['validation_and_build_ms']/record['total_ms']
            record['check_overhead_ms'] = record['validation_and_build_ms']-record['build_only_ms']
            cases.append(record)
        group = dict(corridors=n, successful_cases=len(cases), cases=cases,
                     statistics={key: summary([case[key] for case in cases]) for key in
                                 ('total_ms', 'validation_and_build_ms', 'build_only_ms',
                                  'check_overhead_ms', 'validation_share_percent')})
        report['groups'].append(group)
        print(f'{n} corridors, {len(cases)} successes:')
        for key, stats in group['statistics'].items():
            print(f'  {key}: median={stats["median"]:.4f}, mean={stats["mean"]:.4f}, '
                  f'p95={stats["p95"]:.4f}')
    output = OUTPUT_DIRECTORY / 'validation_timing.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
