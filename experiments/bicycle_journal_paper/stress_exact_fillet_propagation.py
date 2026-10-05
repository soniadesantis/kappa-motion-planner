"""Compare exact and bounded baseline construction on identical random corridors.

Run in VS Code or use --cases/--corridors/--seed. Generation uses the existing
extended random-walk distribution, conditional on exact polyline feasibility.
Timings exclude generation, plotting, refinement and endpoint arc alternatives.
Full construction includes its usual independent geometric validation.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np

import example_random_baseline_construction as generator
from kappa_planner.baseline_construction import (
    analyze_orthogonal_polyline_feasibility, compute_filleted_baseline,
    compute_filleted_baseline_exact,
)
from kappa_planner.helpers.fillet_reachability import (
    RoundedCornerConstraint, _make_set, propagate_fillet_regions,
)


OUTPUT = Path(__file__).parent / 'results/random_baseline/exact_propagation/stress_report.json'


def timed(function, repetitions):
    function()  # Warm-up; order alternates at the caller between methods.
    samples = []
    for _ in range(repetitions):
        start = perf_counter_ns()
        function()
        samples.append((perf_counter_ns()-start)/1e6)
    return float(np.median(samples))


def statistics(values):
    if not values:
        return None
    return dict(median=float(np.median(values)), mean=float(np.mean(values)),
                p95=float(np.percentile(values, 95)), maximum=float(np.max(values)))


def synthetic_constraint_timings(repetitions):
    """Adversarial intersections: many curved constraints survive box pruning."""
    records = []
    for count in (1, 4, 8, 16, 32, 64):
        # Centers fan around the SW diagonal. None of the rounded inequalities
        # holds on the entire tightened box; all survive the cheap pruning.
        angles = np.linspace(np.pi/6, np.pi/3, count)
        constraints = [RoundedCornerConstraint((-float(np.cos(a)), -float(np.sin(a))),
                                                (1, 1), 1.05) for a in angles]
        result = _make_set((-2, 2, -2, 2), constraints, 1e-9)
        if result is None or not result.contains((0., 0.)):
            raise RuntimeError('Lost known synthetic witness.')
        records.append(dict(input_constraints=count, retained_constraints=len(result.constraints),
                            intersection_ms=timed(lambda: _make_set(
                                (-2, 2, -2, 2), constraints, 1e-9), repetitions)))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=100)
    parser.add_argument('--corridors', type=int, nargs='+', default=[5, 8, 12, 20])
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--max-generated', type=int, default=20000)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.cases, args.repetitions, args.max_generated) < 1 or min(args.corridors) < 3:
        parser.error('Positive counts and at least three corridors required.')
    robot = SimpleNamespace(r=generator.ROBOT_RADIUS, R=generator.TURNING_RADIUS)
    report = dict(seed=args.seed, requested_per_group=args.cases,
                  repetitions=args.repetitions, r=robot.r, R=robot.R,
                  widths=generator.WIDTH_RANGE, lengths=generator.LENGTH_RANGE,
                  overhangs=generator.OVERHANG_RANGE,
                  straight_probability=generator.STRAIGHT_PROBABILITY,
                  extend_perpendicular=generator.EXTEND_PERPENDICULAR_CORRIDORS,
                  note='Conditional random-walk sample. Failure certificates concern local A model. '
                       'Bounded unresolved is not a false negative unless exact succeeds.', groups=[])
    for n in args.corridors:
        generator.NUMBER_OF_CORRIDORS = n
        rng = np.random.default_rng(args.seed)
        cases, rejected = [], Counter()
        for generated in range(1, args.max_generated+1):
            bounds, _ = generator.sample_corridors(rng)
            corridors = generator.corridors_from_bounds(bounds)
            analysis = analyze_orthogonal_polyline_feasibility(corridors, robot, compute_viable=False)
            if not analysis.feasible:
                rejected[analysis.status] += 1
                continue
            exact = compute_filleted_baseline_exact(corridors, robot)
            heuristic = compute_filleted_baseline(corridors, robot, use_joint_solver=False)
            if heuristic.feasible and not exact.feasible:
                raise RuntimeError(f'Lost heuristic witness: n={n}, generated={generated}: {exact.reason}')
            if exact.feasible:
                for point, reachable in zip(exact.polyline, exact.fillet_reachability.reachable_sets):
                    if not reachable.contains(point):
                        raise RuntimeError('Reconstructed point outside F_j.')
            jobs = [('exact_ms', lambda: compute_filleted_baseline_exact(corridors, robot)),
                    ('heuristic_ms', lambda: compute_filleted_baseline(corridors, robot, use_joint_solver=False))]
            timings = {}
            for name, job in jobs if generated % 2 else reversed(jobs):
                timings[name] = timed(job, args.repetitions)
            reachability = exact.fillet_reachability
            if reachability is not None:
                timings['propagation_ms'] = timed(lambda: propagate_fillet_regions(
                    exact.feasibility, exact.fillet_regions, robot.R), args.repetitions)
            cases.append(dict(case=len(cases)+1, generated=generated, bounds=bounds,
                              exact_status=exact.status, heuristic_status=heuristic.status,
                              exact_success=exact.feasible, heuristic_success=heuristic.feasible,
                              max_constraints=max((len(f.constraints) for f in reachability.reachable_sets
                                                   if f is not None), default=0) if reachability else 0,
                              **timings))
            if len(cases) == args.cases:
                break
        group = dict(corridors=n, accepted=len(cases), generated=generated,
                     polyline_rejections=dict(rejected),
                     exact_statuses=dict(Counter(c['exact_status'] for c in cases)),
                     heuristic_statuses=dict(Counter(c['heuristic_status'] for c in cases)),
                     heuristic_misses=sum(c['exact_success'] and not c['heuristic_success'] for c in cases),
                     max_constraints=max((c['max_constraints'] for c in cases), default=0),
                     timings_ms={name: statistics([c[name] for c in cases if name in c])
                                 for name in ('exact_ms', 'heuristic_ms', 'propagation_ms')}, cases=cases)
        report['groups'].append(group)
        print(json.dumps({k: v for k, v in group.items() if k != 'cases'}, indent=2), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    report['synthetic_constraint_timings'] = synthetic_constraint_timings(args.repetitions)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Synthetic intersections:', json.dumps(report['synthetic_constraint_timings'], indent=2))
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
