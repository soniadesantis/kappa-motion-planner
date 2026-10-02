"""Lightweight timing of circles -> repairs -> tangent skips -> D_j checks.

Build each bp once outside timing. One warm-up pass, then five timed passes.
No figures, strict quarter-chain construction, endpoint maneuvers or profiling.
"""
import csv
from pathlib import Path
import platform
import runpy
from statistics import median
from time import perf_counter

from kappa_planner.helpers.bp_circle_sequence import build_bp_circle_sequence
from kappa_planner.helpers.bp_tangent_polyline import build_bp_tangent_polyline


def main():
    here = Path(__file__).resolve().parent
    example = runpy.run_path(str(here/'examples_maps_polyline.py'))
    cases, excluded = [], []
    for number in example['EXAMPLE_NUMBERS']:
        corridors, start, end, vehicle = example['example_corridor_sequence'](number)
        geometry = example['build_trajectory_geometry'](
            corridors, vehicle.width/2, vehicle.max_radius, start_pose=start, end_pose=end)
        if geometry['feasible']:
            cases.append((number, corridors, vehicle, geometry))
        else:
            excluded.append(number)

    def measure(case):
        number, corridors, vehicle, geometry = case
        t0 = perf_counter()
        placement = build_bp_circle_sequence(corridors, vehicle, geometry,
            shift_same_turn=True, shift_opposite_turn=True, connect_tangents=False)
        t1 = perf_counter()
        polyline = build_bp_tangent_polyline(placement, geometry, skip_intersections=True)
        t2 = perf_counter()
        return dict(example=number, circles=len(placement['sequence']),
                    skipped=len(polyline['skipped']),
                    placement_ms=1000*placement['placement_time'],
                    same_turn_ms=1000*placement['shifting']['computation_time'],
                    opposite_turn_ms=1000*placement['opposite_shifting']['computation_time'],
                    circles_and_repairs_ms=1000*(t1-t0),
                    tangents_skips_Dj_ms=1000*(t2-t1), total_ms=1000*(t2-t0))

    for case in cases:
        measure(case)
    samples = []
    for repeat in range(1, 6):
        for case in cases:
            samples.append(dict(repeat=repeat, **measure(case)))
    summaries = []
    for number, *_ in cases:
        rows = [r for r in samples if r['example'] == number]
        summary = dict(example=number, circles=rows[0]['circles'], skipped=rows[0]['skipped'])
        for key in rows[0]:
            if key.endswith('_ms'):
                summary[key] = median(r[key] for r in rows)
        summary.update(min_total_ms=min(r['total_ms'] for r in rows),
                       max_total_ms=max(r['total_ms'] for r in rows))
        summaries.append(summary)
    output = here/'figures'/'bp_circles'/'timings'
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (('samples.csv', samples), ('summary.csv', summaries)):
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    typical = median(r['total_ms'] for r in summaries)
    slowest = max(summaries, key=lambda r: r['total_ms'])
    batch = median(sum(r['total_ms'] for r in samples if r['repeat'] == repeat)
                   for repeat in range(1, 6))
    lines = ['# Circle-to-tangent-polyline timing', '',
        f'Python {platform.python_version()}, {platform.machine()}. One warm-up pass; '
        f'five measured passes over {len(cases)} feasible-bp examples. '
        'Medians of wall-clock perf_counter measurements, in milliseconds.', '',
        'Includes fresh initial circle placement, initial diagnostics, same-turn and '
        'opposite-turn repairs and their diagnostics, unrestricted directed tangents, '
        'intersection-triggered block skips, final consecutive-tangent intersection checks, '
        'polyline vertex construction and D_j membership. All placement and repair rules '
        'are the current defaults used by the tangent-polyline overview.', '',
        'Excludes imports, bp construction, plotting, file output, start/end pose maneuvers, '
        'and the separate stricter quarter-arc tangent-chain algorithm. This measures '
        'the current internal geometric construction, not a complete certified trajectory.', '',
        f'Excluded examples (no feasible bp): {excluded}.', '',
        f'Median across per-example median totals: {typical:.3f} ms. '
        f'Slowest median: example {slowest["example"]}, {slowest["total_ms"]:.3f} ms. '
        f'Median sum of timed calls for the full batch: {batch:.3f} ms.', '',
        '| Example | Circles | Skipped | Circles + repairs [ms] | Tangents + skips + D_j [ms] | Total [ms] |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in summaries:
        lines.append(f"| {r['example']} | {r['circles']} | {r['skipped']} | "
                     f"{r['circles_and_repairs_ms']:.3f} | {r['tangents_skips_Dj_ms']:.3f} | {r['total_ms']:.3f} |")
    lines += ['', 'Component medians need not sum exactly to the median total. '
              'CSV details include raw samples and observed min/max totals. '
              'These are short local warm timings, not worst-case guarantees.', '',
              'Reproduce: `MPLBACKEND=Agg python experiments/bicycle_journal_paper/benchmark_bp_circle_to_polyline.py`.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
