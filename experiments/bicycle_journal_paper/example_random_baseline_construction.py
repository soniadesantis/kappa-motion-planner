"""Small reproducible experiment on random axis-aligned corridor sequences.

Press Run in VS Code. Edit the configuration below to change the distribution.
Keep only sequences passing the exact internal 2R-polyline analysis, then run
the fillet heuristic with the joint solver disabled. No refinement or boundary
pose connections are included. This is a conditional sample from a connected
random-walk generator, not a uniform sample of all corridor arrangements.

CLI: python experiments/bicycle_journal_paper/example_random_baseline_construction.py
     --cases 100 --seed 7 --no-plot
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from example_baseline_construction import (
    benchmark_check, draw_fillet_region, draw_region, plot_example, plot_filleted_baseline,
)
from kappa_planner.baseline_construction import (
    analyze_orthogonal_polyline_feasibility, compute_filleted_baseline,
    compute_local_boundary_fillets,
)

from kappa_planner.helpers.corridor_extension import extend_perpendicular_corridors

SEED = 7
NUMBER_OF_CORRIDORS = 8  # Consecutive corridors per case; edit before pressing Run.
ACCEPTED_CASES = 100
MAX_GENERATED = 10000  # Stop even if the requested accepted sample is not reached.
ROBOT_RADIUS = .5
TURNING_RADIUS = 2.
WIDTH_RANGE = (1.1, 5.)  # Metres, transverse to the corridor's intended direction.
LENGTH_RANGE = (3., 14.)  # Metres between consecutive random-walk vertices.
OVERHANG_RANGE = (0., 2.)  # Extra longitudinal extent at EACH corridor end.
STRAIGHT_PROBABILITY = .15  # Other transitions split equally between left and right.
EXTEND_PERPENDICULAR_CORRIDORS = True
MAX_BACKTRACKING_ATTEMPTS = 128
PLOT_CASE = 83  # One-based accepted case number; None selects an informative case.
PLOT_SINGLE_CASE = False  # Enable the two detailed figures in addition to the overview.
OVERVIEW_START = 1  # First accepted case displayed in the overview.
OVERVIEW_COUNT = 20
OUTPUT_DIRECTORY = (Path(__file__).parent / 'results' / 'random_baseline'
                    / 'local_boundary_fillets' / f'n{NUMBER_OF_CORRIDORS}')


def corridors_from_bounds(bounds):
    """Minimal rectangle interface consumed by the baseline module."""
    return [SimpleNamespace(corners=np.array([[a, c], [b, c], [b, d], [a, d]], dtype=float))
            for a, b, c, d in bounds]



def sample_corridors(rng):
    """Grow a connected orthogonal walk and place one rectangle per segment.

    Width, walk length and both overhangs are drawn independently and uniformly.
    Before extension, physical length equals walk length plus both overhangs.
    After sampling, fully covered edges extend into perpendicular neighbors.
    Final dimensions may exceed the sampling ranges; both are saved for replay.
    Lengths are not forced to satisfy the 2R condition.
    The exact analyzer subsequently decides which sequences enter the sample.
    """
    widths = rng.uniform(*WIDTH_RANGE, NUMBER_OF_CORRIDORS)
    lengths = rng.uniform(*LENGTH_RANGE, NUMBER_OF_CORRIDORS)
    overhangs = rng.uniform(*OVERHANG_RANGE, (NUMBER_OF_CORRIDORS, 2))
    axes = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    heading = int(rng.integers(4))
    point = np.zeros(2)
    bounds, headings = [], []
    for j in range(NUMBER_OF_CORRIDORS):
        if j:
            heading = (heading + int(rng.choice(
                [0, 1, -1], p=[STRAIGHT_PROBABILITY,
                               (1-STRAIGHT_PROBABILITY)/2,
                               (1-STRAIGHT_PROBABILITY)/2]))) % 4
        direction = axes[heading]
        endpoint = point + lengths[j]*direction
        first = point - overhangs[j, 0]*direction
        last = endpoint + overhangs[j, 1]*direction
        transverse = np.abs(np.array([-direction[1], direction[0]]))
        low = np.minimum(first, last) - .5*widths[j]*transverse
        high = np.maximum(first, last) + .5*widths[j]*transverse
        bounds.append((float(low[0]), float(high[0]), float(low[1]), float(high[1])))
        headings.append(heading)
        point = endpoint
    original_bounds = list(bounds)
    changes = []
    if EXTEND_PERPENDICULAR_CORRIDORS:
        bounds, changes = extend_perpendicular_corridors(bounds, headings)
    spans = [(b-a, d-c) for a, b, c, d in bounds]
    return bounds, dict(widths=widths.tolist(), walk_lengths=lengths.tolist(),
                        overhangs=overhangs.tolist(), headings=headings,
                        original_bounds=original_bounds, edge_extensions=changes,
                        final_widths=[span[1-h % 2] for span, h in zip(spans, headings)],
                        final_lengths=[span[h % 2] for span, h in zip(spans, headings)])


def validate_configuration():
    if NUMBER_OF_CORRIDORS < 3 or not 0 < ROBOT_RADIUS < TURNING_RADIUS:
        raise ValueError('Require at least three corridors and 0 < r < R.')
    for name, interval in (('width', WIDTH_RANGE), ('length', LENGTH_RANGE),
                           ('overhang', OVERHANG_RANGE)):
        if len(interval) != 2 or not np.all(np.isfinite(interval)) or not 0 <= interval[0] <= interval[1]:
            raise ValueError(f'Invalid {name} range.')
        if name != 'overhang' and interval[0] == 0:
            raise ValueError(f'{name} must be positive.')
    if not 0 <= STRAIGHT_PROBABILITY <= 1:
        raise ValueError('STRAIGHT_PROBABILITY must lie in [0,1].')


def run_experiment(seed, count, max_generated, timing_repeats=7):
    validate_configuration()
    if count < 1 or max_generated < 1 or timing_repeats < 1:
        raise ValueError('Case count and generation limit must be positive.')
    rng = np.random.default_rng(seed)
    robot = SimpleNamespace(r=ROBOT_RADIUS, R=TURNING_RADIUS)
    rejected, cases, reports, boundary_reports = Counter(), [], [], []
    for generated in range(1, max_generated+1):
        bounds, sampled = sample_corridors(rng)
        corridors = corridors_from_bounds(bounds)
        started = perf_counter_ns()
        analysis = analyze_orthogonal_polyline_feasibility(corridors, robot, compute_viable=False)
        analysis_ms = (perf_counter_ns()-started)/1e6
        if not analysis.feasible:
            rejected[analysis.status] += 1
            continue
        started = perf_counter_ns()
        result = compute_filleted_baseline(
            corridors, robot, use_joint_solver=False,
            max_backtracking_attempts=MAX_BACKTRACKING_ATTEMPTS)
        construction_ms = (perf_counter_ns()-started)/1e6
        alternatives = compute_local_boundary_fillets(result, robot) if result.feasible else None
        samples = {'internal': [], 'local_boundary': []}
        for _ in range(timing_repeats):
            started = perf_counter_ns()
            compute_filleted_baseline(corridors, robot, use_joint_solver=False,
                                     max_backtracking_attempts=MAX_BACKTRACKING_ATTEMPTS)
            samples['internal'].append((perf_counter_ns()-started)/1e6)
            if result.feasible:
                started = perf_counter_ns()
                compute_local_boundary_fillets(result, robot)
                samples['local_boundary'].append((perf_counter_ns()-started)/1e6)
        boundary_reports.append(alternatives)
        outcome = ('first_midpoints' if result.midpoint_only_success else
                   'alternative_candidates' if result.feasible else result.status)
        cases.append(dict(case=len(cases)+1, generated_index=generated, bounds=bounds,
                          **sampled, passage_directions=list(analysis.passage_directions),
                          outcome=outcome, status=result.status, reason=result.reason,
                          fillet_success=result.feasible,
                          certified_infeasible_local_model=result.certified_infeasible,
                          attempts=result.backtracking_attempts,
                          first_rejected_waypoint=result.first_rejected_waypoint,
                          polyline_analysis_ms=analysis_ms,
                          fillet_construction_ms=construction_ms,
                          internal_median_ms=median(samples['internal']),
                          local_boundary_median_ms=(median(samples['local_boundary'])
                                                    if samples['local_boundary'] else None),
                          timing_samples_ms=samples,
                          initial_arc_directions=([d for d,_ in alternatives.initial] if alternatives else []),
                          final_arc_directions=([d for d,_ in alternatives.final] if alternatives else []),
                          boundary_unresolved=(alternatives.unresolved if alternatives else [])))
        reports.append(result)
        if len(cases) == count:
            break
    return dict(seed=seed, requested_cases=count, generated=generated,
                accepted=len(cases), polyline_rejections=dict(rejected),
                outcomes=dict(Counter(case['outcome'] for case in cases)),
                configuration=dict(corridors=NUMBER_OF_CORRIDORS, r=ROBOT_RADIUS, R=TURNING_RADIUS,
                                   width_range=WIDTH_RANGE, walk_length_range=LENGTH_RANGE,
                                   overhang_range=OVERHANG_RANGE,
                                   straight_probability=STRAIGHT_PROBABILITY,
                                   extend_perpendicular_corridors=EXTEND_PERPENDICULAR_CORRIDORS,
                                   max_attempts=MAX_BACKTRACKING_ATTEMPTS, use_joint_solver=False),
                timing_repeats=timing_repeats,
                timing_note='Per-case warmed medians: internal includes exact analysis; local_boundary is the '
                            'additional fixed-endpoint check. Excludes generation/plotting. Joint solver disabled.',
                index_note='case is one-based; first_rejected_waypoint is zero-based.',
                cases=cases), reports, boundary_reports, robot


def plot_case_overview(cases, reports, seed, boundary_reports=None):
    """Compact panels with independent equal-aspect axes and a shared legend."""
    columns = min(5, len(cases))
    rows = (len(cases)+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(4*columns, 3.4*rows+1), squeeze=False)
    labels = {'first_midpoints': 'First midpoints', 'alternative_candidates': 'Alternatives used',
              'empty_fillet_region': 'Empty local region', 'backtracking_unresolved': 'Unresolved'}
    for panel, (ax, case, internal_result) in enumerate(zip(axes.flat, cases, reports)):
        alternatives = boundary_reports[panel] if boundary_reports is not None else None
        result = internal_result
        for a, b, c, d in case['bounds']:
            ax.add_patch(Rectangle((a, c), b-a, d-c, facecolor='#e2e8f0',
                                   edgecolor='black', linewidth=.5, alpha=.65, zorder=1))
        for door in result.feasibility.safe_overlaps:
            draw_region(ax, door, '#a855f7', alpha=.2, zorder=2)
        for j, region in enumerate(result.fillet_regions):
            if region is None:
                continue
            if region['empty']:
                a, b, c, d = result.feasibility.safe_overlaps[j]
                ax.scatter((a+b)/2, (c+d)/2, marker='x', color='#b91c1c', s=35, zorder=7)
            else:
                draw_fillet_region(ax, region)
        points = result.polyline if result.feasible else result.orthogonal_polyline
        if points is not None:
            ax.plot(points[:, 0], points[:, 1], '--o', color='#2563eb',
                    linewidth=.9, markersize=2.5, zorder=4)
        if result.feasible:
            for j in range(len(points)-1):
                start = result.fillets[j].outgoing_tangent if result.fillets[j] else points[j]
                end = result.fillets[j+1].incoming_tangent if result.fillets[j+1] else points[j+1]
                ax.plot([start[0], end[0]], [start[1], end[1]], color='#172033', linewidth=1.5, zorder=5)
            for fillet in result.fillets:
                if fillet is None:
                    continue
                radial = fillet.incoming_tangent-fillet.center
                angles = np.arctan2(radial[1], radial[0]) + np.linspace(0, fillet.signed_angle, 61)
                arc = fillet.center + fillet.radius*np.column_stack((np.cos(angles), np.sin(angles)))
                ax.plot(arc[:, 0], arc[:, 1], color='#ea580c', linewidth=2, zorder=6)
        if alternatives is not None:
            for options, color in ((alternatives.initial, '#0891b2'),
                                   (alternatives.final, '#db2777')):
                for direction, fillet in options:
                    radial = fillet.incoming_tangent-fillet.center
                    angles = np.arctan2(radial[1], radial[0]) + np.linspace(0, fillet.signed_angle, 61)
                    arc = fillet.center + fillet.radius*np.column_stack((np.cos(angles), np.sin(angles)))
                    ax.plot(*arc.T, color=color, linewidth=2, zorder=8)
            boundary_label = f'Local arcs: initial {len(alternatives.initial)}, final {len(alternatives.final)}'
        else:
            boundary_label = 'Endpoint arcs not checked (no internal solution)'
        outcome = labels.get(case['outcome'], case['outcome'])
        ax.set_title(f"Case {case['case']} · internal: {outcome}\n{boundary_label}", fontsize=9,
                     color='#15803d' if result.feasible else '#b91c1c')
        ax.set_aspect('equal', adjustable='box')
        ax.autoscale_view()
        ax.margins(.08)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(cases):]:
        ax.axis('off')
    fig.suptitle(f'{len(cases)} random cases · {NUMBER_OF_CORRIDORS} corridors each · seed {seed}\n'
                 'Local endpoint arcs at fixed polyline vertices; joint solver disabled. Metres.', fontsize=13)
    fig.legend(handles=[Patch(facecolor='#e2e8f0', edgecolor='black', label='Corridors'),
                        Patch(facecolor='#a855f7', alpha=.3, label='Safe overlaps D_j'),
                        Patch(facecolor='#16a34a', alpha=.4, label='Local fillet regions A_j'),
                        Line2D([], [], color='#2563eb', linestyle='--', marker='o', markersize=3,
                               label='Orthogonal polyline'),
                        Line2D([], [], color='#172033', label='Trimmed straights'),
                        Line2D([], [], color='#ea580c', linewidth=2, label='Fillets'),
                        Line2D([], [], color='#b91c1c', marker='x', linestyle='none', label='Empty A_j'),
                        Line2D([], [], color='#0891b2', linewidth=2, label='Initial fillet alternatives'),
                        Line2D([], [], color='#db2777', linewidth=2, label='Final fillet alternatives')],
               loc='lower center', ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .94), h_pad=2)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--cases', type=int, default=ACCEPTED_CASES)
    parser.add_argument('--max-generated', type=int, default=MAX_GENERATED)
    parser.add_argument('--plot-case', type=int, default=PLOT_CASE if PLOT_SINGLE_CASE else None)
    parser.add_argument('--overview-start', type=int, default=OVERVIEW_START)
    parser.add_argument('--overview-count', type=int, default=OVERVIEW_COUNT)
    parser.add_argument('--timing-repeats', type=int, default=7)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--save-figures', action='store_true')
    parser.add_argument('--output-directory', type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    experiment, reports, boundary_reports, robot = run_experiment(
        args.seed, args.cases, args.max_generated, args.timing_repeats)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    output = args.output_directory / f'random_baseline_seed{args.seed}.json'
    output.write_text(json.dumps(experiment, indent=2, allow_nan=False) + '\n')
    print(f"Seed {args.seed}: {experiment['accepted']}/{experiment['generated']} generated sequences "
          f"passed the exact internal 2R-polyline test.")
    print('Rejected before fillet construction:', experiment['polyline_rejections'])
    print('Fully contained edges extended for perpendicular consecutive corridors:',
          EXTEND_PERPENDICULAR_CORRIDORS)
    print('Heuristic-only outcomes:', experiment['outcomes'])
    print('Empty local regions reject the local fillet model; bounded search failure is unresolved.')
    if experiment['accepted'] < args.cases:
        print(f'Generation limit reached; requested {args.cases} accepted cases.')
    cases = experiment['cases']
    if cases:
        success = sum(case['fillet_success'] for case in cases)
        print(f'Fillet success among polyline-feasible cases: {success}/{len(cases)} ({100*success/len(cases):.1f}%).')
        for field in ('polyline_analysis_ms', 'fillet_construction_ms'):
            print(f'{field}: median {median(c[field] for c in cases):.3f} ms (single calls).')
    for label, subset in (
        ('All accepted cases', cases),
        ('Internal successes', [c for c in cases if c['fillet_success']]),
    ):
        if not subset:
            continue
        print(f'{label}: {len(subset)} cases')
        for field in ('internal_median_ms', 'local_boundary_median_ms'):
            values = np.array([c[field] for c in subset if c[field] is not None])
            if not len(values):
                continue
            print(f'  {field}: mean={values.mean():.3f}, median={np.median(values):.3f}, '
                  f'p95={np.percentile(values,95):.3f}, max={values.max():.3f} ms; '
                  f'sum={values.sum():.3f} ms')
    print('Cases with local arcs at both endpoints:',
          sum(bool(c['initial_arc_directions'] and c['final_arc_directions']) for c in cases))
    print(f'Saved replayable corridor bounds and diagnostics to {output}')
    if not cases or (args.no_plot and not args.save_figures):
        return
    if not 1 <= args.overview_start <= len(cases) or args.overview_count < 1:
        parser.error('Choose an overview start within the accepted cases and a positive count.')
    first = args.overview_start-1
    last = min(len(cases), first+args.overview_count)
    overview = plot_case_overview(cases[first:last], reports[first:last], args.seed,
                                  boundary_reports[first:last])
    if args.save_figures:
        stem = args.output_directory / f'seed{args.seed}_cases{first+1}-{last}_overview'
        for extension in ('.png', '.pdf'):
            overview.savefig(stem.with_suffix(extension), dpi=170)
        print(f'Saved overview to {stem}.png and .pdf')
    if args.plot_case is None and not PLOT_SINGLE_CASE:
        if args.no_plot:
            plt.close(overview)
        else:
            plt.show()
        return
    if args.plot_case is not None:
        if not 1 <= args.plot_case <= len(cases):
            parser.error(f'--plot-case must be between 1 and {len(cases)}')
        selected = args.plot_case-1
    else:
        priority = {'backtracking_unresolved': 0, 'alternative_candidates': 1,
                    'empty_fillet_region': 2, 'first_midpoints': 4}
        selected = min(range(len(cases)), key=lambda j: priority.get(cases[j]['outcome'], 3))
    case, result = cases[selected], reports[selected]
    print(f"Plotting accepted case {case['case']} (generated sequence {case['generated_index']}): {case['outcome']}")
    corridors = corridors_from_bounds(case['bounds'])
    analysis = analyze_orthogonal_polyline_feasibility(corridors, robot)
    timings = benchmark_check(corridors, robot, repetitions=10, batches=3, warmup=1)
    label = f"random {case['case']} / seed {args.seed}"
    figures = [plot_example(label, analysis, robot, timings),
               plot_filleted_baseline(label, result, case['fillet_construction_ms'], robot)]
    if args.save_figures:
        for figure, suffix in zip(figures, ('feasibility', 'fillets')):
            figure.savefig(args.output_directory / f'seed{args.seed}_case{case["case"]}_{suffix}.png', dpi=160)
    if args.no_plot:
        plt.close(overview)
        for figure in figures:
            plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
