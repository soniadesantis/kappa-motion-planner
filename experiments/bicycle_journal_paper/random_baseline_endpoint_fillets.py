"""Random exact baselines with endpoint arcs and a sweep of sampled widths.

Run with VS Code's Run button or --no-show. Incoming/outgoing virtual directions
are the sampled first/last corridor headings. Require quarter turns at both
ends, then include their local A sets in exact propagation. No prescribed
boundary positions, poses or boundary segment lengths are introduced.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

import example_random_baseline_construction as generator
from kappa_planner.baseline_construction import (
    analyze_orthogonal_polyline_feasibility, compute_filleted_baseline_exact,
)


SEED = 37
CASES = 100
CORRIDOR_COUNTS = (5, 8, 12, 20)
WIDTH_RANGES = ((1.1, 5.), (1.1, 8.), (1.1, 12.))
DIRECTIONS = ('right', 'up', 'left', 'down')
OUTPUT = Path(__file__).parent / 'results/random_baseline/endpoint_widths'


def stats(values):
    return dict(median=float(np.median(values)), mean=float(np.mean(values)),
                p95=float(np.percentile(values, 95)), maximum=float(np.max(values))) if values else None


def is_quarter_turn(first, second):
    return (DIRECTIONS.index(first)-DIRECTIONS.index(second)) % 2 == 1


def experiment(n, widths, args, robot):
    generator.NUMBER_OF_CORRIDORS = n
    generator.WIDTH_RANGE = widths
    rng = np.random.default_rng(args.seed)
    cases, rejections, results = [], Counter(), []
    for generated in range(1, args.max_generated+1):
        bounds, sampled = generator.sample_corridors(rng)
        corridors = generator.corridors_from_bounds(bounds)
        analysis = analyze_orthogonal_polyline_feasibility(corridors, robot, compute_viable=False)
        if not analysis.feasible:
            rejections[analysis.status] += 1
            continue
        incoming, outgoing = DIRECTIONS[sampled['headings'][0]], DIRECTIONS[sampled['headings'][-1]]
        if not (is_quarter_turn(incoming, analysis.passage_directions[0])
                and is_quarter_turn(analysis.passage_directions[-1], outgoing)):
            rejections['endpoint_headings_not_both_quarter_turns'] += 1
            continue
        result = compute_filleted_baseline_exact(corridors, robot,
                                                 initial_direction=incoming, final_direction=outgoing)
        if result.feasible and (result.fillets[0] is None or result.fillets[-1] is None):
            raise RuntimeError('Successful case missing a requested endpoint arc.')
        timings = []
        for _ in range(args.repetitions):
            start = perf_counter_ns()
            compute_filleted_baseline_exact(corridors, robot,
                                            initial_direction=incoming, final_direction=outgoing)
            timings.append((perf_counter_ns()-start)/1e6)
        cases.append(dict(case=len(cases)+1, generated=generated, bounds=bounds,
                          headings=sampled['headings'], sampled_widths=sampled['widths'],
                          final_widths=sampled['final_widths'],
                          initial_direction=incoming, final_direction=outgoing,
                          status=result.status, reason=result.reason, success=result.feasible,
                          empty_waypoint=(result.fillet_reachability.empty_waypoint
                                          if result.fillet_reachability else None),
                          polyline=result.polyline.tolist() if result.feasible else None,
                          fillet_indices=[j for j,f in enumerate(result.fillets) if f] if result.feasible else [],
                          median_ms=float(np.median(timings))))
        results.append(result)
        if len(cases) == args.cases:
            break
    group = dict(corridors=n, width_range=widths, accepted=len(cases), generated=generated,
                 generation_cap=args.max_generated,
                 generation_rejections=dict(rejections),
                 statuses=dict(Counter(c['status'] for c in cases)),
                 successes=sum(c['success'] for c in cases),
                 full_construction_ms=stats([c['median_ms'] for c in cases]),
                 successful_construction_ms=stats([c['median_ms'] for c in cases if c['success']]),
                 cases=cases)
    print(json.dumps({k:v for k,v in group.items() if k != 'cases'}, indent=2), flush=True)
    return group, results


def overview(group, results, count):
    count = min(count, len(results))
    columns = min(5, count)
    rows = (count+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(4*columns, 3.5*rows), squeeze=False)
    for ax, case, result in zip(axes.flat, group['cases'][:count], results[:count]):
        for a,b,c,d in case['bounds']:
            ax.add_patch(Rectangle((a,c),b-a,d-c,facecolor='#EDF5FC',edgecolor='.4',linewidth=.4))
        if result.feasible:
            points = result.polyline
            ax.plot(*points.T, '--o', color='#64748b', lw=.6, markersize=2, alpha=.6)
            for j in range(len(points)-1):
                f, g = result.fillets[j:j+2]
                a = f.outgoing_tangent if f else points[j]
                b = g.incoming_tangent if g else points[j+1]
                ax.plot([a[0],b[0]],[a[1],b[1]],color='#333333',lw=1.3)
            for j, f in enumerate(result.fillets):
                if not f:
                    continue
                radial = f.incoming_tangent-f.center
                angles = np.arctan2(radial[1],radial[0])+np.linspace(0,f.signed_angle,81)
                arc = f.center+f.radius*np.column_stack((np.cos(angles),np.sin(angles)))
                ax.plot(*arc.T,color='#D55E00' if j in (0,len(points)-1) else '#009E73',lw=1.8)
            label = f'case {case["case"]} · {case["median_ms"]:.2f} ms'
        else:
            label = f'case {case["case"]}\n{case["status"]}'
        ax.set_title(label,fontsize=9,color='#333333' if result.feasible else '#A13E00')
        ax.autoscale_view()
        ax.set_aspect('equal',adjustable='box')
        ax.margins(.08)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[count:]:
        ax.set_axis_off()
    fig.suptitle(f'{group["corridors"]} corridors · sampled widths {group["width_range"]} m\n'
                 'Endpoint directions follow the first/last corridor headings',fontsize=15)
    fig.legend(handles=[Line2D([],[],color='#D55E00',lw=2,label='Endpoint fillets'),
                        Line2D([],[],color='#009E73',lw=2,label='Internal fillets'),
                        Line2D([],[],color='#333333',lw=2,label='Straights')],
               loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.035,1,.94))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--cases',type=int,default=CASES)
    parser.add_argument('--corridors',type=int,nargs='+',default=CORRIDOR_COUNTS)
    parser.add_argument('--width-maxima',type=float,nargs='+',default=[w[1] for w in WIDTH_RANGES])
    parser.add_argument('--max-generated',type=int,default=50000)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--overview-count',type=int,default=20)
    parser.add_argument('--no-show',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    args = parser.parse_args()
    if min(args.cases,args.max_generated,args.repetitions,args.overview_count) < 1 or min(args.corridors) < 3:
        parser.error('Positive counts and at least three corridors required.')
    if any(not np.isfinite(w) or w < 1.1 for w in args.width_maxima):
        parser.error('Width maxima must be finite and at least 1.1.')
    robot = SimpleNamespace(r=generator.ROBOT_RADIUS,R=generator.TURNING_RADIUS)
    report = dict(seed=args.seed,r=robot.r,R=robot.R,repetitions=args.repetitions,
                  cases_requested=args.cases,length_range=generator.LENGTH_RANGE,
                  overhang_range=generator.OVERHANG_RANGE,
                  straight_probability=generator.STRAIGHT_PROBABILITY,
                  note='Conditional on exact internal polyline feasibility and quarter-turn endpoint '
                       'headings. Local endpoint A sets included before propagation. No boundary poses. '
                       'Timings include full construction/validation, exclude generation and plotting. '
                       'Extended corridor widths may exceed sampled ranges.',groups=[])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    figures = []
    for maximum in args.width_maxima:
        for n in args.corridors:
            group, results = experiment(n,(1.1,maximum),args,robot)
            report['groups'].append(group)
            (args.output_dir/f'endpoint_width_sweep_seed{args.seed}.json').write_text(json.dumps(report,indent=2)+'\n')
            if len(results) < args.cases:
                print(f'Generation cap reached: n={n}, widths=(1.1,{maximum}), accepted={len(results)}',flush=True)
            if results and n in (5,8):
                fig = overview(group,results,args.overview_count)
                name = f'endpoint_fillets_n{n}_width{maximum:g}_seed{args.seed}'
                for suffix in ('pdf','png'):
                    fig.savefig(args.output_dir/f'{name}.{suffix}',dpi=160,bbox_inches='tight')
                if args.no_show:
                    plt.close(fig)
                else:
                    figures.append(fig)
    if figures:
        plt.show()
    print('Saved results in',args.output_dir)


if __name__ == '__main__':
    main()
