"""Independent circle placement and tangent-graph refinement; VS Code runnable.

Defaults to the thesis running example. --map selects a legacy corridor example.
--benchmark-json replays saved random-test bounds without generating new cases.
The baseline is retained whenever the graph fails, the global audit is unresolved,
or the proposed path is longer. No local circle-repair rules run here.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from example_planner_thesis import build_example
from kappa_planner.baseline_construction import compute_filleted_baseline
from kappa_planner.refinement import refine_internal_baseline

ADD_ALIGNED_TURNS = True
MAX_SKIPPED_DOORS = None  # None: complete graph; integer: explicitly limited search.
OUTPUT_DIRECTORY = Path(__file__).parent / 'results' / 'internal_refinement'


def draw_primitive(ax, primitive, color, linewidth, linestyle='-'):
    if primitive['kind'] == 'line':
        points = np.array([primitive['start'], primitive['end']])
    else:
        angles = np.linspace(primitive['enter'], primitive['leave'], 100)
        region = primitive['region']
        points = primitive['center'] + primitive['radius'] * (
            -np.cos(angles)[:, None]*region['outgoing']
            + np.sin(angles)[:, None]*region['incoming'])
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth,
            linestyle=linestyle, zorder=5)



def solve(corridors, robot, max_skip):
    baseline = compute_filleted_baseline(corridors, robot, use_joint_solver=False)
    if not baseline.feasible:
        return baseline, None
    return baseline, refine_internal_baseline(
        baseline, robot, add_aligned_turns=ADD_ALIGNED_TURNS, max_skipped_doors=max_skip)


def plot_result(result):
    fig, axes = plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    for ax in axes:
        for a,b,c,d in result.problem.corridor_bounds:
            ax.add_patch(Rectangle((a,c),b-a,d-c,facecolor='#e2e8f0',
                                   edgecolor='black',linewidth=.6,alpha=.7))
        for primitive in result.problem.baseline_primitives:
            draw_primitive(ax,primitive,'#94a3b8',4)
        for anchor in (result.problem.start,result.problem.end):
            ax.scatter(*anchor.point,marker='D',color='black',zorder=8)
        ax.set_aspect('equal'); ax.autoscale_view(); ax.margins(.08)
        ax.grid(alpha=.15); ax.set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    for circle in result.problem.circles:
        alpha = np.linspace(0,np.pi/2,100)
        points = circle.center+circle.radius*(-np.cos(alpha)[:,None]*circle.region['outgoing']
                                             +np.sin(alpha)[:,None]*circle.region['incoming'])
        color = '#db2777' if circle.origin == 'aligned_option' else '#2563eb'
        axes[0].plot(*points.T,'--',color=color,lw=1.5)
        axes[0].scatter(*circle.center,marker='+',color=color)
    for state in result.graph.tangent_states:
        axes[0].plot([state.start[0],state.end[0]],[state.start[1],state.end[1]],
                     color='#16a34a',lw=.8,alpha=.5)
    for primitive in result.primitives:
        draw_primitive(axes[1],primitive,'#ea580c',2.5)
    axes[0].set_title(f'Placed quarters and {len(result.graph.tangent_states)} tangent states')
    axes[1].set_title(f'{result.status}: {result.graph.status}')
    fig.legend(handles=[Line2D([],[],color='#94a3b8',lw=4,label='Baseline'),
                        Line2D([],[],color='#2563eb',ls='--',label='Preferred quarters'),
                        Line2D([],[],color='#db2777',ls='--',label='Optional aligned-door turns'),
                        Line2D([],[],color='#ea580c',lw=2,label='Returned path')],
               loc='outside lower center',ncol=4)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map',type=int)
    parser.add_argument('--max-skipped-doors',type=int,default=MAX_SKIPPED_DOORS)
    parser.add_argument('--benchmark-json',type=Path)
    parser.add_argument('--no-show',action='store_true')
    parser.add_argument('--output-directory',type=Path,default=OUTPUT_DIRECTORY)
    args=parser.parse_args()
    args.output_directory.mkdir(parents=True,exist_ok=True)
    if args.benchmark_json:
        data=json.loads(args.benchmark_json.read_text())
        robot=SimpleNamespace(r=data['configuration']['r'],R=data['configuration']['R'])
        rows=[]
        for case in data['cases']:
            corridors=[SimpleNamespace(corners=np.array([[a,c],[b,c],[b,d],[a,d]]))
                       for a,b,c,d in case['bounds']]
            baseline,result=solve(corridors,robot,args.max_skipped_doors)
            row=dict(case=case['case'],baseline_status=baseline.status)
            if result:
                row.update(status=result.status,graph_status=result.graph.status,
                           circles=len(result.problem.circles),aligned_options=result.aligned_options,
                           attempted_pairs=result.graph.attempted_pairs,
                           tangent_states=len(result.graph.tangent_states),
                           placement_ms=result.placement_ms,connection_ms=result.connection_ms,
                           baseline_length=result.problem.baseline_length,length=result.length,
                           rejected_pairs=result.graph.rejected_pairs,
                           contact_order_blocks=result.graph.contact_order_blocks)
            rows.append(row)
        successful=[row for row in rows if 'status' in row]
        print('Outcomes:',dict(Counter(row.get('status',row['baseline_status']) for row in rows)))
        print('Graph statuses:',dict(Counter(row['graph_status'] for row in successful)))
        for key in ('placement_ms','connection_ms','attempted_pairs','circles'):
            values=np.array([row[key] for row in successful])
            if len(values):
                print(key,dict(median=float(np.median(values)),p95=float(np.percentile(values,95)),
                               maximum=float(values.max())))
        output=args.output_directory/'benchmark.json'
        output.write_text(json.dumps(dict(source=str(args.benchmark_json),
            max_skipped_doors=args.max_skipped_doors,add_aligned_turns=ADD_ALIGNED_TURNS,
            timing_note='Single calls; refinement only, excludes baseline and plotting.',cases=rows),indent=2)+'\n')
        print('Saved:',output)
        return
    if args.map is None:
        corridors,robot=build_example()
    else:
        from examples_maps_polyline import example_corridor_sequence
        corridors,_,_,robot=example_corridor_sequence(args.map)
    baseline,result=solve(corridors,robot,args.max_skipped_doors)
    if result is None:
        print('Baseline failed:',baseline.status); return
    print('Status:',result.status,result.graph.status)
    print('Lengths:',result.length,'baseline:',result.problem.baseline_length)
    print('Placement/graph ms:',result.placement_ms,result.connection_ms)
    print('Circle candidates:',len(result.problem.circles),'aligned options:',result.aligned_options)
    print('Tested pairs:',result.graph.attempted_pairs,'tangent states:',len(result.graph.tangent_states))
    print('Rejected pairs:',result.graph.rejected_pairs)
    print('Contact-order blocks:',result.graph.contact_order_blocks)
    fig=plot_result(result)
    for suffix in ('.png','.pdf'):
        fig.savefig(args.output_directory/('internal_refinement'+suffix),dpi=180,
                    bbox_inches='tight')
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
