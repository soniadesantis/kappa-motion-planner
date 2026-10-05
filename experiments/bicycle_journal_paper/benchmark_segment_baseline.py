"""Warmed timings of generic and certified-segment baseline construction.

Run directly or select --examples 21. Median per-call batch timings exclude
imports, map creation, plotting and refinement. Full construction includes
local-region preparation, propagation, reconstruction and final validation.
Propagation-only calls share prepared regions but reconstruct their own chains.
Validation-only calls use each algorithm's own reconstructed chain. Methods
are interleaved and their execution order reverses between timing batches.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import matplotlib.pyplot as plt
import numpy as np

from examples_maps_polyline import EXAMPLE_NUMBERS, example_corridor_sequence
from kappa_planner.baseline_construction import (
    compute_filleted_baseline, compute_filleted_baseline_exact,
    compute_filleted_baseline_segment, _validate_and_build_fillets, _robot_radii,
)
from kappa_planner.helpers.fillet_reachability import propagate_fillet_regions
from kappa_planner.helpers.fillet_segment_reachability import propagate_fillet_regions_segment

ROOT=Path(__file__).parent
OUTPUT=ROOT/'results/baseline_segment_timing.json'


def measure(job,repetitions):
    start=perf_counter_ns()
    for _ in range(repetitions):job()
    return (perf_counter_ns()-start)/repetitions/1e6


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples',type=int,nargs='+',default=list(EXAMPLE_NUMBERS))
    parser.add_argument('--batches',type=int,default=7)
    parser.add_argument('--repetitions',type=int,default=10)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.batches<1 or args.repetitions<1 or args.warmup<0:
        parser.error('Require positive batches and repetitions and nonnegative warmup.')
    report=dict(batches=args.batches,repetitions_per_batch=args.repetitions,warmup=args.warmup,
                units='milliseconds per call',boundary_directions=True,
                timing_scope='Warmed per-call batch medians. Full includes preparation, propagation, '
                'reconstruction and independent validation. Imports, map creation, plotting and refinement excluded.',
                cases=[],skipped=[])
    for number in args.examples:
        corridors,start,end,robot=example_corridor_sequence(number)
        kwargs=dict(initial_position=start[:2],final_position=end[:2])
        generic=compute_filleted_baseline_exact(corridors,robot,**kwargs)
        segment=compute_filleted_baseline_segment(corridors,robot,**kwargs)
        if generic.feasible!=segment.feasible or generic.status!=segment.status:
            raise RuntimeError(f'Baseline disagreement on example {number}.')
        if not segment.feasible:
            report['skipped'].append(dict(example=number,status=segment.status));continue
        prepared=compute_filleted_baseline(corridors,robot,max_backtracking_attempts=0,
                                           use_joint_solver=False,**kwargs)
        data,regions=prepared.feasibility,prepared.fillet_regions
        r,R=_robot_radii(robot)
        jobs=[
            ('generic_full_ms',lambda:compute_filleted_baseline_exact(corridors,robot,**kwargs)),
            ('segment_full_ms',lambda:compute_filleted_baseline_segment(corridors,robot,**kwargs)),
            ('generic_propagation_ms',lambda:propagate_fillet_regions(data,regions,R)),
            ('segment_propagation_ms',lambda:propagate_fillet_regions_segment(data,regions,R)),
            ('preparation_ms',lambda:compute_filleted_baseline(corridors,robot,
                     max_backtracking_attempts=0,use_joint_solver=False,**kwargs)),
            ('generic_validation_ms',lambda:_validate_and_build_fillets(generic.polyline,data,regions,r,R,1e-9)),
            ('segment_validation_ms',lambda:_validate_and_build_fillets(segment.polyline,data,regions,r,R,1e-9)),
        ]
        for _,job in jobs:
            for _ in range(args.warmup):job()
        samples={key:[] for key,_ in jobs}
        for batch in range(args.batches):
            for key,job in jobs if batch%2==0 else reversed(jobs):
                samples[key].append(measure(job,args.repetitions))
        timings={key:float(np.median(values))for key,values in samples.items()}
        record=dict(example=number,corridors=len(corridors),overlaps=len(data.safe_overlaps),
                    arcs=sum(f is not None for f in segment.fillets),r=r,R=R,**timings,
                    batch_samples=samples)
        record['segment_full_over_generic']=timings['segment_full_ms']/timings['generic_full_ms']
        record['segment_propagation_over_generic']=timings['segment_propagation_ms']/timings['generic_propagation_ms']
        report['cases'].append(record)
        print(f'Example {number}: {len(corridors)} corridors, {record["arcs"]} arcs; '
              f'full generic={timings["generic_full_ms"]:.3f} ms, segment={timings["segment_full_ms"]:.3f} ms; '
              f'propagation generic={timings["generic_propagation_ms"]:.3f} ms, '
              f'segment={timings["segment_propagation_ms"]:.3f} ms',flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    metrics=[key for key in report['cases'][0]if key.endswith('_ms')or key.endswith('_over_generic')]
    report['summary']={key:dict(median=float(np.median([c[key]for c in report['cases']])),
                                 minimum=float(min(c[key]for c in report['cases'])),
                                 maximum=float(max(c[key]for c in report['cases'])))for key in metrics}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('SUMMARY:',json.dumps(report['summary'],indent=2),flush=True)
    plt.rcParams.update({'font.family':'serif','font.size':13,'mathtext.fontset':'cm'})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6))
    for ax,stage,title in zip(axes,('full','propagation'),('Complete baseline construction','Propagation and reconstruction only')):
        x=[c['overlaps']for c in report['cases']]
        for method,color,marker in [('generic','#167DA7','o'),('segment','#A64A43','s')]:
            ax.scatter(x,[c[f'{method}_{stage}_ms']for c in report['cases']],
                       color=color,marker=marker,label='Generic exact'if method=='generic'else'Certified segments',alpha=.8)
        ax.set_title(title);ax.set_xlabel('Number of overlap sets');ax.set_ylabel('Time per call [ms]')
        ax.grid(alpha=.25);ax.legend(frameon=False)
    fig.tight_layout()
    figure=ROOT/'figures/planner_thesis/baseline_segment_timing.pdf'
    figure.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(figure,bbox_inches='tight');plt.close(fig)
    print('Saved',args.output,'and',figure)


if __name__=='__main__':main()
