"""Compare certified segment propagation with the unchanged generic reference.

Thousands of conditional random corridor walks; both free boundary directions
and explicit entry/exit fillets are tested. Full-set comparisons use tight
bounds AND identical analytic constraints, not just sampled membership or
Boolean feasibility. Slice sampling independently checks the extreme-segment
claim, alongside its exact convex-endpoint certificate. Witnesses receive the
baseline's independent fillet validation. Synthetic arbitrary rounded sets
are deliberately outside the structural guarantee.
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
    _validate_and_build_fillets,
)
from kappa_planner.helpers.fillet_reachability import (
    RoundedCornerConstraint, _make_set, propagate_fillet_regions, _DIRECTIONS,
)
from kappa_planner.helpers.fillet_segment_reachability import (
    extreme_segment_certificate, propagate_fillet_regions_segment, SegmentHypothesisError,
)

OUTPUT=Path(__file__).parent/'results/random_baseline/exact_propagation/segment_report.json'


def arbitrary_set_counterexample():
    previous=_make_set((0,1,0,1),(RoundedCornerConstraint((1,0),(-1,1),1.),),1e-10)
    # At y=.8 the true upstream coordinate is .4, although global xmin is 0.
    return dict(scope='Arbitrary rounded sets; not a nominal outgoing fillet region',
                direction='right',bounds=list(previous.bounds),
                endpoint_certificate_valid=extreme_segment_certificate(previous,'right').valid,
                slice_y=.8,true_min_x=previous.slice_interval(1,.8)[0],global_min_x=0.,
                minimum_separation=2.,naive_false_positive=[2.2,.8],
                true_propagated_min_x_at_y=2.4)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=int,default=250,help='Accepted walks per seed and length')
    parser.add_argument('--corridors',type=int,nargs='+',default=[5,8,12,20])
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,37,2026])
    parser.add_argument('--r',type=float,default=.5)
    parser.add_argument('--R',type=float,default=2.)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.cases<1 or min(args.corridors)<3 or not 0<args.r<args.R:
        parser.error('Require positive cases, at least three corridors and 0<r<R.')
    robot=SimpleNamespace(r=args.r,R=args.R)
    tol=1e-9
    report=dict(cases_per_group=args.cases,seeds=args.seeds,corridor_counts=args.corridors,
                r=args.r,R=args.R,tol=tol,slice_samples_per_face=17,
                sampling='Conditional on rectangular internal-polyline feasibility',
                groups=[],counterexamples=[],arbitrary_set_counterexample=arbitrary_set_counterexample())
    direction_names=('right','up','left','down')
    for n in args.corridors:
        generator.NUMBER_OF_CORRIDORS=n
        for seed in args.seeds:
            rng=np.random.default_rng(seed)
            counts=Counter(failures=0,nonmonotone_constraint_faces=0);directions=Counter();rejections=Counter()
            generic_times=[];segment_times=[];max_error=0.;max_slice_variation=0.
            generated=accepted=0
            while accepted<args.cases and generated<args.cases*200:
                generated+=1
                bounds,meta=generator.sample_corridors(rng)
                corridors=generator.corridors_from_bounds(bounds)
                analysis=analyze_orthogonal_polyline_feasibility(corridors,robot,compute_viable=False)
                if not analysis.feasible:
                    rejections[analysis.status]+=1;continue
                accepted+=1
                for mode in ('free_boundaries','entry_exit_fillets'):
                    kwargs={} if mode=='free_boundaries' else dict(
                        initial_direction=direction_names[meta['headings'][0]],
                        final_direction=direction_names[meta['headings'][-1]])
                    prepared=compute_filleted_baseline(corridors,robot,max_backtracking_attempts=0,
                                                       use_joint_solver=False,**kwargs)
                    if prepared.status not in ('backtracking_unresolved','empty_fillet_region','feasible'):
                        counts['unresolved_local_geometry_'+prepared.status]+=1;continue
                    counts['comparisons']+=1
                    data,regions=prepared.feasibility,prepared.fillet_regions
                    jobs=[('generic',lambda:propagate_fillet_regions(data,regions,args.R)),
                          ('segment',lambda:propagate_fillet_regions_segment(data,regions,args.R))]
                    outcomes={};failure=None
                    for name,job in jobs if accepted%2 else reversed(jobs):
                        start=perf_counter_ns()
                        try:outcomes[name]=job()
                        except SegmentHypothesisError as error:failure=str(error)
                        elapsed=(perf_counter_ns()-start)/1e6
                        (generic_times if name=='generic' else segment_times).append(elapsed)
                    if failure is None:
                        generic,segment=outcomes['generic'],outcomes['segment']
                        counts['generic_'+generic.status]+=1;counts['segment_'+segment.status]+=1
                        if (generic.status!=segment.status or generic.empty_waypoint!=segment.empty_waypoint
                                or len(generic.reachable_sets)!=len(segment.reachable_sets)):
                            failure='Status, empty index or number of sets differs'
                        for j,(first,second) in enumerate(zip(generic.reachable_sets,segment.reachable_sets)):
                            if (first is None)!=(second is None):failure='Set emptiness differs';break
                            if first is None:continue
                            counts['sets_compared']+=1
                            error=float(np.max(np.abs(np.array(first.bounds)-second.bounds)))
                            max_error=max(max_error,error)
                            if error>1e-8 or set(first.constraints)!=set(second.constraints):
                                failure=f'Analytic set representation differs at waypoint {j}';break
                            if j<len(data.passage_directions):
                                direction=data.passage_directions[j]
                                cert=extreme_segment_certificate(first,direction)
                                counts['face_certificates']+=1;directions[direction]+=1
                                if not cert.valid:failure=f'Endpoint certificate failed at {j}';break
                                axis,sign=_DIRECTIONS[direction];transverse=1-axis
                                if any(g.signs[axis]!=sign for g in first.constraints):
                                    counts['nonmonotone_constraint_faces']+=1
                                for value in np.linspace(*first.bounds[2*transverse:2*transverse+2],17):
                                    sl=first.slice_interval(transverse,value)
                                    if sl is None:failure=f'Missing transverse slice at {j}';break
                                    variation=abs(sl[0 if sign>0 else 1]-first.bounds[2*axis+(sign<0)])
                                    max_slice_variation=max(max_slice_variation,variation)
                                    counts['slice_checks']+=1
                                    if variation>1e-8:failure=f'Nonconstant extreme boundary at {j}';break
                        if segment.feasible:
                            counts['witnesses_validated']+=1
                            if _validate_and_build_fillets(segment.polyline,data,regions,args.r,args.R,tol) is None:
                                failure='Segment reconstruction failed independent fillet validation'
                    if failure:
                        counts['failures']+=1
                        if len(report['counterexamples'])<20:
                            report['counterexamples'].append(dict(n=n,seed=seed,accepted_case=accepted,
                                generated=generated,mode=mode,bounds=bounds,error=failure))
                if accepted%100==0:
                    print(f'n={n}, seed={seed}: {accepted}/{args.cases} walks; '
                          f"comparisons={counts['comparisons']}, failures={counts['failures']}",flush=True)
            group=dict(corridors=n,seed=seed,accepted=accepted,generated=generated,requested_fulfilled=accepted==args.cases,
                       counts=dict(counts),directions=dict(directions),rejections=dict(rejections),
                       max_projection_difference=max_error,max_slice_variation=max_slice_variation,
                       generic_median_ms=float(np.median(generic_times)) if generic_times else None,
                       segment_median_ms=float(np.median(segment_times)) if segment_times else None)
            report['groups'].append(group)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(group),flush=True)
    totals=Counter()
    for g in report['groups']:totals.update(g['counts'])
    report['totals']=dict(totals)
    report['accepted_walks']=sum(g['accepted']for g in report['groups'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('TOTAL:',json.dumps(report['totals']),flush=True)
    print('Saved',args.output)
    if totals['failures']:raise SystemExit(1)


if __name__=='__main__':main()
