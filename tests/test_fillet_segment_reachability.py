"""Extreme-face certificates, nominal corridors, and a deliberate counterexample."""
import unittest
from types import SimpleNamespace

import numpy as np

from kappa_planner.baseline_construction import compute_filleted_baseline_exact, compute_filleted_baseline_segment
from kappa_planner.helpers.fillet_reachability import RoundedCornerConstraint, _make_set, propagate_fillet_regions
from kappa_planner.helpers.fillet_segment_reachability import (
    extreme_segment_certificate, propagate_fillet_regions_segment, propagate_reachable_set, SegmentHypothesisError,
)


class SegmentReachabilityTest(unittest.TestCase):
    def test_curved_upstream_face_is_rejected_in_all_directions(self):
        for direction, transform in (
            ('right', np.eye(2)), ('left', np.diag([-1,1])),
            ('up', np.array([[0,1],[1,0]])), ('down', np.array([[0,1],[-1,0]])),
        ):
            corners=np.array([[0,0],[1,0],[0,1],[1,1]])@transform.T
            lo,hi=corners.min(axis=0),corners.max(axis=0)
            c=tuple(transform@np.array([1,0]))
            signs=tuple(map(int,transform@np.array([-1,1])))
            s=_make_set((lo[0],hi[0],lo[1],hi[1]),(RoundedCornerConstraint(c,signs,1.),),1e-10)
            self.assertFalse(extreme_segment_certificate(s,direction).valid)

    def test_monotone_outgoing_constraint_certifies_full_face(self):
        for direction, axis, sign in (('right',0,1),('left',0,-1),('up',1,1),('down',1,-1)):
            signs=[1,1];signs[axis]=sign
            s=_make_set((-1,1,-1,1),(RoundedCornerConstraint((0,0),tuple(signs),1.),),1e-10)
            self.assertTrue(extreme_segment_certificate(s,direction).valid)
            transverse=1-axis
            for value in np.linspace(*s.bounds[2*transverse:2*transverse+2],31):
                sl=s.slice_interval(transverse,value)
                self.assertAlmostEqual(sl[0 if sign>0 else 1],s.bounds[2*axis+(sign<0)])

    def test_singleton_and_segment_certificates(self):
        for bounds in ((2,2,3,3),(2,2,3,5),(2,4,3,3)):
            s=_make_set(bounds,(),1e-10)
            for direction in ('right','left','up','down'):
                self.assertTrue(extreme_segment_certificate(s,direction).valid)

    def test_counterexample_is_not_reported_as_infeasible(self):
        # Generic propagation forms a curved lens; a bounding-box half-strip
        # would incorrectly include (2.2,.9), whose predecessor doesn't exist.
        boxes=((0.,1.,0.,1.),(2.,3.,0.,1.))
        data=SimpleNamespace(feasible=True,safe_overlaps=boxes,passage_directions=('right',),
            x_reachable=[b[:2]for b in boxes],y_reachable=[b[2:]for b in boxes])
        regions=[dict(low=np.array([b[0],b[2]]),high=np.array([b[1],b[3]]),
            offset=-np.array(c),frame=np.diag(s),radius=1.,empty=False)
            for b,c,s in zip(boxes,((1.,0.),(2.,0.)),((-1,1),(1,1)))]
        reference=propagate_fillet_regions(data,regions,1.)
        self.assertTrue(reference.feasible)
        local_next=_make_set(boxes[1],(RoundedCornerConstraint((2.,0.),(1,1),1.),),1e-10)
        self.assertTrue(local_next.contains((2.2,.9)))
        self.assertFalse(reference.reachable_sets[-1].contains((2.2,.9)))
        with self.assertRaises(SegmentHypothesisError) as caught:
            propagate_fillet_regions_segment(data,regions,1.)
        self.assertEqual(caught.exception.waypoint,0)

    def test_real_corridors_and_boundary_fillets_all_rotations(self):
        original=[(-7,-2.9,-1.3,.2),(-4.4,-2.9,-1.3,4.5),(-4.2,2.4,2,4.5),
                  (-.1,3,-.9,4.2),(.2,4.6,-2.3,1)]
        transforms=(np.eye(2),np.diag([-1,1]),np.array([[0,1],[1,0]]),np.array([[0,1],[-1,0]]))
        for direction,transform in zip(('right','left','up','down'),transforms):
            corridors=[SimpleNamespace(corners=np.array([[a,c],[b,c],[b,d],[a,d]])@transform.T)
                       for a,b,c,d in original]
            reference=compute_filleted_baseline_exact(corridors,SimpleNamespace(r=.5,R=2.),
                initial_direction=direction,final_direction=direction)
            self.assertTrue(reference.feasible,reference.reason)
            constructed=compute_filleted_baseline_segment(corridors,SimpleNamespace(r=.5,R=2.),
                initial_direction=direction,final_direction=direction)
            self.assertTrue(constructed.feasible,constructed.reason)
            self.assertEqual(constructed.selection_method,'segment_set_propagation')
            result=constructed.fillet_reachability
            self.assertTrue(result.feasible)
            for first,second,p in zip(reference.fillet_reachability.reachable_sets,result.reachable_sets,result.polyline):
                np.testing.assert_allclose(first.bounds,second.bounds)
                self.assertEqual(first.constraints,second.constraints)
                self.assertTrue(first.contains(p))

    def test_constructor_runs_segment_propagation_for_straight_corridors(self):
        corridors=[SimpleNamespace(corners=np.array([[a,0],[b,0],[b,2],[a,2]],float))
                   for a,b in ((0,2),(0,5),(3,5))]
        result=compute_filleted_baseline_segment(corridors,SimpleNamespace(r=1,R=1.5))
        self.assertTrue(result.feasible,result.reason)
        self.assertEqual(result.selection_method,'segment_set_propagation')
        self.assertIsNotNone(result.fillet_reachability)
        self.assertEqual(len(result.fillet_reachability.reachable_sets),2)

    def test_stepwise_api_preserves_curved_projection_cutoff(self):
        previous=_make_set((-3.518033988749895,-3.4,-.8,-.681966011250105),
            (RoundedCornerConstraint((-2.4,-1.8),(-1,1),1.5),),1e-10)
        local=_make_set((-3.7,-3.4,2.5,4),
            (RoundedCornerConstraint((-4.9,4),(1,-1),1.5),),1e-10)
        next_set=propagate_reachable_set(local,previous,'up',2.)
        self.assertAlmostEqual(next_set.bounds[2],3.416759103157646)
        self.assertTrue(extreme_segment_certificate(next_set,'right').valid)

    def test_empty_intersection_still_detected(self):
        boxes=((0,1,0,1),(1.5,2,2,3))
        data=SimpleNamespace(feasible=True,safe_overlaps=boxes,passage_directions=('right',),
            x_reachable=[b[:2]for b in boxes],y_reachable=[b[2:]for b in boxes])
        result=propagate_fillet_regions_segment(data,(None,None),1.)
        self.assertFalse(result.feasible)
        self.assertEqual(result.empty_waypoint,1)


if __name__=='__main__':unittest.main()
