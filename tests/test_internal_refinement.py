"""Independent placement, optional aligned turns, and explicit fallback semantics."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from kappa_planner.baseline_construction import compute_filleted_baseline
from kappa_planner.refinement import (
    FixedTurningCircle, FixedCircleConnectionProblem, InternalPathAnchor,
    connect_fixed_circles, refine_internal_baseline, _quarter_in_eroded_pair,
)


def corridors(boxes):
    return [SimpleNamespace(corners=np.array([[a,c],[b,c],[b,d],[a,d]],float))
            for a,b,c,d in boxes]


def optional(index, center, u, v):
    u,v=np.array(u,float),np.array(v,float)
    return FixedTurningCircle(index,np.array(center,float),2.,int(u[0]*v[1]-u[1]*v[0]),
                              dict(incoming=u,outgoing=v,radius=1.),'aligned_option')


class RefinementTest(unittest.TestCase):
    def test_independent_placement_produces_shorter_valid_chain(self):
        boxes=[(-2,2,0,14),(-2.5,2.5,12,19),(-6,2.5,16.5,18.5),
               (-6,-2,2,18),(-9,-4,1.5,6)]
        robot=SimpleNamespace(r=.5,R=2)
        baseline=compute_filleted_baseline(corridors(boxes),robot,use_joint_solver=False)
        result=refine_internal_baseline(baseline,robot)
        self.assertTrue(result.graph.feasible,result.graph.reason)
        self.assertFalse(result.used_fallback)
        self.assertAlmostEqual(result.problem.baseline_length-result.length,.5)
        np.testing.assert_allclose(result.primitives[0]['start'],result.problem.start.point)
        np.testing.assert_allclose(result.primitives[-1]['end'],result.problem.end.point)
        for first,second in zip(result.primitives,result.primitives[1:]):
            np.testing.assert_allclose(first['end'],second['start'],atol=1e-9)

    def test_optional_turns_connect_offset_parallel_anchors(self):
        # Two extra opposite quarter arcs create a lateral shift. A straight
        # source-to-sink segment cannot satisfy both fixed horizontal headings.
        turns=(optional(1,(2,2),(1,0),(0,1)),optional(2,(6,2),(0,1),(1,0)))
        problem=FixedCircleConnectionProblem(
            InternalPathAnchor(np.array([0.,0.]),np.array([1.,0.]),0,1),
            InternalPathAnchor(np.array([8.,4.]),np.array([1.,0.]),3,3),
            (),((-20.,20.,-20.,20.),)*5,((-10.,10.,-10.,10.),)*4,1.)
        self.assertFalse(connect_fixed_circles(problem).local_feasible)
        augmented=replace(problem,circles=turns)
        result=connect_fixed_circles(augmented)
        self.assertTrue(result.feasible,result.reason)
        self.assertEqual(result.retained_waypoints,(1,2))
        self.assertAlmostEqual(result.length,4+2*np.pi)
        # Mutually exclusive alternatives at the same door must not create a
        # within-door transition or invalidate the ordering of the graph.
        extra=optional(1,(2,5),(1,0),(0,1))
        result=connect_fixed_circles(replace(problem,circles=(turns[0],extra,turns[1])))
        self.assertTrue(result.feasible,result.reason)
        for edge in result.tangent_states:
            if edge.first and edge.second < 4:
                indices=[1,1,2]
                self.assertLess(indices[edge.first-1],indices[edge.second-1])

    def test_aligned_generation_keeps_straight_baseline(self):
        c=corridors([(-4,2,-10,10),(0,12,-10,10),(10,22,-10,10),(20,26,-10,10)])
        robot=SimpleNamespace(r=1,R=2)
        baseline=compute_filleted_baseline(c,robot,use_joint_solver=False)
        original=baseline.polyline.copy()
        result=refine_internal_baseline(baseline,robot)
        self.assertGreater(result.aligned_options,0)
        self.assertTrue(result.graph.exhaustive)
        self.assertFalse(result.used_fallback)
        self.assertAlmostEqual(result.length,result.problem.baseline_length)
        np.testing.assert_array_equal(baseline.polyline,original)
        self.assertEqual(result.graph.retained_waypoints,())
        for circle in result.problem.circles:
            self.assertTrue(_quarter_in_eroded_pair(circle,
                result.problem.corridor_bounds[circle.waypoint_index:circle.waypoint_index+2],1,1e-9))
        limited=refine_internal_baseline(baseline,robot,add_aligned_turns=False,max_skipped_doors=0)
        self.assertTrue(limited.used_fallback)
        self.assertFalse(limited.graph.exhaustive)
        self.assertFalse(limited.graph.certified_locally_infeasible)
        self.assertEqual(limited.graph.status,'limited_search_unresolved')

    def test_unsafe_quarter_rejected_between_safe_endpoints(self):
        circle=optional(1,(2,2),(1,0),(0,1))
        # End contacts (2,0), (4,2) fit separate eroded boxes; the middle of
        # the arc traverses their gap and must be rejected analytically.
        pair=((0,3.2,-2,1.2),(2.8,6,.8,4))
        self.assertFalse(_quarter_in_eroded_pair(circle,pair,1,1e-9))

    def test_preferred_positions_are_independent_and_fallback_is_unchanged(self):
        c=corridors([(0,4,0,8),(0,13,0,4),(9,19,-1,5),
                     (14,18,-2,12),(7,17,9,13),(7,11,6,13)])
        robot=SimpleNamespace(r=1,R=3)
        baseline=compute_filleted_baseline(c,robot,use_joint_solver=False)
        original=baseline.polyline.copy()
        a=refine_internal_baseline(baseline,robot,add_aligned_turns=False)
        b=refine_internal_baseline(baseline,robot,add_aligned_turns=True)
        for x,y in zip(a.placements,b.placements):
            np.testing.assert_allclose(x['preferred_center'],y['preferred_center'])
        np.testing.assert_array_equal(original,baseline.polyline)
        self.assertTrue(b.used_fallback)
        self.assertEqual(b.primitives,b.problem.baseline_primitives)
        self.assertEqual(b.length,b.problem.baseline_length)
        self.assertIn('anchor_heading',','.join(reason for _,_,reason in b.graph.rejected_pairs))


if __name__ == '__main__':
    unittest.main()
