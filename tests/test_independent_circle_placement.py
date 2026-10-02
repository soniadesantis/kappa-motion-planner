"""Five-priority placement is local, A_j-consistent, and independent of a graph."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from kappa_planner.baseline_construction import compute_filleted_baseline
from kappa_planner.refinement import place_refinement_circles
from kappa_planner.helpers.fillet_safety import region_contains_point


class IndependentPlacementTest(unittest.TestCase):
    def build(self, transform=np.eye(2), shift=np.zeros(2)):
        boxes=[(-8,4,0,4),(0,16,0,4),(12,16,0,16),(12,28,12,16)]
        c=[SimpleNamespace(corners=np.array([[a,c],[b,c],[b,d],[a,d]])@transform.T+shift)
           for a,b,c,d in boxes]
        r=SimpleNamespace(r=1.,R=3.)
        return compute_filleted_baseline(c,r,use_joint_solver=False),r

    def test_nominal_and_no_graph(self):
        baseline,robot=self.build()
        original=baseline.polyline.copy()
        with patch('kappa_planner.refinement.connect_fixed_circles',side_effect=AssertionError):
            result=place_refinement_circles(baseline,robot)
        self.assertEqual(result.status,'placed')
        self.assertEqual(result.skipped_waypoints,(0,2))
        self.assertEqual(len(result.circles),1)
        circle=result.circles[0]
        self.assertEqual(circle.rule,'forty_five_safe_half')
        u,v=circle.region['incoming'],circle.region['outgoing']
        np.testing.assert_allclose(circle.vertex,circle.center+robot.R*u-robot.R*v)
        self.assertTrue(region_contains_point(circle.vertex,circle.region))
        self.assertFalse(region_contains_point(circle.center,circle.region))
        np.testing.assert_array_equal(baseline.polyline,original)
        for tangent,b in zip((circle.vertex-robot.R*u,circle.vertex+robot.R*v),
                             baseline.feasibility.corridor_bounds[1:3]):
            self.assertTrue(np.all(tangent >= np.array([b[0],b[2]])+robot.r-1e-9))
            self.assertTrue(np.all(tangent <= np.array([b[1],b[3]])-robot.r+1e-9))

    def test_outside_Aj_candidates_fall_back_without_clamping(self):
        baseline,robot=self.build()
        proposals=[(name,np.array([100.,100.])) for name in
                   ('forty_five_safe_half','safe_half_shifted','forty_five_basic','basic_shifted')]
        with patch('kappa_planner.helpers.arc_feasibility.generate_local_center_candidates',
                   return_value=proposals):
            result=place_refinement_circles(baseline,robot)
        circle=result.circles[0]
        self.assertEqual(circle.rule,'baseline')
        self.assertEqual(len(circle.rejected_candidates),4)
        self.assertTrue(all(item['reason']=='outside_Aj' for item in circle.rejected_candidates))
        np.testing.assert_array_equal(circle.center,baseline.fillets[1].center)
        np.testing.assert_array_equal(circle.vertex,baseline.polyline[1])

    def test_priority_after_geometric_filter(self):
        baseline,robot=self.build()
        for name in ('safe_half_shifted','forty_five_basic','basic_shifted'):
            with patch('kappa_planner.helpers.arc_feasibility.generate_local_center_candidates',
                       return_value=[(name,np.array([1.,1.]))]):
                result=place_refinement_circles(baseline,robot)
            self.assertEqual(result.circles[0].rule,name)
            self.assertTrue(result.circles[0].in_admissible_region)

    def test_reflections_and_rotations(self):
        baseline,robot=self.build()
        original=place_refinement_circles(baseline,robot).circles[0]
        shift=np.array([21.,-13.])
        for transform in (np.diag([-1.,1.]),-np.eye(2),np.array([[0.,-1.],[1.,0.]])):
            b,r=self.build(transform,shift)
            c=place_refinement_circles(b,r).circles[0]
            self.assertEqual(c.rule,original.rule)
            np.testing.assert_allclose(c.center,original.center@transform.T+shift)
            np.testing.assert_allclose(c.vertex,original.vertex@transform.T+shift)


if __name__ == '__main__':
    unittest.main()
