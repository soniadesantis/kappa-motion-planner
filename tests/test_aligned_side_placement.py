"""Identify all boundary intersections by side before local circle placement."""
import unittest
import numpy as np
from kappa_planner.refinement import _place_aligned_sides, _corridor_boundary_intersections
from kappa_planner.helpers.fillet_safety import region_contains_point


class AlignedSidePlacementTest(unittest.TestCase):
    def place(self,pair=((0,13,0,4),(9,19,-1,5)),door=(10,12,1,3),point=(10,2)):
        return _place_aligned_sides(1,np.array(point,float),np.array([1.,0.]),
                                   door,pair,1.,3.,1e-9)

    def test_two_sides_with_separate_regions(self):
        circles,decisions=self.place()
        self.assertEqual([d.status for d in decisions],['placed','placed'])
        self.assertEqual([c.side for c in circles],['left','right'])
        self.assertEqual([c.rule for c in circles],['basic_shifted','basic_shifted'])
        for c in circles:
            self.assertTrue(region_contains_point(c.vertex,c.region))
            np.testing.assert_allclose(c.vertex,c.center+3*c.region['incoming']-3*c.region['outgoing'])
        np.testing.assert_allclose(decisions[0].corner,[9,4])
        np.testing.assert_allclose(decisions[1].corner,[9,0])

    def test_shared_edge_is_not_a_unique_intersection(self):
        pair=((0,13,0,4),(9,19,0,5))
        circles,decisions=self.place(pair)
        self.assertEqual([c.side for c in circles],['left'])
        self.assertEqual(decisions[1].status,'shared_boundary')
        points,segments=_corridor_boundary_intersections(*pair,1e-9)
        self.assertEqual(len(segments),1)
        self.assertEqual(len(points),3)

    def test_multiple_or_no_intersections_are_not_arbitrarily_chosen(self):
        _,decisions=self.place(((0,10,0,4),(3,7,-4,8)),(4,6,1,3),(5,2))
        self.assertEqual([d.status for d in decisions],['ambiguous_intersection']*2)
        self.assertEqual([d.intersection_count for d in decisions],[2,2])
        _,decisions=self.place(((0,20,0,10),(5,15,2,8)),(6,14,3,7),(10,5))
        self.assertEqual([d.status for d in decisions],['no_intersection']*2)

    def test_reflection_swaps_sides_rotation_preserves_them(self):
        original,_=self.place()
        pair=((0,13,0,4),(9,19,-1,5)); door=(10,12,1,3)
        shift=np.array([22.,-17.])
        for t in (np.array([[0.,-1.],[1.,0.]]),np.diag([1.,-1.])):
            def transform(box):
                a,b,c,d=box
                points=np.array([[a,c],[b,c],[b,d],[a,d]])@t.T+shift
                return points[:,0].min(),points[:,0].max(),points[:,1].min(),points[:,1].max()
            circles,_=_place_aligned_sides(1,np.array([10.,2.])@t.T+shift,
                         np.array([1.,0.])@t.T,transform(door),tuple(map(transform,pair)),1.,3.,1e-9)
            for c in original:
                side=c.side if np.linalg.det(t)>0 else {'left':'right','right':'left'}[c.side]
                transformed=next(x for x in circles if x.side==side)
                np.testing.assert_allclose(transformed.center,c.center@t.T+shift)
                self.assertTrue(transformed.in_admissible_region)


if __name__=='__main__':
    unittest.main()
