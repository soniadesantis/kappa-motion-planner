"""Analytic capsule checks at concave corners, seams, holes and boundary contact."""

import unittest

import numpy as np

from kappa_planner.helpers.corridor_union import CorridorUnion


class SegmentCorridorUnionTest(unittest.TestCase):
    def test_rectangle_contains_end_caps_and_admits_exact_contact(self):
        union = CorridorUnion(((0., 6., 0., 4.),))
        self.assertTrue(union.contains_capsule((1., 1.), (5., 1.), 1., tol=0.))
        self.assertIsNone(union._boundary)  # Convexity fast path builds no boundary.
        self.assertFalse(union.contains_capsule((.5, 2.), (5., 2.), 1., tol=0.))
        self.assertFalse(union.contains_capsule((1., 1.), (5., 1.), 1.01, tol=0.))

    def test_shared_seam_and_duplicate_rectangles_are_not_obstacles(self):
        boxes = ((-2., 0., -1., 1.), (0., 2., -1., 1.), (-2., 0., -1., 1.))
        union = CorridorUnion(boxes)
        self.assertTrue(union.contains_capsule((-1., 0.), (1., 0.), .9, tol=0.))
        self.assertEqual(len(union.boundary), 4)
        self.assertFalse(np.any(np.all(union.boundary[:, :, 0] == 0., axis=1)))
        boundary = union.boundary
        self.assertTrue(union.contains_capsule((0., -.1), (0., .1), .8, tol=0.))
        self.assertIs(union.boundary, boundary)

    def test_safe_concave_wedge_outside_individual_erosions(self):
        union = CorridorUnion(((-5., 1., -1., 1.), (-1., 1., -1., 5.)))
        # Each point is too close to both internal rectangle walls, but its
        # footprint clears the true concave corner (-1,1).
        self.assertTrue(union.contains_capsule((-.8, .75), (-.75, .8), .3, tol=0.))
        self.assertTrue(union.contains_capsule((-.8, .8), (-.8, .8), .25, tol=0.))
        self.assertFalse(union.contains_capsule((-.8, .8), (-.8, .8), .3, tol=0.))

    def test_safe_endpoints_do_not_certify_middle_clearance(self):
        union = CorridorUnion(((-5., 1., -1., 1.), (-1., 1., -1., 5.)))
        a, b = (-.8, 1.), (-1., .8)
        for point in (a, b):
            self.assertTrue(union.contains_capsule(point, point, .18, tol=0.))
        self.assertTrue(union.contains_capsule(a, b, 0., tol=0.))
        self.assertFalse(union.contains_capsule(a, b, .18, tol=0.))

    def test_holes_and_disconnected_components_reject_centerline_gaps(self):
        frame = CorridorUnion(((0., 1., 0., 4.), (3., 4., 0., 4.),
                               (0., 4., 0., 1.), (0., 4., 3., 4.)))
        self.assertFalse(frame.contains_capsule((.5, 2.), (3.5, 2.), .25, tol=0.))
        self.assertFalse(frame.contains_capsule((.5, 2.), (3.5, 2.), 0., tol=0.))
        self.assertTrue(frame.contains_capsule((.5, .5), (3.5, .5), .5, tol=0.))
        disconnected = CorridorUnion(((0., 1., 0., 1.), (2., 3., 0., 1.)))
        self.assertFalse(disconnected.contains_capsule((.5, .5), (2.5, .5), 0., tol=0.))

    def test_translation_and_reversed_segment_preserve_clearance(self):
        boxes = np.array(((-5., 1., -1., 1.), (-1., 1., -1., 5.)))
        start, end = np.array((-.8, .75)), np.array((-.75, .8))
        shift = np.array((10000., -70000.))
        union = CorridorUnion(boxes+shift[[0, 0, 1, 1]])
        self.assertTrue(union.contains_capsule(start+shift, end+shift, .3))
        self.assertTrue(union.contains_capsule(end+shift, start+shift, .3))

    def test_invalid_geometry_and_parameters_are_rejected(self):
        for bounds in ((), ((0., 0., 0., 1.),), ((0., np.inf, 0., 1.),)):
            with self.assertRaises(ValueError):
                CorridorUnion(bounds)
        union = CorridorUnion(((0., 1., 0., 1.),))
        for radius, tol in ((-1., 0.), (np.nan, 0.), (.1, -1.)):
            with self.assertRaises(ValueError):
                union.contains_capsule((.5, .5), (.5, .5), radius, tol=tol)


if __name__ == '__main__':
    unittest.main()
