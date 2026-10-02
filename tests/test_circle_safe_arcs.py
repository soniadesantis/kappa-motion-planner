"""Analytic circle clipping, including disconnected and degenerate safe sets."""
from types import SimpleNamespace
import unittest
import numpy as np
from kappa_planner.refinement import circle_safe_angular_intervals, compute_circle_safe_arcs


class CircleSafeArcsTest(unittest.TestCase):
    def test_quarter_and_three_quarters(self):
        quarter = circle_safe_angular_intervals((0, 0), 1., [(0, 2, 0, 2)])
        np.testing.assert_allclose(quarter, ((0., np.pi/2),), atol=1e-12)
        arcs = circle_safe_angular_intervals((0, 0), 1., [(-2, 2, 0, 2), (-2, 0, -2, 2)])
        np.testing.assert_allclose(arcs, ((0., 3*np.pi/2),), atol=1e-12)

    def test_wraparound(self):
        arcs = circle_safe_angular_intervals((0, 0), 1., [(0, 2, -2, 2)])
        np.testing.assert_allclose(arcs, ((3*np.pi/2, 5*np.pi/2),), atol=1e-12)

    def test_full_empty_and_external_tangency(self):
        self.assertEqual(circle_safe_angular_intervals((0, 0), 1., [(-2, 2, -2, 2)]), ((0., 2*np.pi),))
        self.assertEqual(circle_safe_angular_intervals((0, 0), 1., [(2, 3, 2, 3)]), ())
        self.assertEqual(circle_safe_angular_intervals((0, 0), 1., [(1, 2, -1, 1)]), ((0., 0.),))
        self.assertEqual(circle_safe_angular_intervals((0, 0), 1., [(2, 1, -1, 1)]), ())

    def test_degenerate_rectangle_retains_isolated_contacts(self):
        arcs = circle_safe_angular_intervals((0, 0), 1., [(0, 0, -2, 2)])
        np.testing.assert_allclose(arcs, ((np.pi/2, np.pi/2), (3*np.pi/2, 3*np.pi/2)))

    def test_disconnected_intervals(self):
        arcs = circle_safe_angular_intervals((0, 0), 1., [(-2, 2, -.5, .5)])
        self.assertEqual(len(arcs), 2)
        np.testing.assert_allclose(arcs, ((5*np.pi/6, 7*np.pi/6), (11*np.pi/6, 13*np.pi/6)))

    def test_random_membership_matches_rectangles(self):
        rng = np.random.default_rng(42)
        angles = rng.uniform(0, 2*np.pi, 400)
        for _ in range(80):
            center = rng.uniform(-3, 3, 2)
            radius = rng.uniform(.2, 3)
            raw = np.sort(rng.uniform(-5, 5, (2, 2, 2)), axis=2)
            boxes = [tuple(x.ravel()) for x in raw]
            arcs = circle_safe_angular_intervals(center, radius, boxes)
            points = center+radius*np.column_stack((np.cos(angles), np.sin(angles)))
            expected = np.array([any(a <= p[0] <= b and c <= p[1] <= d for a,b,c,d in boxes) for p in points])
            actual = np.array([any(a <= t <= b or a <= t+2*np.pi <= b for a,b in arcs) for t in angles])
            np.testing.assert_array_equal(actual, expected)

    def test_coincident_circles_use_separate_eroded_pairs(self):
        circles = tuple(SimpleNamespace(waypoint_index=j, center=np.zeros(2), radius=2.) for j in (0, 1))
        bounds = ((-4, 4, 1, 4), (10, 12, 10, 12), (-4, 4, -4, -1))
        baseline = SimpleNamespace(feasibility=SimpleNamespace(corridor_bounds=bounds))
        results = compute_circle_safe_arcs(SimpleNamespace(circles=circles), baseline,
                                          SimpleNamespace(r=.5, R=2.), include_certified_fillet=False)
        self.assertNotEqual(results[0].intervals, results[1].intervals)
        self.assertEqual(results[0].eroded_corridors[0], (-3.5, 3.5, 1.5, 3.5))
        self.assertEqual(results[1].circle_index, 1)

    def test_certified_quarter_bridges_gap_in_both_turn_directions(self):
        from kappa_planner.refinement import _merge_circle_intervals
        raw = np.radians(((47., 132.), (138., 222.)))
        np.testing.assert_allclose(_merge_circle_intervals((*raw, (np.pi/2, np.pi))),
                                   np.radians(((47., 222.),)))
        raw = np.radians(((227., 312.), (318., 402.)))
        np.testing.assert_allclose(_merge_circle_intervals((*raw, (3*np.pi/2, 2*np.pi))),
                                   np.radians(((227., 402.),)))
        # The larger complementary gap is still excluded.
        self.assertLess(_merge_circle_intervals((*raw, (3*np.pi/2, 2*np.pi)))[0][1], 3*np.pi)


if __name__ == '__main__':
    unittest.main()
