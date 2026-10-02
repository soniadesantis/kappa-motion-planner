"""Consecutive overlap diagnostics, including aligned alternatives."""
import unittest
from types import SimpleNamespace
import numpy as np

from kappa_planner.refinement import find_consecutive_circle_overlaps


def circle(j, x, y=0.):
    return SimpleNamespace(waypoint_index=j, center=np.array([x, y]), radius=1.)


class ConsecutiveOverlapTest(unittest.TestCase):
    def test_blocks_and_missing_waypoints(self):
        circles = [circle(0, 0), circle(2, 1), circle(3, 2),
                   circle(4, 8), circle(5, 9)]
        self.assertEqual(find_consecutive_circle_overlaps(circles),
                         (((0, 1), (1, 2), (3, 4)), ((0, 1, 2), (3, 4))))

    def test_tangency_coincidence_and_tolerance(self):
        self.assertEqual(find_consecutive_circle_overlaps([circle(0, 0), circle(1, 2)]), ((), ()))
        self.assertEqual(find_consecutive_circle_overlaps([circle(0, 0), circle(1, 0)]),
                         (((0, 1),), ((0, 1),)))
        self.assertEqual(find_consecutive_circle_overlaps([circle(0, 0), circle(1, 2-1e-10)]), ((), ()))

    def test_alternatives_compared_to_both_neighbors(self):
        circles = [circle(0, 0), circle(1, 1), circle(1, 10), circle(2, 11)]
        self.assertEqual(find_consecutive_circle_overlaps(circles),
                         (((0, 1), (2, 3)), ((0, 1), (2, 3))))
        self.assertEqual(find_consecutive_circle_overlaps([circle(1, 0), circle(1, 0)]), ((), ()))

    def test_nonconsecutive_overlap_ignored(self):
        self.assertEqual(find_consecutive_circle_overlaps(
            [circle(0, 0), circle(1, 10), circle(2, 0)]), ((), ()))
        self.assertEqual(find_consecutive_circle_overlaps([]), ((), ()))
        with self.assertRaises(ValueError):
            find_consecutive_circle_overlaps([circle(2, 0), circle(1, 0)])


if __name__ == '__main__':
    unittest.main()
