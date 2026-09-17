"""Regression checks for tangents between circles separated by exactly 2R."""
import math
import unittest

from kappa_planner.helpers.primitives import compute_extreme_poses_arc_line


class TouchingCircleTangentsTest(unittest.TestCase):
    def test_equal_turns_keep_external_tangent(self):
        for radius in (1.0, 2.0):
            for turn in (-1, 1):
                with self.subTest(radius=radius, turn=turn):
                    x1, y1, heading, x2, y2, _ = compute_extreme_poses_arc_line(
                        0, 0, 0, 2 * radius, turn, turn, radius)
                    self.assertAlmostEqual(math.hypot(x2 - x1, y2 - y1), 2 * radius)
                    self.assertAlmostEqual(x1, turn * radius)
                    self.assertAlmostEqual(y1, 0)
                    self.assertAlmostEqual(x2, turn * radius)
                    self.assertAlmostEqual(y2, 2 * radius)
                    self.assertAlmostEqual(heading, math.pi / 2)

    def test_opposite_turns_share_touching_point(self):
        for turn in (-1, 1):
            with self.subTest(turn=turn):
                x1, y1, heading, x2, y2, _ = compute_extreme_poses_arc_line(
                    0, 0, 0, 4, turn, -turn, 2)
                self.assertAlmostEqual(x1, 0)
                self.assertAlmostEqual(y1, 2)
                self.assertAlmostEqual(x2, x1)
                self.assertAlmostEqual(y2, y1)
                self.assertAlmostEqual(heading, (math.pi / 2 + turn * math.pi / 2) % (2 * math.pi))


if __name__ == '__main__':
    unittest.main()
