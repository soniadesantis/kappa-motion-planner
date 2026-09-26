"""Check disk-erosion clearance against analytic union geometries."""

import runpy
import unittest
from pathlib import Path

import numpy as np


EXAMPLE = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "experiments/bicycle_journal_paper/examples_maps_polyline.py"
))
C = EXAMPLE["CorridorWorld"]
CLEARANCE = EXAMPLE["union_signed_clearance"]


class UnionErosionTest(unittest.TestCase):
    def test_single_rectangle_and_duplicate(self):
        rectangle = C(2, 4, [0, 0], 0)
        points = [[0, 0], [1.8, 0], [0, 1], [3, 2]]
        expected = [1, 0.2, 0, -np.sqrt(2)]
        np.testing.assert_allclose(CLEARANCE(points, [rectangle]), expected, atol=1e-9)
        np.testing.assert_allclose(CLEARANCE(points, [rectangle, rectangle]),
                                   expected, atol=1e-9)

    def test_shared_seam_is_not_a_union_boundary(self):
        rectangles = [C(2, 2, [-1, 0], 0), C(2, 2, [1, 0], 0)]
        # The common edge x=0 is inside the union, not distance zero.
        np.testing.assert_allclose(CLEARANCE([[0, 0], [0, 0.8]], rectangles),
                                   [1, 0.2], atol=1e-9)

    def test_concave_corner_produces_circular_erosion(self):
        rectangles = [C(2, 6, [-2, 0], 0), C(2, 6, [0, 2], np.pi/2)]
        points = np.array([[-0.8, 0.8], [-1.2, 1.2], [-2, 0], [0, 2]])
        expected = [np.sqrt(0.08), -0.2, 1, 1]
        np.testing.assert_allclose(CLEARANCE(points, rectangles), expected, atol=1e-9)
        # This first point survives erosion by 0.25 even though it is only
        # 0.2 from each rectangle's internal edge: separate erosion is wrong.
        self.assertGreater(CLEARANCE(points[:1], rectangles)[0], 0.25)

    def test_hole_boundary_is_retained(self):
        frame = [C(4, 1, [0.5, 2], 0), C(4, 1, [3.5, 2], 0),
                 C(1, 4, [2, 0.5], 0), C(1, 4, [2, 3.5], 0)]
        np.testing.assert_allclose(
            CLEARANCE([[2, 2], [0.5, 2], [0.9, 0.9]], frame),
            [-1, 0.5, np.sqrt(0.02)], atol=1e-9,
        )

    def test_rotated_union_has_same_clearance(self):
        rectangles = [C(2, 6, [-2, 0], 0), C(2, 6, [0, 2], np.pi/2)]
        points = np.array([[-0.8, 0.8], [-1.2, 1.2], [-2, 0], [0, 2]])
        angle = 0.37
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        shift = np.array([10, -7])
        rotated = [C(c.width, c.height, rotation @ c.center + shift, c.tilt + angle)
                   for c in rectangles]
        np.testing.assert_allclose(
            CLEARANCE(points @ rotation.T + shift, rotated),
            CLEARANCE(points, rectangles), atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
