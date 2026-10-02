"""Centered-overlap validation, independent of fillet turning radius."""

import unittest

import numpy as np

from kappa_planner.corridor import CorridorWorld as C
from kappa_planner.helpers.centroid_polyline_validity import (
    check_centroid_polyline,
    check_orthogonal_polyline,
)


class CentroidPolylineValidityTest(unittest.TestCase):
    def test_centered_sequence_at_exact_diameter_threshold(self):
        corridors = [C(2, 6, [0, 0], 0), C(2, 6, [2, 2], np.pi/2),
                     C(2, 6, [2, 4], 0), C(2, 6, [4, 4], np.pi/2)]
        points, report = check_centroid_polyline(corridors, r=1)
        self.assertTrue(report["valid"], report["errors"])
        np.testing.assert_allclose(points, [[2, 0], [2, 4], [4, 4]], atol=1e-9)
        self.assertEqual(report["turn_directions"][0]["tau"], -1)
        self.assertEqual([item["centerline"] for item in report["corridors"]],
                         ["horizontal", "vertical", "horizontal", "vertical"])

    def test_off_center_overlap_returns_no_polyline(self):
        points, report = check_centroid_polyline(
            [C(2, 6, [0, 0], 0), C(2, 6, [1, 1], 0)], r=0.1
        )
        self.assertEqual(points.shape, (0, 2))
        self.assertFalse(report["valid"])
        self.assertTrue(report["overlap_dimensions_ok"])
        self.assertFalse(report["all_overlap_centroids_on_both_centerlines"])
        np.testing.assert_allclose(report["overlaps"][0]["centroid"], [0.5, 0.5])

    def test_square_can_choose_either_axis_but_must_be_consistent(self):
        first = C(2, 6, [-2, 0], 0)
        square = C(2, 2, [0, 0], 0)
        points, report = check_centroid_polyline([first, square], r=0.1)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(points.shape, (1, 2))
        points, report = check_centroid_polyline(
            [C(2, 2, [-1, 0], 0), square, C(2, 2, [0, 1], 0)], r=0.1
        )
        self.assertFalse(report["valid"])
        self.assertEqual(points.shape, (0, 2))
        self.assertIsNone(report["corridors"][1]["centerline"])

    def test_small_dimensions_missing_overlap_and_rotation(self):
        cases = [
            ([C(1, 4, [0, 0], 0), C(1, 4, [1, 0], 0)], 0.6),
            ([C(2, 4, [0, 0], 0), C(2, 4, [3.5, 0], 0)], 0.3),
            ([C(2, 4, [0, 0], 0), C(2, 4, [4, 0], 0)], 0),
            ([C(2, 4, [0, 0], 0), C(2, 4, [0, 0], 0.1)], 0.1),
        ]
        for corridors, radius in cases:
            with self.subTest(radius=radius, centers=[c.center for c in corridors]):
                points, report = check_centroid_polyline(corridors, radius)
                self.assertFalse(report["valid"])
                self.assertEqual(points.shape, (0, 2))
                self.assertTrue(report["errors"])

    def test_degenerate_centroids_are_reported_for_simplification(self):
        rectangle = C(2, 6, [0, 0], 0)
        points, report = check_centroid_polyline([rectangle]*4, 0.1)
        self.assertTrue(report["valid"])
        self.assertEqual(points.shape, (3, 2))
        self.assertTrue(report["turn_directions"][0]["degenerate"])

    def test_invalid_arguments(self):
        rectangle = C(2, 6, [0, 0], 0)
        for radius in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                check_centroid_polyline([rectangle]*2, radius)
        with self.assertRaises(ValueError):
            check_centroid_polyline([rectangle], 0.1)


class CommonRangePolylineTest(unittest.TestCase):
    @staticmethod
    def rectangle(xmin, xmax, ymin, ymax):
        return C(ymax-ymin, xmax-xmin, [(xmin+xmax)/2, (ymin+ymax)/2], 0)

    def test_common_midpoint_propagates_across_entire_run(self):
        corridors = [self.rectangle(0, 3, 0, 1), self.rectangle(2, 6, 0, 1.5),
                     self.rectangle(5, 9, 0.5, 2.2), self.rectangle(8, 11, 0.8, 2.2)]
        points, report = check_orthogonal_polyline(corridors, 0.1)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["waypoint_method"], "common_range_midpoints")
        np.testing.assert_allclose(points, [[2.5, 0.9], [5.5, 0.9], [8.5, 0.9]])
        np.testing.assert_allclose(report["alignment_runs"][0]["common_range"], [0.8, 1])

    def test_pairwise_intersections_are_not_enough_for_a_run(self):
        corridors = [self.rectangle(0, 3, 0, 1), self.rectangle(2, 6, 0, 1.5),
                     self.rectangle(5, 9, 0.5, 2.2), self.rectangle(8, 11, 1.2, 2.2)]
        points, report = check_orthogonal_polyline(corridors, 0.1)
        self.assertFalse(report["valid"])
        self.assertEqual(points.shape, (0, 2))
        self.assertIn("globally consistent", report["errors"][0])

    def test_centered_centroids_stay_unchanged(self):
        corridors = [C(2, 6, [0, 0], 0), C(2, 6, [2, 2], np.pi/2),
                     C(2, 6, [2, 4], 0)]
        original, _ = check_centroid_polyline(corridors, 0.2)
        points, report = check_orthogonal_polyline(corridors, 0.2)
        np.testing.assert_array_equal(points, original)
        self.assertEqual(report["waypoint_method"], "centroids")

    def test_dimensions_are_still_required(self):
        points, report = check_orthogonal_polyline(
            [C(1, 4, [0, 0], 0), C(1, 4, [1, 0], 0)], 0.6
        )
        self.assertFalse(report["valid"])
        self.assertEqual(points.shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
