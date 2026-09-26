"""Geometry guarantees for the auxiliary-corridor example construction."""

import runpy
import unittest
from pathlib import Path

import numpy as np


EXAMPLE = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "experiments/bicycle_journal_paper/examples_maps_polyline.py"
))


def bounds(corridor):
    corners = np.asarray(corridor.corners)
    return corners.min(axis=0), corners.max(axis=0)


def intersection_area(*rectangles):
    boxes = [bounds(rectangle) for rectangle in rectangles]
    low = np.max([box[0] for box in boxes], axis=0)
    high = np.min([box[1] for box in boxes], axis=0)
    return np.prod(np.maximum(high - low, 0))


class OrthogonalPolylineTest(unittest.TestCase):
    def test_offset_rectangles_need_only_one_bridge(self):
        rectangle = EXAMPLE["CorridorWorld"]
        original = [rectangle(2, 6, [0, 0], 0), rectangle(2, 6, [1, 1], 0)]
        sequence, auxiliary = EXAMPLE["build_orthogonal_corridor_sequence"](original)
        self.assertEqual(auxiliary, [1])
        self.assertEqual(len(sequence), 3)
        self.assertAlmostEqual(sequence[1].tilt, np.pi / 2)
        np.testing.assert_allclose(bounds(sequence[1]), [[-2, -1], [3, 2]])

    def test_nominal_examples_and_returned_trajectory(self):
        accepted = 0
        for number in range(1, 35):
            with self.subTest(example=number):
                corridors, _, _, vehicle = EXAMPLE["example_corridor_sequence"](number)
                original = [np.copy(c.corners) for c in corridors]
                result = EXAMPLE["build_trajectory_geometry"](
                    corridors, vehicle.width/2, vehicle.max_radius
                )
                check = result["check"]
                if not check["feasible"]:
                    self.assertIsNone(result["polyline"])
                    self.assertFalse(result["fillets"])
                    self.assertTrue(check["reason"])
                    continue
                accepted += 1
                points = result["polyline"]
                smoothed = result['smoothed_check']
                if smoothed['feasible']:
                    np.testing.assert_array_equal(points, smoothed['points'])
                    self.assertTrue(all(value >= vehicle.width/2-1e-8
                                        for value in smoothed['arc_clearances']))
                    for j, region in enumerate(smoothed['regions']):
                        if region is not None:
                            self.assertGreaterEqual(EXAMPLE['region_margin'](points[j], region), -1e-8)
                    for arc in result['arcs']:
                        self.assertTrue(np.all(EXAMPLE['union_signed_clearance'](arc, corridors)
                                               >= vehicle.width/2-1e-8))
                else:
                    np.testing.assert_array_equal(points, check['polyline'])
                    self.assertFalse(result['fillets'])
                    self.assertFalse(result['arcs'])
                self.assertEqual(len(points), len(corridors)-1)
                for i, point in enumerate(points):
                    for corridor in corridors[i:i+2]:
                        self.assertTrue(np.all(
                            np.r_[point, 1] @ corridor.W <= -vehicle.width/2+1e-8
                        ))
                delta = np.diff(points, axis=0)
                self.assertTrue(np.all(np.min(abs(delta), axis=1) <= 1e-8))
                self.assertTrue(np.all(np.linalg.norm(delta, axis=1)
                                       >= 2*vehicle.max_radius-1e-8))
                np.testing.assert_allclose(abs(result["turn_angles"]), np.pi/2,
                                           atol=1e-8, rtol=0)
                for corridor, corners in zip(corridors, original):
                    np.testing.assert_array_equal(corridor.corners, corners)
        self.assertGreater(accepted, 0)

    def test_oblique_corridors_are_not_silently_snapped(self):
        for number in (3, 14):
            corridors, _, _, vehicle = EXAMPLE["example_corridor_sequence"](number)
            # Keep an explicitly oblique fixture now that both maps are aligned.
            corridor = corridors[-1]
            center = np.asarray(corridor.center)
            angle = .001
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            corridor.corners = (np.asarray(corridor.corners)-center) @ rotation.T + center
            result = EXAMPLE["build_trajectory_geometry"](
                corridors, vehicle.width/2, vehicle.max_radius
            )
            self.assertEqual(result["check"]["status"], "unsupported_geometry")
            self.assertIsNone(result["polyline"])

    def test_ambiguous_maps_report_all_intersecting_safe_doors(self):
        for number, indices in ((27, [2, 4]), (28, [4, 6, 8]), (32, [8])):
            with self.subTest(example=number):
                corridors, _, _, vehicle = EXAMPLE['example_corridor_sequence'](number)
                result = EXAMPLE['check_orthogonal_polyline'](
                    corridors, vehicle.width/2, vehicle.max_radius)
                self.assertEqual(result['status'], 'ambiguous_direction')
                self.assertEqual([d['corridor'] for d in result['ambiguous_overlaps']],
                                 indices)
                for diagnostic in result['ambiguous_overlaps']:
                    self.assertTrue(diagnostic['positive_area'])
                    a, b = [result['doors'][j] for j in diagnostic['overlaps']]
                    expected = (max(a['x'][0], b['x'][0]), min(a['x'][1], b['x'][1]),
                                max(a['y'][0], b['y'][0]), min(a['y'][1], b['y'][1]))
                    np.testing.assert_allclose(diagnostic['safe_intersection'], expected)

    def test_collinear_backtracking_and_duplicates_are_removed(self):
        points = np.array([[0., 0.], [2., 0.], [1., 0.], [1., 0.], [1., 2.]])
        simplified, kept = EXAMPLE["simplify_orthogonal_polyline"](points)
        np.testing.assert_allclose(simplified, [[0., 0.], [1., 0.], [1., 2.]])
        np.testing.assert_array_equal(simplified, points[kept])



if __name__ == "__main__":
    unittest.main()
