"""Local safe-door spacing diagnostics, including failure and tolerance cases."""

import unittest

import numpy as np

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.nominal_polyline import (
    check_orthogonal_polyline, classify_door_spacing,
)
from kappa_planner.helpers.endpoint_polyline import extend_nominal_with_pose_coordinates


class SpacingCouplingTest(unittest.TestCase):
    def test_longitudinal_extrema_match_aligned_choices(self):
        for direction in ("H", "V"):
            for reverse in (False, True):
                for interval, expected in (((3, 4), "guaranteed"),
                                           ((2, 4), "coupled"),
                                           ((1.1, 1.5), "impossible")):
                    axis = "x" if direction == "H" else "y"
                    other = "y" if direction == "H" else "x"
                    doors = [{axis: (0, 1), other: (0, 2)},
                             {axis: interval, other: (1, 3)}]
                    if reverse:
                        doors.reverse()
                    pair = classify_door_spacing(doors, [direction], 1)[0]
                    self.assertEqual(pair['status'], expected)
                    a = np.linspace(*doors[0][axis], 101)
                    b = np.linspace(*doors[1][axis], 101)
                    lengths = np.abs(a[:, None] - b[None, :])
                    self.assertAlmostEqual(pair['minimum_length'], lengths.min())
                    self.assertAlmostEqual(pair['maximum_length'], lengths.max())

    def test_exact_threshold_and_tolerance(self):
        doors = [dict(x=(0, 0), y=(0, 1)), dict(x=(2, 2), y=(0, 1))]
        self.assertEqual(classify_door_spacing(doors, ['H'], 1)[0]['status'],
                         'guaranteed')
        doors[1]['x'] = (2-5e-10, 2-5e-10)
        self.assertEqual(classify_door_spacing(doors, ['H'], 1)[0]['status'],
                         'guaranteed')
        self.assertEqual(classify_door_spacing(doors, ['H'], 1, tol=0)[0]['status'],
                         'impossible')

    def test_unclassified_is_not_guaranteed(self):
        door = dict(x=(0, 1), y=(0, 1))
        for first, second, direction in ((None, door, 'UNAVAILABLE'),
                                         (door, door, 'AMBIGUOUS'),
                                         (door, door, 'INFEASIBLE')):
            pair = classify_door_spacing([first, second], [direction], 1)[0]
            self.assertEqual(pair['status'], 'unavailable')
            self.assertIsNone(pair['minimum_length'])

    def test_rejected_nominal_still_has_pair_diagnostics(self):
        corridors = [CorridorWorld(2, 2, [x, 1], 0) for x in (1, 2, 3)]
        result = check_orthogonal_polyline(corridors, .1, 2)
        self.assertEqual(result['status'], 'spacing')
        self.assertEqual(result['spacing_pairs'][0]['status'], 'impossible')
        self.assertEqual(result['spacing_pairs'][0]['overlaps'], (0, 1))
        self.assertEqual(result['spacing_pairs'][0]['corridor'], 1)

    def test_endpoint_extension_preserves_original_pair_indices(self):
        corridors = [CorridorWorld(2, 4, [2, 1], 0),
                     CorridorWorld(8, 2, [3, 4], 0),
                     CorridorWorld(2, 4, [4, 7], 0)]
        nominal = check_orthogonal_polyline(corridors, .1, .25)
        self.assertTrue(nominal['feasible'])
        extended = extend_nominal_with_pose_coordinates(
            nominal, corridors, [.5, 1, 0], [5.5, 7, 0], .1, .25,
            exact_start=True, exact_end=True)
        self.assertTrue(extended['feasible'])
        self.assertEqual(len(extended['polyline']), len(nominal['doors'])+2)
        self.assertEqual(extended['spacing_pairs'], nominal['spacing_pairs'])
        self.assertEqual(extended['spacing_pairs'][0]['overlaps'], (0, 1))


if __name__ == '__main__':
    unittest.main()
