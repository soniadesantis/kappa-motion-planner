"""Nominal-class rejection, interval reachability, and witness guarantees."""

import itertools
import unittest

import numpy as np

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.nominal_polyline import (
    check_orthogonal_polyline, _propagate_coordinate, _backtrack_coordinate,
)


def rectangle(a, b, c=0, d=2):
    return CorridorWorld(d-c, b-a, [(a+b)/2, (c+d)/2], 0)


class NominalPolylineTest(unittest.TestCase):
    def test_alternating_witness(self):
        corridors = [rectangle(0, 2), rectangle(0, 6),
                     rectangle(4, 6, 0, 6), rectangle(4, 10, 4, 6)]
        result = check_orthogonal_polyline(corridors, .25, 1)
        self.assertTrue(result['feasible'])
        self.assertEqual(result['directions'], ['H', 'V'])
        delta = np.diff(result['polyline'], axis=0)
        self.assertAlmostEqual(delta[0, 1], 0)
        self.assertAlmostEqual(delta[1, 0], 0)
        self.assertTrue(np.all(np.linalg.norm(delta, axis=1) >= 2))

    def test_point_doors_and_exact_length(self):
        corridors = [rectangle(0, 2), rectangle(0, 4), rectangle(2, 4)]
        result = check_orthogonal_polyline(corridors, 1, 1)
        self.assertTrue(result['feasible'])
        np.testing.assert_allclose(result['polyline'], [[1, 1], [3, 1]])
        self.assertEqual(check_orthogonal_polyline(corridors, 1, 1.01)['status'],
                         'spacing')
        self.assertEqual(check_orthogonal_polyline(corridors, 1.01, 1)['status'],
                         'empty_overlap')

    def test_nominal_rejections(self):
        cases = [
            ([rectangle(0, 4)]*3, 'ambiguous_direction'),
            ([rectangle(0, 2), rectangle(0, 5), rectangle(4, 9),
              rectangle(8, 10)], 'nonalternating'),
            ([rectangle(0, 2), rectangle(0, 6, 0, 6),
              rectangle(4, 6, 4, 6)], 'no_orthogonal_connection'),
            ([rectangle(0, 2), rectangle(3, 5), rectangle(4, 6)],
             'empty_overlap'),
        ]
        for corridors, status in cases:
            with self.subTest(status=status):
                result = check_orthogonal_polyline(corridors, .1, .1)
                self.assertEqual(result['status'], status)
                self.assertIsNone(result['polyline'])

    def test_ambiguous_overlap_intersections(self):
        for last, expected, positive_area in (
            (rectangle(1, 3, 1, 3), (1, 2, 1, 2), True),
            (rectangle(2, 4, 0, 2), (2, 2, 0, 2), False),
            (rectangle(2, 4, 2, 4), (2, 2, 2, 2), False),
        ):
            with self.subTest(intersection=expected):
                result = check_orthogonal_polyline(
                    [rectangle(0, 2, 0, 2), rectangle(0, 4, 0, 4), last], 0, .1)
                self.assertEqual(result['status'], 'ambiguous_direction')
                self.assertIsNone(result['polyline'])
                diagnostic, = result['ambiguous_overlaps']
                self.assertEqual(diagnostic['corridor'], 1)
                self.assertEqual(diagnostic['overlaps'], (0, 1))
                np.testing.assert_allclose(diagnostic['safe_intersection'], expected)
                self.assertEqual(diagnostic['positive_area'], positive_area)

    def test_raw_overlap_contact_can_disappear_after_erosion(self):
        result = check_orthogonal_polyline(
            [rectangle(0, 2), rectangle(0, 4), rectangle(2, 4)], .1, .1)
        self.assertEqual(result['directions'], ['H'])
        self.assertEqual(result['ambiguous_overlaps'], [])

    def test_split_and_gap_are_preserved(self):
        reachable = _propagate_coordinate([(0, 2), (0, 2), (.9, 1.1)],
                                           ['separate', 'equal'], 1.5)
        self.assertEqual(reachable[1], [(0, .5), (1.5, 2)])
        self.assertEqual(reachable[2], [])

    def test_coordinate_propagation_against_exhaustive_integer_grid(self):
        # Integer endpoints and spacing make integer witnesses sufficient.
        domains = [(0, 1), (0, 3), (2, 4)]
        for intervals in itertools.product(domains, repeat=3):
            for relations in itertools.product(['equal', 'separate'], repeat=2):
                witnesses = [values for values in itertools.product(
                    *(range(a, b+1) for a, b in intervals)
                ) if all((values[j] == values[j+1] if relation == 'equal'
                          else abs(values[j]-values[j+1]) >= 2)
                         for j, relation in enumerate(relations))]
                reachable = _propagate_coordinate(intervals, relations, 2)
                self.assertEqual(bool(reachable[-1]), bool(witnesses))
                values = _backtrack_coordinate(reachable, relations, 2)
                if values is None:
                    continue
                for value, (a, b) in zip(values, intervals):
                    self.assertTrue(a <= value <= b)
                for j, relation in enumerate(relations):
                    if relation == 'equal':
                        self.assertEqual(values[j], values[j+1])
                    else:
                        self.assertGreaterEqual(abs(values[j]-values[j+1]), 2)

    def test_invalid_arguments(self):
        for r, R in [(-1, 1), (0, 0), (0, float('nan'))]:
            with self.assertRaises(ValueError):
                check_orthogonal_polyline([rectangle(0, 2)]*3, r, R)
        with self.assertRaises(ValueError):
            check_orthogonal_polyline([rectangle(0, 2)]*2, 0, 1)


if __name__ == '__main__':
    unittest.main()
