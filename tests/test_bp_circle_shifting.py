"""Allocation, rollback, state consistency, and example-wide shift regressions."""
import unittest

import numpy as np

from test_bp_circle_diagnostics import fixture
from test_bp_circle_sequence import EXAMPLE
from kappa_planner.helpers.bp_circle_diagnostics import analyze_bp_circles
from kappa_planner.helpers.bp_circle_sequence import build_bp_circle_sequence
from kappa_planner.helpers.bp_circle_shifting import shift_same_turn_circles


def setup(left=False, right=False):
    p, g = fixture([([0, 0], [1, 0], [0, 1]),
                    ([0, -.2], [0, 1], [-1, 0])])
    for c in p['sequence']:
        c.index += 1
    p['circle_slots'] = [None, *p['sequence'], None]
    g['internal_check'] = dict(spacing_pairs=[
        dict(overlaps=(0, 1), status='coupled' if left else 'guaranteed'),
        dict(overlaps=(2, 3), status='coupled' if right else 'guaranteed')])
    p['diagnostics'] = analyze_bp_circles(p, g)
    return p, g


class SameTurnShiftingTest(unittest.TestCase):
    def test_neighbor_allocations_and_common_coordinate(self):
        for left, right, y in ((False, False, -.1), (True, False, 0.),
                               (False, True, -.2), (True, True, -.1)):
            with self.subTest(left=left, right=right):
                p, g = setup(left, right)
                result = shift_same_turn_circles(p, g)
                self.assertEqual(result['accepted'], 1)
                for c in p['sequence']:
                    np.testing.assert_allclose([c.center.x, c.center.y], [0, y])
                    self.assertEqual(c.xc, c.center.x)
                    self.assertEqual(c.yc, c.center.y)
                self.assertTrue(p['diagnostics']['pairs'][0]['shared_join'])
                self.assertEqual(p['diagnostics']['flags']['same_turn_arc_conflict'], [])
                # Re-running does not shift an already corrected pair.
                self.assertEqual(shift_same_turn_circles(p, g)['accepted'], 0)

    def test_Aj_violation_rolls_back(self):
        p, g = setup(True, True)
        g['smoothed_check']['regions'][1]['low'][1] = -1.05
        p['diagnostics'] = analyze_bp_circles(p, g)
        result = shift_same_turn_circles(p, g)
        self.assertEqual(result['rejected'], 1)
        self.assertEqual(result['steps'][0]['new_flags']['outside_Aj'], [1])
        self.assertEqual(p['sequence'][0].center.y, 0.)
        self.assertEqual(p['sequence'][1].center.y, -.2)

    def test_unaligned_pair_is_explicitly_unresolved(self):
        p, g = setup()
        p['sequence'][1].center.x = .001
        p['diagnostics'] = analyze_bp_circles(p, g)
        result = shift_same_turn_circles(p, g)
        self.assertEqual(result['accepted'], 0)
        self.assertEqual(result['rejected'], 1)
        self.assertIn('aligned', result['steps'][0]['reason'])

    def test_new_opposite_neighbor_overlap_rolls_back(self):
        p, g = fixture([([-1.999, -.1], [0, 1], [1, 0]),
                        ([0, 0], [1, 0], [0, 1]),
                        ([0, -.2], [0, 1], [-1, 0])])
        p['circle_slots'] = list(p['sequence'])
        g['internal_check'] = dict(spacing_pairs=[
            dict(overlaps=(0, 1), status='guaranteed')])
        p['diagnostics'] = analyze_bp_circles(p, g)
        result = shift_same_turn_circles(p, g)
        self.assertEqual(result['rejected'], 1)
        self.assertEqual(result['steps'][0]['new_flags']['opposite_turn_overlap'], [(0, 1)])
        self.assertEqual(p['diagnostics']['flags']['opposite_turn_overlap'], [])

    def test_all_examples_preserve_bp_and_resolve_ten_pairs(self):
        accepted = 0
        for number in EXAMPLE['EXAMPLE_NUMBERS']:
            c, s, e, v = EXAMPLE['example_corridor_sequence'](number)
            g = EXAMPLE['build_trajectory_geometry'](
                c, v.width/2, v.max_radius, start_pose=s, end_pose=e)
            original = None if not g['feasible'] else g['polyline'].copy()
            p = build_bp_circle_sequence(c, v, g, shift_same_turn=True)
            if not g['feasible']:
                continue
            result = p['shifting']
            accepted += result['accepted']
            self.assertEqual(result['rejected'], 0, number)
            self.assertEqual(p['diagnostics']['flags']['same_turn_arc_conflict'], [], number)
            for flag in ('outside_Aj', 'opposite_turn_overlap'):
                self.assertEqual(p['diagnostics']['flags'][flag],
                                 result['initial_diagnostics']['flags'][flag], number)
            np.testing.assert_array_equal(g['polyline'], original)
            for circle in p['sequence']:
                self.assertIs(p['circle_slots'][circle.index], circle)
        self.assertEqual(accepted, 10)


if __name__ == '__main__':
    unittest.main()
