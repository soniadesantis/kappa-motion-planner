"""Opposite-circle repair, certified-bp restoration, and rollback."""
import unittest

import numpy as np

from test_bp_circle_diagnostics import fixture
from test_bp_circle_sequence import EXAMPLE
from kappa_planner.helpers.bp_circle_diagnostics import analyze_bp_circles
from kappa_planner.helpers.bp_circle_sequence import build_bp_circle_sequence
from kappa_planner.helpers.bp_opposite_circle_shifting import shift_opposite_turn_circles


def setup():
    p, g = fixture([([-.9, 0], [1, 0], [0, 1]),
                    ([.9, 0], [0, 1], [1, 0])])
    p['circle_slots'] = list(p['sequence'])
    g['polyline'] = np.array([[-3., -1.], [0., -1.], [0., 1.], [3., 1.]])
    p['diagnostics'] = analyze_bp_circles(p, g)
    return p, g


class OppositeShiftingTest(unittest.TestCase):
    def test_radial_touch_and_idempotence(self):
        p, g = setup()
        result = shift_opposite_turn_circles(p, g)
        self.assertEqual(result['accepted'], 1)
        self.assertEqual(result['steps'][0]['method'], 'radial')
        self.assertEqual(p['diagnostics']['flags']['opposite_turn_overlap'], [])
        self.assertAlmostEqual(p['sequence'][0].center.x, -1.)
        self.assertAlmostEqual(p['sequence'][1].center.x, 1.)
        self.assertEqual(shift_opposite_turn_circles(p, g)['accepted'], 0)

    def test_inadmissible_candidates_are_rolled_back(self):
        p, g = setup()
        for c in p['diagnostics']['circles']:
            region = g['smoothed_check']['regions'][c['bp_vertex']]
            region['low'] = c['implied_vertex'].copy()
            region['high'] = c['implied_vertex'].copy()
        result = shift_opposite_turn_circles(p, g)
        self.assertEqual(result['rejected'], 1)
        self.assertEqual(p['diagnostics']['flags']['opposite_turn_overlap'], [(0, 1)])
        self.assertEqual(p['sequence'][0].center.x, -.9)
        self.assertEqual(p['sequence'][0].xc, -.9)

    def test_coincident_opposite_centers_have_baseline_fallback(self):
        p, g = setup()
        for circle in p['sequence']:
            circle.center.x = 0.
        p['diagnostics'] = analyze_bp_circles(p, g)
        self.assertEqual(shift_opposite_turn_circles(p, g)['accepted'], 1)
        self.assertEqual(p['diagnostics']['flags']['opposite_turn_overlap'], [])

    def test_example_35_restores_bp_centers_and_all_examples_keep_constraints(self):
        total = 0
        for number in EXAMPLE['EXAMPLE_NUMBERS']:
            c, s, e, v = EXAMPLE['example_corridor_sequence'](number)
            g = EXAMPLE['build_trajectory_geometry'](
                c, v.width/2, v.max_radius, start_pose=s, end_pose=e)
            bp = g['polyline'].copy() if g['feasible'] else None
            report = build_bp_circle_sequence(c, v, g, shift_same_turn=True,
                                              shift_opposite_turn=True)
            if not g['feasible']:
                continue
            shift = report['opposite_shifting']
            total += shift['accepted']
            self.assertEqual(shift['rejected'], 0, number)
            self.assertEqual(report['diagnostics']['flags']['opposite_turn_overlap'], [], number)
            self.assertEqual(report['diagnostics']['flags']['outside_Aj'],
                             shift['initial_diagnostics']['flags']['outside_Aj'], number)
            self.assertEqual(report['diagnostics']['flags']['same_turn_arc_conflict'], [], number)
            np.testing.assert_array_equal(g['polyline'], bp)
            if number == 35:
                self.assertEqual(shift['steps'][0]['method'], 'bp_both_full')
                for circle in report['sequence']:
                    region = g['smoothed_check']['regions'][circle.bp_vertex_index]
                    expected = bp[circle.bp_vertex_index]-circle.radius*region['incoming']+circle.radius*region['outgoing']
                    np.testing.assert_allclose([circle.center.x, circle.center.y], expected)
                    self.assertEqual(circle.xc, circle.center.x)
        self.assertEqual(total, 1)


if __name__ == '__main__':
    unittest.main()
