"""Analytic relevant-quarter checks and center-to-A_j mapping."""

from types import SimpleNamespace
import unittest

import numpy as np

from kappa_planner.geometry import Point
from kappa_planner.helpers.bp_circle_diagnostics import analyze_bp_circles


def fixture(specifications):
    circles, regions = [], [None]
    for i, (center, incoming, outgoing) in enumerate(specifications):
        u, v = np.array(incoming, dtype=float), np.array(outgoing, dtype=float)
        regions.append(dict(incoming=u, outgoing=v, low=np.array([-10., -10.]),
                            high=np.array([10., 10.]), offset=np.zeros(2),
                            frame=np.eye(2), radius=-1., empty=False))
        circles.append(SimpleNamespace(index=i, bp_vertex_index=i+1, center=Point(*center),
                        radius=1., turn_direction=int(u[0]*v[1]-u[1]*v[0]),
                        construction_rule='fixture'))
    regions.append(None)
    return dict(sequence=circles), dict(smoothed_check=dict(regions=regions))


class CircleDiagnosticsTest(unittest.TestCase):
    def test_overlap_with_disjoint_same_turn_quarters(self):
        p, g = fixture([([0, 0], [1, 0], [0, 1]), ([1, 0], [1, 0], [0, 1])])
        pair = analyze_bp_circles(p, g)['pairs'][0]
        self.assertEqual(pair['circle_relation'], 'overlap')
        self.assertFalse(pair['quarters_intersect'])
        self.assertTrue(pair['same_turn_local_ok'])

    def test_same_turn_quarters_cross(self):
        p, g = fixture([([0, 0], [1, 0], [0, 1]), ([1, 0], [0, -1], [1, 0])])
        pair = analyze_bp_circles(p, g)['pairs'][0]
        self.assertTrue(pair['quarter_conflict'])
        self.assertFalse(pair['same_turn_local_ok'])

    def test_opposite_turn_overlap_is_separate_from_arc_intersection(self):
        p, g = fixture([([0, 0], [1, 0], [0, 1]), ([1, 0], [0, 1], [1, 0])])
        pair = analyze_bp_circles(p, g)['pairs'][0]
        self.assertTrue(pair['opposite_turn_overlap'])
        self.assertFalse(pair['quarters_intersect'])

    def test_touch_and_coincident_shared_join(self):
        for center, incoming, outgoing in (([2, 0], [0, -1], [1, 0]),
                                          ([0, 0], [0, 1], [-1, 0])):
            p, g = fixture([([0, 0], [1, 0], [0, 1]), (center, incoming, outgoing)])
            pair = analyze_bp_circles(p, g)['pairs'][0]
            self.assertTrue(pair['shared_join'])
            self.assertTrue(pair['quarters_intersect'])
            self.assertFalse(pair['quarter_conflict'])

    def test_coincident_arcs_and_nonconsecutive_join(self):
        p, g = fixture([([0, 0], [1, 0], [0, 1]), ([0, 0], [1, 0], [0, 1])])
        self.assertTrue(analyze_bp_circles(p, g)['pairs'][0]['quarter_conflict'])
        p, g = fixture([([0, 0], [1, 0], [0, 1]), ([9, 9], [1, 0], [0, 1]),
                        ([0, 0], [0, 1], [-1, 0])])
        pair = analyze_bp_circles(p, g, all_pairs=True)['pairs'][1]
        self.assertFalse(pair['consecutive'])
        self.assertTrue(pair['quarter_conflict'])
        self.assertFalse(pair['shared_join'])

    def test_membership_uses_implied_vertex_not_circle_center(self):
        p, g = fixture([([0, 0], [1, 0], [0, 1])])
        g['smoothed_check']['regions'][1]['high'][0] = .9
        row = analyze_bp_circles(p, g)['circles'][0]
        np.testing.assert_allclose(row['implied_vertex'], [1, -1])
        self.assertFalse(row['in_Aj'])
        self.assertAlmostEqual(row['Aj_margin'], -.1)

    def test_consecutive_flags_match_all_pairs_without_stale_state(self):
        p, g = fixture([([0, 0], [1, 0], [0, 1]),
                        ([1, 0], [0, 1], [1, 0]),
                        ([0, 0], [1, 0], [0, 1])])
        local = analyze_bp_circles(p, g)
        full = analyze_bp_circles(p, g, all_pairs=True)
        self.assertEqual(len(local['pairs']), 2)
        self.assertEqual(len(full['pairs']), 3)
        self.assertEqual(local['pairs'], [pair for pair in full['pairs'] if pair['consecutive']])
        self.assertEqual(local['flags']['opposite_turn_overlap'], [(0, 1), (1, 2)])
        self.assertEqual(local['flags'], full['flags'])
        p['sequence'][1].center = Point(9, 9)
        updated = analyze_bp_circles(p, g)
        self.assertEqual(updated['flags']['opposite_turn_overlap'], [])


if __name__ == '__main__':
    unittest.main()
