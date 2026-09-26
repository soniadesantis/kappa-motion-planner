"""Directed contacts, transactional skips, clearance, and fixed circle state."""
import unittest

import numpy as np

from test_bp_circle_diagnostics import fixture
from test_bp_circle_sequence import EXAMPLE
from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.bp_circle_diagnostics import analyze_bp_circles
from kappa_planner.helpers.bp_circle_sequence import build_bp_circle_sequence
from kappa_planner.helpers.bp_tangent_chain import build_bp_tangent_chain, _covered, _conflict, _union_clear


def setup(centers):
    p, g = fixture([(center, [1, 0], [0, 1]) for center in centers])
    p['feasible'] = g['feasible'] = True
    p['diagnostics'] = analyze_bp_circles(p, g)
    corridors = [CorridorWorld(30, 30, [0, 0], 0) for _ in range(len(centers)+1)]
    return p, g, corridors


class TangentChainTest(unittest.TestCase):
    def test_skip_reversed_contacts_and_preserve_centers(self):
        p, g, c = setup([[0, 0], [1, 2], [3, 3]])
        result = build_bp_tangent_chain(p, g, c, .1)
        self.assertTrue(result['feasible'], result['reason'])
        self.assertEqual(result['retained'], [0, 2])
        self.assertEqual(result['skipped'], [1])
        self.assertEqual([(a.center.x, a.center.y) for a in p['sequence']], [(0, 0), (1, 2), (3, 3)])
        self.assertTrue(all(0 <= a['leave']-a['enter'] <= np.pi/2 for a in result['arcs']))

    def test_defer_unconnectable_middle_circle(self):
        p, g, c = setup([[0, 0], [1, -2], [3, 0]])
        result = build_bp_tangent_chain(p, g, c, .1)
        self.assertTrue(result['feasible'])
        self.assertEqual(result['skipped'], [1])

    def test_one_extension_can_skip_two_retained_circles(self):
        p, g, c = setup([[0, 0], [1, 1], [2, 3], [5, 3]])
        result = build_bp_tangent_chain(p, g, c, .1)
        self.assertTrue(result['feasible'], result['reason'])
        self.assertEqual(result['retained'], [0, 3])
        self.assertEqual(result['skipped'], [1, 2])
        self.assertEqual(result['events'], [dict(target=3, bypassed=[1, 2], replacement=(0, 3))])

    def test_union_cover_requires_entire_segment(self):
        segment = dict(start=np.array([0., 0.]), end=np.array([3., 0.]))
        boxes = [(np.array([-1., -1.]), np.array([1., 1.])),
                 (np.array([2., -1.]), np.array([4., 1.]))]
        self.assertFalse(_covered(segment, boxes, 1e-8))
        boxes[1][0][0] = 1.
        self.assertTrue(_covered(segment, boxes, 1e-8))

    def test_crossings_overlap_and_allowed_shared_join(self):
        def line(a, b):
            return dict(kind='line', start=np.array(a, float), end=np.array(b, float))
        a = line([0, 0], [1, 1])
        self.assertTrue(_conflict(a, line([0, 1], [1, 0]), True, 1e-8))
        self.assertTrue(_conflict(a, line([.5, .5], [2, 2]), True, 1e-8))
        self.assertFalse(_conflict(a, line([1, 1], [2, 2]), True, 1e-8))

    def test_union_corner_clearance_ignores_internal_seams(self):
        boxes = [(np.array([-2., -1.]), np.array([1., 0.])),
                 (np.array([0., -1.]), np.array([1., 2.]))]
        segment = dict(start=np.array([-.2, -.2]), end=np.array([.2, 0.]))
        self.assertFalse(_covered(segment, [(a+.08, b-.08) for a, b in boxes], 1e-8))
        self.assertTrue(_union_clear(segment, boxes, .08, 1e-8))
        segment['end'][1] = .12
        self.assertFalse(_union_clear(segment, boxes, .08, 1e-8))

    def test_all_examples_have_bounded_arcs_and_honest_status(self):
        connected = []
        skipped = []
        for n in EXAMPLE['EXAMPLE_NUMBERS']:
            c, s, e, v = EXAMPLE['example_corridor_sequence'](n)
            g = EXAMPLE['build_trajectory_geometry'](c, v.width/2, v.max_radius,
                                                     start_pose=s, end_pose=e)
            p = build_bp_circle_sequence(c, v, g, shift_same_turn=True,
                                        shift_opposite_turn=True, connect_tangents=True)
            result = p['tangent_chain']
            if result['feasible']:
                connected.append(n)
                skipped.extend(result['skipped'])
                self.assertFalse(result['intersections'])
                self.assertEqual(result['retained'][0], p['sequence'][0].index)
                self.assertEqual(result['retained'][-1], p['sequence'][-1].index)
                self.assertTrue(all(-1e-8 <= a['leave']-a['enter'] <= np.pi/2+1e-8
                                    for a in result['arcs']))
                for first, second in zip(result['primitives'], result['primitives'][1:]):
                    np.testing.assert_allclose(first['end'], second['start'], atol=1e-8)
        self.assertEqual(len(connected), 20)
        self.assertTrue(skipped)

    def test_existing_examples_zero_length_join_and_opposite_connection(self):
        for n in (3, 35):
            c, s, e, v = EXAMPLE['example_corridor_sequence'](n)
            g = EXAMPLE['build_trajectory_geometry'](c, v.width/2, v.max_radius,
                                                     start_pose=s, end_pose=e)
            p = build_bp_circle_sequence(c, v, g, shift_same_turn=True, shift_opposite_turn=True)
            result = build_bp_tangent_chain(p, g, c, v.width/2)
            self.assertTrue(result['feasible'], result['reason'])
            self.assertFalse(result['endpoint_connections_checked'])
            if n == 3:
                self.assertLess(np.linalg.norm(result['tangents'][0]['end']-result['tangents'][0]['start']), 1e-8)
            else:
                np.testing.assert_allclose(result['tangents'][0]['start'], [0, .8], atol=1e-8)
                np.testing.assert_allclose(result['tangents'][0]['end'], [0, 1.2], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
