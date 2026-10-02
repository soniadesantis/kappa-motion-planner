"""Geometry-only connections retain direction and tangent-history constraints."""
import unittest
import numpy as np
from test_consecutive_circle_connection import circle, connect
from kappa_planner.refinement import DirectedTangentState, _nonintersecting_tangent_chain


class GeometryConnectionTest(unittest.TestCase):
    def test_longer_forward_arc_is_allowed(self):
        # Short tangents do not reach their line-line intersection; the long
        # forward arc is therefore allowed by this tangent-only audit.
        circles = [circle(0, (0, 0)), circle(1, (.1, .1)), circle(2, (.2, .125))]
        self.assertFalse(connect(circles).feasible)
        result = connect(circles, geometry_only=True)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.status, 'geometry_connected_pending_containment')
        self.assertTrue(result.tangent_intersections_checked)
        self.assertTrue(any(p['leave']-p['enter'] > np.pi/2
                            for p in result.primitives if p['kind'] == 'arc'))
        for p in result.primitives:
            if p['kind'] == 'arc':
                self.assertTrue(0 < p['leave']-p['enter'] < 2*np.pi)

    def test_outside_quarter_and_missing_overlap_do_not_reject(self):
        circles = [circle(0, (0, 0)), circle(2, (8, -4))]
        doors = [(-40., 40., -40., 40.), (15., 16., 15., 16.), (-40., 40., -40., 40.)]
        self.assertFalse(connect(circles, doors).feasible)
        result = connect(circles, doors, geometry_only=True)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.tangent_states[0].crossings, ())

    def test_search_limit_is_unresolved(self):
        result = connect([circle(0, (0, 0)), circle(1, (4, 4)), circle(2, (8, 8))],
                         geometry_only=True, max_search_expansions=1)
        self.assertFalse(result.feasible)
        self.assertEqual(result.status, 'geometry_search_unresolved')

    def test_opposite_turn_overlap_still_has_no_tangent(self):
        result = connect([circle(0, (0, 0)), circle(1, (0, 1), (0, 1), (1, 0))], geometry_only=True)
        self.assertFalse(result.feasible)
        self.assertEqual(result.rejected_pairs[0][2], 'no_directed_tangent')

    def test_intersection_search_keeps_different_histories(self):
        def state(i, j, a, b):
            a, b = np.array(a, dtype=float), np.array(b, dtype=float)
            length = float(np.linalg.norm(b-a))
            return DirectedTangentState(i, j, a, b, (b-a)/length, 0., 0., length, ())
        # The cheap prefix crosses the final tangent. Both prefixes reach the
        # SAME tangent state (2->3); retaining only its cheapest history fails.
        states = [state(0, 1, (0, 0), (4, 0)), state(1, 2, (4, 0), (4, 4)),
                  state(0, 2, (14, 0), (4, 4)), state(2, 3, (4, 4), (3, 3)),
                  state(3, 4, (3, 3), (0, -1))]
        outgoing = [[0, 2], [1], [3], [4], []]
        chosen, _, _, limited = _nonintersecting_tangent_chain(
            states, [(0,), (1,), (2,), (3,), (4,)], outgoing, 2., 1e-9, 100)
        self.assertEqual(chosen, (2, 3, 4))
        self.assertFalse(limited)
        # Remove the alternative: the intersecting chain must not be accepted.
        chosen, length, _, limited = _nonintersecting_tangent_chain(
            states, [(0,), (1,), (2,), (3,), (4,)], [[0], [1], [3], [4], []], 2., 1e-9, 100)
        self.assertEqual(chosen, ())
        self.assertIsNone(length)
        self.assertFalse(limited)


if __name__ == '__main__':
    unittest.main()
