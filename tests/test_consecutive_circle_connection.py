"""Consecutive tangent graph: branching, contact order, coincidence, and safety."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np

from kappa_planner.refinement import (
    IndependentCirclePlacement, IndependentCirclePlacements, connect_consecutive_circles,
    connect_refinement_circles,
)


def circle(j, center, incoming=(1, 0), outgoing=(0, 1), side=None):
    """Locally safe synthetic quarter in generous rectangular corridors."""
    u, v, center = (np.asarray(x, dtype=float) for x in (incoming, outgoing, center))
    R, r = 2., 1.
    vertex = center+R*u-R*v
    corner = vertex+10*(-u+v)
    region = dict(low=np.array([-40., -40.]), high=np.array([40., 40.]),
                  incoming=u, outgoing=v, corner=corner, frame=np.column_stack((-u, v)),
                  offset=-R*u+R*v-corner, radius=R-r, empty=False, witness=vertex)
    return IndependentCirclePlacement(j, 'synthetic', center, vertex, R, region, True, side=side)


def connect(circles, doors=None, *, allow_skipping=False, geometry_only=False, **kwargs):
    count = max((c.waypoint_index for c in circles), default=0)+1
    data = SimpleNamespace(corridor_bounds=((-50., 50., -50., 50.),)*(count+1),
                           safe_overlaps=tuple(doors) if doors else ((-40., 40., -40., 40.),)*count)
    baseline = SimpleNamespace(feasible=True, feasibility=data)
    if geometry_only:
        return connect_refinement_circles(IndependentCirclePlacements('placed', tuple(circles)),
            baseline, SimpleNamespace(r=1., R=2.), allow_skipping=allow_skipping,
            geometry_only=True, **kwargs)
    function = connect_refinement_circles if allow_skipping else connect_consecutive_circles
    return function(IndependentCirclePlacements('placed', tuple(circles)),
                    baseline, SimpleNamespace(r=1., R=2.))


def brute_length(result):
    """Enumerate all state chains rather than reuse the prefix-minimum DP."""
    lengths = []
    def visit(circle_index, arrival, cost):
        if circle_index in result.groups[-1]:
            lengths.append(cost+2*(np.pi/2-arrival))
            return
        for state in result.tangent_states:
            if state.first == circle_index and state.departure_angle >= arrival-1e-10:
                visit(state.second, state.arrival_angle,
                      cost+2*max(0., state.departure_angle-arrival)+state.length)
    for i in result.groups[0]:
        visit(i, 0., 0.)
    return min(lengths) if lengths else None


class ConsecutiveCircleConnectionTest(unittest.TestCase):
    def test_pairwise_valid_but_wrong_contact_order(self):
        result = connect([circle(0, (0, 0)), circle(1, (4, 4)), circle(2, (8, 5))])
        self.assertEqual(len(result.tangent_states), 2)
        self.assertEqual(result.rejected_pairs, ())
        self.assertFalse(result.feasible)
        self.assertEqual(result.status, 'no_consecutive_chain')
        self.assertEqual(result.contact_order_blocks, (1,))
        # A direct first-to-last tangent would work, but skipping is excluded.
        self.assertFalse(any(s.first == 0 and s.second == 2 for s in result.tangent_states))

    def test_aligned_alternatives_branch_without_connecting_to_each_other(self):
        result = connect([circle(0, (0, 0)), circle(1, (4, 4), side='left'),
                          circle(1, (4, 1), side='right'), circle(2, (8, 5))])
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.groups, ((0,), (1, 2), (3,)))
        self.assertEqual(result.selected_circles, (0, 2, 3))
        self.assertEqual(result.attempted_pairs, 4)
        self.assertFalse(any(s.first == 1 and s.second == 2 for s in result.tangent_states))
        self.assertAlmostEqual(result.length, brute_length(result))
        for a, b in zip(result.primitives, result.primitives[1:]):
            np.testing.assert_allclose(a['end'], b['start'], atol=1e-9)

    def test_coincident_contacts_include_incident_interior_tangencies(self):
        result = connect([circle(0, (-4, -1)), circle(1, (0, 0)),
                          circle(2, (0, 0)), circle(3, (4, 4))])
        self.assertTrue(result.feasible, result.reason)
        joins = [s for s in result.tangent_states if (s.first, s.second) == (1, 2)]
        self.assertTrue(any(abs(s.departure_angle-np.pi/4) < 1e-9 for s in joins))
        self.assertTrue(any(s.length == 0. for s in joins))
        self.assertAlmostEqual(result.length, brute_length(result))

    def test_coincident_adjacent_quarters(self):
        result = connect([circle(0, (0, 0)), circle(1, (0, 0), (0, 1), (-1, 0))])
        self.assertTrue(result.feasible, result.reason)
        self.assertAlmostEqual(result.length, 2*np.pi)
        self.assertTrue(all(p['kind'] == 'arc' for p in result.primitives))

    def test_missing_circle_still_requires_safe_door_crossing(self):
        circles = [circle(0, (0, 0)), circle(2, (8, 8))]
        doors = [(-40., 40., -40., 40.), (4., 6., 2., 4.), (-40., 40., -40., 40.)]
        self.assertTrue(connect(circles, doors).feasible)
        doors[1] = (7., 8., 7., 8.)
        result = connect(circles, doors)
        self.assertFalse(result.feasible)
        self.assertIn('ordered_safe_overlap_crossing', result.rejected_pairs[0][2])

    def test_empty_and_single_group(self):
        self.assertEqual(connect([]).status, 'no_circles')
        result = connect([circle(0, (0, 0))])
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.length, np.pi)
        self.assertEqual(len(result.primitives), 1)

    def test_dag_matches_brute_force_for_random_alternatives(self):
        rng = np.random.default_rng(17)
        for _ in range(30):
            circles = [circle(0, (0, 0)), circle(1, rng.uniform(1, 4, 2)),
                       circle(1, rng.uniform(1, 4, 2)), circle(2, rng.uniform(4, 7, 2)),
                       circle(3, (8, 8))]
            result = connect(circles)
            expected = brute_length(result)
            self.assertEqual(result.feasible, expected is not None)
            if expected is not None:
                self.assertAlmostEqual(result.length, expected, places=9)

    def test_invalid_radius_rejected(self):
        with self.assertRaises(ValueError):
            connect([replace(circle(0, (0, 0)), radius=3.)])

    def test_skipping_recovers_contact_order_failure(self):
        circles = [circle(0, (0, 0)), circle(1, (4, 4)), circle(2, (8, 5))]
        self.assertFalse(connect(circles).feasible)
        result = connect(circles, allow_skipping=True)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.selected_circles, (0, 2))
        self.assertEqual(result.skipped_waypoints, (1,))
        self.assertEqual(result.attempted_pairs, 3)
        state = result.tangent_states[result.chosen_states[0]]
        self.assertEqual(tuple(j for j, _ in state.crossings), (1,))
        self.assertAlmostEqual(result.length, brute_length(result))

    def test_shortcut_must_cross_the_skipped_circles_door(self):
        circles = [circle(0, (0, 0)), circle(1, (4, 4)), circle(2, (8, 5))]
        # Contains the middle circle's vertex (6,2), but misses the 0->2 tangent.
        doors = [(-40., 40., -40., 40.), (5.9, 6.1, 1.9, 2.1), (-40., 40., -40., 40.)]
        result = connect(circles, doors, allow_skipping=True)
        self.assertFalse(result.feasible)
        self.assertIn((0, 2, 'ordered_safe_overlap_crossing'), result.rejected_pairs)

    def test_can_skip_multiple_groups_while_retaining_end_groups(self):
        result = connect([circle(0, (0, 0)), circle(1, (2, 4)),
                          circle(2, (4, 5)), circle(3, (8, 6))], allow_skipping=True)
        self.assertTrue(result.feasible)
        self.assertEqual(result.selected_circles, (0, 3))
        self.assertEqual(result.skipped_waypoints, (1, 2))
        state = result.tangent_states[result.chosen_states[0]]
        self.assertEqual(tuple(j for j, _ in state.crossings), (1, 2))

    def test_skipped_doors_must_be_crossed_in_sequence(self):
        circles = [circle(0, (0, 0)), circle(1, (5, 5)),
                   circle(2, (1, 1)), circle(3, (8, 8))]
        doors = [(-40., 40., -40., 40.), (6.9, 7.1, 2.9, 4.4),
                 (2.9, 3.1, -1.1, .4), (-40., 40., -40., 40.)]
        result = connect(circles, doors, allow_skipping=True)
        self.assertIn((0, 3, 'ordered_safe_overlap_crossing'), result.rejected_pairs)

    def test_skipping_still_rejects_contacts_outside_quarters(self):
        result = connect([circle(0, (0, 0)), circle(1, (4, 4)), circle(2, (8, -4))],
                         allow_skipping=True)
        self.assertFalse(any(s.first == 0 and s.second == 2 for s in result.tangent_states))
        self.assertIn((0, 2, 'contact_outside_quarter'), result.rejected_pairs)

    def test_skipping_with_alternatives_matches_exhaustive_search(self):
        rng = np.random.default_rng(31)
        for _ in range(30):
            circles = [circle(0, (0, 0)), circle(1, rng.uniform(1, 7, 2)),
                       circle(1, rng.uniform(1, 7, 2)), circle(2, rng.uniform(1, 7, 2)),
                       circle(3, (8, 8))]
            consecutive = connect(circles)
            result = connect(circles, allow_skipping=True)
            self.assertTrue(result.feasible)
            self.assertEqual(result.attempted_pairs, 9)  # Excludes same-door alternatives.
            self.assertAlmostEqual(result.length, brute_length(result), places=9)
            if consecutive.feasible:
                self.assertLessEqual(result.length, consecutive.length+1e-9)
            for group in result.groups:
                self.assertLessEqual(sum(i in result.selected_circles for i in group), 1)


if __name__ == '__main__':
    unittest.main()
