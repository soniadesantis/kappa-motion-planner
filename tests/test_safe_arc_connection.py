"""Directed compatibility on safe circular components and consecutive chains."""
from types import SimpleNamespace
import unittest
import numpy as np
from test_consecutive_circle_connection import circle
from kappa_planner.refinement import (
    CircleSafeArcs, IndependentCirclePlacements, connect_safe_arc_circles,
    safe_directed_arc_sweep,
    _safe_arc_predecessors,
    ordered_safe_overlap_crossings,
)


def connect_green(circles, intervals, *, allow_skipping=False):
    count = max(c.waypoint_index for c in circles)+1
    baseline = SimpleNamespace(feasible=True, feasibility=SimpleNamespace(
        corridor_bounds=((-50., 50., -50., 50.),)*(count+1)))
    safe = tuple(CircleSafeArcs(i, c.waypoint_index, tuple(intervals[i]), ()) for i, c in enumerate(circles))
    return connect_safe_arc_circles(IndependentCirclePlacements('placed', tuple(circles)),
                                  baseline, SimpleNamespace(r=1., R=2.), safe_arcs=safe,
                                  allow_skipping=allow_skipping)


class SafeArcCompatibilityTest(unittest.TestCase):
    def test_wraparound_both_turns(self):
        intervals = np.radians(((350., 400.),))
        a, b = np.radians((355., 20.))
        self.assertAlmostEqual(safe_directed_arc_sweep(intervals, a, b, 1), np.radians(25.))
        self.assertAlmostEqual(safe_directed_arc_sweep(intervals, b, a, -1), np.radians(25.))
        self.assertIsNone(safe_directed_arc_sweep(intervals, a, b, -1))
        self.assertIsNone(safe_directed_arc_sweep(intervals, b, a, 1))

    def test_safe_contacts_do_not_allow_crossing_a_gap(self):
        intervals = np.radians(((0., 20.), (90., 110.)))
        for turn in (-1, 1):
            self.assertIsNone(safe_directed_arc_sweep(intervals, np.radians(10.), np.radians(100.), turn))

    def test_more_than_quarter_and_boundary_contacts(self):
        intervals = ((0., np.pi),)
        self.assertAlmostEqual(safe_directed_arc_sweep(intervals, 0., np.pi, 1), np.pi)
        self.assertAlmostEqual(safe_directed_arc_sweep(intervals, np.pi, 0., -1), np.pi)
        self.assertEqual(safe_directed_arc_sweep(((1., 1.),), 1., 1., 1), 0.)
        self.assertIsNone(safe_directed_arc_sweep((), 0., 0., 1))

    def test_green_arc_order_blocks_chain(self):
        circles = [circle(0, (0, 0)), circle(1, (4, 4)), circle(2, (8, 5))]
        intervals = [((3*np.pi/2, 2*np.pi),)]*3
        result = connect_green(circles, intervals)
        self.assertFalse(result.feasible)
        self.assertEqual(len(result.tangent_states), 2)
        self.assertEqual(result.contact_order_blocks, (1,))
        # Full-circle permission genuinely changes the allowable middle arc.
        result = connect_green(circles, [((0., 2*np.pi),)]*3)
        self.assertTrue(result.feasible, result.reason)
        self.assertTrue(any(p['leave']-p['enter'] > np.pi for p in result.primitives if p['kind']=='arc'))

    def test_branch_chooses_compatible_aligned_alternative(self):
        circles = [circle(0, (0, 0)), circle(1, (4, 4), side='left'),
                   circle(1, (4, 1), side='right'), circle(2, (8, 5))]
        result = connect_green(circles, [((3*np.pi/2, 2*np.pi),)]*4)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.selected_circles, (0, 2, 3))
        self.assertEqual(result.attempted_pairs, 4)
        self.assertTrue(result.safe_arcs_checked)
        for a, b in zip(result.primitives, result.primitives[1:]):
            np.testing.assert_allclose(a['end'], b['start'], atol=1e-9)

    def test_coincident_circles_join_at_safe_shared_contact(self):
        circles = [circle(0, (0, 0)), circle(1, (0, 0), (0, 1), (-1, 0))]
        result = connect_green(circles, [((3*np.pi/2, 2*np.pi),), ((0., np.pi/2),)])
        self.assertTrue(result.feasible, result.reason)
        self.assertAlmostEqual(result.length, 2*np.pi)
        self.assertTrue(all(p['kind']=='arc' for p in result.primitives))

    def test_tangent_contact_outside_green_is_rejected(self):
        result = connect_green([circle(0, (0, 0)), circle(1, (4, 4))],
                               [((0., .1),), ((0., 2*np.pi),)])
        self.assertFalse(result.feasible)
        self.assertEqual(result.rejected_pairs, ((0, 1, 'contact_outside_safe_arcs'),))

    def test_skipping_recovers_chain_and_reports_groups(self):
        circles = [circle(0, (0, 0)), circle(1, (2, 4)), circle(2, (4, 5)), circle(3, (8, 6))]
        arcs = [((3*np.pi/2, 2*np.pi),)]*4
        self.assertFalse(connect_green(circles, arcs).feasible)
        result = connect_green(circles, arcs, allow_skipping=True)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.selected_circles, (0, 3))
        self.assertEqual(result.skipped_waypoints, (1, 2))
        self.assertEqual(result.attempted_pairs, 6)

    def test_tangent_crosses_full_overlap_outside_safe_overlap(self):
        circles = [circle(0, (-.2, .2)), circle(2, (7.8, 8.2))]
        bounds = ((-4., 4., -4., 4.), (-4., 6., -4., 4.),
                  (4., 14., 2., 12.), (6., 14., 4., 12.))
        baseline = SimpleNamespace(feasible=True, feasibility=SimpleNamespace(
            corridor_bounds=bounds))
        arcs = tuple(CircleSafeArcs(i, c.waypoint_index, ((0., 2*np.pi),), ())
                     for i, c in enumerate(circles))
        result = connect_safe_arc_circles(
            IndependentCirclePlacements('placed', tuple(circles)), baseline,
            SimpleNamespace(r=1., R=2.), safe_arcs=arcs)
        self.assertTrue(result.feasible, result.reason)
        self.assertTrue(result.ordered_overlap_crossings_checked)
        self.assertTrue(result.tangent_containment_checked)
        state = result.tangent_states[result.chosen_states[0]]
        self.assertEqual((state.start_corridor, state.end_corridor), (None, None))
        self.assertEqual(tuple(j for j, _ in state.crossings), (1,))
        # Full overlap is [4,6] x [2,4]; erosion leaves only (5,3).
        self.assertIsNotNone(ordered_safe_overlap_crossings(
            state.start, state.end, ((4., 6., 2., 4.),), tol=0.))
        self.assertIsNone(ordered_safe_overlap_crossings(
            state.start, state.end, ((5., 5., 3., 3.),), tol=0.))

    def test_tangent_missing_full_overlap_is_rejected(self):
        circles = [circle(0, (0, 0)), circle(2, (8, 8))]
        bounds = ((-4., 4., -4., 4.), (-4., 6., -4., 4.),
                  (4., 14., 3.5, 12.), (6., 14., 4., 12.))
        baseline = SimpleNamespace(feasible=True, feasibility=SimpleNamespace(
            corridor_bounds=bounds))
        arcs = tuple(CircleSafeArcs(i, c.waypoint_index, ((0., 2*np.pi),), ())
                     for i, c in enumerate(circles))
        result = connect_safe_arc_circles(
            IndependentCirclePlacements('placed', tuple(circles)), baseline,
            SimpleNamespace(r=1., R=2.), safe_arcs=arcs)
        self.assertFalse(result.feasible)
        self.assertEqual(result.rejected_pairs, ((0, 1, 'ordered_overlap_crossing'),))

    def test_green_bridge_contacts_do_not_require_individual_eroded_containment(self):
        # Like map 7's O6 -> O8: both contacts lie on green fillet portions
        # bridging the eroded rectangles, and the shortcut crosses the middle door.
        circles = [circle(0, (0, 0)), circle(2, (8, 8))]
        bounds = ((-4., 2.2, -4., 4.), (-4., 7., -2.2, 5.),
                  (3., 10.2, 1., 12.), (6., 14., 5.8, 12.))
        baseline = SimpleNamespace(feasible=True, feasibility=SimpleNamespace(
            corridor_bounds=bounds))
        arcs = tuple(CircleSafeArcs(i, c.waypoint_index, ((3*np.pi/2, 2*np.pi),), ())
                     for i, c in enumerate(circles))
        result = connect_safe_arc_circles(
            IndependentCirclePlacements('placed', tuple(circles)), baseline,
            SimpleNamespace(r=1., R=2.), safe_arcs=arcs)
        self.assertTrue(result.feasible, result.reason)
        state = result.tangent_states[result.chosen_states[0]]
        for point, pair in ((state.start, bounds[:2]), (state.end, bounds[2:])):
            self.assertFalse(any(x+1 <= point[0] <= xx-1 and y+1 <= point[1] <= yy-1
                                 for x, xx, y, yy in pair))
        self.assertEqual(tuple(j for j, _ in state.crossings), (1,))

    def test_raw_overlap_is_not_enough_for_footprint_safety(self):
        circles = [circle(0, (0, 0)), circle(2, (8, 8))]
        bounds = ((-4., 4., -4., 4.), (-4., 6., -4., 4.),
                  (4., 14., 2., 12.), (6., 14., 4., 12.))
        baseline = SimpleNamespace(feasible=True, feasibility=SimpleNamespace(
            corridor_bounds=bounds))
        arcs = tuple(CircleSafeArcs(i, c.waypoint_index, ((0., 2*np.pi),), ())
                     for i, c in enumerate(circles))
        result = connect_safe_arc_circles(
            IndependentCirclePlacements('placed', tuple(circles)), baseline,
            SimpleNamespace(r=1., R=2.), safe_arcs=arcs)
        self.assertFalse(result.feasible)
        self.assertEqual(result.rejected_pairs, ((0, 1, 'straight_footprint_clearance'),))

    def test_shortcut_can_cross_an_unused_supporting_disk(self):
        circles = [circle(0, (0, 0)), circle(1, (4, 2)), circle(2, (8, 8))]
        result = connect_green(circles, [((3*np.pi/2, 2*np.pi),)]*3,
                               allow_skipping=True)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.selected_circles, (0, 2))
        state = result.tangent_states[result.chosen_states[0]]
        t = np.clip((circles[1].center-state.start) @ state.direction/state.length, 0., 1.)
        distance = np.linalg.norm(circles[1].center-(state.start+t*(state.end-state.start)))
        self.assertLess(distance, circles[1].radius)

    def test_opposite_turn_external_tangency_allows_zero_length_link(self):
        circles = [circle(0, (0, 0)), circle(1, (4, 0), (0, 1), (1, 0))]
        result = connect_green(circles, [((3*np.pi/2, 2*np.pi),), ((np.pi/2, np.pi),)])
        self.assertTrue(result.feasible, result.reason)
        self.assertTrue(all(state.length < 1e-9 for state in result.tangent_states))
        self.assertTrue(all(p['kind'] == 'arc' for p in result.primitives))
        self.assertAlmostEqual(result.length, 2*np.pi)

    def test_prefix_queries_match_exhaustive_predecessors(self):
        rng = np.random.default_rng(810)
        for intervals in (((0., 2*np.pi),), ((5., 7.5),), ((0., 1.), (2., 4.)), ((1., 1.),)):
            for turn in (-1, 1):
                angles = np.r_[rng.uniform(0, 2*np.pi, 30), [v % (2*np.pi) for p in intervals for v in p]]
                arrivals = [(i, float(rng.uniform(0, 10)), a) for i, a in enumerate(angles)]
                departures = list(enumerate(np.r_[rng.uniform(0, 2*np.pi, 30), angles]))
                best = _safe_arc_predecessors(arrivals, departures, intervals, turn, 2., 0.)
                for idx, departure in departures:
                    choices = [(cost+2*delta, i) for i, cost, arrival in arrivals
                               if (delta := safe_directed_arc_sweep(intervals, arrival, departure, turn,
                                                                   angular_tol=0.)) is not None]
                    self.assertEqual(idx in best, bool(choices))
                    if choices:
                        self.assertAlmostEqual(best[idx][0], min(choices)[0], places=10)

    def test_skipping_dag_matches_exhaustive_chains_with_alternatives(self):
        rng = np.random.default_rng(181)
        for _ in range(30):
            circles = [circle(0, (0, 0)), circle(1, rng.uniform(1, 6, 2)),
                       circle(1, rng.uniform(1, 6, 2)), circle(2, rng.uniform(1, 6, 2)),
                       circle(3, (8, 8))]
            arcs = [((3*np.pi/2, 2*np.pi),)]*5
            result = connect_green(circles, arcs, allow_skipping=True)
            lengths = []
            def visit(i, arrival, cost):
                if i in result.groups[-1]:
                    delta = safe_directed_arc_sweep(arcs[i], arrival, 0., 1)
                    if delta is not None:
                        lengths.append(cost+2*delta)
                    return
                for state in result.tangent_states:
                    if state.first == i:
                        delta = safe_directed_arc_sweep(arcs[i], arrival, state.departure_angle, 1)
                        if delta is not None:
                            visit(state.second, state.arrival_angle, cost+2*delta+state.length)
            visit(0, 3*np.pi/2, 0.)
            self.assertEqual(result.feasible, bool(lengths))
            self.assertAlmostEqual(result.length, min(lengths), places=9)
            self.assertEqual(result.attempted_pairs, 9)
            self.assertFalse(any(s.first == 1 and s.second == 2 for s in result.tangent_states))


if __name__ == '__main__':
    unittest.main()
