"""Independent interval/LP and exhaustive-chain checks of fixed-circle search."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linprog

from kappa_planner.baseline_construction import compute_filleted_baseline
from kappa_planner.refinement import (
    FixedCircleConnectionProblem, FixedTurningCircle, InternalPathAnchor,
    connect_fixed_circles, freeze_internal_baseline,
    ordered_safe_overlap_crossings, segment_rectangle_parameter_interval,
)


def circle(index, center, incoming=(1, 0), outgoing=(0, 1)):
    """Synthetic safe quarter in a very large rectangle union."""
    u, v, center = map(lambda x: np.array(x, dtype=float), (incoming, outgoing, center))
    R, r = 2., 1.
    vertex = center+R*u-R*v
    corner = vertex+10*(-u+v)
    region = dict(low=np.array([-49., -49.]), high=np.array([49., 49.]),
                  incoming=u, outgoing=v, corner=corner, frame=np.column_stack((-u, v)),
                  offset=-R*u+R*v-corner, radius=R-r, empty=False, witness=vertex)
    return FixedTurningCircle(index, center, R, int(u[0]*v[1]-u[1]*v[0]), region)


def problem(circles, start, end, start_direction=(1, 0), end_direction=(0, 1), doors=None):
    count = max((c.waypoint_index for c in circles), default=0)+2
    if doors is None:
        doors = ((-40., 40., -40., 40.),)*count
    return FixedCircleConnectionProblem(
        InternalPathAnchor(np.array(start, dtype=float), np.array(start_direction, dtype=float), 0, 1),
        InternalPathAnchor(np.array(end, dtype=float), np.array(end_direction, dtype=float), count-1, count-1),
        tuple(circles), ((-50., 50., -50., 50.),)*(count+1), tuple(doors), 1.)


def brute_shortest(report):
    """Enumerate every compatible state chain independently of the prefix DP."""
    states, problem = report.tangent_states, report.problem
    sink, costs = len(problem.circles)+1, []
    def visit(node, enter, length):
        if node == sink:
            costs.append(length)
            return
        for state in states:
            if state.first != node or (node and state.departure_angle < enter-1e-10):
                continue
            arc = problem.circles[node-1].radius*(state.departure_angle-enter) if node else 0.
            visit(state.second, state.arrival_angle, length+max(0., arc)+state.length)
    visit(0, 0., 0.)
    return min(costs) if costs else None


class FixedCircleConnectionTest(unittest.TestCase):
    def test_slab_clipping_degenerate_and_zero_length(self):
        self.assertEqual(segment_rectangle_parameter_interval([0, 0], [10, 0], (2, 4, 0, 0), tol=0),
                         (.2, .4))
        self.assertEqual(segment_rectangle_parameter_interval([10, 0], [0, 0], (2, 4, 0, 0), tol=0),
                         (.6, .8))
        self.assertEqual(segment_rectangle_parameter_interval([2, 0], [2, 0], (2, 2, 0, 0), tol=0),
                         (0., 1.))
        self.assertIsNone(segment_rectangle_parameter_interval([0, 0], [10, 0], (2, 4, 1, 1), tol=0))

    def test_ordered_crossings_reject_reverse_order_and_allow_contacts(self):
        self.assertIsNone(ordered_safe_overlap_crossings([0, 0], [10, 0],
                                                       [(7, 8, 0, 0), (2, 3, 0, 0)], tol=0))
        self.assertEqual(ordered_safe_overlap_crossings([0, 0], [10, 0],
                                                       [(2, 3, 0, 0), (3, 3, 0, 0)], tol=0), (.2, .3))

    def test_ordered_propagation_matches_independent_linear_program(self):
        rng = np.random.default_rng(91)
        for _ in range(150):
            start, end = rng.uniform(-4, 4, (2, 2))
            doors = []
            for _ in range(4):
                bounds = np.sort(rng.uniform(-4, 4, (2, 2)), axis=0)
                doors.append((bounds[0, 0], bounds[1, 0], bounds[0, 1], bounds[1, 1]))
            # LP directly encodes XY box inequalities and t_h<=t_{h+1}.
            inequalities, upper = [], []
            for j, door in enumerate(doors):
                for axis, (lo, hi) in enumerate((door[:2], door[2:])):
                    row = np.zeros(4)
                    row[j] = end[axis]-start[axis]
                    inequalities.extend((row, -row))
                    upper.extend((hi-start[axis], start[axis]-lo))
            for j in range(3):
                row = np.zeros(4)
                row[j], row[j+1] = 1, -1
                inequalities.append(row)
                upper.append(0)
            expected = linprog(np.zeros(4), A_ub=inequalities, b_ub=upper,
                               bounds=[(0, 1)]*4, method='highs').success
            crossings = ordered_safe_overlap_crossings(start, end, doors, tol=0)
            self.assertEqual(crossings is not None, expected)

    def test_shortest_dag_matches_exhaustive_chain_enumeration(self):
        circles = [circle(j+1, center) for j, center in enumerate(((2, 2), (4, 4), (6, 5), (8, 8)))]
        result = connect_fixed_circles(problem(circles, (-5, 0), (10, 15)))
        self.assertTrue(result.local_feasible, result.reason)
        self.assertAlmostEqual(result.length, brute_shortest(result), places=10)
        recomputed = sum(np.linalg.norm(p['end']-p['start']) if p['kind'] == 'line'
                         else p['radius']*(p['leave']-p['enter']) for p in result.primitives)
        self.assertAlmostEqual(result.length, recomputed)
        for a, b in zip(result.primitives, result.primitives[1:]):
            np.testing.assert_allclose(a['end'], b['start'], atol=1e-9)

    def test_random_fixed_placements_match_exhaustive_search(self):
        rng = np.random.default_rng(73)
        for _ in range(40):
            centers = [(2, 2), *rng.uniform(2, 8, (3, 2)), (8, 8)]
            circles = [circle(j+1, center) for j, center in enumerate(centers)]
            result = connect_fixed_circles(problem(circles, (-5, 0), (10, 15)), audit_global=False)
            expected = brute_shortest(result)
            self.assertEqual(result.local_feasible, expected is not None)
            if expected is not None:
                self.assertAlmostEqual(result.length, expected, places=9)

    def test_unreachable_circle_is_skipped_but_its_door_is_crossed(self):
        circles = [circle(1, (-4, -4), (0, 1), (-1, 0)), circle(2, (0, 0))]
        doors = [(-40, 40, -40, 40), (-2.5, -1.5, -2.5, -1.5),
                 (-40, 40, -40, 40), (-40, 40, -40, 40)]
        result = connect_fixed_circles(problem(circles, (-5, -2), (2, 5), doors=doors))
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.retained_waypoints, (2,))
        self.assertEqual(result.skipped_waypoints, (1,))
        first = result.tangent_states[result.chosen_states[0]]
        self.assertEqual(first.crossings[0][0], 1)
        bad = list(doors)
        # Move the first circle together with its associated door, so its new
        # placement remains locally valid but the shortcut cannot cross the door.
        bad[1] = (-2.5, -1.5, 10, 11)
        bad_circles = [circle(1, (-4, 8.5), (0, 1), (-1, 0)), circles[1]]
        rejected = connect_fixed_circles(problem(bad_circles, (-5, -2), (2, 5), doors=bad))
        self.assertFalse(rejected.local_feasible)
        self.assertTrue(rejected.certified_locally_infeasible)

    def test_opposite_turn_overlap_has_no_required_common_tangent(self):
        circles = [circle(1, (0, 0)), circle(2, (0, 1), (0, 1), (1, 0))]
        result = connect_fixed_circles(problem(circles, (-5, -2), (5, 3), end_direction=(1, 0)))
        self.assertFalse(any(s.first == 1 and s.second == 2 for s in result.tangent_states))
        self.assertIn((1, 2, 'opposite_turn_overlap'), result.rejected_pairs)

    def test_coincident_same_turn_quarters_can_join_at_zero_length(self):
        circles = [circle(1, (0, 0)), circle(2, (0, 0), (0, 1), (-1, 0))]
        result = connect_fixed_circles(problem(circles, (-5, -2), (-5, 2), end_direction=(-1, 0)))
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.retained_waypoints, (1, 2))
        self.assertAlmostEqual(result.length, 10+2*np.pi)
        self.assertTrue(any(result.tangent_states[i].length == 0 for i in result.chosen_states))

    def test_coincident_join_includes_incident_interior_contacts(self):
        circles = [circle(1, (0, 0)), circle(2, (0, 0)), circle(3, (4, 4))]
        result = connect_fixed_circles(problem(circles, (-5, -2), (6, 9)))
        joins = [s for s in result.tangent_states if s.first == 1 and s.second == 2]
        self.assertTrue(any(abs(s.departure_angle-np.pi/4) < 1e-10 for s in joins))
        self.assertAlmostEqual(result.length, brute_shortest(result), places=10)

    def test_global_intersections_are_unresolved_not_local_infeasibility(self):
        circles = [circle(1, (0, 0)), circle(2, (0, 6), (0, 1), (-1, 0)),
                   circle(3, (-6, 6), (-1, 0), (0, -1)),
                   circle(4, (-6, 0), (0, -1), (1, 0))]
        doors = [(-40, 40, -40, 40), (2, 2, -2, -2), (2, 2, 8, 8),
                 (-8, -8, 8, 8), (-8, -8, -2, -2), (-40, 40, -40, 40)]
        p = problem(circles, (-10, -2), (10, -2), end_direction=(1, 0), doors=doors)
        result = connect_fixed_circles(p)
        self.assertTrue(result.local_feasible)
        self.assertFalse(result.feasible)
        self.assertEqual(result.status, 'global_audit_unresolved')
        self.assertFalse(result.certified_locally_infeasible)
        self.assertTrue(result.intersections)
        self.assertTrue(connect_fixed_circles(p, audit_global=False).feasible)

    def test_no_circle_straight_case_and_anchor_heading_rejection(self):
        p = problem([], (-5, 0), (5, 0), end_direction=(1, 0))
        result = connect_fixed_circles(p)
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.length, 10)
        p = replace(p, end=replace(p.end, direction=np.array([-1., 0.])))
        result = connect_fixed_circles(p)
        self.assertFalse(result.local_feasible)

    def test_endpoint_containment_is_required_even_without_intermediate_doors(self):
        # The candidate departure at (0,-2) is inside the circle's incoming
        # corridor but outside its outgoing corridor. The direct skip is valid.
        p = problem([circle(1, (0, 0))], (-5, -2), (5, -2), end_direction=(1, 0))
        bounds = list(p.corridor_bounds)
        bounds[-2] = (0, 50, -50, 50)
        doors = list(p.safe_overlaps)
        doors[1], doors[2] = (1.9, 2.1, -2.1, -1.9), (5, 5, -2, -2)
        p = replace(p, corridor_bounds=tuple(bounds), safe_overlaps=tuple(doors))
        result = connect_fixed_circles(p)
        self.assertTrue(result.local_feasible)
        self.assertFalse(any(s.first == 1 and s.second == 2 for s in result.tangent_states))
        self.assertTrue(any('endpoint_containment' in reason for _, _, reason in result.rejected_pairs))

    def test_frozen_baseline_boundary_arcs_stay_outside_refinement(self):
        bounds = [(0, 4, -8, 4), (0, 16, 0, 4), (12, 16, -8, 4)]
        corridors = [SimpleNamespace(corners=np.array([[a, c], [b, c], [b, d], [a, d]]))
                     for a, b, c, d in bounds]
        robot = SimpleNamespace(r=1, R=3)
        baseline = compute_filleted_baseline(corridors, robot,
                                             initial_position=[2, -1], final_position=[14, -1])
        p = freeze_internal_baseline(baseline, robot)
        self.assertEqual(p.circles, ())
        np.testing.assert_array_equal(p.start.point, baseline.fillets[0].outgoing_tangent)
        np.testing.assert_array_equal(p.end.point, baseline.fillets[-1].incoming_tangent)
        result = connect_fixed_circles(p)
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.length, p.baseline_length)
        saved = p.start.point.copy()
        baseline.polyline[:] = 1000
        np.testing.assert_array_equal(p.start.point, saved)
        with self.assertRaises(ValueError):
            p.start.point[0] = 10

    def test_locally_valid_displacement_can_disconnect_but_preserves_fallback(self):
        bounds = [(0, 4, 0, 8), (0, 13, 0, 4), (9, 19, -1, 5),
                  (14, 18, 0, 12), (8, 18, 9, 13), (7, 11, 6, 13)]
        corridors = [SimpleNamespace(corners=np.array([[a, c], [b, c], [b, d], [a, d]]))
                     for a, b, c, d in bounds]
        robot = SimpleNamespace(r=1, R=3)
        baseline = compute_filleted_baseline(corridors, robot, use_joint_solver=False)
        frozen = freeze_internal_baseline(baseline, robot)
        self.assertEqual([c.waypoint_index for c in frozen.circles], [2, 3])
        self.assertTrue(connect_fixed_circles(frozen).feasible)
        center = baseline.fillets[2].center + [.1, 0]
        modified = freeze_internal_baseline(baseline, robot, circle_centers={2: center})
        center[:] = 1000
        result = connect_fixed_circles(modified)
        self.assertFalse(result.local_feasible)
        self.assertTrue(result.certified_locally_infeasible)
        self.assertAlmostEqual(modified.baseline_length, frozen.baseline_length)
        for a, b in zip(modified.baseline_primitives, frozen.baseline_primitives):
            np.testing.assert_array_equal(a['start'], b['start'])
            np.testing.assert_array_equal(a['end'], b['end'])
        with self.assertRaises(ValueError):
            freeze_internal_baseline(baseline, robot, circle_centers={1: [0, 0]})
        with self.assertRaises(ValueError):
            freeze_internal_baseline(baseline, robot, circle_centers={2: [100, 100]})


if __name__ == '__main__':
    unittest.main()
