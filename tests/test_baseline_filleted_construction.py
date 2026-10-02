"""Signed construction, bounded-search semantics and independent arc geometry."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from kappa_planner.baseline_construction import (
    OrthogonalPolylineFeasibility,
    _backtrack_filleted_polyline,
    compute_filleted_baseline,
    compute_orthogonal_polyline,
)
from kappa_planner.helpers.fillet_safety import region_contains_point


def rectangle(xmin, xmax, ymin=0, ymax=4):
    return SimpleNamespace(corners=np.array([
        [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax],
    ], dtype=float))


def thesis_example():
    return [rectangle(*bounds) for bounds in (
        (0, 4, 0, 8), (0, 13, 0, 4), (9, 19, -1, 5),
        (14, 18, 0, 12), (8, 18, 9, 13), (7, 11, 6, 13),
    )], SimpleNamespace(r=1, R=3)


class BaselineFilletedConstructionTest(unittest.TestCase):
    def assert_valid_chain(self, result, robot):
        self.assertTrue(result.feasible, result.reason)
        self.assertFalse(result.certified_infeasible)
        data, points = result.feasibility, result.polyline
        vectors = {'right': [1, 0], 'left': [-1, 0], 'up': [0, 1], 'down': [0, -1]}
        for j, (point, door) in enumerate(zip(points, data.safe_overlaps)):
            self.assertTrue(door[0] - 1e-8 <= point[0] <= door[1] + 1e-8)
            self.assertTrue(door[2] - 1e-8 <= point[1] <= door[3] + 1e-8)
            for k, reachable in enumerate((data.x_reachable, data.y_reachable)):
                self.assertTrue(reachable[j][0] - 1e-8 <= point[k] <= reachable[j][1] + 1e-8)
        for delta, direction in zip(np.diff(points, axis=0), data.passage_directions):
            vector = np.array(vectors[direction])
            length = delta @ vector
            self.assertGreaterEqual(length, 2 * robot.R - 1e-8)
            np.testing.assert_allclose(delta, length * vector, atol=1e-8)
        self.assertTrue(np.all(result.remaining_lengths >= -1e-8))
        # Independent geometric audit: sample each quarter arc and the circular
        # footprint perimeter against the ORIGINAL local rectangle union.
        for j, fillet in enumerate(result.fillets):
            if fillet is None:
                continue
            region = result.fillet_regions[j]
            self.assertTrue(region_contains_point(points[j], region))
            self.assertEqual(fillet.waypoint_index, j)
            self.assertAlmostEqual(abs(fillet.signed_angle), np.pi / 2)
            incoming = np.array(vectors[data.passage_directions[j - 1]])
            outgoing = np.array(vectors[data.passage_directions[j]])
            np.testing.assert_allclose(fillet.center,
                                       points[j] - robot.R * incoming + robot.R * outgoing)
            angles = np.linspace(0, np.pi / 2, 101)
            arc = fillet.center + robot.R * (
                -np.cos(angles)[:, None] * outgoing + np.sin(angles)[:, None] * incoming)
            np.testing.assert_allclose(arc[0], fillet.incoming_tangent)
            np.testing.assert_allclose(arc[-1], fillet.outgoing_tangent)
            footprint_angles = np.linspace(0, 2 * np.pi, 97)
            offsets = robot.r * np.column_stack((np.cos(footprint_angles),
                                                 np.sin(footprint_angles)))
            probes = (arc[:, None, :] + offsets).reshape(-1, 2)
            contained = np.zeros(len(probes), dtype=bool)
            for xmin, xmax, ymin, ymax in data.corridor_bounds[j:j + 2]:
                contained |= ((probes[:, 0] >= xmin - 1e-8)
                              & (probes[:, 0] <= xmax + 1e-8)
                              & (probes[:, 1] >= ymin - 1e-8)
                              & (probes[:, 1] <= ymax + 1e-8))
            self.assertTrue(contained.all(), f'Footprint leaves local union at arc {j}')

    def test_thesis_example_keeps_aligned_waypoint_and_two_fillets(self):
        corridors, robot = thesis_example()
        with patch('kappa_planner.baseline_construction._joint_fillet_fallback',
                   side_effect=AssertionError('Unnecessary joint solver')):
            result = compute_filleted_baseline(corridors, robot)
        self.assert_valid_chain(result, robot)
        self.assertEqual(result.selection_method, 'midpoint_backtracking')
        self.assertEqual(result.backtracking_attempts, 5)
        self.assertTrue(result.midpoint_only_success)
        self.assertIsNone(result.first_rejected_waypoint)
        self.assertIsNone(result.fillet_regions[1])
        self.assertIsNone(result.fillets[1])
        self.assertEqual([j for j, f in enumerate(result.fillets) if f], [2, 3])

    def test_rotations_reflections_and_translation(self):
        original, robot = thesis_example()
        for transform in (np.eye(2), -np.eye(2), np.diag([-1., 1.]),
                          np.array([[0., -1.], [1., 0.]])):
            corridors = [SimpleNamespace(corners=c.corners @ transform.T + [37, -19])
                         for c in original]
            result = compute_filleted_baseline(corridors, robot)
            self.assert_valid_chain(result, robot)

    def test_forced_joint_solver_fallback(self):
        corridors, robot = thesis_example()
        result = compute_filleted_baseline(corridors, robot, max_backtracking_attempts=0)
        self.assert_valid_chain(result, robot)
        self.assertEqual(result.selection_method, 'joint_solver_fallback')
        self.assertEqual(result.backtracking_attempts, 0)
        self.assertFalse(result.midpoint_only_success)

    def test_budget_exhaustion_without_fallback_is_unresolved(self):
        corridors, robot = thesis_example()
        result = compute_filleted_baseline(corridors, robot,
                                          max_backtracking_attempts=1, use_joint_solver=False)
        self.assertFalse(result.feasible)
        self.assertFalse(result.certified_infeasible)
        self.assertEqual(result.status, 'backtracking_unresolved')
        self.assertEqual(result.backtracking_attempts, 1)
        self.assertIsNone(result.polyline)
        self.assertIsNotNone(result.orthogonal_polyline)

    def test_all_collinear_has_no_fillet_regions_or_arcs(self):
        corridors = [rectangle(0, 4), rectangle(0, 10),
                     rectangle(6, 16), rectangle(12, 16)]
        robot = SimpleNamespace(r=1, R=2)
        result = compute_filleted_baseline(corridors, robot)
        self.assert_valid_chain(result, robot)
        self.assertEqual(result.fillet_regions, (None,) * 3)
        self.assertEqual(result.fillets, (None,) * 3)

    def test_single_segment_and_degenerate_doors_at_exact_2R(self):
        corridors = [rectangle(0, 2, 0, 2), rectangle(0, 5, 0, 2),
                     rectangle(3, 5, 0, 2)]
        robot = SimpleNamespace(r=1, R=1.5)
        result = compute_filleted_baseline(corridors, robot)
        self.assert_valid_chain(result, robot)
        np.testing.assert_allclose(result.polyline, [[1, 1], [4, 1]])
        np.testing.assert_allclose(compute_orthogonal_polyline(corridors, robot),
                                   result.polyline)

    def test_empty_local_region_does_not_erase_polyline_feasibility(self):
        corridors = [rectangle(0, 3, 0, 2.2), rectangle(0, 8, 0, 2.2),
                     rectangle(6, 8, 0, 9), rectangle(6, 14, 6, 9)]
        robot = SimpleNamespace(r=1, R=1.5)
        result = compute_filleted_baseline(corridors, robot)
        self.assertTrue(result.feasibility.feasible)
        self.assertFalse(result.feasible)
        self.assertTrue(result.certified_infeasible)
        self.assertEqual(result.status, 'empty_fillet_region')
        self.assertIsNotNone(result.orthogonal_polyline)

    def test_nominal_assumption_failure_is_not_path_infeasibility(self):
        result = compute_filleted_baseline([rectangle(0, 8)] * 3,
                                          SimpleNamespace(r=1, R=2))
        self.assertFalse(result.feasible)
        self.assertFalse(result.certified_infeasible)
        self.assertEqual(result.status, 'intersecting_safe_overlaps')
        self.assertIsNone(result.orthogonal_polyline)

    def test_missing_connection_and_spacing_failure_are_distinguished(self):
        robot = SimpleNamespace(r=1, R=2)
        for corridors, status in (
            ([rectangle(0, 3, 0, 3), rectangle(0, 9, 0, 9),
              rectangle(6, 9, 6, 9)], 'no_orthogonal_connection'),
            ([rectangle(0, 2, 0, 2), rectangle(0, 5, 0, 2),
              rectangle(3, 5, 0, 2)], 'insufficient_spacing'),
        ):
            result = compute_filleted_baseline(corridors, robot)
            self.assertEqual(result.status, status)
            self.assertTrue(result.certified_infeasible)
            self.assertIsNone(compute_orthogonal_polyline(corridors, robot))

    def test_unresolved_corner_geometry(self):
        corridors, robot = thesis_example()
        with patch('kappa_planner.helpers.sequence_geometry.axis_aligned_concave_corners',
                   return_value=np.empty((0, 2))):
            result = compute_filleted_baseline(corridors, robot)
        self.assertEqual(result.status, 'corner_unresolved')
        self.assertFalse(result.certified_infeasible)

    def test_joint_solver_numerical_failure_is_unresolved(self):
        corridors, robot = thesis_example()
        with patch('kappa_planner.baseline_construction._joint_fillet_fallback',
                   return_value=dict(feasible=False, status='NUMERICAL_SEARCH_UNRESOLVED',
                                     reason='Not converged')):
            result = compute_filleted_baseline(corridors, robot, max_backtracking_attempts=0)
        self.assertFalse(result.feasible)
        self.assertFalse(result.certified_infeasible)
        self.assertEqual(result.status, 'numerical_search_unresolved')

    def test_joint_solver_false_witness_is_not_accepted(self):
        corridors, robot = thesis_example()
        with patch('kappa_planner.baseline_construction._joint_fillet_fallback',
                   return_value=dict(feasible=True, points=np.zeros((5, 2)))):
            result = compute_filleted_baseline(corridors, robot, max_backtracking_attempts=0)
        self.assertFalse(result.feasible)
        self.assertFalse(result.certified_infeasible)
        self.assertEqual(result.status, 'validation_failed')

    def test_curved_slice_and_terminal_alternatives(self):
        data = OrthogonalPolylineFeasibility(
            True, 'feasible', safe_overlaps=((0, 1, 0, 0), (3, 4, 0, 0), (3, 4, 2, 3)),
            passage_directions=('right', 'up'),
            x_reachable=((0, 1), (3, 4), (3, 4)), y_reachable=((0, 0), (0, 0), (2, 3)))
        region = dict(low=np.array([3., 0.]), high=np.array([4., 0.]), empty=False,
                      radius=1., frame=np.eye(2), offset=np.array([-3., .95]))
        diagnostics = {}
        points, attempts = _backtrack_filleted_polyline(
            data, [None, region, None], 1, 1e-9, 128, diagnostics)
        self.assertIsNotNone(points)
        self.assertGreater(attempts, 3)
        self.assertLessEqual(attempts, 128)
        self.assertTrue(region_contains_point(points[1], region))
        self.assertLess(points[1, 0], 3.5)
        self.assertEqual(diagnostics['first_rejected_waypoint'], 1)

    def test_long_chain_avoids_recursion_limit(self):
        count = 1100
        x = tuple((2. * j, 2. * j) for j in range(count))
        y = ((0., 0.),) * count
        data = OrthogonalPolylineFeasibility(
            True, 'feasible', safe_overlaps=tuple((*a, *b) for a, b in zip(x, y)),
            passage_directions=('right',) * (count - 1), x_reachable=x, y_reachable=y)
        points, attempts = _backtrack_filleted_polyline(data, [None] * count, 1, 0, count)
        self.assertEqual(attempts, count)
        np.testing.assert_array_equal(points[:, 0], np.arange(count) * 2)

    def test_invalid_budgets(self):
        corridors, robot = thesis_example()
        for budget in (-1, 1.5, True, float('nan')):
            with self.assertRaises(ValueError):
                compute_filleted_baseline(corridors, robot, max_backtracking_attempts=budget)


if __name__ == '__main__':
    unittest.main()
