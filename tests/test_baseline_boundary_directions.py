"""Virtual boundary directions add local arcs without boundary segment constraints."""

import unittest
from types import SimpleNamespace

import numpy as np

from kappa_planner.baseline_construction import (
    analyze_orthogonal_polyline_feasibility, compute_filleted_baseline,
    infer_initial_boundary_direction, infer_final_boundary_direction,
)


def rectangle(bounds, transform=np.eye(2), shift=(0, 0)):
    a, b, c, d = bounds
    return SimpleNamespace(corners=np.array([[a, c], [b, c], [b, d], [a, d]])
                           @ transform.T + shift)


class BoundaryDirectionTest(unittest.TestCase):
    def setUp(self):
        self.bounds = [(0, 4, -8, 4), (0, 16, 0, 4), (12, 16, -8, 4)]
        self.corridors = [rectangle(b) for b in self.bounds]
        self.robot = SimpleNamespace(r=1., R=3.)

    def test_signed_inference_and_unresolved_cases(self):
        door = (1, 3, 2, 4)
        for point, entry, exit_direction in (
            ((0, 3), 'right', 'left'), ((4, 3), 'left', 'right'),
            ((2, 1), 'up', 'down'), ((2, 5), 'down', 'up'),
            ((0, 1), None, None), ((2, 3), None, None), ((1, 2), None, None),
        ):
            self.assertEqual(infer_initial_boundary_direction(point, door), entry)
            self.assertEqual(infer_final_boundary_direction(door, point), exit_direction)
        self.assertIsNone(infer_initial_boundary_direction((1-1e-10, 3), door))
        self.assertEqual(infer_initial_boundary_direction((0, 2-1e-10), door), 'right')
        self.assertEqual(infer_initial_boundary_direction((0, 2), (1, 1, 2, 2)), 'right')

    def test_input_validation(self):
        for point in ([1], [1, 2, 3], [np.nan, 2], [1, np.inf]):
            with self.assertRaises(ValueError):
                infer_initial_boundary_direction(point, (1, 3, 2, 4))
        with self.assertRaises(ValueError):
            infer_final_boundary_direction((3, 1, 2, 4), (0, 3))
        with self.assertRaises(ValueError):
            infer_initial_boundary_direction((0, 3), (1, 3, 2, 4), tol=-1)

    def test_both_endpoint_arcs_and_no_boundary_alignment_or_distance(self):
        plain = compute_filleted_baseline(self.corridors, self.robot, use_joint_solver=False)
        start, end = [1.1, -100], [14.9, -100]
        result = compute_filleted_baseline(self.corridors, self.robot,
                                          initial_position=start, final_position=end,
                                          use_joint_solver=False)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.segment_directions, ('up', 'right', 'down'))
        self.assertEqual(result.initial_direction, 'up')
        self.assertEqual(result.final_direction, 'down')
        self.assertTrue(all(f is not None for f in result.fillets))
        self.assertNotAlmostEqual(result.polyline[0, 0], start[0])
        self.assertNotAlmostEqual(result.polyline[-1, 0], end[0])
        self.assertEqual(result.feasibility, plain.feasibility)
        np.testing.assert_array_equal(result.orthogonal_polyline, plain.orthogonal_polyline)
        length = np.linalg.norm(result.polyline[1] - result.polyline[0])
        self.assertAlmostEqual(result.remaining_lengths[0], length - 2*self.robot.R)
        # Verify endpoint tangents against the proper adjacent eroded corridors.
        for j, fillet in enumerate(result.fillets):
            for tangent, bounds in zip((fillet.incoming_tangent, fillet.outgoing_tangent),
                                       self.bounds[j:j+2]):
                a, b, c, d = bounds
                self.assertTrue(np.all(tangent >= [a+1, c+1]))
                self.assertTrue(np.all(tangent <= [b-1, d-1]))
            # Independently audit both boundary arcs against the original union.
            radial = fillet.incoming_tangent - fillet.center
            angles = np.arctan2(radial[1], radial[0]) + np.linspace(0, fillet.signed_angle, 101)
            arc = fillet.center + self.robot.R * np.column_stack((np.cos(angles), np.sin(angles)))
            theta = np.linspace(0, 2*np.pi, 97)
            probes = (arc[:, None, :] + self.robot.r
                      * np.column_stack((np.cos(theta), np.sin(theta)))).reshape(-1, 2)
            contained = np.zeros(len(probes), dtype=bool)
            for a, b, c, d in self.bounds[j:j+2]:
                contained |= ((probes[:, 0] >= a-1e-9) & (probes[:, 0] <= b+1e-9)
                              & (probes[:, 1] >= c-1e-9) & (probes[:, 1] <= d+1e-9))
            self.assertTrue(contained.all())

    def test_single_boundary_and_collinear_boundary(self):
        result = compute_filleted_baseline(self.corridors, self.robot,
                                          initial_position=[2, -1], use_joint_solver=False)
        self.assertTrue(result.feasible, result.reason)
        self.assertIsNotNone(result.fillets[0])
        self.assertIsNone(result.fillets[-1])
        result = compute_filleted_baseline(self.corridors, self.robot,
                                          initial_position=[0, 2], final_position=[16, 2],
                                          use_joint_solver=False)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.fillets, (None, None))

    def test_unresolved_and_reversal_leave_exact_feasibility_intact(self):
        for point, status in (([-1, -1], 'boundary_direction_unresolved'),
                              ([2, 2], 'boundary_direction_unresolved'),
                              ([4, 2], 'unsupported_turn')):
            result = compute_filleted_baseline(self.corridors, self.robot,
                                              initial_position=point, use_joint_solver=False)
            self.assertEqual(result.status, status)
            self.assertFalse(result.certified_infeasible)
            self.assertTrue(result.feasibility.feasible)
            self.assertIsNotNone(result.orthogonal_polyline)

    def test_terminal_curved_region_midpoint_failure_uses_witness(self):
        result = compute_filleted_baseline(self.corridors, SimpleNamespace(r=1, R=6),
                                          initial_position=[2, -1], final_position=[14, -1],
                                          use_joint_solver=False)
        self.assertTrue(result.feasible, result.reason)
        self.assertFalse(result.midpoint_only_success)
        self.assertEqual(result.first_rejected_waypoint, 1)
        self.assertGreater(result.backtracking_attempts, 2)
        self.assertTrue(np.all(result.remaining_lengths >= 0))

    def test_forced_joint_fallback_handles_endpoint_regions(self):
        result = compute_filleted_baseline(self.corridors, self.robot,
                                          initial_position=[2, -1], final_position=[14, -1],
                                          max_backtracking_attempts=0)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.selection_method, 'joint_solver_fallback')
        self.assertTrue(all(f is not None for f in result.fillets))

    def test_signed_transformations_preserve_boundary_construction(self):
        for transform in (np.eye(2), -np.eye(2), np.diag([-1., 1.]),
                          np.array([[0., -1.], [1., 0.]])):
            shift = np.array([23., -17.])
            corridors = [rectangle(b, transform, shift) for b in self.bounds]
            result = compute_filleted_baseline(
                corridors, self.robot, initial_position=np.array([2, -1]) @ transform.T + shift,
                final_position=np.array([14, -1]) @ transform.T + shift, use_joint_solver=False)
            self.assertTrue(result.feasible, result.reason)
            self.assertTrue(all(f is not None for f in result.fillets))
            self.assertEqual(result.feasibility,
                             analyze_orthogonal_polyline_feasibility(corridors, self.robot,
                                                                    compute_viable=False))


if __name__ == '__main__':
    unittest.main()
