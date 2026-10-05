"""Centroid turn signs and boundary-aware baseline construction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from kappa_planner import baseline_construction as baseline


def rectangles(bounds, transform=np.eye(2)):
    return [SimpleNamespace(corners=np.array([[a, c], [b, c], [b, d], [a, d]]) @ transform.T)
            for a, b, c, d in bounds]


class PoseDirectedBaselineTest(unittest.TestCase):
    def setUp(self):
        self.bounds = [(0, 4, -8, 4), (0, 16, 0, 4), (12, 16, -8, 4)]
        self.corridors = rectangles(self.bounds)
        self.robot = SimpleNamespace(r=1., R=3.)

    def test_diagonal_segment_turn_sign_for_every_internal_direction(self):
        door = (1., 3., 1., 3.)
        centroid = np.array([2., 2.])
        for direction, internal in baseline._DIRECTION_VECTORS.items():
            normal = np.array([-internal[1], internal[0]])
            for turn in (-1, 1):
                incoming = internal-turn*normal
                outgoing = internal+turn*normal
                initial = baseline.infer_initial_pose_boundary_direction(
                    [*(centroid-incoming), .7], door, direction)
                final = baseline.infer_final_pose_boundary_direction(
                    door, [*(centroid+outgoing), -.8], direction)
                u, v = baseline._DIRECTION_VECTORS[initial], baseline._DIRECTION_VECTORS[final]
                self.assertEqual(u[0]*internal[1]-u[1]*internal[0], turn)
                self.assertEqual(internal[0]*v[1]-internal[1]*v[0], turn)
                self.assertEqual(u @ internal, 0.)
                self.assertEqual(internal @ v, 0.)

    def test_collinear_coincident_and_heading_independence(self):
        door = (1., 3., 1., 3.)
        for heading in (-2., 0., 2.):
            self.assertEqual(baseline.infer_initial_pose_boundary_direction(
                [-1., 2., heading], door, 'right'), 'right')
            self.assertEqual(baseline.infer_initial_pose_boundary_direction(
                [4., 2., heading], door, 'right'), 'left')
            self.assertEqual(baseline.infer_final_pose_boundary_direction(
                door, [4., 2., heading], 'right'), 'right')
            self.assertEqual(baseline.infer_final_pose_boundary_direction(
                door, [-1., 2., heading], 'right'), 'left')
            self.assertIsNone(baseline.infer_initial_pose_boundary_direction(
                [2., 2., heading], door, 'right'))

    def test_pose_rule_is_used_by_all_construction_methods(self):
        for method in ('heuristic', 'exact', 'segment'):
            result = baseline.compute_boundary_directed_baseline(
                self.corridors, self.robot, initial_pose=[-2., -2., .4],
                final_pose=[18., -2., 2.], method=method, use_joint_solver=False)
            self.assertTrue(result.feasible, result.reason)
            self.assertEqual(result.segment_directions, ('up', 'right', 'down'))
            self.assertTrue(all(f is not None for f in result.fillets))
            # Poses select virtual directions; they are not pinned waypoints.
            self.assertFalse(np.array_equal(result.polyline[0], [-2., -2.]))

    def test_missing_poses_prefer_perpendicular_then_straight(self):
        result = baseline.compute_boundary_directed_baseline(
            self.corridors, self.robot, use_joint_solver=False)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.segment_directions, ('up', 'right', 'down'))
        for bottom in (0., -.1):
            corridors = rectangles([(0, 4, bottom, 4), (0, 16, 0, 4), (12, 16, bottom, 4)])
            result = baseline.compute_boundary_directed_baseline(
                corridors, self.robot, use_joint_solver=False)
            self.assertTrue(result.feasible, result.reason)
            self.assertEqual(result.segment_directions, ('right', 'right', 'right'))
            self.assertEqual(result.fillets, (None, None))

    def test_single_pose_and_centroid_fallback(self):
        for pose in ([-2., -2., .4], [2., 2., .4]):
            result = baseline.compute_boundary_directed_baseline(
                self.corridors, self.robot, initial_pose=pose, use_joint_solver=False)
            self.assertTrue(result.feasible, result.reason)
            self.assertEqual(result.segment_directions, ('up', 'right', 'down'))

    def test_pose_selected_turn_is_not_replaced_on_failure(self):
        result = baseline.compute_boundary_directed_baseline(
            self.corridors, self.robot, initial_pose=[2., 8., 0.], use_joint_solver=False)
        self.assertFalse(result.feasible)
        self.assertEqual(result.initial_direction, 'down')
        self.assertTrue(result.feasibility.feasible)
        reversal = baseline.compute_boundary_directed_baseline(
            self.corridors, self.robot, initial_pose=[5., 2., 0.], use_joint_solver=False)
        self.assertEqual(reversal.status, 'unsupported_turn')
        self.assertFalse(reversal.certified_infeasible)

    def test_transformations_preserve_pose_turns(self):
        for transform in (np.eye(2), -np.eye(2), np.diag([-1., 1.]),
                          np.array([[0., -1.], [1., 0.]])):
            result = baseline.compute_boundary_directed_baseline(
                rectangles(self.bounds, transform), self.robot,
                initial_pose=[*(transform @ [-2., -2.]), .4],
                final_pose=[*(transform @ [18., -2.]), 2.], use_joint_solver=False)
            self.assertTrue(result.feasible, result.reason)
            np.testing.assert_array_equal(baseline._DIRECTION_VECTORS[result.initial_direction],
                                          transform @ [0., 1.])
            np.testing.assert_array_equal(baseline._DIRECTION_VECTORS[result.final_direction],
                                          transform @ [0., -1.])

    def test_fallback_pairs_share_internal_analysis_and_geometry(self):
        corridors = rectangles([(0, 4, -.1, 4), (0, 16, 0, 4), (12, 16, -.1, 4)])
        with patch.object(baseline, 'analyze_orthogonal_polyline_feasibility',
                          wraps=baseline.analyze_orthogonal_polyline_feasibility) as analyze:
            result = baseline.compute_boundary_directed_baseline(
                corridors, self.robot, use_joint_solver=False)
        self.assertTrue(result.feasible)
        analyze.assert_called_once()

    def test_invalid_pose_and_method(self):
        for pose in ([1., 2.], [1., 2., np.nan]):
            with self.assertRaises(ValueError):
                baseline.compute_boundary_directed_baseline(
                    self.corridors, self.robot, initial_pose=pose)
        with self.assertRaises(ValueError):
            baseline.compute_boundary_directed_baseline(self.corridors, self.robot, method='bad')


if __name__ == '__main__':
    unittest.main()
