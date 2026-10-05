"""Optional physical connections: continuity, safe joins and independent failures."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from kappa_planner.baseline_construction import compute_baseline
from kappa_planner import bicycle_boundary_connections as boundary
from kappa_planner.vehicle import Bicycle


def rectangles(bounds):
    return [SimpleNamespace(corners=np.array([[a, c], [b, c], [b, d], [a, d]], float))
            for a, b, c, d in bounds]


class BaselineBoundaryConnectionsTest(unittest.TestCase):
    def setUp(self):
        self.vehicle = Bicycle(width=.1, length=.1, wheelbase=.25,
                               v_max=1., v_min=-1., delta_max=.5, delta_min=-.5)
        self.corridors = rectangles([(0, 4, -8, 4), (0, 16, 0, 4), (12, 16, -8, 4)])
        self.start = np.array([2., -6., np.pi/2])
        self.goal = np.array([14., -6., -np.pi/2])

    def compute(self, **options):
        return compute_baseline(self.corridors, self.vehicle, self.start, self.goal, **options)

    def assert_pose(self, actual, expected):
        np.testing.assert_allclose(actual[:2], expected[:2], atol=1e-8)
        self.assertAlmostEqual(np.sin(actual[2]-expected[2]), 0., places=8)
        self.assertGreater(np.cos(actual[2]-expected[2]), .99999999)

    def test_disabled_does_not_run_connections_or_change_baseline(self):
        with patch.object(boundary, 'build_baseline_boundary_connections',
                          side_effect=AssertionError('Disabled stage executed')):
            result = self.compute()
        self.assertTrue(result.feasible)
        self.assertFalse(result.boundary_connections_requested)
        self.assertIsNone(result.initial_connection)
        self.assertIsNone(result.final_connection)
        self.assertGreaterEqual(result.total_time_ms, result.baseline_time_ms)

    def test_both_ends_join_the_internal_baseline_with_matching_heading(self):
        without = self.compute()
        result = self.compute(connect_boundaries=True)
        np.testing.assert_array_equal(result.polyline, without.polyline)
        self.assertTrue(result.initial_connection.connected)
        self.assertTrue(result.final_connection.connected)
        self.assertGreater(result.total_time_ms, result.baseline_time_ms)
        incoming = result.initial_connection.maneuvers
        outgoing = result.final_connection.maneuvers
        self.assert_pose(incoming[0].start_pose, self.start)
        self.assert_pose(outgoing[-1].end_pose, self.goal)
        self.assert_pose(incoming[-1].end_pose, [*result.fillets[0].outgoing_tangent, 0.])
        self.assert_pose(outgoing[0].start_pose, [*result.fillets[-1].incoming_tangent, 0.])
        for pieces in (incoming, outgoing):
            for first, second in zip(pieces, pieces[1:]):
                self.assert_pose(first.end_pose, second.start_pose)
                self.assertAlmostEqual(first.tf, second.t0)

    def test_one_failed_end_keeps_the_other_and_baseline(self):
        original = boundary.build_pose_to_circle_connection
        def fail_initial(c1, c2, pose, *args, **kwargs):
            return None if pose[0] < 8 else original(c1, c2, pose, *args, **kwargs)
        with patch.object(boundary, 'build_pose_to_circle_connection', side_effect=fail_initial):
            result = self.compute(connect_boundaries=True)
        self.assertTrue(result.feasible)
        self.assertEqual(result.initial_connection.status, 'failed')
        self.assertEqual(result.initial_connection.maneuvers, ())
        self.assertTrue(result.final_connection.connected)

    def test_missing_pose_and_infeasible_baseline_skip_physical_work(self):
        with patch.object(boundary, 'build_pose_to_circle_connection',
                          side_effect=AssertionError('Unnecessary physical construction')):
            missing = compute_baseline(self.corridors, self.vehicle, connect_boundaries=True)
            self.assertEqual(missing.initial_connection.status, 'missing_pose')
            self.assertEqual(missing.final_connection.status, 'missing_pose')
            invalid = compute_baseline(rectangles([(0, 4, 0, 4)]*3), self.vehicle,
                                       self.start, self.goal, connect_boundaries=True)
            self.assertFalse(invalid.feasible)
            self.assertEqual(invalid.initial_connection.status, 'baseline_unavailable')
            self.assertEqual(invalid.final_connection.status, 'baseline_unavailable')

    def test_collinear_boundaries_connect_without_endpoint_fillets(self):
        result = compute_baseline(
            rectangles([(0, 4, 0, 4), (0, 16, 0, 4), (12, 16, 0, 4)]), self.vehicle,
            [.1, 2., 0.], [15.9, 2., 0.], connect_boundaries=True)
        self.assertTrue(result.feasible)
        self.assertEqual(result.fillets, (None, None))
        self.assertTrue(result.initial_connection.connected)
        self.assertTrue(result.final_connection.connected)
        self.assert_pose(result.initial_connection.maneuvers[-1].end_pose, [*result.polyline[0], 0.])
        self.assert_pose(result.final_connection.maneuvers[0].start_pose, [*result.polyline[-1], 0.])

    def test_target_circle_arc_cannot_cross_an_unsafe_gap(self):
        radius = self.vehicle.max_radius
        circle = SimpleNamespace(center=np.zeros(2), radius=radius, turn=1)
        contact = SimpleNamespace(xf=radius, yf=0.)
        completed = boundary._complete_circle_connection(
            [contact], circle, [-radius, 0.], ((0., .1), (np.pi-.1, np.pi+.1)),
            self.vehicle, 1e-9)
        self.assertIsNone(completed)


if __name__ == '__main__':
    unittest.main()
