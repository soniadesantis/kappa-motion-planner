"""Boundary geometry, legacy agreement and forward-only physical recovery."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from kappa_planner import bicycle_boundary_connections as boundary
from kappa_planner.baseline_construction import QuarterCircleFillet
from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.pose_to_circle_bicycle import compute_traj_to_circle_free_space_bicycle
from kappa_planner.refinement import FixedTurningCircle, IndependentCirclePlacement
from kappa_planner.vehicle import Bicycle


class BicycleBoundaryConnectionsTest(unittest.TestCase):
    def setUp(self):
        self.bicycle = Bicycle(
            [0, 0, 0], width=.1, length=.1, wheelbase=.25,
            v_max=1., v_min=-1., delta_max=.5, delta_min=-.5)
        self.radius = self.bicycle.max_radius
        self.region = dict(corner=np.array([1., 1.]), incoming=np.array([1., 0.]),
                           outgoing=np.array([0., 1.]))
        self.circle = FixedTurningCircle(3, np.array([0., 0.]), self.radius, 1, self.region)
        self.corridor = CorridorWorld(10., 20., [0., 0.], 0.)

    def assert_continuous_tangent_connection(self, maneuvers, start, circle):
        self.assertIsNotNone(maneuvers)
        np.testing.assert_allclose(maneuvers[0].start_pose, start, atol=1e-9)
        for first, second in zip(maneuvers, maneuvers[1:]):
            np.testing.assert_allclose(first.end_pose, second.start_pose, atol=1e-9)
            self.assertAlmostEqual(first.tf, second.t0)
        last = maneuvers[-1]
        radial = np.array([last.xf, last.yf])-circle.center
        heading = np.array([np.cos(last.thetaf), np.sin(last.thetaf)])
        self.assertAlmostEqual(np.linalg.norm(radial), circle.radius)
        self.assertAlmostEqual(radial @ heading, 0.)
        self.assertGreater(circle.turn*(radial[0]*heading[1]-radial[1]*heading[0]), 0.)

    def test_free_space_matches_original_for_both_turns_and_backward_candidate(self):
        for turn in (-1, 1):
            circle = FixedTurningCircle(3, np.array([0., 0.]), self.radius, turn, self.region)
            old_circle = boundary._boundary_circle(circle, self.bicycle)
            for heading in (0., 2.5, -2.5):
                pose = [-4., 0., heading]
                old = compute_traj_to_circle_free_space_bicycle(pose, self.bicycle, old_circle)
                new = boundary.build_free_space_minimum_time_candidate(pose, self.bicycle, circle)
                self.assertEqual([p.label for p in old], [p.label for p in new])
                for a, b in zip(old, new):
                    np.testing.assert_allclose(a.end_pose, b.end_pose)
                    self.assertAlmostEqual(a.tf, b.tf)
                self.assert_continuous_tangent_connection(new, pose, circle)

    def test_baseline_and_refinement_types_have_identical_connections(self):
        R = self.radius
        fillet = QuarterCircleFillet(3, np.array([0., 0.]), np.array([0., -R]),
                                    np.array([R, 0.]), R, np.pi/2)
        proposal = IndependentCirclePlacement(3, 'test', np.array([0., 0.]),
                                             np.array([R, -R]), R, self.region, True)
        results = []
        for circle in (fillet, proposal, self.circle):
            results.append(boundary.build_pose_to_circle_connection(
                self.corridor, self.corridor, [-4., 0., 0.], self.bicycle,
                circle, corner_point=self.region['corner']))
        for result in results:
            self.assert_continuous_tangent_connection(result, [-4., 0., 0.], self.circle)
            self.assertAlmostEqual(result[-1].tf, results[0][-1].tf)

    def test_recovery_aligns_forward_and_reverses_for_extra_separation(self):
        pose = [-.1, 0., .3]
        with patch.object(boundary, '_compute_alignment_state',
                          wraps=boundary._compute_alignment_state) as align:
            result = boundary.build_pose_to_circle_connection(
                self.corridor, self.corridor, pose, self.bicycle, self.circle)
        align.assert_called_once()
        self.assertEqual(align.call_args.kwargs['preferred_direction'], 1)
        self.assert_continuous_tangent_connection(result, pose, self.circle)
        self.assertAlmostEqual(result[0].thetaf, self.corridor.tilt)
        self.assertEqual(result[1].label, 'segment')
        self.assertLess(result[1].v, 0.)
        np.testing.assert_array_equal(self.circle.center, [0., 0.])

    def test_inside_target_can_recover_but_disabled_recovery_fails(self):
        pose = [0., 0., 0.]
        self.assertIsNone(boundary.build_free_space_minimum_time_candidate(
            pose, self.bicycle, self.circle))
        self.assertIsNone(boundary.build_pose_to_circle_connection(
            self.corridor, self.corridor, pose, self.bicycle, self.circle, recovery=False))
        result = boundary.build_pose_to_circle_connection(
            self.corridor, self.corridor, pose, self.bicycle, self.circle)
        self.assert_continuous_tangent_connection(result, pose, self.circle)

    def test_existing_wall_tangent_backward_correction_is_retained(self):
        corridor = CorridorWorld(1., 12., [0., 0.], 0.)
        circle = FixedTurningCircle(3, np.array([0., 0.]), self.radius, 1,
                                    dict(self.region, corner=np.array([.5, .5])))
        pose = [-3., .4, .5]
        with patch.object(boundary, 'compute_backward_arc',
                          wraps=boundary.compute_backward_arc) as correction:
            result = boundary.build_pose_to_circle_connection(
                corridor, corridor, pose, self.bicycle, circle, recovery=False)
        correction.assert_called_once()
        self.assertEqual(result[0].label, 'backward arc')
        self.assert_continuous_tangent_connection(result, pose, circle)
        for arc in result[:-1]:
            self.assertFalse(boundary.check_arc_collision(arc, corridor)[0])

    def test_original_construction_has_no_extra_segment_containment_check(self):
        circle = FixedTurningCircle(3, np.array([20., 0.]), self.radius, 1, self.region)
        self.assertIsNotNone(boundary.build_free_space_minimum_time_candidate(
            [-4., 0., 0.], self.bicycle, circle))
        result = boundary.build_pose_to_circle_connection(
            self.corridor, self.corridor, [-4., 0., 0.], self.bicycle,
            circle, recovery=False)
        self.assert_continuous_tangent_connection(result, [-4., 0., 0.], circle)

    def test_insufficient_room_for_recovery_returns_none(self):
        corridor = CorridorWorld(10., .4, [0., 0.], 0.)
        self.assertIsNone(boundary.build_pose_to_circle_connection(
            corridor, self.corridor, [0., 0., 0.], self.bicycle, self.circle))

    def test_invalid_inputs_raise_instead_of_reporting_geometric_failure(self):
        for pose in ([0., 0.], [0., 0., np.nan]):
            with self.assertRaises(ValueError):
                boundary.build_free_space_minimum_time_candidate(pose, self.bicycle, self.circle)
        circle = SimpleNamespace(center=[0., 0.], radius=2*self.radius, turn=1)
        with self.assertRaises(ValueError):
            boundary.build_free_space_minimum_time_candidate([-4., 0., 0.], self.bicycle, circle)


if __name__ == '__main__':
    unittest.main()
