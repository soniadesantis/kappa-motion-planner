"""Cache reuse must preserve safety when public geometry or robot data changes."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from kappa_planner import baseline_construction as construction
from kappa_planner import refinement
from kappa_planner.helpers.sequence_geometry import sequence_geometry
from kappa_planner.helpers.tangent_refinement import _directed_tangent_lines
from test_consecutive_circle_connection import circle


class GeometryReuseTest(unittest.TestCase):
    def build(self):
        boxes = [(-8, 4, 0, 4), (0, 16, 0, 4), (12, 16, 0, 16), (12, 28, 12, 16)]
        corridors = [SimpleNamespace(corners=np.array([[a, c], [b, c], [b, d], [a, d]]))
                     for a, b, c, d in boxes]
        robot = SimpleNamespace(r=1., R=3.)
        return construction.compute_filleted_baseline(corridors, robot, use_joint_solver=False), robot

    def test_unchanged_baseline_reuses_validation(self):
        baseline, robot = self.build()
        self.assertTrue(baseline.feasible)
        with patch.object(construction, '_validate_and_build_fillets', side_effect=AssertionError):
            self.assertEqual(refinement.place_refinement_circles(baseline, robot).status, 'placed')

    def test_mutated_waypoints_do_not_reuse_validation(self):
        baseline, robot = self.build()
        baseline.polyline[1] = 1000.
        with self.assertRaisesRegex(ValueError, 'Baseline is not valid'):
            refinement.place_refinement_circles(baseline, robot)

    def test_changed_region_robot_or_tolerance_revalidates(self):
        baseline, robot = self.build()
        for r, R, tol in ((1.1, 3., 1e-9), (1., 3.1, 1e-9), (1., 3., 1e-8)):
            with patch.object(construction, '_validate_and_build_fillets', return_value=None) as validate:
                self.assertFalse(construction._baseline_is_valid(baseline, r, R, tol))
                validate.assert_called_once()
        baseline.fillet_regions[1]['corner'][0] += 1.
        with patch.object(construction, '_validate_and_build_fillets', return_value=None) as validate:
            self.assertFalse(construction._baseline_is_valid(baseline, robot.r, robot.R, 1e-9))
            validate.assert_called_once()

    def test_geometry_context_keys_include_bounds_radius_and_tolerance(self):
        baseline, robot = self.build()
        data = baseline.feasibility
        context = sequence_geometry(data, robot.r, 1e-9)
        self.assertIs(sequence_geometry(data, robot.r, 1e-9), context)
        self.assertIsNot(sequence_geometry(data, 1.1, 1e-9), context)
        self.assertIsNot(sequence_geometry(data, robot.r, 1e-8), context)
        changed = replace(data, corridor_bounds=((-9, 4, 0, 4), *data.corridor_bounds[1:]))
        self.assertIsNot(sequence_geometry(changed, robot.r, 1e-9), context)
        self.assertIs(context.union(0, 2), context.union(0, 2))

    def test_circle_clipping_reuses_intervals_but_not_after_movement(self):
        baseline, robot = self.build()
        placed = refinement.place_refinement_circles(baseline, robot)
        with patch.object(refinement, 'circle_safe_angular_intervals',
                          wraps=refinement.circle_safe_angular_intervals) as clip:
            a = refinement.compute_circle_safe_arcs(placed, baseline, robot)
            b = refinement.compute_circle_safe_arcs(placed, baseline, robot)
            self.assertEqual(a, b)
            self.assertEqual(clip.call_count, len(placed.circles))
            moved = replace(placed, circles=(replace(placed.circles[0],
                            center=placed.circles[0].center+np.array([.1, 0.])),))
            refinement.compute_circle_safe_arcs(moved, baseline, robot, include_certified_fillet=False)
            self.assertEqual(clip.call_count, len(placed.circles)+1)

    def test_local_repair_checks_each_unchanged_pair_once(self):
        placed = refinement.IndependentCirclePlacements('placed', (circle(0, (0, 0)), circle(1, (1, 0))))
        baseline = SimpleNamespace()
        with patch.object(refinement, 'has_safe_circle_transition', return_value=True) as check:
            repaired = refinement.repair_same_turn_overlaps(placed, baseline, SimpleNamespace(r=1., R=2.))
            self.assertEqual(check.call_count, 1)
            self.assertEqual(repaired.same_turn_transitions, ((0, 1, True),))

    def test_raw_tangent_cache_invalidates_after_center_changes(self):
        baseline, robot = self.build()
        geometry = sequence_geometry(baseline.feasibility, robot.r, 1e-9)
        a = dict(index=0, center=np.array([0., 0.]), radius=2., turn=1, region=None)
        b = dict(index=1, center=np.array([4., 4.]), radius=2., turn=1, region=None)
        with patch('kappa_planner.helpers.tangent_refinement._directed_tangent_lines',
                   wraps=_directed_tangent_lines) as compute:
            first = geometry.tangents(a, b, compute)
            self.assertIs(geometry.tangents(a, b, compute), first)
            b['center'][1] += 1.
            second = geometry.tangents(a, b, compute)
            self.assertEqual(compute.call_count, 2)
            self.assertFalse(np.array_equal(first[0]['end'], second[0]['end']))

    def test_corridor_intersections_and_corners_are_computed_once(self):
        baseline, robot = self.build()
        geometry = sequence_geometry(baseline.feasibility, robot.r, 1e-9)
        with patch.object(refinement, '_corridor_boundary_intersections',
                          wraps=refinement._corridor_boundary_intersections) as compute:
            points = geometry.intersections(0, compute)
            self.assertIs(geometry.intersections(0, compute), points)
            self.assertEqual(compute.call_count, 1)
            self.assertIs(geometry.corners(0), geometry.corners(0))

    def test_capsule_check_is_reused_only_for_identical_endpoints(self):
        baseline, robot = self.build()
        geometry = sequence_geometry(baseline.feasibility, robot.r, 1e-9)
        union = geometry.union(0, 1)
        with patch.object(union, 'contains_capsule', wraps=union.contains_capsule) as check:
            self.assertTrue(geometry.contains_capsule(0, 1, (1., 2.), (10., 2.)))
            self.assertTrue(geometry.contains_capsule(0, 1, (1., 2.), (10., 2.)))
            self.assertTrue(geometry.contains_capsule(0, 1, (1., 2.), (11., 2.)))
            self.assertEqual(check.call_count, 2)

    def test_graph_timing_does_not_change_results(self):
        baseline, robot = self.build()
        placed = refinement.place_refinement_circles(baseline, robot)
        times = {}
        result = refinement.connect_safe_arc_circles(placed, baseline, robot, timings=times)
        self.assertTrue(result.feasible)
        self.assertEqual(set(times), {'setup_ms', 'tangent_generation_ms', 'candidate_checks_ms',
                                     'straight_containment_ms', 'graph_search_ms'})
        self.assertTrue(all(t >= 0 for t in times.values()))


if __name__ == '__main__':
    unittest.main()
