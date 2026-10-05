"""Direct pipeline behavior and independent geometry checks outside runtime."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from kappa_planner import baseline_construction as baseline
from kappa_planner.helpers.fillet_reachability import RoundedCornerConstraint, _make_set
from kappa_planner.helpers.fillet_segment_reachability import intersect_baseline_region


def rectangles(bounds):
    return [SimpleNamespace(corners=np.array([[a, c], [b, c], [b, d], [a, d]], float))
            for a, b, c, d in bounds]


class DirectBaselineTest(unittest.TestCase):
    def setUp(self):
        self.corridors = rectangles([(0, 4, -8, 4), (0, 16, 0, 4), (12, 16, -8, 4)])
        self.vehicle = SimpleNamespace(r=1., R=3.)
        self.poses = dict(initial_pose=[-2., -2., .4], final_pose=[18., -2., 2.])

    def test_direct_steps_do_not_invoke_legacy_checks_or_construction(self):
        legacy = ('analyze_orthogonal_polyline_feasibility', '_recover_orthogonal_polyline',
                  '_backtrack_filleted_polyline', '_joint_fillet_fallback',
                  '_validate_and_build_fillets', 'propagate_fillet_regions_segment')
        patches = [patch.object(baseline, name, side_effect=AssertionError(name)) for name in legacy]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        result = baseline.compute_baseline(self.corridors, self.vehicle, **self.poses)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.selection_method, 'segment_midpoints')
        self.assertEqual(result.backtracking_attempts, 0)
        self.assertIsNone(result.max_violation)

    def test_stored_intersections_supply_the_selected_corners_and_shared_cache(self):
        with patch.object(baseline, 'corridor_boundary_intersections',
                          wraps=baseline.corridor_boundary_intersections) as intersections:
            result = baseline.compute_baseline(self.corridors, self.vehicle, **self.poses)
        self.assertEqual(intersections.call_count, 2)
        self.assertEqual(result.overlaps, ((0., 4., 0., 4.), (12., 16., 0., 4.)))
        self.assertEqual(result.feasibility.safe_overlaps, ((1., 3., 1., 3.), (13., 15., 1., 3.)))
        np.testing.assert_array_equal(result.corner_points, [[4., 0.], [12., 0.]])
        for j, corner in enumerate(result.corner_points):
            points, _ = result.boundary_intersections[j]
            self.assertTrue(any(np.array_equal(corner, point) for point in points))
            cached = result.feasibility._geometry.intersections(
                j, lambda *args: self.fail('Stored intersections must be reused.'))
            self.assertIs(cached[0], points)

    def test_pipeline_matches_reference_and_passes_independent_final_audit(self):
        for transform in (np.eye(2), -np.eye(2), np.diag([-1., 1.]),
                          np.array([[0., -1.], [1., 0.]])):
            corridors = [SimpleNamespace(corners=c.corners @ transform.T) for c in self.corridors]
            kwargs = dict(initial_pose=[*(transform @ [-2., -2.]), .4],
                          final_pose=[*(transform @ [18., -2.]), 2.])
            result = baseline.compute_baseline(corridors, self.vehicle, **kwargs)
            reference = baseline.compute_boundary_directed_baseline(
                corridors, self.vehicle, method='segment', **kwargs)
            self.assertTrue(result.feasible, result.reason)
            np.testing.assert_allclose(result.polyline, reference.polyline)
            self.assertIsNotNone(baseline._validate_and_build_fillets(
                result.polyline, result.feasibility, result.fillet_regions, 1., 3., 1e-9))
            self.assertTrue(all(s.contains(p) for s, p in
                                zip(result.fillet_reachability.reachable_sets, result.polyline)))

    def test_optional_poses_choose_perpendicular_directions(self):
        result = baseline.compute_baseline(self.corridors, self.vehicle)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.segment_directions, ('up', 'right', 'down'))
        straight = baseline.compute_baseline(
            rectangles([(0, 4, 0, 4), (0, 16, 0, 4), (12, 16, 0, 4)]), self.vehicle)
        self.assertTrue(straight.feasible)
        self.assertEqual(straight.segment_directions, ('right', 'right', 'right'))
        self.assertEqual(straight.fillets, (None, None))

    def test_empty_safe_overlap_stops_before_directions(self):
        corridors = rectangles([(0, 1, 0, 1), (0, 5, 0, 1), (4, 5, 0, 1)])
        with patch.object(baseline, '_determine_internal_directions',
                          side_effect=AssertionError('Cannot continue with an empty D_j.')):
            result = baseline.compute_baseline(corridors, self.vehicle)
        self.assertEqual(result.status, 'empty_safe_overlap')
        self.assertTrue(result.boundary_intersections)
        self.assertFalse(result.feasible)

    def test_ambiguous_and_unavailable_passages_stop_at_direction_stage(self):
        for bounds, status in (
            ([(0, 4, 0, 4)]*3, 'ambiguous_passage_direction'),
            ([(0, 4, 0, 4), (0, 12, 0, 12), (8, 12, 8, 12)], 'no_orthogonal_connection'),
        ):
            result = baseline.compute_baseline(rectangles(bounds), self.vehicle)
            self.assertEqual(result.status, status)
            self.assertFalse(result.admissible_sets)

    def test_empty_local_region_does_not_start_propagation_or_search_other_directions(self):
        corridors = rectangles([(0, 4, -.1, 4), (0, 16, 0, 4), (12, 16, -.1, 4)])
        with patch.object(baseline, '_propagate_admissible_regions',
                          side_effect=AssertionError('Cannot propagate an empty A_j.')):
            result = baseline.compute_baseline(corridors, self.vehicle)
        self.assertEqual(result.status, 'empty_fillet_region')
        self.assertEqual(result.initial_direction, 'up')
        self.assertIsNone(result.polyline)

    def test_insufficient_spacing_is_rejected_by_segment_propagation(self):
        result = baseline.compute_baseline(
            rectangles([(0, 4, -8, 4), (0, 7, 0, 4), (3, 7, -8, 4)]), self.vehicle)
        self.assertEqual(result.status, 'fillet_reachability_empty')
        self.assertTrue(all(region is not None for region in result.admissible_sets))
        self.assertIsNone(result.fillet_reachability.reachable_sets[-1])
        self.assertIsNone(result.polyline)

    def test_invalid_external_inputs_raise_once_at_entry(self):
        for kwargs in (dict(initial_pose=[0., 0.]), dict(final_pose=[0., np.nan, 0.]), dict(tol=-1.)):
            with self.assertRaises(ValueError):
                baseline.compute_baseline(self.corridors, self.vehicle, **kwargs)


class SingleRoundedRegionTest(unittest.TestCase):
    def test_closed_form_projection_matches_general_analytic_intersection(self):
        rng = np.random.default_rng(81)
        for _ in range(500):
            a, b = sorted(rng.uniform(-3., 3., 2))
            c, d = sorted(rng.uniform(-3., 3., 2))
            constraint = RoundedCornerConstraint(tuple(rng.uniform(-2., 2., 2)),
                tuple(map(int, rng.choice((-1, 1), 2))), float(rng.uniform(.1, 3.)))
            fast = intersect_baseline_region((a, b, c, d), constraint)
            reference = _make_set((a, b, c, d), (constraint,), 1e-9)
            self.assertEqual(fast is None, reference is None)
            if fast is not None:
                np.testing.assert_allclose(fast.bounds, reference.bounds, atol=1e-8, rtol=0.)
                self.assertTrue(fast.contains(fast.witness))
                for point in rng.uniform([a, c], [b, d], (10, 2)):
                    self.assertEqual(fast.contains(point), reference.contains(point))


if __name__ == '__main__':
    unittest.main()
