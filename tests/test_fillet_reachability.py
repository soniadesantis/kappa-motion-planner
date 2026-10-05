"""Analytic projection, coupled propagation, reconstruction and planner regression."""

import unittest
from math import sqrt
from types import SimpleNamespace

import numpy as np

from kappa_planner.baseline_construction import (
    OrthogonalPolylineFeasibility,
    compute_filleted_baseline_exact,
)
from kappa_planner.helpers.fillet_reachability import (
    RoundedCornerConstraint,
    _make_set,
    _local_set,
    _circle_intersections,
    propagate_fillet_regions,
)
from kappa_planner.helpers.fillet_safety import region_contains_point


def region(bounds, center, signs, radius=1.):
    a, b, c, d = bounds
    return dict(low=np.array([a, c]), high=np.array([b, d]),
                offset=-np.array(center, dtype=float), frame=np.diag(signs),
                radius=radius, empty=False)


class FilletReachabilityTest(unittest.TestCase):
    def test_local_conversion_rounds_frame_noise_and_rejects_real_rotation(self):
        data = SimpleNamespace(x_reachable=((-2., 2.),), y_reachable=((-2., 2.),))
        local = region((-2, 2, -2, 2), (0, 0), (-1, 1))
        local['frame'] = local['frame'].astype(float)*(1.-1e-16)
        bounds, constraints = _local_set(data, [local], 0)
        self.assertEqual(constraints[0].signs, (-1, 1))
        converted = _make_set(bounds, constraints, 1e-10)
        for point in np.random.default_rng(91).uniform(-2, 2, size=(1000, 2)):
            self.assertEqual(converted.contains(point), region_contains_point(point, local))
        local['frame'] = np.array([[.8, -.6], [.6, .8]])
        with self.assertRaisesRegex(ValueError, 'signed-permutation'):
            _local_set(data, [local], 0)
        local['frame'] = np.array([[1., 0.], [1., 0.]])
        with self.assertRaises(ValueError):
            _local_set(data, [local], 0)

    def test_near_coincident_distinct_centers_are_not_discarded(self):
        first = RoundedCornerConstraint((0., 0.), (1, 1), 1.)
        second = RoundedCornerConstraint((1e-12, 0.), (-1, 1), 1.)
        events = list(_circle_intersections(first, second, 1e-15))
        self.assertTrue(events)
        np.testing.assert_allclose(events[0], [5e-13, 1.], atol=1e-15)
        # Geometry smaller than default tol must not be replaced by coincidence.
        first = RoundedCornerConstraint((0., 0.), (1, 1), 1e-10)
        second = RoundedCornerConstraint((1e-10, 0.), (-1, 1), 1e-10)
        result = _make_set((0., 1e-10, 0., 1e-10), (first, second), 1e-16)
        self.assertAlmostEqual(result.bounds[3]/1e-10, sqrt(3)/2, places=12)

    def test_thin_lenses_and_near_tangency_over_scales(self):
        for scale in (1e-6, 1., 1e6):
            for gap in (1e-4, 1e-8, 1e-12):
                distance = (2.-gap)*scale
                constraints = (RoundedCornerConstraint((0., 0.), (1, 1), scale),
                               RoundedCornerConstraint((distance, 0.), (-1, 1), scale))
                result = _make_set((0, distance, 0, scale), constraints, scale*1e-15)
                self.assertIsNotNone(result)
                expected = sqrt((scale-distance/2)*(scale+distance/2))
                self.assertAlmostEqual(result.bounds[3]/scale, expected/scale, places=10)
                self.assertTrue(result.contains(result.witness, scale*1e-15))
                outside = (2.+gap)*scale
                separated = (constraints[0], RoundedCornerConstraint((outside, 0.), (-1, 1), scale))
                self.assertIsNone(_make_set((0, outside, 0, scale), separated, scale*1e-15))

    def test_arc_arc_event_gives_projection_extremum(self):
        constraints = (RoundedCornerConstraint((0., 0.), (1, 1), 1.),
                       RoundedCornerConstraint((1., 0.), (-1, 1), 1.))
        result = _make_set((0, 1, 0, 1), constraints, 1e-10)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.bounds, [0, 1, 0, sqrt(3)/2])
        self.assertTrue(result.contains((.5, sqrt(3)/2)))
        self.assertFalse(result.contains((.5, .9)))
        self.assertTrue(result.contains(result.witness))

    def test_arc_arc_tangency_keeps_singleton(self):
        constraints = (RoundedCornerConstraint((0., 0.), (1, 1), 1.),
                       RoundedCornerConstraint((2., 0.), (-1, 1), 1.))
        result = _make_set((0, 2, 0, 1), constraints, 1e-10)
        np.testing.assert_allclose(result.bounds, [1, 1, 0, 0])
        self.assertEqual(result.slice_interval(1, 0), (1, 1))

    def test_disjoint_curved_regions_are_empty_despite_overlapping_boxes(self):
        constraints = (RoundedCornerConstraint((0., 0.), (1, 1), 1.),
                       RoundedCornerConstraint((2.1, 0.), (-1, 1), 1.))
        self.assertIsNone(_make_set((0, 2.1, 0, 1), constraints, 1e-10))

    def test_positive_part_preserves_far_arm(self):
        result = _make_set((-4, 1, -3, 1),
                           (RoundedCornerConstraint((0., 0.), (1, 1), 1.),), 1e-10)
        self.assertTrue(result.contains((-4, 1)))
        self.assertTrue(result.contains((1, -3)))
        self.assertFalse(result.contains((1, 1)))
        self.assertEqual(result.slice_interval(0, -4), (-3, 1))

    def test_arc_line_projection_and_reflections(self):
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            xs = sorted([sx*.6, sx*.9])
            ys = sorted([0., sy*1.])
            result = _make_set((*xs, *ys),
                               (RoundedCornerConstraint((0., 0.), (sx, sy), 1.),), 1e-10)
            self.assertTrue(result.contains((sx*.6, sy*.8)))
            self.assertFalse(result.contains((sx*.6, sy*.81)))
            self.assertAlmostEqual(result.bounds[3 if sy > 0 else 2], sy*.8)

    def test_zero_radius_reduces_to_half_planes(self):
        result = _make_set((-1, 1, -1, 1),
                           (RoundedCornerConstraint((0., 0.), (1, 1), 0.),), 1e-10)
        np.testing.assert_allclose(result.bounds, [-1, 0, -1, 0])

    def test_full_coupled_propagation_in_all_four_directions(self):
        # After separation 2, opposing circular bounds form a lens. Its exact
        # transverse upper limit is sqrt(3)/2, not the original box bound 1.
        for direction, swap, reflection in (
            ('right', False, 1), ('left', False, -1),
            ('up', True, 1), ('down', True, -1),
        ):
            boxes = [(0., 1., 0., 1.), (2., 3., 0., 1.)]
            centers = [(1., 0.), (2., 0.)]
            signs = [(-1, 1), (1, 1)]
            transform = np.array([[reflection, 0], [0, 1]])
            if swap:
                transform = np.array([[0, 1], [1, 0]]) @ transform
            transformed_boxes, regions = [], []
            for box, center, sign in zip(boxes, centers, signs):
                corners = np.array([[x, y] for x in box[:2] for y in box[2:]]) @ transform.T
                low, high = corners.min(axis=0), corners.max(axis=0)
                new_box = (low[0], high[0], low[1], high[1])
                transformed_boxes.append(new_box)
                regions.append(region(new_box, transform @ center, transform @ sign))
            data = OrthogonalPolylineFeasibility(True, 'feasible',
                safe_overlaps=tuple(transformed_boxes), passage_directions=(direction,),
                x_reachable=tuple(b[:2] for b in transformed_boxes),
                y_reachable=tuple(b[2:] for b in transformed_boxes))
            result = propagate_fillet_regions(data, regions, 1.)
            self.assertTrue(result.feasible, direction)
            transverse = 0 if swap else 1
            self.assertAlmostEqual(result.reachable_sets[-1].bounds[2*transverse+1], sqrt(3)/2)
            for point, reachable, local in zip(result.polyline, result.reachable_sets, regions):
                self.assertTrue(reachable.contains(point))
                self.assertTrue(region_contains_point(point, local))
            delta = result.polyline[1]-result.polyline[0]
            np.testing.assert_allclose(delta[transverse], 0., atol=1e-10)
            self.assertGreaterEqual(abs(delta[1-transverse]), 2.-1e-10)

    def test_exact_constructor_collinear_degenerate_spacing(self):
        def rectangle(a, b):
            return SimpleNamespace(corners=np.array([[a, 0], [b, 0], [b, 2], [a, 2]], dtype=float))
        result = compute_filleted_baseline_exact([rectangle(0, 2), rectangle(0, 5), rectangle(3, 5)],
                                                SimpleNamespace(r=1, R=1.5))
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.selection_method, 'exact_set_propagation')
        self.assertEqual(result.backtracking_attempts, 0)
        np.testing.assert_allclose(result.polyline, [[1, 1], [4, 1]])

    def test_exact_constructor_keeps_thesis_fillets_under_transforms(self):
        bounds = [(0, 4, 0, 8), (0, 13, 0, 4), (9, 19, -1, 5),
                  (14, 18, 0, 12), (8, 18, 9, 13), (7, 11, 6, 13)]
        for transform in (np.eye(2), -np.eye(2), np.diag([-1., 1.]),
                          np.array([[0., -1.], [1., 0.]])):
            corridors = [SimpleNamespace(corners=np.array([[a,c],[b,c],[b,d],[a,d]])
                                          @ transform.T + [37., -19.]) for a,b,c,d in bounds]
            result = compute_filleted_baseline_exact(corridors, SimpleNamespace(r=1, R=3))
            self.assertTrue(result.feasible, result.reason)
            self.assertEqual([j for j, arc in enumerate(result.fillets) if arc], [2, 3])
            for point, local in zip(result.polyline, result.fillet_regions):
                if local:
                    self.assertTrue(region_contains_point(point, local))
            self.assertTrue(np.all(result.remaining_lengths >= -1e-9))

    def test_case66_empty_full_region_despite_feasible_unfilleted_polyline(self):
        # Recorded five-corridor random case 66, seed 7. The transverse values
        # reachable through the first passage exclude every safe second turn.
        bounds = [
            (-11.238428543344417, .163055674652947, -1.9248286869161242, 1.9248286869161242),
            (-11.513071847164486, -10.116540338643887, -1.482906656693728, 11.688262320773516),
            (-11.876310867792148, 1.3071850773269504, 10.15480027038249, 11.688262320773516),
            (-2.3524653101581148, 1.3071850773269504, 10.058981256874862, 21.668416733443102),
            (-1.8439201550176605, 11.52272070164637, 18.138122469454267, 21.861999390417516),
        ]
        corridors = [SimpleNamespace(corners=np.array([[a,c],[b,c],[b,d],[a,d]]))
                     for a,b,c,d in bounds]
        result = compute_filleted_baseline_exact(corridors, SimpleNamespace(r=.5, R=2.))
        self.assertTrue(result.feasibility.feasible)
        self.assertEqual(result.status, 'fillet_reachability_empty')
        self.assertTrue(result.certified_infeasible)
        self.assertEqual(result.fillet_reachability.empty_waypoint, 1)
        self.assertIsNone(result.polyline)

    def test_exact_constructor_handles_curved_endpoint_regions(self):
        bounds = [(0, 4, -4, 4), (0, 16, 0, 4), (12, 16, -4, 4)]
        corridors = [SimpleNamespace(corners=np.array([[a,c],[b,c],[b,d],[a,d]], dtype=float))
                     for a,b,c,d in bounds]
        result = compute_filleted_baseline_exact(corridors, SimpleNamespace(r=1, R=3),
                                                initial_position=[2, -100], final_position=[14, -100])
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.segment_directions, ('up', 'right', 'down'))
        self.assertTrue(all(arc is not None for arc in result.fillets))
        for point, local in zip(result.polyline, result.fillet_regions):
            self.assertTrue(region_contains_point(point, local))
        self.assertTrue(np.all(result.remaining_lengths >= -1e-9))

    def test_explicit_endpoint_directions_match_position_inference(self):
        bounds = [(0, 4, -4, 4), (0, 16, 0, 4), (12, 16, -4, 4)]
        corridors = [SimpleNamespace(corners=np.array([[a,c],[b,c],[b,d],[a,d]], dtype=float))
                     for a,b,c,d in bounds]
        robot = SimpleNamespace(r=1, R=3)
        explicit = compute_filleted_baseline_exact(corridors, robot,
                                                   initial_direction='up', final_direction='down')
        inferred = compute_filleted_baseline_exact(corridors, robot,
                                                   initial_position=[2, -100], final_position=[14, -100])
        self.assertTrue(explicit.feasible, explicit.reason)
        np.testing.assert_allclose(explicit.polyline, inferred.polyline)
        self.assertIsNotNone(explicit.fillets[0])
        self.assertIsNotNone(explicit.fillets[-1])
        self.assertTrue(all(f.contains(p) for f, p in
                            zip(explicit.fillet_reachability.reachable_sets, explicit.polyline)))
        with self.assertRaises(ValueError):
            compute_filleted_baseline_exact(corridors, robot, initial_direction='north')
        with self.assertRaises(ValueError):
            compute_filleted_baseline_exact(corridors, robot, initial_direction='up',
                                            initial_position=[2, -100])


if __name__ == '__main__':
    unittest.main()
