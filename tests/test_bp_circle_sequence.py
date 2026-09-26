"""Legacy placement priorities with bp-derived geometry and immutable inputs."""

from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.arc_feasibility import compute_vehicle_clearance_radii
from kappa_planner.helpers.bp_circle_sequence import (
    build_bp_circle_sequence, corridor_along_bp,
)

EXAMPLE = runpy.run_path(str(Path(__file__).resolve().parents[1]
    / 'experiments/bicycle_journal_paper/examples_maps_polyline.py'))


def fixture(w1, w2):
    corridors = [CorridorWorld(w1, 20, [0, -w1/2], 0),
                 CorridorWorld(w2, 20, [w2/2, 0], np.pi/2)]
    region = dict(incoming=np.array([1., 0.]), outgoing=np.array([0., 1.]),
                  corner=np.zeros(2), low=np.array([-10., -10.]),
                  high=np.array([10., 10.]), offset=np.zeros(2), frame=np.eye(2),
                  radius=-1., empty=False)
    points = np.array([[-5., -.2], [.2, -.2], [.2, 5.]])
    geometry = dict(feasible=True, polyline=points, check=dict(corridor_offset=-1),
                    smoothed_check=dict(regions=[None, region, None]))
    vehicle = SimpleNamespace(width=.2, max_radius=1., length=.3,
                              rectangular_footprint=False)
    return corridors, vehicle, geometry


class BpCircleSequenceTest(unittest.TestCase):
    def test_all_four_placement_priorities(self):
        for widths, rule in (((2, 2), 'forty_five_safe_half'),
                             ((.65, 1), 'safe_half_shifted'),
                             ((.6, .6), 'forty_five_basic'),
                             ((.4, .8), 'basic_shifted')):
            with self.subTest(rule=rule):
                corridors, vehicle, geometry = fixture(*widths)
                original = geometry['polyline'].copy()
                result = build_bp_circle_sequence(corridors, vehicle, geometry)
                self.assertTrue(result['feasible'])
                self.assertFalse(result['connections_checked'])
                circle = result['sequence'][0]
                self.assertEqual(circle.construction_rule, rule)
                self.assertEqual(circle.corridor_index_start, 0)
                self.assertEqual(circle.corridor_index_end, 1)
                self.assertEqual(circle.bp_vertex_index, 1)
                x, y = circle.geometry_result['center_local']
                np.testing.assert_allclose([circle.center.x, circle.center.y], [-y, x], atol=1e-9)
                np.testing.assert_array_equal(geometry['polyline'], original)

    def test_failed_placement_keeps_its_slot(self):
        corridors, vehicle, geometry = fixture(.1, .1)
        result = build_bp_circle_sequence(corridors, vehicle, geometry)
        self.assertEqual(result['status'], 'PLACEMENT_INCOMPLETE')
        self.assertEqual(result['circle_slots'], [None])
        self.assertEqual(result['placements'][0]['geometry']['reason'], 'width_pair_infeasible')

    def test_radius_override_does_not_modify_vehicle(self):
        corridors, vehicle, geometry = fixture(2, 2)
        vehicle.rectangular_footprint = True
        result = build_bp_circle_sequence(corridors, vehicle, geometry, radius=.8)
        self.assertTrue(result['feasible'])
        circle = result['sequence'][0]
        self.assertEqual(circle.radius, .8)
        self.assertEqual(vehicle.max_radius, 1.)
        self.assertAlmostEqual(circle.swept_radius, np.hypot(.8+.1, vehicle.length))

    def test_equivalent_corridors_follow_each_signed_bp_axis(self):
        original = CorridorWorld(2, 8, [3, 4], np.pi)
        for direction in ([1, 0], [-1, 0], [0, 1], [0, -1]):
            effective = corridor_along_bp(original, direction)
            np.testing.assert_allclose(effective.unit_vector, direction, atol=1e-9)
            np.testing.assert_allclose(np.min(effective.corners, axis=0),
                                       np.min(original.corners, axis=0))
            np.testing.assert_allclose(np.max(effective.corners, axis=0),
                                       np.max(original.corners, axis=0))
        self.assertEqual(original.tilt, np.pi)

    def test_all_examples_keep_bp_and_corridor_mapping(self):
        accepted = 0
        rules = set()
        for number in EXAMPLE['EXAMPLE_NUMBERS']:
            corridors, start, end, vehicle = EXAMPLE['example_corridor_sequence'](number)
            geometry = EXAMPLE['build_trajectory_geometry'](
                corridors, vehicle.width/2, vehicle.max_radius,
                start_pose=start, end_pose=end)
            bp = None if geometry['polyline'] is None else geometry['polyline'].copy()
            before = [(c.width, c.height, c.tilt) for c in corridors]
            report = build_bp_circle_sequence(corridors, vehicle, geometry)
            if not geometry['feasible']:
                self.assertEqual(report['status'], 'NO_FEASIBLE_BP')
                continue
            accepted += 1
            self.assertTrue(report['feasible'], (number, report))
            np.testing.assert_array_equal(geometry['polyline'], bp)
            self.assertEqual(before, [(c.width, c.height, c.tilt) for c in corridors])
            regions = geometry['smoothed_check']['regions']
            self.assertEqual(len(report['sequence']), sum(r is not None for r in regions))
            for circle in report['sequence']:
                j = circle.bp_vertex_index
                self.assertEqual(circle.corridor_index_start, j-1)
                self.assertIs(report['circle_slots'][j-1], circle)
                u, v = regions[j]['incoming'], regions[j]['outgoing']
                tau = int(np.sign(np.linalg.det(np.array([u, v]))))
                self.assertEqual(circle.turn_direction, tau)
                np.testing.assert_allclose(circle.ex, tau*np.array([-u[1], u[0]]), atol=1e-9)
                np.testing.assert_allclose(circle.ey, tau*np.array([-v[1], v[0]]), atol=1e-9)
                rules.add(circle.construction_rule)
                r, R, S, D = compute_vehicle_clearance_radii(vehicle)
                local = circle.geometry_result['center_local']
                self.assertLessEqual(local @ local, D*D+1e-8)
                self.assertGreaterEqual(local[0], circle.lower_bound_x-1e-8)
                self.assertGreaterEqual(local[1], circle.lower_bound_y-1e-8)
            self.assertFalse(report['connections_checked'])
        self.assertEqual(accepted, 26)
        self.assertIn('safe_half_shifted', rules)

    def test_example_35_has_nominal_opposite_turn_overlap(self):
        corridors, start, end, vehicle = EXAMPLE['example_corridor_sequence'](35)
        geometry = EXAMPLE['build_trajectory_geometry'](
            corridors, vehicle.width/2, vehicle.max_radius,
            start_pose=start, end_pose=end, validate_arcs=True)
        self.assertTrue(geometry['feasible'])
        report = build_bp_circle_sequence(corridors, vehicle, geometry, shift_same_turn=True)
        self.assertEqual(report['diagnostics']['flags']['opposite_turn_overlap'], [(0, 1)])
        self.assertEqual(report['diagnostics']['flags']['outside_Aj'], [])
        self.assertEqual(report['shifting']['accepted'], 0)
        self.assertEqual([c.turn_direction for c in report['sequence']], [1, -1])
        self.assertTrue(all(c.construction_rule == 'forty_five_basic'
                            for c in report['sequence']))
        self.assertAlmostEqual(report['diagnostics']['pairs'][0]['center_distance'],
                               1.774472668025395)

    def test_cached_bp_geometry_avoids_rebuilding_corridors(self):
        corridors, start, end, vehicle = EXAMPLE['example_corridor_sequence'](23)
        geometry = EXAMPLE['build_trajectory_geometry'](
            corridors, vehicle.width/2, vehicle.max_radius, start_pose=start, end_pose=end)
        module = 'kappa_planner.helpers.bp_circle_sequence.'
        with patch(module+'_bounds', side_effect=AssertionError('Repeated bounds check')), \
             patch(module+'CorridorWorld', side_effect=AssertionError('Rebuilt corridor')), \
             patch(module+'compute_vehicle_clearance_radii', wraps=compute_vehicle_clearance_radii) as radii:
            report = build_bp_circle_sequence(corridors, vehicle, geometry)
        radii.assert_called_once()
        self.assertEqual(report['diagnostics']['pair_scope'], 'consecutive')
        self.assertEqual(len(report['diagnostics']['pairs']), len(report['sequence'])-1)
        self.assertEqual(report['diagnostics']['flags']['same_turn_arc_conflict'], [(4, 5)])


if __name__ == '__main__':
    unittest.main()
