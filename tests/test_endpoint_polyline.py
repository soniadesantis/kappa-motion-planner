"""Pose-coordinate entry/exit vertices, spacing, containment and indexing."""

import runpy
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.endpoint_polyline import extend_nominal_with_pose_coordinates
from kappa_planner.helpers.nominal_polyline import check_orthogonal_polyline
from kappa_planner.helpers.tangent_refinement import refine_by_blocks

EXAMPLE = runpy.run_path(str(Path(__file__).resolve().parents[1]
    / 'experiments/bicycle_journal_paper/examples_maps_polyline.py'))


class EndpointPolylineTest(unittest.TestCase):
    def test_exact_positions_are_preferred_and_projection_is_explicit(self):
        exact_sides = 0
        projected_sides = 0
        for number in (1, 2, 3, 15, 24, 26, 29):
            corridors, start, end, vehicle = EXAMPLE['example_corridor_sequence'](number)
            result = EXAMPLE['build_trajectory_geometry'](corridors, vehicle.width/2,
                vehicle.max_radius, start_pose=start, end_pose=end)
            self.assertTrue(result['feasible'])
            for point, pose, mode in zip(result['polyline'][[0, -1]], (start, end), result['check']['endpoint_modes']):
                if mode == 'exact':
                    np.testing.assert_allclose(point, pose[:2], atol=1e-8, rtol=0)
                    exact_sides += 1
                else:
                    projected_sides += 1
            attempts = result['endpoint_attempts']
            self.assertTrue(attempts[0]['exact_start'] and attempts[0]['exact_end'])
            self.assertTrue(all(not attempt['feasible'] for attempt in attempts[:-1]))
        self.assertGreater(exact_sides, 0)
        self.assertGreater(projected_sides, 0)

    def test_example_extensions_and_certified_refinement(self):
        for number in (1, 2, 3, 15, 24, 26, 29):
            corridors, start, end, vehicle = EXAMPLE['example_corridor_sequence'](number)
            r, R = vehicle.width/2, vehicle.max_radius
            geometry = EXAMPLE['build_trajectory_geometry'](corridors, r, R, start_pose=start, end_pose=end)
            self.assertTrue(geometry['feasible'], (number, geometry['check']['reason'], geometry['smoothed_check']['reason']))
            points = geometry['polyline']
            self.assertEqual(len(points), len(corridors)+1)
            for point, corridor, pose in ((points[0], corridors[0], start), (points[-1], corridors[-1], end)):
                k = int(np.argmax(np.abs(corridor.unit_vector)))
                self.assertAlmostEqual(point[k], pose[k])
            for j, (a, b) in enumerate(zip(points, points[1:])):
                self.assertLessEqual(min(abs(b-a)), 1e-8)
                self.assertGreaterEqual(np.linalg.norm(b-a), 2*R-1e-8)
                corridor = corridors[j]
                for point in (a, b):
                    self.assertTrue(np.all(point @ corridor.W[:2]+corridor.W[2]
                        <= -r*np.linalg.norm(corridor.W[:2], axis=0)+1e-8))
            for arc in geometry['arcs']:
                if arc is not None:
                    self.assertTrue(np.all(EXAMPLE['union_signed_clearance'](arc, corridors) >= r-1e-8))
            self.assertEqual(geometry['check']['corridor_offset'], -1)
            result = refine_by_blocks(points, geometry['smoothed_check']['regions'], corridors, r, R,
                                      corridor_offset=-1)
            self.assertTrue(result['feasible'], (number, result))
            np.testing.assert_array_equal(result['points'][[0, -1]], points[[0, -1]])
            for tangent in result['tangents']:
                self.assertTrue(0 <= tangent['corridor'] < len(corridors))

    def test_short_exit_reports_failure_without_changing_original_check(self):
        corridors, start, end, vehicle = EXAMPLE['example_corridor_sequence'](30)
        nominal = check_orthogonal_polyline(corridors, vehicle.width/2, vehicle.max_radius)
        initial = nominal['polyline'].copy()
        result = extend_nominal_with_pose_coordinates(nominal, corridors, start, end,
                                                      vehicle.width/2, vehicle.max_radius)
        self.assertTrue(nominal['feasible'])
        np.testing.assert_array_equal(nominal['polyline'], initial)
        self.assertFalse(result['feasible'])
        self.assertEqual(result['status'], 'endpoint_spacing')
        self.assertIsNone(result['polyline'])

    def test_exact_2R_and_one_pose_coordinate_only(self):
        # H -> V -> H, point doors at (0,0) and (0,2), radius r=0.
        corridors = [CorridorWorld(2, 4, [-2, -1], 0),
                     CorridorWorld(2, 4, [1, 1], np.pi/2),
                     CorridorWorld(2, 4, [2, 3], 0)]
        nominal = dict(feasible=True, polyline=np.array([[0., 0.], [0., 2.]]),
            doors=[dict(x=(0., 0.), y=(0., 0.)), dict(x=(0., 0.), y=(2., 2.))],
            directions=['V'], computation_time=0., reason=None, status='feasible')
        result = extend_nominal_with_pose_coordinates(nominal, corridors,
            [-2., -1., .7], [2., 3., -.8], 0., 1.)
        self.assertTrue(result['feasible'], result)
        np.testing.assert_allclose(result['polyline'], [[-2, 0], [0, 0], [0, 2], [2, 2]])
        np.testing.assert_allclose(np.linalg.norm(np.diff(result['polyline'], axis=0), axis=1), 2.)

    def test_endpoint_coordinate_outside_corridor(self):
        corridors, start, end, vehicle = EXAMPLE['example_corridor_sequence'](1)
        start = np.array(start)
        start[int(np.argmax(np.abs(corridors[0].unit_vector)))] = 1e6
        result = EXAMPLE['build_trajectory_geometry'](corridors, vehicle.width/2,
            vehicle.max_radius, start_pose=start, end_pose=end)
        self.assertFalse(result['feasible'])
        self.assertEqual(result['check']['status'], 'endpoint_outside_corridor')

    def test_main_success_and_endpoint_failure_render_three_figures(self):
        original_show = plt.show
        plt.show = lambda: None
        try:
            for number in (24, 30):
                plt.close('all')
                EXAMPLE['main'](number)
                self.assertEqual(len(plt.get_fignums()), 3)
        finally:
            plt.show = original_show
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
