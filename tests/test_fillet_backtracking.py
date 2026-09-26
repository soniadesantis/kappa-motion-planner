"""Analytic curved slices, cheap witness selection, and bounded fallback."""

import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from kappa_planner.helpers.fillet_backtracking import region_slice, backtrack_fillet_waypoints
from kappa_planner.helpers.fillet_safety import region_margin
from kappa_planner.helpers.smoothed_polyline import solve_fillet_waypoints

EXAMPLE = runpy.run_path(str(Path(__file__).resolve().parents[1]
    / 'experiments/bicycle_journal_paper/examples_maps_polyline.py'))


class FilletBacktrackingTest(unittest.TestCase):
    def test_rounded_slices_and_far_arm(self):
        region = dict(low=np.array([-2., -2.]), high=np.array([2., 2.]),
                      radius=1., frame=np.eye(2), offset=np.zeros(2), empty=False)
        np.testing.assert_allclose(region_slice(region, 1, .6), [-2., .8])
        np.testing.assert_allclose(region_slice(region, 0, -.7), [-2., 1.])
        self.assertIsNone(region_slice(region, 1, 1.2))
        region['frame'] = np.diag([-1., 1.])
        np.testing.assert_allclose(region_slice(region, 1, .6), [-.8, 2.])
        region['radius'] = 0.
        np.testing.assert_allclose(region_slice(region, 1, 0.), [0., 2.])
        region['radius'] = -1.
        np.testing.assert_allclose(region_slice(region, 1, 0.), [-2., 2.])

    def fixture(self):
        corridors, _, _, vehicle = EXAMPLE['example_corridor_sequence'](30)
        R = vehicle.max_radius
        geometry = EXAMPLE['build_trajectory_geometry'](corridors, vehicle.width/2, R)
        return geometry['check'], geometry['smoothed_check']['regions'], R

    def test_map_30_avoids_optimizer_and_preserves_nominal_supports(self):
        nominal, regions, R = self.fixture()
        initial = nominal['polyline'].copy()
        supports = repr((nominal['x_reachable'], nominal['y_reachable']))
        with patch('kappa_planner.helpers.smoothed_polyline.minimize', side_effect=AssertionError('optimizer called')):
            with patch('kappa_planner.helpers.smoothed_polyline.linprog', side_effect=AssertionError('LP called')):
                result = solve_fillet_waypoints(nominal, regions, R)
        self.assertTrue(result['feasible'])
        self.assertEqual(result['selection_method'], 'MIDPOINT_BACKTRACKING')
        self.assertEqual(result['backtracking_attempts'], len(nominal['doors']))
        np.testing.assert_array_equal(nominal['polyline'], initial)
        self.assertEqual(repr((nominal['x_reachable'], nominal['y_reachable'])), supports)
        points = result['points']
        for j, axis in enumerate(nominal['directions']):
            k = 0 if axis == 'H' else 1
            self.assertEqual(points[j, 1-k], points[j+1, 1-k])
            self.assertGreaterEqual(abs(points[j+1, k]-points[j, k]), 2*R-1e-8)
        for point, region in zip(points, regions):
            if region is not None:
                self.assertGreaterEqual(region_margin(point, region), -1e-8)

    def test_bounded_failure_is_unresolved_and_joint_fallback_succeeds(self):
        nominal, regions, R = self.fixture()
        quick = backtrack_fillet_waypoints(nominal, regions, R, max_attempts=1)
        self.assertFalse(quick['feasible'])
        self.assertEqual(quick['status'], 'BACKTRACKING_UNRESOLVED')
        self.assertLessEqual(quick['backtracking_attempts'], 1)
        result = solve_fillet_waypoints(nominal, regions, R, max_backtracking_attempts=1)
        self.assertTrue(result['feasible'], result)
        self.assertEqual(result['selection_method'], 'JOINT_SOLVER_FALLBACK')

    def test_exact_2R_and_singleton_intervals(self):
        nominal = dict(doors=[dict(x=(0, 0), y=(0, 0)), dict(x=(2, 2), y=(0, 0)),
                              dict(x=(2, 2), y=(2, 2))], directions=['H', 'V'],
                       polyline=np.array([[0., 0.], [2., 0.], [2., 2.]]))
        region = dict(low=np.array([2., 0.]), high=np.array([2., 0.]), radius=0.,
                      frame=np.eye(2), offset=np.array([-2., 0.]), empty=False)
        result = backtrack_fillet_waypoints(nominal, [None, region, None], 1.)
        self.assertTrue(result['feasible'], result)
        np.testing.assert_array_equal(result['points'], nominal['polyline'])


if __name__ == '__main__':
    unittest.main()
