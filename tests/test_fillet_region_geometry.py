"""Analytic region/slice equivalence and corner invariance under waypoint motion."""

import unittest
from types import SimpleNamespace

import numpy as np

from kappa_planner.baseline_construction import compute_filleted_baseline
from kappa_planner.helpers.fillet_backtracking import region_slice
from kappa_planner.helpers.fillet_safety import (
    axis_aligned_concave_corners, fillet_vertex_region,
    region_contains_point, revised_corner_condition,
)


class FilletRegionGeometryTest(unittest.TestCase):
    def test_regions_and_slices_match_original_conditions_for_all_signed_turns(self):
        rng = np.random.default_rng(42)
        axes = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        canonical = [np.array([[-8, -3], [3, -3], [3, 0], [-8, 0]]),
                     np.array([[-3, -3], [0, -3], [0, 8], [-3, 8]])]
        for u in axes:
            for v in axes:
                if u @ v:
                    continue
                frame = np.column_stack((-u, v))
                corner = np.array([17., -9.])
                pair = [SimpleNamespace(corners=c @ frame.T + corner) for c in canonical]
                overlap = np.array([[-2.8, -2.8], [-.2, -.2]]) @ frame.T + corner
                low, high = overlap.min(axis=0), overlap.max(axis=0)
                door = dict(x=(low[0], high[0]), y=(low[1], high[1]))
                for R in (1., 3., 9.):
                    region = fillet_vertex_region(door, pair, corner, u, v, .2, R, tol=0)
                    points = rng.uniform(low - .1, high + .1, (400, 2))
                    expected = np.all((points >= low) & (points <= high), axis=1)
                    for corridor, tangents in zip(pair, (points - R*u, points + R*v)):
                        expected &= np.all((tangents >= corridor.corners.min(axis=0) + .2)
                                           & (tangents <= corridor.corners.max(axis=0) - .2), axis=1)
                    expected &= np.all((points-corner) @ frame <= -.2, axis=1)
                    expected &= revised_corner_condition(points, corner, u, v, R, .2, 0)['predicted']
                    actual = [region_contains_point(p, region, 0) for p in points]
                    np.testing.assert_array_equal(actual, expected)
                    if not region['empty']:
                        self.assertTrue(region_contains_point(region['witness'], region))
                    for axis in (0, 1):
                        for fixed in np.linspace(low[axis]-.1, high[axis]+.1, 21):
                            interval = region_slice(region, axis, fixed, tol=0)
                            free = np.linspace(low[1-axis]-.1, high[1-axis]+.1, 101)
                            probes = np.empty((len(free), 2))
                            probes[:, axis], probes[:, 1-axis] = fixed, free
                            expected = [region_contains_point(p, region, 0) for p in probes]
                            actual = (np.zeros(len(free), dtype=bool) if interval is None else
                                      (free >= interval[0]) & (free <= interval[1]))
                            np.testing.assert_array_equal(actual, expected)
                            if interval is not None:
                                for endpoint in interval:
                                    p = np.empty(2)
                                    p[axis], p[1-axis] = fixed, endpoint
                                    self.assertTrue(region_contains_point(p, region, 1e-12))

    def test_corner_is_unique_throughout_each_examples_safe_door(self):
        import runpy
        from pathlib import Path
        examples = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                     / 'experiments/bicycle_journal_paper/examples_maps_polyline.py'))
        for number in examples['EXAMPLE_NUMBERS']:
            corridors, _, _, robot = examples['example_corridor_sequence'](number)
            result = compute_filleted_baseline(corridors, robot, use_joint_solver=False)
            for j, region in enumerate(result.fillet_regions):
                if region is None:
                    continue
                door = result.feasibility.safe_overlaps[j]
                u, v = region['incoming'], region['outgoing']
                turn = np.linalg.det(np.stack((u, v)))
                corners = axis_aligned_concave_corners(*result.feasibility.corridor_bounds[j:j+2])
                for x in np.linspace(*door[:2], 9):
                    for y in np.linspace(*door[2:], 9):
                        delta = corners - [x, y]
                        mask = ((turn*(u[0]*delta[:, 1]-u[1]*delta[:, 0]) > 0)
                                & (turn*(v[0]*delta[:, 1]-v[1]*delta[:, 0]) > 0))
                        self.assertEqual(mask.sum(), 1, (number, j, x, y))
                        np.testing.assert_array_equal(corners[mask][0], region['corner'])


if __name__ == '__main__':
    unittest.main()
