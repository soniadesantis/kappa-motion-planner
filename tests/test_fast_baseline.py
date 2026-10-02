"""Fast axis-aligned geometry agrees with independent geometric auditing."""

import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.fillet_safety import (
    axis_aligned_concave_corners, region_contains_point, region_margin,
)

EXAMPLE = runpy.run_path(str(Path(__file__).resolve().parents[1]
    / 'experiments/bicycle_journal_paper/examples_maps_polyline.py'))


def bounds(corridor):
    low, high = np.min(corridor.corners, axis=0), np.max(corridor.corners, axis=0)
    return low[0], high[0], low[1], high[1]


class FastBaselineTest(unittest.TestCase):
    def test_direct_corners_match_general_boundary_geometry(self):
        rng = np.random.default_rng(19)
        for _ in range(150):
            pair = [CorridorWorld(float(rng.integers(1, 7)), float(rng.integers(1, 7)),
                                 rng.integers(-3, 4, size=2), int(rng.integers(4))*np.pi/2)
                    for _ in range(2)]
            candidates = EXAMPLE['boundary_intersections'](*pair)
            expected = [point for point in candidates if EXAMPLE['is_union_concave_corner'](point, *pair)]
            actual = axis_aligned_concave_corners(*(bounds(c) for c in pair))
            self.assertEqual({tuple(np.round(p, 8)) for p in expected},
                             {tuple(np.round(p, 8)) for p in actual})

    def test_squared_membership_matches_complete_region_margin(self):
        rng = np.random.default_rng(21)
        for frame in (np.eye(2), np.diag([-1., 1.]), np.array([[0., -1.], [1., 0.]])):
            for radius in (-.1, 0., 1.):
                region = dict(low=np.array([-1., -2.]), high=np.array([2., 1.]),
                              frame=frame, offset=np.array([.4, -.1]), radius=radius, empty=False)
                points = rng.uniform(-3, 3, (500, 2))
                np.testing.assert_array_equal([region_contains_point(p, region) for p in points],
                                              region_margin(points, region) >= -1e-8)

    def test_normal_baseline_does_not_call_expensive_audits_or_optimizers(self):
        function = EXAMPLE['build_trajectory_geometry']
        c, s, e, v = EXAMPLE['example_corridor_sequence'](7)
        def forbidden(*args, **kwargs):
            raise AssertionError('Expensive geometry or optimizer used on the fast baseline.')
        with patch.dict(function.__globals__, corridor_union_boundary=forbidden,
                        continuous_arc_clearance=forbidden, compute_inside_corners=forbidden):
            with patch('kappa_planner.helpers.smoothed_polyline._solve_fillet_waypoints_joint', forbidden):
                result = function(c, v.width/2, v.max_radius, start_pose=s, end_pose=e)
        self.assertTrue(result['feasible'])
        self.assertEqual(result['smoothed_check']['validation_mode'], 'region_membership')
        self.assertEqual(result['smoothed_check']['arc_clearances'], [])
        self.assertEqual(len(result['internal_check']['corridor_bounds']), len(c))

    def test_all_maps_fast_and_audited_modes_agree(self):
        accepted = 0
        for number in range(1, 35):
            c, s, e, v = EXAMPLE['example_corridor_sequence'](number)
            args = dict(start_pose=s, end_pose=e)
            fast = EXAMPLE['build_trajectory_geometry'](c, v.width/2, v.max_radius, **args)
            audited = EXAMPLE['build_trajectory_geometry'](c, v.width/2, v.max_radius,
                                                          validate_arcs=True, **args)
            self.assertEqual(fast['feasible'], audited['feasible'], number)
            if fast['feasible']:
                accepted += 1
                np.testing.assert_array_equal(fast['polyline'], audited['polyline'])
                count = sum(r is not None for r in audited['smoothed_check']['regions'])
                self.assertEqual(len(audited['smoothed_check']['arc_clearances']), count)
                self.assertTrue(all(d >= v.width/2-1e-8 for d in audited['smoothed_check']['arc_clearances']))
        self.assertEqual(accepted, 25)

    def test_local_regions_are_reused_within_geometry_cache(self):
        c, _, _, v = EXAMPLE['example_corridor_sequence'](7)
        nominal = EXAMPLE['check_orthogonal_polyline'](c, v.width/2, v.max_radius)
        check = EXAMPLE['check_smoothed_polyline']
        cache = {}
        first = check(c, v.width/2, v.max_radius, nominal=nominal, _cache=cache)
        self.assertTrue(first['feasible'])
        def forbidden(*args, **kwargs):
            raise AssertionError('Rebuilt cached local geometry.')
        with patch.dict(check.__globals__, fillet_vertex_region=forbidden, axis_aligned_concave_corners=forbidden):
            second = check(c, v.width/2, v.max_radius, nominal=nominal, _cache=cache)
        np.testing.assert_array_equal(first['points'], second['points'])


if __name__ == '__main__':
    unittest.main()
