"""Joint fillet selection, failure modes, and revised far-arm acceptance."""

import runpy
import unittest
from pathlib import Path

import numpy as np

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.fillet_safety import region_margin, revised_corner_condition, fillet_vertex_region
from kappa_planner.helpers.smoothed_polyline import solve_fillet_waypoints

EXAMPLE = runpy.run_path(str(Path(__file__).resolve().parents[1]
                            / 'experiments/bicycle_journal_paper/examples_maps_polyline.py'))


def rectangle(x0, x1, y0, y1):
    return CorridorWorld(y1-y0, x1-x0, [(x0+x1)/2, (y0+y1)/2], 0)


class SmoothedPolylineTest(unittest.TestCase):
    def test_reselection_recovers_example_30_and_preserves_far_arm_branch(self):
        corridors, _, _, vehicle = EXAMPLE['example_corridor_sequence'](30)
        result = EXAMPLE['build_trajectory_geometry'](corridors, vehicle.width/2, vehicle.max_radius)
        second = result['smoothed_check']
        self.assertTrue(second['feasible'], second['reason'])
        initial = result['check']['polyline']
        self.assertTrue(any(region_margin(initial[j], region) < -1e-8
                            for j, region in enumerate(second['regions']) if region is not None))
        self.assertGreater(np.max(abs(result['polyline']-initial)), .01)
        self.assertTrue(any(np.linalg.norm((result['polyline'][j]+region['offset']) @ region['frame'])
                            > region['radius']+1e-6
                            for j, region in enumerate(second['regions']) if region is not None))

    def test_empty_local_region_rejects_fillets_but_not_polyline(self):
        corridors = [rectangle(-4, -3, -.6, 0), rectangle(-6, 0, -.6, 0),
                     rectangle(-.6, 0, -.6, 6), rectangle(-.6, 0, 3, 4)]
        result = EXAMPLE['build_trajectory_geometry'](corridors, .2, 1.)
        self.assertTrue(result['check']['feasible'])
        self.assertEqual(result['smoothed_check']['status'], 'NO_SAFE_FILLET_REGION')
        self.assertFalse(result['feasible'])
        self.assertEqual(result['fillets'], [])
        self.assertEqual(result['arcs'], [])

    def test_nonempty_regions_can_have_incompatible_shared_coordinate(self):
        doors = [dict(x=(0, 1), y=(0, 1)), dict(x=(3, 4), y=(0, 1)),
                 dict(x=(3, 4), y=(2, 3)), dict(x=(6, 7), y=(2, 3))]
        # Box-only regions occur when R<r and the safe canonical coordinates
        # already imply the far-arm condition. Individually nonempty, these
        # two boxes impose contradictory x bounds on the same vertical run.
        first = dict(low=np.array([3., 0.]), high=np.array([3.2, 1.]), empty=False, radius=-1.)
        second = dict(low=np.array([3.8, 2.]), high=np.array([4., 3.]), empty=False, radius=-1.)
        nominal = dict(doors=doors, directions=['H', 'V', 'H'],
                       polyline=np.array([[.5, .5], [3.5, .5], [3.5, 2.5], [6.5, 2.5]]))
        result = solve_fillet_waypoints(nominal, [None, first, second, None], .5)
        self.assertEqual(result['status'], 'FILLET_REGIONS_GLOBALLY_INCOMPATIBLE')
        self.assertIsNone(result['points'])

    def test_convex_region_matches_original_or_rule_on_prerequisite_box(self):
        pair = [rectangle(-6, 0, -2, 0), rectangle(-2, 0, -2, 6)]
        door = dict(x=(-1.8, -.2), y=(-1.8, -.2))
        rng = np.random.default_rng(33)
        for R in (.1, .2, 1., 1.4):
            region = fillet_vertex_region(door, pair, [-2, 0], [1, 0], [0, 1], .2, R)
            points = rng.uniform(region['low'], region['high'], (500, 2))
            expected = revised_corner_condition(points, [-2, 0], [1, 0], [0, 1], R, .2)
            np.testing.assert_array_equal(region_margin(points, region) >= -1e-9,
                                          expected['predicted'])

    def test_curved_regions_can_conflict_even_when_linear_ranges_overlap(self):
        doors = [dict(x=(0, 1), y=(.5, .5)), dict(x=(3, 4), y=(0, .5)),
                 dict(x=(3.2, 4), y=(2, 2.5)), dict(x=(6, 7), y=(2.5, 2.5))]
        nominal = dict(doors=doors, directions=['H', 'V', 'H'],
                       polyline=np.array([[.5, .5], [3.6, .5], [3.6, 2.5], [6.5, 2.5]]))
        first = dict(low=np.array([3., 0.]), high=np.array([4., .5]), empty=False,
                     radius=.5, offset=np.array([-3.5, 0.]), frame=np.eye(2))
        second = dict(low=np.array([3.2, 2.]), high=np.array([4., 2.5]), empty=False,
                      radius=.5, offset=np.array([-3.7, -2.]), frame=np.diag([-1., 1.]))
        # Neighbors fix each y at its curved limit. The first region then needs
        # shared x<=3.5, and the second needs shared x>=3.7.
        result = solve_fillet_waypoints(nominal, [None, first, second, None], .5)
        self.assertEqual(result['status'], 'FILLET_REGIONS_GLOBALLY_INCOMPATIBLE')
        self.assertIn('outer relaxation', result['reason'])

    def test_boundary_contact_and_singleton_waypoints(self):
        doors = [dict(x=(0, 0), y=(0, 0)), dict(x=(2, 2), y=(0, 0)),
                 dict(x=(2, 2), y=(2, 2))]
        nominal = dict(doors=doors, directions=['H', 'V'],
                       polyline=np.array([[0., 0.], [2., 0.], [2., 2.]]))
        region = dict(low=np.array([2., 0.]), high=np.array([2., 0.]), empty=False,
                      radius=0., offset=np.array([-2., 0.]), frame=np.eye(2))
        result = solve_fillet_waypoints(nominal, [None, region, None], 1.)
        self.assertTrue(result['feasible'], result)
        np.testing.assert_allclose(result['points'], nominal['polyline'])

    def test_ambiguous_map_never_enters_second_stage(self):
        corridors, _, _, vehicle = EXAMPLE['example_corridor_sequence'](32)
        result = EXAMPLE['build_trajectory_geometry'](corridors, vehicle.width/2, vehicle.max_radius)
        self.assertEqual(result['check']['status'], 'ambiguous_direction')
        self.assertEqual(result['smoothed_check']['status'], 'NO_ORTHOGONAL_POLYLINE')
        self.assertIsNone(result['smoothed_check']['solver'])


if __name__ == '__main__':
    unittest.main()
