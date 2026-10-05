"""Central terminal selection and midpoint reconstruction under directed spacing."""

import unittest
from types import SimpleNamespace

import numpy as np

from kappa_planner.baseline_construction import compute_boundary_directed_baseline
from kappa_planner.helpers.fillet_reachability import (
    FilletReachableSet, RoundedCornerConstraint, _midpoint_polyline,
)


class MidpointBaselineSelectionTest(unittest.TestCase):
    def test_terminal_and_clipped_slice_midpoints_under_rotations(self):
        boxes = ((-8., 3., -2., 2.), (4., 8., -2., 2.), (4., 8., 5., 9.))
        expected = np.array([[-2.5, 0.], [6., 0.], [6., 7.]])
        # First compatible interval is [-8, min(3, 6-2)] = [-8, 3].
        for transform, directions in (
            (np.eye(2), ('right', 'up')),
            (-np.eye(2), ('left', 'down')),
            (np.array([[0., -1.], [1., 0.]]), ('up', 'left')),
            (np.diag([-1., 1.]), ('left', 'up')),
        ):
            sets = []
            for a, b, c, d in boxes:
                corners = np.array([[a, c], [b, c], [b, d], [a, d]]) @ transform.T
                low, high = corners.min(axis=0), corners.max(axis=0)
                sets.append(FilletReachableSet((low[0], high[0], low[1], high[1]), (), tuple(low)))
            points = _midpoint_polyline(sets, directions, 1., 1e-9)
            np.testing.assert_allclose(points, expected @ transform.T)

    def test_spacing_clips_the_slice_before_midpoint_selection(self):
        sets = (FilletReachableSet((-8., 8., -1., 1.), (), (-8., -1.)),
                FilletReachableSet((4., 6., -1., 1.), (), (4., -1.)))
        points = _midpoint_polyline(sets, ('right',), 1., 1e-9)
        # The successor midpoint is (5,0); predecessor x must be <=3.
        np.testing.assert_allclose(points, [[-2.5, 0.], [5., 0.]])

    def test_curved_terminal_uses_center_instead_of_event_witness(self):
        reachable = FilletReachableSet(
            (0., 1., 0., 1.), (RoundedCornerConstraint((0., 0.), (1, 1), 1.),), (0., 0.))
        points = _midpoint_polyline((reachable,), (), 1., 1e-9)
        np.testing.assert_allclose(points, [[.5, .5]])
        self.assertTrue(reachable.contains(points[0]))

    def test_exact_and_segment_select_the_same_central_chain(self):
        bounds = [(0, 4, -8, 4), (0, 16, 0, 4), (12, 16, -8, 4)]
        corridors = [SimpleNamespace(corners=np.array([[a, c], [b, c], [b, d], [a, d]]))
                     for a, b, c, d in bounds]
        kwargs = dict(initial_pose=[-2., -2., .4], final_pose=[18., -2., 2.])
        robot = SimpleNamespace(r=1., R=3.)
        exact = compute_boundary_directed_baseline(corridors, robot, method='exact', **kwargs)
        segment = compute_boundary_directed_baseline(corridors, robot, method='segment', **kwargs)
        self.assertTrue(exact.feasible, exact.reason)
        self.assertTrue(segment.feasible, segment.reason)
        np.testing.assert_allclose(exact.polyline, segment.polyline)
        for reachable, point in zip(segment.fillet_reachability.reachable_sets, segment.polyline):
            self.assertTrue(reachable.contains(point))
        xmin, xmax, ymin, ymax = segment.fillet_reachability.reachable_sets[-1].bounds
        np.testing.assert_allclose(segment.polyline[-1], [(xmin+xmax)/2, (ymin+ymax)/2])


if __name__ == '__main__':
    unittest.main()
