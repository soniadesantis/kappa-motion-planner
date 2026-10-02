"""Exact-support and witness tests for rectangle-union propagation."""

import itertools
import unittest

import numpy as np

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.waypoint_regions import (
    compute_corridor_point_sets,
    propagate_waypoint_regions,
)


def rectangle(xmin, xmax, ymin=0, ymax=2):
    return CorridorWorld(ymax-ymin, xmax-xmin, [(xmin+xmax)/2, (ymin+ymax)/2], 0)


def contains(region, point):
    x, y = point
    return any(a-1e-9 <= x <= b+1e-9 and c-1e-9 <= y <= d+1e-9
               for a, b, c, d in region)


class WaypointRegionTest(unittest.TestCase):
    def test_split_is_not_replaced_by_bounding_box(self):
        result = propagate_waypoint_regions([rectangle(0, 2)]*3, 0, 0.75,
                                            axes=["horizontal"])
        self.assertTrue(result["feasible"])
        for region in result["feasible_regions"]:
            self.assertEqual(len(region), 2)
            self.assertTrue(contains(region, [0.25, 1]))
            self.assertTrue(contains(region, [1.75, 1]))
            self.assertFalse(contains(region, [1, 1]))
        self.assertGreaterEqual(abs(np.diff(result["waypoints"], axis=0)[0, 0]), 1.5)

    def test_backward_pass_removes_unsupported_initial_points(self):
        result = propagate_waypoint_regions(
            [rectangle(0, 4), rectangle(0, 4), rectangle(3, 4)],
            0, 1.25, axes=["horizontal"],
        )
        self.assertTrue(contains(result["forward_regions"][0], [3, 1]))
        self.assertFalse(contains(result["feasible_regions"][0], [3, 1]))
        np.testing.assert_allclose(result["feasible_regions"][0], [[0, 1.5, 0, 2]])

    def test_locally_feasible_pairs_can_have_no_global_solution(self):
        result = propagate_waypoint_regions(
            [rectangle(0, 1), rectangle(0, 4), rectangle(0, 4), rectangle(3, 4)],
            0, 1.25, axes=["horizontal", "horizontal"],
        )
        self.assertFalse(result["feasible"])
        self.assertTrue(all(len(region) == 0 for region in result["feasible_regions"]))
        self.assertEqual(result["waypoints"].shape, (0, 2))

    def test_point_domains_and_exact_2R_equality_are_preserved(self):
        result = propagate_waypoint_regions(
            [rectangle(0, 2), rectangle(0, 4), rectangle(2, 4)],
            1, 1, axes=["horizontal"],
        )
        self.assertTrue(result["feasible"])
        np.testing.assert_allclose(result["waypoints"], [[1, 1], [3, 1]])
        self.assertFalse(propagate_waypoint_regions([rectangle(0, 2)]*2, 1.1, 1)["feasible"])

    def test_single_overlap_has_no_internal_segment_constraint(self):
        result = propagate_waypoint_regions([rectangle(0, 2)]*2, 1, 100)
        self.assertTrue(result["feasible"])
        np.testing.assert_allclose(result["waypoints"], [[1, 1]])

    def test_matches_exhaustive_integer_grid_chain_support(self):
        rng = np.random.default_rng(2)
        grid = set(itertools.product(range(7), repeat=2))
        for trial in range(20):
            corridors = [rectangle(int(rng.integers(0, 3)), int(rng.integers(3, 7)),
                                   int(rng.integers(0, 3)), int(rng.integers(3, 7)))
                         for _ in range(4)]
            axes = [str(rng.choice(["horizontal", "vertical", "either"])) for _ in range(2)]
            result = propagate_waypoint_regions(corridors, 0, 0.5, axes=axes)
            domains = [{p for p in grid if contains(region, p)} for region in result["safe_regions"]]

            def compatible(a, b, axis):
                return ((axis in ("horizontal", "either") and a[1] == b[1]
                         and abs(a[0]-b[0]) >= 1)
                        or (axis in ("vertical", "either") and a[0] == b[0]
                            and abs(a[1]-b[1]) >= 1))

            forward = [domains[0]]
            for j in range(1, 3):
                forward.append({b for b in domains[j]
                                if any(compatible(a, b, axes[j-1]) for a in forward[-1])})
            backward = [None, None, domains[-1]]
            for j in (1, 0):
                backward[j] = {a for a in domains[j]
                               if any(compatible(a, b, axes[j]) for b in backward[j+1])}
            with self.subTest(trial=trial, axes=axes):
                self.assertEqual(result["feasible"], bool(forward[-1]))
                for j, region in enumerate(result["feasible_regions"]):
                    self.assertEqual({p for p in grid if contains(region, p)},
                                     forward[j] & backward[j])
                if result["feasible"]:
                    for j, point in enumerate(result["waypoints"]):
                        self.assertTrue(contains(result["safe_regions"][j], point))
                    for j, (a, b) in enumerate(zip(result["waypoints"], result["waypoints"][1:])):
                        self.assertTrue(compatible(a, b, axes[j]))


class CorridorPointSetsTest(unittest.TestCase):
    def test_every_cross_pair_is_separated_not_just_some_partners(self):
        result = compute_corridor_point_sets([rectangle(0, 4)]*3, 0, 1)
        self.assertTrue(result["feasible"])
        self.assertEqual(len(result["candidates"][0]), 4)
        for candidate in result["candidates"][0]:
            a, b = candidate["entrance"], candidate["exit"]
            dx = max(a[0]-b[1], b[0]-a[1], 0)
            dy = max(a[2]-b[3], b[2]-a[3], 0)
            self.assertGreaterEqual(np.hypot(dx, dy), 1-1e-9)
            k = 2 if candidate["axis"] == "horizontal" else 0
            np.testing.assert_allclose(a[k:k+2], b[k:k+2])
        selected = result["selected_pairs"][0]
        np.testing.assert_allclose(selected["entrance"], [0, 1.5, 0, 2], atol=1e-9)
        np.testing.assert_allclose(selected["exit"], [2.5, 4, 0, 2], atol=1e-9)

    def test_already_separated_sets_are_not_trimmed(self):
        result = compute_corridor_point_sets(
            [rectangle(0, 1), rectangle(0, 5), rectangle(4, 5)], 0, 1
        )
        self.assertTrue(result["feasible"])
        pair = result["selected_pairs"][0]
        np.testing.assert_allclose(pair["entrance"], [0, 1, 0, 2])
        np.testing.assert_allclose(pair["exit"], [4, 5, 0, 2])

    def test_nonempty_neighbor_intersections_do_not_prove_global_equality(self):
        ybounds = [(0, 1), (0, 2), (0, 3), (0.5, 3), (1.5, 3)]
        corridors = [rectangle(3*j, 3*j+4, *ybounds[j]) for j in range(5)]
        result = compute_corridor_point_sets(corridors, 0, 0.5)
        self.assertTrue(result["local_intersections_nonempty"])
        self.assertFalse(result["feasible"])
        self.assertEqual(result["waypoints"].shape, (0, 2))

    def test_erosion_and_R_equality(self):
        result = compute_corridor_point_sets(
            [rectangle(0, 2), rectangle(0, 4), rectangle(2, 4)], 1, 2
        )
        self.assertTrue(result["feasible"])
        np.testing.assert_allclose(result["waypoints"], [[1, 1], [3, 1]])
        for point, region in zip(result["waypoints"], result["shared_regions"]):
            self.assertTrue(contains(region, point))


if __name__ == "__main__":
    unittest.main()
