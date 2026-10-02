"""Regression checks using the corridor suite's same-turn overlap example."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/bicycle_journal_paper'))
from example_baseline_construction import example_corridor_sequence
from kappa_planner.baseline_construction import compute_filleted_baseline
from kappa_planner.refinement import (
    place_refinement_circles, restore_opposite_turn_overlaps,
    repair_same_turn_overlaps, has_safe_circle_transition,
)
from kappa_planner.helpers.fillet_safety import region_contains_point


class SameTurnOverlapTest(unittest.TestCase):
    def build(self, number=5):
        corridors, start, end, robot = example_corridor_sequence(number)
        baseline = compute_filleted_baseline(corridors, robot, use_joint_solver=False,
                                            initial_position=start[:2], final_position=end[:2])
        placed = restore_opposite_turn_overlaps(place_refinement_circles(baseline, robot), baseline)
        return baseline, robot, placed

    def test_repairs_both_pairs_and_preserves_safety(self):
        baseline, robot, placed = self.build()
        centers = [c.center.copy() for c in placed.circles]
        for i, k in placed.overlap_pairs:
            self.assertFalse(has_safe_circle_transition(placed.circles[i], placed.circles[k], baseline, robot))
        result = repair_same_turn_overlaps(placed, baseline, robot)
        self.assertEqual(result.coincident_waypoints, (0, 5))
        self.assertTrue(all(valid for _, _, valid in result.same_turn_transitions))
        for i, k, _ in result.same_turn_transitions:
            np.testing.assert_allclose(result.circles[i].center, result.circles[k].center)
        for old, center, new in zip(placed.circles, centers, result.circles):
            np.testing.assert_array_equal(old.center, center)
            self.assertTrue(region_contains_point(new.vertex, new.region))

    def test_valid_coincident_transitions_are_kept(self):
        baseline, robot, placed = self.build()
        first = repair_same_turn_overlaps(placed, baseline, robot)
        second = repair_same_turn_overlaps(first, baseline, robot)
        for a, b in zip(first.circles, second.circles):
            self.assertIs(a, b)

    def test_no_overlap_no_shift(self):
        baseline, robot, placed = self.build(29)
        result = repair_same_turn_overlaps(placed, baseline, robot)
        self.assertEqual(result.coincident_waypoints, ())
        for a, b in zip(placed.circles, result.circles):
            self.assertIs(a, b)

    def test_unsafe_common_centers_restore_baseline(self):
        baseline, robot, placed = self.build()
        # Reject shifted proposals while retaining membership of the actual
        # certified baseline vertices used by the fallback validation.
        def baseline_only(vertex, region, tol):
            return any(np.array_equal(vertex, p) for p in baseline.polyline)
        with patch('kappa_planner.refinement.region_contains_point', side_effect=baseline_only):
            result = repair_same_turn_overlaps(placed, baseline, robot)
        self.assertEqual(result.coincident_waypoints, ())
        self.assertEqual(result.restored_waypoints, (0, 1, 4, 5))
        self.assertTrue(all(valid for _, _, valid in result.same_turn_transitions))
        for j in result.restored_waypoints:
            self.assertEqual(result.circles[j].rule, 'baseline')
            np.testing.assert_array_equal(result.circles[j].center, baseline.fillets[j].center)
        for j in (2, 3):
            self.assertIs(placed.circles[j], result.circles[j])


if __name__ == '__main__':
    unittest.main()
