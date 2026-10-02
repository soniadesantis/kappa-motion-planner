import unittest
from types import SimpleNamespace
import numpy as np

from kappa_planner.refinement import (
    IndependentCirclePlacement, IndependentCirclePlacements,
    restore_opposite_turn_overlaps,
)


class OppositeOverlapRestorationTest(unittest.TestCase):
    def setup_case(self, positions, turns, fallback):
        regions = [dict(incoming=np.array([1., 0.]), outgoing=np.array([0., t]),
                        empty=False, low=(-100., -100.), high=(100., 100.), radius=-1.)
                   for t in turns]
        circles = tuple(IndependentCirclePlacement(j, 'nominal', np.array([x, 0.]),
                        np.array([x, 0.]), 1., regions[j], True)
                        for j, x in enumerate(positions))
        baseline = SimpleNamespace(
            fillets=[SimpleNamespace(center=np.array([x, 0.])) for x in fallback],
            polyline=np.array([[x, 0.] for x in fallback]), fillet_regions=regions)
        return IndependentCirclePlacements('placed', circles), baseline

    def test_opposite_restored_same_turn_unchanged(self):
        placements, baseline = self.setup_case([0., 1., 10., 11.], [1, -1, 1, 1], [-2., 3., 10., 11.])
        result = restore_opposite_turn_overlaps(placements, baseline)
        self.assertEqual(result.restored_waypoints, (0, 1))
        self.assertEqual(result.overlap_pairs, ((2, 3),))
        self.assertEqual(result.circles[0].rule, 'baseline')
        self.assertEqual(placements.circles[0].rule, 'nominal')
        np.testing.assert_array_equal(result.circles[0].center, baseline.fillets[0].center)
        self.assertIs(result.circles[2], placements.circles[2])

    def test_new_conflict_after_restoration(self):
        placements, baseline = self.setup_case([0., 1., 5.], [1, -1, 1], [-2., 4., 8.])
        result = restore_opposite_turn_overlaps(placements, baseline)
        self.assertEqual(result.restored_waypoints, (0, 1, 2))
        self.assertEqual(result.overlap_pairs, ())

    def test_aligned_option_returns_to_no_circle(self):
        from dataclasses import replace
        placements, baseline = self.setup_case([0., 1.], [1, -1], [-2., 4.])
        placements = replace(placements, circles=(replace(placements.circles[0], side='left'),
                                                  placements.circles[1]))
        baseline.fillets[0] = None
        result = restore_opposite_turn_overlaps(placements, baseline)
        self.assertEqual(result.removed_aligned_options, ((0, 'left'),))
        self.assertEqual(result.skipped_waypoints, (0,))
        self.assertEqual(len(result.circles), 1)

    def test_surviving_baseline_overlap_terminates(self):
        placements, baseline = self.setup_case([0., 1.], [1, -1], [0., 1.])
        result = restore_opposite_turn_overlaps(placements, baseline)
        self.assertEqual(result.overlap_pairs, ((0, 1),))
        self.assertEqual(result.restored_waypoints, (0, 1))


if __name__ == '__main__':
    unittest.main()
