"""Covered-edge extension preserves the environment and excludes aligned pairs."""

import runpy
import sys
import unittest
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / 'experiments' / 'bicycle_journal_paper'
sys.path.insert(0, str(EXAMPLES))
try:
    SCRIPT = runpy.run_path(str(EXAMPLES / 'example_random_baseline_construction.py'))
finally:
    sys.path.pop(0)
extend = SCRIPT['extend_perpendicular_corridors']


class CorridorExtensionTest(unittest.TestCase):
    def test_full_edge_reaches_neighbor_wall(self):
        before = [(0, 6, -1, 1), (4, 8, -2, 8)]
        after, changes = extend(before, [0, 1])
        self.assertEqual(after, [(0, 8, -1, 1), before[1]])
        self.assertEqual(changes[0]['edge'], 'xmax')
        self.assertEqual(before[0][1], 6)  # Input bounds are not mutated.

    def test_aligned_and_partially_covered_edges_are_unchanged(self):
        before = [(0, 6, -1, 1), (4, 8, -2, 8)]
        self.assertEqual(extend(before, [0, 2]), (before, []))
        partial = [(0, 6, -1, 1), (4, 8, 0, 8)]
        self.assertEqual(extend(partial, [0, 1]), (partial, []))

    def test_all_four_outward_directions(self):
        for first, second, heading, expected in (
            ((0, 6, -1, 1), (-2, 2, -2, 8), 0, (-2, 6, -1, 1)),
            ((-1, 1, 0, 6), (-2, 8, 4, 8), 1, (-1, 1, 0, 8)),
            ((-1, 1, 0, 6), (-2, 8, -2, 2), 1, (-1, 1, -2, 6)),
        ):
            after, _ = extend([first, second], [heading, 1-heading])
            self.assertEqual(after[0], expected)

    def test_random_extensions_preserve_union_and_reach_fixed_point(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            after, metadata = SCRIPT['sample_corridors'](rng)
            before, headings = metadata['original_bounds'], metadata['headings']
            self.assertEqual(extend(after, headings), (after, []))
            for a, b in zip(before, after):
                self.assertTrue(b[0] <= a[0] <= a[1] <= b[1])
                self.assertTrue(b[2] <= a[2] <= a[3] <= b[3])
            for change in metadata['edge_extensions']:
                self.assertEqual((headings[change['corridor']]-headings[change['neighbor']]) % 2, 1)
            # Rectangular arrangement membership is constant inside every cell.
            # Check one midpoint per cell, plus all edges and vertices; this
            # compares the complete closed rectangle unions, not a coarse grid.
            coordinates = []
            for axis in (0, 1):
                walls = sorted({x for box in before+after for x in box[2*axis:2*axis+2]})
                coordinates.append(sorted(walls + [(a+b)/2 for a, b in zip(walls, walls[1:])]))
            xx, yy = np.meshgrid(*coordinates)
            def contains(boxes):
                inside = np.zeros(xx.shape, dtype=bool)
                for a, b, c, d in boxes:
                    inside |= (a <= xx) & (xx <= b) & (c <= yy) & (yy <= d)
                return inside
            np.testing.assert_array_equal(contains(before), contains(after))


if __name__ == '__main__':
    unittest.main()
