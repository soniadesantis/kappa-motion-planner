"""Demo-only revised corner condition and independent whole-arc comparison."""

import runpy
import unittest
from pathlib import Path

import numpy as np

DEMO = runpy.run_path(str(Path(__file__).resolve().parents[1]
                         / 'experiments/bicycle_journal_paper/demo_nominal_orthogonal_polyline.py'))


class RevisedFilletDemoTest(unittest.TestCase):
    def test_canonical_branches_and_boundary(self):
        offset = .8/np.sqrt(2)
        p = np.array([[-1.2, -.2], [-.2, -1.2], [-.2, -.2],
                      [-1+offset, -1+offset], [-1+offset+1e-5, -1+offset+1e-5]])
        result = DEMO['revised_corner_condition'](p, [0, 0], [-1, 0], [0, 1], 1., .2)
        np.testing.assert_allclose(result['local_centers'], p+1)
        np.testing.assert_array_equal(result['predicted'], [True, True, False, True, False])
        np.testing.assert_array_equal(result['far'], [True, True, False, False, False])
        self.assertFalse(result['circular'][0])

    def test_signed_turns_and_translations_use_local_coordinates(self):
        p = np.array([[-1.2, -.2], [-.2, -1.2], [-.2, -.2], [-.5, -.5]])
        axes = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        for u in axes:
            for v in axes:
                if u @ v:
                    continue
                corner = np.array([20., -13.])
                basis = np.column_stack((-u, v))
                world = p @ basis.T+corner
                result = DEMO['revised_corner_condition'](world, corner, u, v, 1., .2)
                np.testing.assert_allclose(result['local_centers'], p+1)
                np.testing.assert_array_equal(result['predicted'], [True, True, False, True])

    def test_negative_R_minus_r_is_not_squared_into_a_disk(self):
        result = DEMO['revised_corner_condition']([-.05, -.05], [0, 0],
                                                  [-1, 0], [0, 1], .1, .2)
        self.assertFalse(result['predicted'])

    def test_all_geometries_and_safe_tangent_filter(self):
        for example in DEMO['local_turn_examples']():
            with self.subTest(example=example['name']):
                result = DEMO['evaluate_local_turn'](example, grid_size=25)
                self.assertEqual(result['counts']['false_positive'], 0)
                self.assertEqual(result['counts']['false_negative'], 0)
                self.assertEqual(sum(result['counts'].values()), int(result['eligible'].sum()))
                self.assertTrue(np.isnan(result['minimum_clearance'][~result['eligible']]).all())
                if example['name'] == 'L: short arms':
                    self.assertTrue((~result['eligible']).any())
                    self.assertGreater(result['counts']['both_safe'], 0)
                if example['name'] == 'L: wide overlap':
                    self.assertTrue((result['eligible'] & result['prediction']['far']
                                     & ~result['prediction']['circular']).any())

    def test_critical_angle_clearance_against_dense_arc_sampling(self):
        geometry = DEMO['_union_geometry']()
        rng = np.random.default_rng(17)
        for example in DEMO['local_turn_examples']():
            result = DEMO['evaluate_local_turn'](example, grid_size=15)
            eligible = np.flatnonzero(result['eligible'])
            indices = rng.choice(eligible, min(12, len(eligible)), replace=False)
            p = result['points'][indices]
            arc = DEMO['quarter_arc_points'](p, example['incoming'], example['outgoing'],
                                             1., np.linspace(0, np.pi/2, 10001))
            dense = geometry.union_signed_clearance(arc, example['corridors']).min(axis=1)
            critical = result['minimum_clearance'][indices]
            np.testing.assert_array_equal(critical >= .2-1e-9, dense >= .2-1e-9)
            contained = critical >= 0
            error = dense[contained]-critical[contained]
            # Signed clearance is 1-Lipschitz; dense sampling may overestimate
            # the continuous minimum by at most half a step times R.
            self.assertTrue(np.all(error >= -1e-9))
            self.assertTrue(np.all(error <= np.pi/40000+1e-9))

    def test_no_eligible_candidates_are_not_claimed_to_be_safe(self):
        result = DEMO['evaluate_local_turn'](DEMO['local_turn_examples']()[3], R=3., grid_size=9)
        self.assertEqual(sum(result['counts'].values()), 0)
        self.assertIsNone(result['selected'])

    def test_explicit_boundary_vertex_has_exact_clearance(self):
        example = DEMO['local_turn_examples']()[-1]
        result = DEMO['evaluate_local_turn'](example, grid_size=9)
        j = result['selected']
        self.assertTrue(result['eligible'][j])
        self.assertTrue(result['prediction']['predicted'][j])
        self.assertTrue(result['actual'][j])
        self.assertAlmostEqual(np.linalg.norm(result['prediction']['local_centers'][j]), .8)
        self.assertAlmostEqual(result['minimum_clearance'][j], .2)
        # Moving toward the missing quadrant crosses the curved limit.
        outside = result['points'][j]+1e-4*(-example['incoming']+example['outgoing'])
        prediction = DEMO['revised_corner_condition'](
            outside, example['corner'], example['incoming'], example['outgoing'], 1., .2)
        self.assertFalse(prediction['predicted'])


if __name__ == '__main__':
    unittest.main()
