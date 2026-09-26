import unittest

import numpy as np

from test_bp_circle_diagnostics import fixture as circle_fixture
from kappa_planner.helpers.bp_tangent_polyline import build_bp_tangent_polyline
from kappa_planner.helpers.tangent_refinement import _directed_tangent_lines


def fixture(specifications):
    p, g = circle_fixture(specifications)
    g['internal_check'] = dict(doors=[dict(x=(-10., 10.), y=(-10., 10.))
                                      for _ in specifications])
    return p, g


class TangentPolylineTest(unittest.TestCase):
    def test_skip_multiple_circles_only_at_segment_intersections(self):
        p, g = fixture([(center, [1, 0], [0, 1])
                        for center in ([0, 0], [1, 1], [2, 3], [5, 3])])
        before = [(c.center.x, c.center.y) for c in p['sequence']]
        raw = build_bp_tangent_polyline(p, g, skip_intersections=False)
        # Removed door indices must not be reused for the surviving last circle.
        g['internal_check']['doors'][1:3] = [None, None]
        result = build_bp_tangent_polyline(p, g)
        self.assertEqual(raw['retained'], [0, 1, 2, 3])
        self.assertEqual(result['retained'], [0, 3])
        self.assertEqual(result['skipped'], [1, 2])
        self.assertEqual(result['remaining_intersections'], [])
        self.assertEqual(result['events'][0]['skipped'], [1, 2])
        self.assertEqual(result['outside_Dj'], [])
        self.assertEqual([(c.center.x, c.center.y) for c in p['sequence']], before)

    def test_original_Dj_not_Aj_or_endpoint_restricted_door(self):
        p, g = fixture([([0, 0], [1, 0], [0, 1])])
        g['internal_check']['doors'][0] = dict(x=(.9, 1.1), y=(-1.1, -.9))
        g['smoothed_check']['regions'][1]['empty'] = True
        g['check'] = dict(endpoint_constraints=True, doors=[None, None, None])
        result = build_bp_tangent_polyline(p, g)
        self.assertTrue(result['vertices'][0]['in_Dj'])
        np.testing.assert_allclose(result['vertices'][0]['point'], [1., -1.])

    def test_Dj_failure_does_not_trigger_skipping(self):
        p, g = fixture([(center, [1, 0], [0, 1])
                        for center in ([0, 0], [3, 1], [5, 4])])
        g['internal_check']['doors'][1] = dict(x=(9., 10.), y=(9., 10.))
        result = build_bp_tangent_polyline(p, g)
        self.assertEqual(result['retained'], [0, 1, 2])
        self.assertEqual(result['skipped'], [])
        self.assertIn(1, result['outside_Dj'])

    def test_actual_intersections_and_Dj_not_implied_vertices(self):
        p, g = fixture([([0, 0], [1, 0], [0, 1]), ([3, 2], [1, 0], [0, 1])])
        g['internal_check']['doors'][0]['x'] = (.9, 10.)
        result = build_bp_tangent_polyline(p, g)
        t = result['tangents'][0]['tangent']
        first = result['vertices'][0]
        self.assertFalse(first['in_Dj'])  # The implied orthogonal vertex (1,-1) is inside.
        self.assertAlmostEqual(first['point'][1], -1.)
        for v in result['vertices']:
            delta = v['point']-t['start']
            self.assertAlmostEqual(delta[0]*t['direction'][1]-delta[1]*t['direction'][0], 0.)

    def test_turn_signs_select_internal_or_external_forward_tangent(self):
        for a in (-1, 1):
            for b in (-1, 1):
                region = dict(incoming=np.array([1., 0.]), outgoing=np.array([0., 1.]))
                first = dict(index=0, center=np.array([0., 0.]), radius=1., turn=a, region=region)
                second = dict(index=1, center=np.array([4., 1.]), radius=1., turn=b, region=region)
                t = _directed_tangent_lines(first, second, 1e-8, restrict_quarters=False)[0]
                u = t['direction']
                for node, contact in ((first, t['start']), (second, t['end'])):
                    radial = contact-node['center']
                    np.testing.assert_allclose(node['turn']*np.array([-radial[1], radial[0]]), u)
                self.assertGreater((t['end']-t['start']) @ u, 0.)

    def test_parallel_tangents_report_nonunique_vertex(self):
        p, g = fixture([([0, 0], [1, 0], [0, 1]), ([3, 0], [1, 0], [0, 1])])
        result = build_bp_tangent_polyline(p, g)
        self.assertEqual(result['vertices'][0]['status'], 'COINCIDENT_LINES')
        self.assertIsNone(result['vertices'][0]['in_Dj'])

    def test_overlapping_opposite_pair_has_no_invented_tangent(self):
        p, g = fixture([([0, 0], [1, 0], [0, 1]), ([1, 0], [0, 1], [1, 0])])
        result = build_bp_tangent_polyline(p, g)
        self.assertIsNone(result['tangents'][0]['tangent'])
        self.assertEqual(result['undefined'], [0, 1])


if __name__ == '__main__':
    unittest.main()
