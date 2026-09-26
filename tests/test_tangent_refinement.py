"""Geometric certificates, ordered block placement, and whole-path fallback."""

import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.fillet_safety import region_margin
from kappa_planner.helpers.tangent_refinement import (
    certified_angle, circle_node, common_tangents, certify_path,
    refine_by_blocks, turn_blocks, _used_arcs_intersect, _region_line_interval, _corner_target,
)

EXAMPLE = runpy.run_path(str(Path(__file__).resolve().parents[1]
    / 'experiments/bicycle_journal_paper/examples_maps_polyline.py'))


def fixed_fixture(same_turn=True):
    points = np.array([[0., 0.], [3., 0.], [3., 3.], [0. if same_turn else 6., 3.]])
    headings = np.sign(np.diff(points, axis=0))
    regions = [None]
    for j in (1, 2):
        regions.append(dict(low=points[j].copy(), high=points[j].copy(), radius=-1.,
            empty=False, corner=points[j]+.1, incoming=headings[j-1], outgoing=headings[j]))
    regions.append(None)
    corridors = [CorridorWorld(20, 20, [0, 0], 0) for _ in range(5)]
    return points, regions, corridors


class TangentRefinementTest(unittest.TestCase):
    def test_maximal_blocks(self):
        points = np.array([[0, 0], [3, 0], [3, 3], [0, 3], [0, 6], [3, 6]])
        self.assertEqual(turn_blocks(points),
            [dict(turn=1, vertices=[1, 2]), dict(turn=-1, vertices=[3, 4])])

    def test_directed_external_and_internal_tangents(self):
        for same_turn in (True, False):
            points, regions, corridors = fixed_fixture(same_turn)
            a, b = [circle_node(j, points[j], regions[j], 1.) for j in (1, 2)]
            choices = common_tangents(a, b, corridors, .1)
            self.assertTrue(choices)
            tangent = choices[0]
            np.testing.assert_allclose(tangent['start'], [3, 1], atol=1e-8)
            np.testing.assert_allclose(tangent['end'], [3, 2], atol=1e-8)
            np.testing.assert_allclose(tangent['direction'], [0, 1], atol=1e-8)
            # Quarter-arc containment is about location as well as angle length.
            self.assertIsNone(certified_angle(a, a['center']+[0, 1]))

    def test_circle_exclusion_and_global_baseline_fallback(self):
        points, regions, corridors = fixed_fixture()
        result = refine_by_blocks(points, regions, corridors, .1, 1., exclude_circle_overlap=True)
        self.assertEqual(result['status'], 'ORTHOGONAL_FALLBACK')
        self.assertTrue(result['feasible'])
        self.assertFalse(result['supporting_circle_exclusion_enforced'])
        np.testing.assert_array_equal(result['points'], points)

    def test_coincident_circles_can_join_at_certified_endpoints(self):
        points, regions, corridors = fixed_fixture()
        points[2:, 1] = 2.
        regions[2]['low'] = regions[2]['high'] = points[2].copy()
        report = certify_path(points, regions, corridors, .1, 1.)
        self.assertTrue(report['feasible'], report)
        np.testing.assert_allclose(report['tangents'][1]['start'], report['tangents'][1]['end'])

    def test_tangent_endpoints_must_be_inside_shared_eroded_corridor(self):
        points, regions, corridors = fixed_fixture()
        corridors[2] = CorridorWorld(.1, .1, [0, 0], 0)
        self.assertFalse(certify_path(points, regions, corridors, .1, 1.)['feasible'])

    def test_skipping_needs_its_own_segment_certificate(self):
        points, regions, corridors = fixed_fixture()
        self.assertTrue(certify_path(points, regions, corridors, .1, 1., skipped=[1, 2])['feasible'])
        for j in range(1, 4):
            corridors[j] = CorridorWorld(.1, .1, [0, 0], 0)
        self.assertFalse(certify_path(points, regions, corridors, .1, 1., skipped=[1, 2])['feasible'])
        self.assertFalse(certify_path(points, regions, corridors, .1, 1., skipped=[0])['feasible'])

    def test_examples_certify_every_piece_and_preserve_baseline(self):
        for number in (1, 6, 15, 30, 34):
            corridors, _, _, vehicle = EXAMPLE['example_corridor_sequence'](number)
            r, R = vehicle.width/2, vehicle.max_radius
            geometry = EXAMPLE['build_trajectory_geometry'](corridors, r, R)
            original = geometry['polyline'].copy()
            regions = geometry['smoothed_check']['regions']
            result = refine_by_blocks(original, regions, corridors, r, R)
            self.assertTrue(result['feasible'], result)
            np.testing.assert_array_equal(original, geometry['polyline'])
            np.testing.assert_array_equal(result['points'][[0, -1]], original[[0, -1]])
            passes = [step['pass_name'] for step in result['history']]
            self.assertEqual(passes, sorted(passes, key=lambda p: p == 'backward'))
            for step in result['history']:
                self.assertIn(step['action'], ('corner_move', 'neighbor_repair', 'skip', 'keep'))
                self.assertLessEqual(len(step.get('neighbor_repair', [])), 1)
                self.assertEqual(len(step['vertices']), 1)
            for tangent in result['tangents']:
                corridor = corridors[tangent['corridor']]
                for point in (tangent['start'], tangent['end']):
                    self.assertTrue(np.all(point @ corridor.W[:2]+corridor.W[2]
                        <= -r*np.linalg.norm(corridor.W[:2], axis=0)+1e-8))
            for arc in result['arcs']:
                self.assertGreaterEqual(arc['enter'], 0.)
                self.assertLessEqual(arc['leave'], np.pi/2)
                self.assertGreaterEqual(arc['leave'], arc['enter'])
                j = arc['index']
                self.assertGreaterEqual(region_margin(result['points'][j], regions[j]), -1e-8)
                # Independently recheck the WHOLE quarter at its new position.
                pair = corridors[j:j+2]
                minimum, _ = EXAMPLE['continuous_arc_clearance'](result['points'][j:j+1],
                    regions[j]['incoming'], regions[j]['outgoing'], R, pair,
                    EXAMPLE['corridor_union_boundary'](pair), EXAMPLE['union_signed_clearance'])
                self.assertGreaterEqual(minimum[0], r-1e-8)
            if number == 34:
                self.assertGreater(np.max(abs(result['points']-original)), .01)
                self.assertTrue(any(arc['leave']-arc['enter'] < np.pi/2-1e-4 for arc in result['arcs']))

    def test_all_endpoint_examples_without_numerical_optimization(self):
        accepted = 0
        for number in range(1, 35):
            corridors, start, end, vehicle = EXAMPLE['example_corridor_sequence'](number)
            r, R = vehicle.width/2, vehicle.max_radius
            geometry = EXAMPLE['build_trajectory_geometry'](
                corridors, r, R, start_pose=start, end_pose=end)
            if not geometry['feasible']:
                continue
            original = geometry['polyline'].copy()
            regions = geometry['smoothed_check']['regions']
            offset = geometry['check'].get('corridor_offset', 0)
            with patch('scipy.optimize.minimize', side_effect=AssertionError('Optimizer called')), \
                 patch('scipy.optimize.linprog', side_effect=AssertionError('LP called')), \
                 patch('numpy.linspace', side_effect=AssertionError('Arc sampling during refinement')):
                result = refine_by_blocks(original, regions, corridors, r, R,
                                          corridor_offset=offset)
            self.assertTrue(result['feasible'], (number, result))
            np.testing.assert_array_equal(geometry['polyline'], original)
            np.testing.assert_array_equal(result['points'][[0, -1]], original[[0, -1]])
            certificate = certify_path(result['points'], regions, corridors, r, R,
                                       skipped=result['skipped'], corridor_offset=offset)
            self.assertTrue(certificate['feasible'], (number, certificate))
            # Audit moved quarters against actual corridor-union geometry,
            # independently of the admissible-region predicate.
            for arc in result['arcs']:
                j = arc['index']
                pair = corridors[j+offset:j+offset+2]
                minimum, _ = EXAMPLE['continuous_arc_clearance'](
                    result['points'][j:j+1], regions[j]['incoming'],
                    regions[j]['outgoing'], R, pair,
                    EXAMPLE['corridor_union_boundary'](pair), EXAMPLE['union_signed_clearance'])
                self.assertGreaterEqual(minimum[0], r-1e-8, (number, j))
            accepted += 1
        self.assertEqual(accepted, 25)

    def test_coincident_arc_intersections_use_angular_intervals(self):
        def arc(start, sweep, turn=1):
            end = start+turn*sweep
            return dict(center=np.zeros(2), radius=1., turn=turn, enter=0., leave=sweep,
                        start=np.array([np.cos(start), np.sin(start)]),
                        end=np.array([np.cos(end), np.sin(end)]))
        for offset in (0., np.pi-.2, -np.pi):
            first = arc(offset, .5)
            self.assertTrue(_used_arcs_intersect(first, arc(offset+.25, .5)))
            self.assertFalse(_used_arcs_intersect(first, arc(offset+.5, .5)))
            self.assertFalse(_used_arcs_intersect(first, arc(offset+.6, .5)))
            self.assertTrue(_used_arcs_intersect(first, arc(offset+.75, .5, -1)))
            self.assertTrue(_used_arcs_intersect(first, first))

    def test_cached_trials_match_independent_certification(self):
        # Includes rejected proposals, restored coordinates and topology changes.
        c, start, end, v = EXAMPLE['example_corridor_sequence'](23)
        r, R = v.width/2, v.max_radius
        g = EXAMPLE['build_trajectory_geometry'](c, r, R, start_pose=start, end_pose=end)
        regions = g['smoothed_check']['regions']
        offset = g['check'].get('corridor_offset', 0)
        cache, rng = {}, np.random.default_rng(42)
        original = g['polyline'].copy()
        candidate = original.copy()
        for trial in range(50):
            candidate[:] = original
            if trial % 3:
                j = int(rng.integers(1, len(candidate)-1))
                candidate[j] += rng.uniform(-.15, .15, 2)
            omitted = [2] if trial % 5 == 0 else []
            args = dict(skipped=omitted, corridor_offset=offset)
            cached = certify_path(candidate, regions, c, r, R, _cache=cache, **args)
            full = certify_path(candidate, regions, c, r, R, **args)
            self.assertEqual((cached['feasible'], cached['reason']),
                             (full['feasible'], full['reason']))
            if full['feasible']:
                self.assertEqual(len(cached['arcs']), len(full['arcs']))
                for a, b in zip(cached['tangents'], full['tangents']):
                    np.testing.assert_array_equal(a['start'], b['start'])
                    np.testing.assert_array_equal(a['end'], b['end'])
                for a, b in zip(cached['arcs'], full['arcs']):
                    self.assertEqual((a['enter'], a['leave']), (b['enter'], b['leave']))
        # Repeated unchanged geometry performs no local geometric checks.
        certify_path(original, regions, c, r, R, _cache=cache, corridor_offset=offset)
        module = 'kappa_planner.helpers.tangent_refinement.'
        with patch(module+'region_margin', side_effect=AssertionError('Repeated membership')), \
             patch(module+'common_tangents', side_effect=AssertionError('Repeated tangent')), \
             patch(module+'_used_arcs_intersect', side_effect=AssertionError('Repeated intersection')):
            result = certify_path(original, regions, c, r, R, _cache=cache, corridor_offset=offset)
        self.assertTrue(result['feasible'])

    def test_analytic_line_clipping_and_cornerward_ray(self):
        region = dict(low=np.array([-2., -2.]), high=np.array([2., 2.]),
                      offset=np.zeros(2), frame=np.eye(2), radius=1., empty=False,
                      corner=np.array([2., 2.]))
        np.testing.assert_allclose(_region_line_interval(np.array([-2., .5]),
                                   np.array([1., 0.]), region), [0., 2+np.sqrt(.75)])
        np.testing.assert_allclose(_region_line_interval(np.zeros(2), np.ones(2), region),
                                   [-2., 1/np.sqrt(2)])
        np.testing.assert_allclose(_corner_target(np.zeros(2), region, 1e-8),
                                   np.ones(2)/np.sqrt(2))
        self.assertIsNone(_region_line_interval(np.array([1.5, 0.]), np.array([0., 1.]), region))
        self.assertIsNone(_region_line_interval(np.zeros(2), np.zeros(2), region))
        # Rotate/translate the same geometry: clipping must not depend on axes.
        rotation = np.array([[0., -1.], [1., 0.]])
        translated = dict(region, low=np.array([1., 2.]), high=np.array([5., 6.]),
                          offset=np.array([-3., -4.]), frame=rotation)
        np.testing.assert_allclose(_region_line_interval(np.array([3., 4.]),
                                   rotation @ np.ones(2), translated), [-2., 1/np.sqrt(2)])

    def test_sequential_skip_removes_redundant_circles_and_keeps_endpoints(self):
        points, regions, corridors = fixed_fixture(same_turn=False)
        result = refine_by_blocks(points, regions, corridors, .1, 1.)
        self.assertTrue(result['feasible'])
        self.assertEqual(result['skipped'], [1, 2])
        self.assertEqual([step['action'] for step in result['history']], ['skip', 'skip'])
        self.assertEqual(len(result['tangents']), 1)
        self.assertEqual(result['arcs'], [])
        np.testing.assert_array_equal(result['points'][[0, -1]], points[[0, -1]])

    def test_third_figure(self):
        corridors, _, _, vehicle = EXAMPLE['example_corridor_sequence'](34)
        r, R = vehicle.width/2, vehicle.max_radius
        geometry = EXAMPLE['build_trajectory_geometry'](corridors, r, R)
        figure, report = EXAMPLE['plot_tangent_refinement'](corridors, geometry, r, R, 34)
        try:
            figure.canvas.draw()
            self.assertTrue(report['feasible'])
        finally:
            plt.close(figure)


if __name__ == '__main__':
    unittest.main()
