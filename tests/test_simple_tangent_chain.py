"""First-chain local repair and the preserved legacy half-plane alternative."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from kappa_planner.refinement import (
    CircleSafeArcs, IndependentCirclePlacements, connect_simple_tangent_chain,
    connect_safe_arc_circles,
)
from test_consecutive_circle_connection import circle


def problem(circles, intervals=None):
    count = max((c.waypoint_index for c in circles), default=0)+1
    baseline = SimpleNamespace(feasible=True, feasibility=SimpleNamespace(
        corridor_bounds=((-50., 50., -50., 50.),)*(count+1)))
    safe = tuple(CircleSafeArcs(i, c.waypoint_index,
                 ((0., 2*np.pi),) if intervals is None else intervals[i], ())
                 for i, c in enumerate(circles))
    return IndependentCirclePlacements('placed', tuple(circles)), baseline, SimpleNamespace(r=1., R=2.), safe


def connect(circles, intervals=None):
    placed, baseline, robot, safe = problem(circles, intervals)
    return connect_simple_tangent_chain(placed, baseline, robot, safe_arcs=safe,
                                        strategy='legacy_halfplane')


class SimpleTangentChainTest(unittest.TestCase):
    def test_bad_triple_proposes_and_selects_skip(self):
        result = connect([circle(0, (-6, 0)), circle(1, (0, 3)), circle(2, (6, 0))])
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.selected_circles, (0, 2))
        self.assertEqual(result.proposed_shortcuts, ((0, 2),))
        self.assertEqual(result.halfplane_blocks, ((0, 2),))
        self.assertAlmostEqual(result.halfplane_triples[0][3], -3.)
        self.assertTrue(result.safe_arcs_checked)
        self.assertTrue(result.tangent_containment_checked)
        self.assertEqual(result.status, 'simple_safe_arcs_connected')
        self.assertFalse(result.ordered_overlap_crossings_checked)

    def test_matching_old_halfplane_rule_for_both_turn_signs(self):
        from kappa_planner.helpers.axis_aligned_int_circle_sequence import circle_halfplane_clearance
        from kappa_planner.helpers.primitives import compute_extreme_poses_arc_line
        from kappa_planner.geometry import Point
        for sign in (-1, 1):
            circles = [circle(i, c, outgoing=(0, sign))
                       for i, c in enumerate(((-6, 0), (0, 3*sign), (6, 0)))]
            result = connect(circles)
            q = compute_extreme_poses_arc_line(-6, 0, 6, 0, sign, sign, 2.)
            old = circle_halfplane_clearance(SimpleNamespace(center=Point(0, 3*sign),
                    turn_direction=sign, radius=2.), 1., q[0], q[1], q[3], q[4])
            self.assertEqual(result.halfplane_triples[0][4], bool(old[0]))
            self.assertAlmostEqual(result.halfplane_triples[0][3], sign*old[3])

    def test_good_triple_has_no_unrestricted_skip(self):
        circles = [circle(0, (-6, 0)), circle(1, (0, -3)), circle(2, (6, 0))]
        simple = connect(circles)
        placed, baseline, robot, safe = problem(circles)
        graph = connect_safe_arc_circles(placed, baseline, robot, safe_arcs=safe,
                    check_straight_containment=False, check_overlap_order=False)
        self.assertTrue(simple.feasible)
        self.assertEqual(simple.selected_circles, (0, 1, 2))
        self.assertEqual(simple.proposed_shortcuts, ())
        self.assertLess(graph.length, simple.length)

    def test_contiguous_bad_triples_allow_skipping_more_than_one_circle(self):
        result = connect([circle(j, c) for j, c in enumerate(((-6, 0), (-2, 3), (2, 3), (6, 0)))])
        self.assertTrue(result.feasible)
        self.assertIn((0, 3), result.proposed_shortcuts)
        self.assertEqual(result.selected_circles, (0, 3))
        self.assertEqual(result.skipped_waypoints, (1, 2))

    def test_aligned_alternatives_are_retained(self):
        circles = [circle(0, (0, 0)), circle(1, (4, 4), side='left'),
                   circle(1, (4, 1), side='right'), circle(2, (8, 5))]
        intervals = [((3*np.pi/2, np.radians(290)),), ((0., 2*np.pi),),
                     ((0., 2*np.pi),), ((np.radians(310), 2*np.pi),)]
        result = connect(circles, intervals)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.groups, ((0,), (1, 2), (3,)))
        self.assertEqual(result.selected_circles, (0, 2, 3))

    def test_green_endpoints_do_not_allow_directed_arc_across_a_gap(self):
        circles = [circle(0, (0, 0)), circle(1, (4, 4)), circle(2, (8, 8))]
        intervals = [((0., 2*np.pi),), ((0., 2*np.pi),), ((0., 2*np.pi),)]
        # Source entry is 270 degrees; departure is 315 degrees. Both contacts
        # are green, but the actual intervening arc crosses an unsafe interval.
        intervals[0] = ((np.radians(260), np.radians(280)), (np.radians(305), np.radians(325)))
        result = connect(circles, intervals)
        self.assertFalse(result.feasible)
        self.assertEqual(result.status, 'simple_chain_unresolved')
        self.assertIn(0, result.contact_order_blocks)

    def test_first_and_last_used_arc_are_checked(self):
        circles = [circle(0, (0, 0)), circle(1, (4, 4))]
        for intervals in [(((3*np.pi/2, 2*np.pi),), ((3*np.pi/2, np.radians(330)),)),
                          (((np.radians(300), 2*np.pi),), ((3*np.pi/2, 2*np.pi),))]:
            self.assertFalse(connect(circles, intervals).feasible)

    def test_footprint_checks_include_shortcuts_but_overlap_order_is_not_checked(self):
        from kappa_planner.helpers.sequence_geometry import SequenceGeometry
        checked = []
        original = SequenceGeometry.contains_capsule
        def contains(geometry, first, last, start, end):
            checked.append((first, last))
            return original(geometry, first, last, start, end)
        with patch.object(SequenceGeometry, 'contains_capsule', contains), \
             patch('kappa_planner.refinement.tangent_corridor_crossings',
                   side_effect=AssertionError('Unexpected overlap check')):
            result = connect([circle(0, (-6, 0)), circle(1, (0, 3)), circle(2, (6, 0))])
            self.assertTrue(result.feasible)
        self.assertIn((0, 2), checked)
        self.assertTrue(result.tangent_containment_checked)
        self.assertFalse(result.ordered_overlap_crossings_checked)

    def test_unsafe_straight_footprint_is_rejected_despite_green_contacts(self):
        placed, baseline, robot, safe = problem([circle(0, (0, 0)), circle(2, (8, 8))])
        baseline.feasibility.corridor_bounds = ((-4., 4., -4., 4.), (-4., 6., -4., 4.),
                                              (4., 14., 2., 12.), (6., 14., 4., 12.))
        full = connect_safe_arc_circles(placed, baseline, robot, safe_arcs=safe)
        simple = connect_simple_tangent_chain(placed, baseline, robot, safe_arcs=safe,
                                              strategy='legacy_halfplane')
        self.assertFalse(full.feasible)
        self.assertFalse(simple.feasible)
        self.assertEqual(simple.status, 'simple_chain_unresolved')
        self.assertIn((0, 1, 'straight_footprint_clearance'), simple.rejected_pairs)
        self.assertTrue(simple.tangent_containment_checked)
        self.assertFalse(simple.ordered_overlap_crossings_checked)
        green_only = connect_simple_tangent_chain(placed, baseline, robot, safe_arcs=safe,
                                                 check_straight_containment=False, strategy='legacy_halfplane')
        self.assertTrue(green_only.feasible)
        self.assertEqual(green_only.status, 'simple_green_arcs_connected')
        self.assertFalse(green_only.tangent_containment_checked)

    def test_green_only_opt_out_performs_no_straight_check(self):
        placed, baseline, robot, safe = problem([circle(0, (0, 0)), circle(1, (4, 4))])
        with patch('kappa_planner.helpers.sequence_geometry.SequenceGeometry.contains_capsule',
                   side_effect=AssertionError('Unexpected footprint check')):
            result = connect_simple_tangent_chain(placed, baseline, robot, safe_arcs=safe,
                                                  check_straight_containment=False, strategy='legacy_halfplane')
        self.assertTrue(result.feasible)

    def test_repeated_connection_reuses_capsule_decisions(self):
        from kappa_planner.helpers.sequence_geometry import sequence_geometry
        placed, baseline, robot, safe = problem([circle(0, (0, 0)), circle(1, (4, 4))])
        geometry = sequence_geometry(baseline.feasibility, robot.r, 1e-9)
        baseline.feasibility._geometry = geometry
        union = geometry.union(0, 1)
        with patch.object(union, 'contains_capsule', wraps=union.contains_capsule) as check:
            first = connect_simple_tangent_chain(placed, baseline, robot, safe_arcs=safe)
            second = connect_simple_tangent_chain(placed, baseline, robot, safe_arcs=safe)
        self.assertTrue(first.feasible)
        self.assertEqual(first.selected_circles, second.selected_circles)
        self.assertEqual(check.call_count, 1)

    def test_coincident_same_turn_and_opposite_external_tangency(self):
        result = connect([circle(0, (0, 0)), circle(1, (0, 0), (0, 1), (-1, 0))])
        self.assertTrue(result.feasible, result.reason)
        self.assertTrue(all(p['kind'] == 'arc' for p in result.primitives))
        result = connect([circle(0, (0, 0)), circle(1, (4, 0), outgoing=(0, -1))])
        self.assertTrue(result.feasible, result.reason)
        self.assertAlmostEqual(result.tangent_states[0].length, 0.)

    def test_empty_and_single_circle(self):
        self.assertEqual(connect([]).status, 'no_circles')
        result = connect([circle(0, (0, 0))])
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.length, np.pi)


class LocalTangentRepairTest(unittest.TestCase):
    def local(self, circles, intervals=None, **options):
        placed, baseline, robot, safe = problem(circles, intervals)
        return connect_simple_tangent_chain(placed, baseline, robot, safe_arcs=safe, **options)

    def test_stop_at_consecutive_chain_even_when_a_shorter_skip_exists(self):
        from kappa_planner.helpers.tangent_refinement import _directed_tangent_lines
        circles = [circle(0, (-6, 0)), circle(1, (0, -3)), circle(2, (6, 0))]
        with patch('kappa_planner.helpers.corridor_union._exposed_boundary',
                   side_effect=AssertionError('Endpoints already fit one eroded corridor')), \
             patch('kappa_planner.helpers.tangent_refinement._directed_tangent_lines',
                   wraps=_directed_tangent_lines) as generate:
            result = self.local(circles)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.selected_circles, (0, 1, 2))
        self.assertEqual(result.attempted_pairs, 2)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result.repair_rounds, ())
        self.assertEqual(result.selection_method, 'local_repair')
        self.assertEqual(result.footprint_scope, 'all')
        self.assertTrue(result.tangent_containment_checked)
        self.assertEqual(result.status, 'simple_local_chain_connected')

    def test_neighboring_occupied_groups_check_footprints_across_aligned_door_gap(self):
        placed, baseline, robot, safe = problem([circle(0, (0, 0)), circle(2, (8, 8))])
        baseline.feasibility.corridor_bounds = ((-4., 4., -4., 4.), (-4., 6., -4., 4.),
                                              (4., 14., 2., 12.), (6., 14., 4., 12.))
        self.assertFalse(connect_safe_arc_circles(placed, baseline, robot, safe_arcs=safe).feasible)
        result = connect_simple_tangent_chain(placed, baseline, robot, safe_arcs=safe)
        self.assertFalse(result.feasible)
        self.assertEqual(result.rejected_pairs, ((0, 1, 'straight_footprint_clearance'),))
        self.assertTrue(result.tangent_containment_checked)
        unchecked = connect_simple_tangent_chain(placed, baseline, robot, safe_arcs=safe,
                                                 check_straight_containment=False)
        self.assertTrue(unchecked.feasible)

    def test_safe_union_bridge_uses_boundary_fallback_instead_of_rejecting(self):
        from kappa_planner.helpers.corridor_union import _exposed_boundary
        placed, baseline, robot, safe = problem([circle(0, (0, 0)), circle(2, (8, 8))],
                                               [((3*np.pi/2, 2*np.pi),)]*2)
        baseline.feasibility.corridor_bounds = ((-4., 2.2, -4., 4.), (-4., 7., -2.2, 5.),
                                              (3., 10.2, 1., 12.), (6., 14., 5.8, 12.))
        with patch('kappa_planner.helpers.corridor_union._exposed_boundary',
                   wraps=_exposed_boundary) as boundary:
            result = connect_simple_tangent_chain(placed, baseline, robot, safe_arcs=safe)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(boundary.call_count, 1)
        self.assertEqual(result.repair_rounds, ())
        self.assertTrue(result.tangent_containment_checked)

    def test_bypass_checks_footprint_and_stops_after_first_repair(self):
        from kappa_planner.helpers.sequence_geometry import SequenceGeometry
        from kappa_planner.helpers.tangent_refinement import _directed_tangent_lines
        circles = [circle(0, (-6, 0)), circle(1, (0, 3)), circle(2, (6, 0))]
        intervals = [((0., 2*np.pi),), (), ((0., 2*np.pi),)]
        checked = []
        original = SequenceGeometry.contains_capsule
        def contains(geometry, first, last, start, end):
            checked.append((first, last))
            return original(geometry, first, last, start, end)
        with patch.object(SequenceGeometry, 'contains_capsule', contains), \
             patch('kappa_planner.helpers.tangent_refinement._directed_tangent_lines',
                   wraps=_directed_tangent_lines) as generate:
            result = self.local(circles, intervals)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.selected_circles, (0, 2))
        self.assertEqual(result.repair_rounds, ((0, 2),))
        self.assertEqual(checked, [(0, 2)])
        self.assertEqual(generate.call_count, 3)

    def test_unsafe_shortcut_is_rejected(self):
        placed, baseline, robot, safe = problem(
            [circle(0, (0, 0)), circle(1, (4, 4)), circle(2, (8, 8))],
            [((0., 2*np.pi),), (), ((0., 2*np.pi),)])
        baseline.feasibility.corridor_bounds = ((-4., 4., -4., 4.), (-4., 6., -4., 4.),
                                              (4., 14., 2., 12.), (6., 14., 4., 12.))
        result = connect_simple_tangent_chain(placed, baseline, robot, safe_arcs=safe)
        self.assertFalse(result.feasible)
        self.assertIn((0, 2, 'straight_footprint_clearance'), result.rejected_pairs)
        unchecked = connect_simple_tangent_chain(placed, baseline, robot, safe_arcs=safe,
                                                 check_straight_containment=False)
        self.assertTrue(unchecked.feasible)
        self.assertEqual(unchecked.footprint_scope, 'none')

    def test_repair_can_bypass_multiple_groups_and_reuses_geometry(self):
        from kappa_planner.helpers.tangent_refinement import _directed_tangent_lines
        circles = [circle(j, c) for j, c in enumerate(((-6, 0), (-2, 3), (2, 3), (6, 0)))]
        intervals = [((0., 2*np.pi),), (), (), ((0., 2*np.pi),)]
        with patch('kappa_planner.helpers.tangent_refinement._directed_tangent_lines',
                   wraps=_directed_tangent_lines) as generate:
            result = self.local(circles, intervals)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.selected_circles, (0, 3))
        self.assertEqual(result.repair_rounds, ((0, 2), (0, 3)))
        self.assertEqual(result.attempted_pairs, 5)
        self.assertEqual(generate.call_count, 5)

    def test_aligned_alternatives_remain_available(self):
        circles = [circle(0, (0, 0)), circle(1, (4, 4), side='left'),
                   circle(1, (4, 1), side='right'), circle(2, (8, 5))]
        intervals = [((3*np.pi/2, np.radians(290)),), ((0., 2*np.pi),),
                     ((0., 2*np.pi),), ((np.radians(310), 2*np.pi),)]
        result = self.local(circles, intervals)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.selected_circles, (0, 2, 3))
        self.assertEqual(result.repair_rounds, ())

    def test_green_gap_still_blocks_neighbor_connection(self):
        intervals = [((np.radians(260), np.radians(280)), (np.radians(305), np.radians(325))),
                     ((0., 2*np.pi),)]
        result = self.local([circle(0, (0, 0)), circle(1, (4, 4))], intervals)
        self.assertFalse(result.feasible)
        self.assertEqual(result.status, 'simple_chain_unresolved')


if __name__ == '__main__':
    unittest.main()
