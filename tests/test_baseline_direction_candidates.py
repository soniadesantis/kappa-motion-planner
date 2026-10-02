"""Boundary alternatives share exact internal analysis but validate each fillet pair."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import runpy
import numpy as np
import kappa_planner.baseline_construction as baseline


def rectangles(bounds):
    return [SimpleNamespace(corners=np.array([[a,c],[b,c],[b,d],[a,d]], dtype=float))
            for a,b,c,d in bounds]


class BoundaryCandidatesTest(unittest.TestCase):
    def test_local_options_keep_vertices_and_do_not_search(self):
        robot = SimpleNamespace(r=1,R=3)
        for first_bottom, first_top, expected in ((-8,12,('up','down')),
                                                  (-8,4,('up',)),
                                                  (0,12,('down',)),
                                                  (0,4,())):
            c = rectangles([(0,4,first_bottom,first_top),(0,16,0,4),(12,16,-8,12)])
            internal = baseline.compute_filleted_baseline(c,robot,use_joint_solver=False)
            original = internal.polyline.copy()
            with patch.object(baseline,'_backtrack_filleted_polyline',
                              side_effect=AssertionError('Local check must not search')):
                options = baseline.compute_local_boundary_fillets(internal,robot)
            self.assertEqual(tuple(d for d,_ in options.initial),expected)
            self.assertEqual(tuple(d for d,_ in options.final),('up','down'))
            np.testing.assert_array_equal(internal.polyline,original)
            for side, arcs in ((0,options.initial),(-1,options.final)):
                for direction, arc in arcs:
                    incoming = baseline._DIRECTION_VECTORS[direction if side == 0 else 'right']
                    outgoing = baseline._DIRECTION_VECTORS['right' if side == 0 else direction]
                    np.testing.assert_allclose(arc.incoming_tangent,original[side]-robot.R*incoming)
                    np.testing.assert_allclose(arc.outgoing_tangent,original[side]+robot.R*outgoing)

    def test_side_signs_and_reversals(self):
        door = (2, 4, 2, 4)
        # Exactly one exposed side, with opposite entry/exit interpretations.
        for corridor, entry, exit_direction in (
            ((0,4,2,4), 'right', 'left'), ((2,6,2,4), 'left', 'right'),
            ((2,4,0,4), 'up', 'down'), ((2,4,2,6), 'down', 'up'),
        ):
            for internal in baseline.CARDINAL_DIRECTIONS:
                forbidden = baseline.OPPOSITE[internal]
                self.assertEqual(baseline.admissible_initial_directions(corridor,door,internal),
                                 () if entry == forbidden else (entry,))
                self.assertEqual(baseline.admissible_final_directions(corridor,door,internal),
                                 () if exit_direction == forbidden else (exit_direction,))
        self.assertEqual(baseline.admissible_initial_directions(door,door,'right'), ())
        self.assertEqual(baseline.admissible_initial_directions((2-1e-10,4,2,4),door,'right'), ())
        self.assertEqual(baseline.admissible_initial_directions((2-1e-10,4,2,4),door,'right',tol=0), ('right',))
        # A degenerate safe door remains valid.
        self.assertEqual(baseline.admissible_initial_directions((0,4,0,4),(2,2,2,2),'right'),
                         ('right','up','down'))

    def test_invalid_inputs(self):
        for corridor,door,direction,tol in (
            ((0,4,0,4),(1,2,1,2),'bad',0),
            ((0,4,0,4),(1,2,1,2),'up',-1),
            ((0,4,0,np.inf),(1,2,1,2),'up',0),
            ((0,4,0,4),(3,2,1,2),'up',0),
            ((0,4,0,4),(-1,2,1,2),'up',0),
        ):
            with self.assertRaises(ValueError):
                baseline.admissible_initial_directions(corridor,door,direction,tol)

    def test_all_nine_pairs_share_one_forward_pass(self):
        corridors = rectangles([(0,10,-10,10),(4,24,-2,2),(18,28,-10,10)])
        robot = SimpleNamespace(r=1,R=3)
        with patch.object(baseline,'analyze_orthogonal_polyline_feasibility',
                          wraps=baseline.analyze_orthogonal_polyline_feasibility) as analyze:
            result = baseline.compute_filleted_baseline_candidates(corridors,robot,use_joint_solver=False)
        self.assertEqual(analyze.call_count,1)
        self.assertEqual(result.initial_directions,('right','up','down'))
        self.assertEqual(result.final_directions,('right','up','down'))
        self.assertEqual(len(result.pair_results),9)
        self.assertTrue(result.feasible)
        self.assertEqual(result.candidates,tuple(r for r in result.pair_results if r.feasible))
        for candidate in result.pair_results:
            self.assertIs(candidate.feasibility,result.feasibility)
        for candidate in result.candidates:
            self.assertLessEqual(candidate.max_violation,1e-9)
            self.assertEqual(candidate.fillets[0] is None,
                             candidate.initial_direction == 'right')
            self.assertEqual(candidate.fillets[-1] is None,
                             candidate.final_direction == 'right')

    def test_geometry_does_not_impose_turning_radius(self):
        # Up is geometrically available, but this shallow corridor cannot fit R=3.
        c = rectangles([(0,4,-.1,4),(0,16,0,4),(12,16,-.1,4)])
        result = baseline.compute_filleted_baseline_candidates(c,SimpleNamespace(r=1,R=3),use_joint_solver=False)
        self.assertTrue(result.feasibility.feasible)
        self.assertEqual(result.initial_directions,('up',))
        self.assertEqual(result.final_directions,('down',))
        self.assertFalse(result.feasible)
        self.assertEqual(result.pair_results[0].status,'empty_fillet_region')

    def test_budget_failure_and_empty_direction_sets(self):
        c = rectangles([(0,4,-8,4),(0,16,0,4),(12,16,-8,4)])
        result = baseline.compute_filleted_baseline_candidates(c,SimpleNamespace(r=1,R=3),
                                   max_backtracking_attempts=0,use_joint_solver=False)
        self.assertEqual(result.status,'boundary_construction_unresolved')
        self.assertFalse(result.pair_results[0].certified_infeasible)
        c = rectangles([(0,4,0,4),(0,16,0,4),(12,16,0,4)])
        result = baseline.compute_filleted_baseline_candidates(c,SimpleNamespace(r=1,R=3))
        self.assertTrue(result.feasibility.feasible)
        self.assertEqual(result.status,'no_admissible_boundary_directions')
        self.assertEqual(result.pair_results,())

    def test_thesis_example_and_internal_rejection(self):
        m = runpy.run_path('experiments/bicycle_journal_paper/example_planner_thesis.py')
        c,r = m['build_example']()
        result = baseline.compute_filleted_baseline_candidates(c,r,use_joint_solver=False)
        self.assertEqual(result.initial_directions,('down',))
        self.assertEqual(result.final_directions,('down',))
        self.assertTrue(result.feasible,result.status)
        self.assertIsNotNone(result.candidates[0].fillets[0])
        self.assertIsNotNone(result.candidates[0].fillets[-1])
        rejected = baseline.compute_filleted_baseline_candidates(rectangles([(0,4,0,4)]*3),r)
        self.assertFalse(rejected.feasibility.feasible)
        self.assertEqual(rejected.pair_results,())


if __name__ == '__main__':
    unittest.main()
