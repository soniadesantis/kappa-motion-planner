"""Thesis geometry cases and independent LP verification of propagation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog

from kappa_planner.baseline_construction import (
    _propagate_coordinate,
    _viable_coordinate,
    analyze_orthogonal_polyline_feasibility,
    check_orthogonal_polyline_feasibility,
)
from kappa_planner.corridor import CorridorWorld
from kappa_planner.vehicle import Bicycle


def rectangle(xmin, xmax, ymin=0, ymax=2):
    return CorridorWorld(ymax - ymin, xmax - xmin,
                         [(xmin + xmax) / 2, (ymin + ymax) / 2], 0)


class BaselineConstructionTest(unittest.TestCase):
    def test_four_signed_directions(self):
        base = [rectangle(0, 2), rectangle(0, 6), rectangle(4, 6)]
        for matrix, direction in (
            (np.eye(2), "right"), (-np.eye(2), "left"),
            (np.array([[0, -1], [1, 0]]), "up"),
            (np.array([[0, 1], [-1, 0]]), "down"),
        ):
            corridors = [SimpleNamespace(corners=c.corners @ matrix.T) for c in base]
            report = analyze_orthogonal_polyline_feasibility(
                corridors, SimpleNamespace(r=.25, R=1))
            self.assertTrue(report.feasible)
            self.assertEqual(report.passage_directions, (direction,))

    def test_collinear_passages_are_allowed(self):
        corridors = [rectangle(0, 2), rectangle(0, 6),
                     rectangle(4, 10), rectangle(8, 10)]
        report = analyze_orthogonal_polyline_feasibility(
            corridors, SimpleNamespace(r=.25, R=1))
        self.assertTrue(report.feasible)
        self.assertEqual(report.passage_directions, ("right", "right"))

    def test_point_doors_boundary_contact_and_exact_spacing(self):
        corridors = [rectangle(0, 2), rectangle(0, 5), rectangle(3, 5)]
        robot = SimpleNamespace(r=1, R=1.5)
        report = analyze_orthogonal_polyline_feasibility(corridors, robot)
        self.assertTrue(report.feasible)
        self.assertEqual(report.safe_overlaps, ((1, 1, 1, 1), (4, 4, 1, 1)))
        robot.R = 1.5001
        self.assertFalse(check_orthogonal_polyline_feasibility(corridors, robot))
        # Nonconsecutive corridor boundary contact is permitted.
        corridors = [rectangle(0, 2), rectangle(0, 4), rectangle(2, 4)]
        self.assertTrue(check_orthogonal_polyline_feasibility(
            corridors, SimpleNamespace(r=.25, R=1)))

    def test_assumptions_and_missing_axis(self):
        robot = SimpleNamespace(r=.25, R=1)
        cases = (
            ([rectangle(0, 4)] * 3, "intersecting_safe_overlaps"),
            ([rectangle(0, 2), rectangle(3, 5), rectangle(4, 6)],
             "empty_safe_overlap"),
            ([rectangle(0, 2), rectangle(0, 6, 0, 6),
              rectangle(4, 6, 4, 6)], "no_orthogonal_connection"),
        )
        for corridors, status in cases:
            self.assertEqual(analyze_orthogonal_polyline_feasibility(
                corridors, robot).status, status)

    def test_global_failure_despite_local_feasibility(self):
        # Each adjacent pair permits separation 3.5, but two increases cannot
        # fit the middle reachable interval: [0,1] -> [2,4] -> [5,6].
        reachable = _propagate_coordinate(
            [(0, 1), (2, 4), (5, 6)], ["increase", "increase"], 3.5, 0)
        self.assertEqual(reachable, ((0, 1), (3.5, 4), None))
        report = analyze_orthogonal_polyline_feasibility(
            [rectangle(0, 2), rectangle(0, 5),
             rectangle(3, 8), rectangle(6, 8)], SimpleNamespace(r=.25, R=2))
        self.assertEqual(report.status, "insufficient_spacing")
        self.assertEqual(report.x_reachable, ((.25, 1.75), (4.25, 4.75), None))
        self.assertEqual(report.x_viable, (None, None, None))
        self.assertEqual(report.y_viable, (None, None, None))

    def test_recurrence_boundaries_and_empty_absorption(self):
        cases = (
            ([(0, 2), (1, 3)], ["equal"], ((0, 2), (1, 2))),
            ([(0, 2), (2, 4)], ["increase"], ((0, 2), (3, 4))),
            ([(2, 4), (0, 2)], ["decrease"], ((2, 4), (0, 1))),
            ([(0, 0), (3, 3)], ["increase"], ((0, 0), (3, 3))),
            ([(0, 0), (1, 2), (0, 5)], ["increase", "equal"],
             ((0, 0), None, None)),
        )
        for intervals, relations, expected in cases:
            self.assertEqual(_propagate_coordinate(intervals, relations, 3, 0),
                             expected)

    def test_boolean_check_never_runs_backward_diagnostic(self):
        with patch("kappa_planner.baseline_construction._viable_coordinate",
                   side_effect=AssertionError("Backward diagnostic called")):
            self.assertTrue(check_orthogonal_polyline_feasibility(
                [rectangle(0, 2), rectangle(0, 6), rectangle(4, 6)],
                SimpleNamespace(r=.25, R=1)))

    def test_nominal_assumptions_are_checked_before_passage_feasibility(self):
        # First pair has no orthogonal passage; the last pair intersects.
        report = analyze_orthogonal_polyline_feasibility(
            [rectangle(0, 2, 0, 2), rectangle(0, 6, 0, 6),
             rectangle(4, 6, 4, 6), rectangle(4, 6, 4, 6)],
            SimpleNamespace(r=.25, R=1))
        self.assertEqual(report.status, "intersecting_safe_overlaps")

    def test_tolerance_is_explicit_boundary_slack(self):
        intervals = [(0., 0.), (3. - 5e-10, 3. - 5e-10)]
        self.assertIsNone(_propagate_coordinate(intervals, ["increase"], 3, 0)[-1])
        self.assertIsNotNone(_propagate_coordinate(
            intervals, ["increase"], 3, 1e-9)[-1])

    def test_nonconsecutive_corridors_may_overlap(self):
        # C0 and C2 overlap, but erosion makes the two safe doors disjoint.
        corridors = [rectangle(0, 2.2), rectangle(0, 4), rectangle(1.8, 4)]
        report = analyze_orthogonal_polyline_feasibility(
            corridors, SimpleNamespace(r=.25, R=1))
        self.assertTrue(report.feasible)
        self.assertEqual(report.passage_directions, ("right",))

    def test_safe_door_boundary_contact_is_rejected(self):
        for last in (rectangle(2, 4), rectangle(2, 4, 2, 4)):
            report = analyze_orthogonal_polyline_feasibility(
                [rectangle(0, 3, 0, 3), rectangle(0, 5, 0, 5), last],
                SimpleNamespace(r=.5, R=1))
            self.assertEqual(report.status, "intersecting_safe_overlaps")

    def test_backward_intervals_remove_values_without_full_continuation(self):
        report = analyze_orthogonal_polyline_feasibility(
            [rectangle(0, 3), rectangle(0, 5), rectangle(3, 5)],
            SimpleNamespace(r=.25, R=1))
        self.assertTrue(report.feasible)
        self.assertEqual(report.x_reachable[0], (.25, 2.75))
        self.assertEqual(report.x_viable[0], (.25, 2.75))
        report = analyze_orthogonal_polyline_feasibility(
            [rectangle(0, 3), rectangle(0, 5), rectangle(3, 5)],
            SimpleNamespace(r=.25, R=1.5))
        self.assertEqual(report.x_viable[0], (.25, 1.75))
        self.assertEqual(report.x_viable[1], (3.25, 4.75))

    def test_propagation_matches_independent_linear_program(self):
        rng = np.random.default_rng(41)
        for _ in range(200):
            count = int(rng.integers(2, 9))
            low = rng.integers(-8, 9, count)
            high = low + rng.integers(0, 9, count)
            intervals = list(zip(low, high))
            relations = rng.choice(["equal", "increase", "decrease"], count - 1)
            spacing = int(rng.integers(1, 5))
            inequalities, limits, equalities = [], [], []
            for j, relation in enumerate(relations):
                row = np.zeros(count)
                row[j], row[j + 1] = 1, -1
                if relation == "equal":
                    equalities.append(row)
                else:
                    inequalities.append(row if relation == "increase" else -row)
                    limits.append(-spacing)
            solution = linprog(
                np.zeros(count),
                A_ub=np.array(inequalities) if inequalities else None,
                b_ub=limits if inequalities else None,
                A_eq=np.array(equalities) if equalities else None,
                b_eq=np.zeros(len(equalities)) if equalities else None,
                bounds=intervals, method="highs",
            )
            reachable = _propagate_coordinate(intervals, relations, spacing, 0)
            self.assertEqual(reachable[-1] is not None, solution.success)
            viable = _viable_coordinate(reachable, relations, spacing, 0)
            if not solution.success:
                self.assertEqual(viable, (None,) * count)
            if solution.success:
                for index, sign, endpoint in (
                    (j, sign, endpoint) for j in range(count)
                    for sign, endpoint in ((1, 0), (-1, 1))
                ):
                    objective = np.zeros(count)
                    objective[index] = sign
                    optimum = linprog(
                        objective,
                        A_ub=np.array(inequalities) if inequalities else None,
                        b_ub=limits if inequalities else None,
                        A_eq=np.array(equalities) if equalities else None,
                        b_eq=np.zeros(len(equalities)) if equalities else None,
                        bounds=intervals, method="highs",
                    )
                    self.assertAlmostEqual(viable[index][endpoint],
                                           optimum.x[index])
                    if index == count - 1:
                        self.assertAlmostEqual(reachable[-1][endpoint],
                                               optimum.x[-1])

    def test_existing_vehicle_convention_and_invalid_inputs(self):
        robot = Bicycle([0, 0, 0], width=.2, length=.2, wheelbase=.25,
                        v_max=1, v_min=0, delta_max=.5, delta_min=-.5)
        corridors = [rectangle(0, 2), rectangle(0, 6), rectangle(4, 6)]
        self.assertTrue(check_orthogonal_polyline_feasibility(corridors, robot))
        for robot in (SimpleNamespace(r=0, R=1), SimpleNamespace(r=1, R=1),
                      SimpleNamespace(r=.1, R=float("inf")), SimpleNamespace()):
            with self.assertRaises(ValueError):
                check_orthogonal_polyline_feasibility(corridors, robot)
        with self.assertRaises(ValueError):
            check_orthogonal_polyline_feasibility(corridors[:2],
                                                SimpleNamespace(r=.1, R=1))
        rotated = CorridorWorld(2, 4, [0, 0], .3)
        with self.assertRaises(ValueError):
            check_orthogonal_polyline_feasibility(
                [rotated] * 3, SimpleNamespace(r=.1, R=1))


if __name__ == "__main__":
    unittest.main()
