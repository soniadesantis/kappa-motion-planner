"""Endpoint sampling and independent boundary outcomes in the random benchmark."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.poses import absolute_to_relative_pose

EXAMPLES = Path(__file__).resolve().parents[1] / "experiments" / "bicycle_journal_paper"
sys.path.insert(0, str(EXAMPLES))
try:
    import benchmark_new_bicycle_baseline as benchmark
finally:
    sys.path.pop(0)


class RandomBaselineBoundaryBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.bicycle = benchmark.make_fixed_bicycle()
        self.rng = np.random.default_rng(7)

    def test_endpoint_poses_are_disk_safe_and_away_from_overlap_in_all_frames(self):
        for heading in range(4):
            tilt = heading * np.pi / 2
            corridor = CorridorWorld(4, 12, [0, 0], tilt)
            direction = np.array([np.cos(tilt), np.sin(tilt)])
            for initial in (True, False):
                neighbor = CorridorWorld(4, 4, (5 if initial else -5) * direction, tilt)
                for _ in range(10):
                    pose = benchmark.sample_endpoint_pose(
                        self.rng, corridor, neighbor, self.bicycle, initial=initial
                    )
                    self.assertIsNotNone(pose)
                    np.testing.assert_allclose(
                        absolute_to_relative_pose(corridor, pose["world"]),
                        pose["relative"], atol=1e-12,
                    )
                    x, y, theta = pose["relative"]
                    self.assertLessEqual(abs(x) + .5, 2)
                    self.assertLessEqual(abs(y) + .5, 6)
                    self.assertGreaterEqual(theta, 0)
                    self.assertLessEqual(theta, np.pi)
                    self.assertGreaterEqual(pose["overlap_clearance"], 2 * self.bicycle.max_radius)
                    self.assertLess(y, 3) if initial else self.assertGreater(y, -3)

    def assert_structured_connection(self, previous, following, heading, change):
        rotation = benchmark.CARDINAL_ROTATIONS[heading]
        corners = np.array([
            [previous[0], previous[2]], [previous[1], previous[2]],
            [previous[1], previous[3]], [previous[0], previous[3]],
        ])
        forward_coordinates = corners @ rotation[:, 0]
        edge = corners[np.isclose(forward_coordinates, forward_coordinates.max())]
        following_center = np.array([
            (following[0] + following[1]) / 2,
            (following[2] + following[3]) / 2,
        ])
        if change:
            for x, y in edge:
                self.assertGreaterEqual(x, following[0] - 1e-9)
                self.assertLessEqual(x, following[1] + 1e-9)
                self.assertGreaterEqual(y, following[2] - 1e-9)
                self.assertLessEqual(y, following[3] + 1e-9)
            outgoing = change * rotation[:, 1]
            following_corners = np.array([
                [following[0], following[2]], [following[1], following[2]],
                [following[1], following[3]], [following[0], following[3]],
            ])
            # The incoming forward edge coincides with the outgoing far side.
            np.testing.assert_allclose(
                edge @ rotation[:, 0],
                (following_corners @ rotation[:, 0]).max(), atol=1e-9,
            )
            # The overlap uses the entire outgoing width, up to the available
            # length of the incoming corridor, and the complete incoming width.
            overlap_width = min(previous[1], following[1]) - max(previous[0], following[0])
            overlap_height = min(previous[3], following[3]) - max(previous[2], following[2])
            previous_length = np.ptp(corners @ rotation[:, 0])
            incoming_width = np.ptp(corners @ rotation[:, 1])
            outgoing_width = np.ptp(following_corners @ rotation[:, 0])
            self.assertAlmostEqual(
                overlap_width * overlap_height,
                min(previous_length, outgoing_width) * incoming_width,
            )
            self.assertGreater((following_corners @ outgoing).max(), (edge @ outgoing).max())
        else:
            previous_center = corners.mean(axis=0)
            self.assertAlmostEqual(
                previous_center @ rotation[:, 1], following_center @ rotation[:, 1]
            )

    def test_canonical_connections_rotate_correctly_for_all_twelve_cases(self):
        for heading in range(4):
            rotation = benchmark.CARDINAL_ROTATIONS[heading]
            corners = np.array([[0, -2], [12, -2], [12, 2], [0, 2]]) @ rotation.T
            low, high = corners.min(axis=0), corners.max(axis=0)
            previous = (low[0], high[0], low[1], high[1])
            for change in (-1, 0, 1):
                following = benchmark.construct_connected_corridor(
                    previous, heading, change, 3, 10, [1, 1.5], 1
                )
                self.assert_structured_connection(previous, following, heading, change)
        with self.assertRaises(ValueError):
            benchmark.construct_connected_corridor((0, 12, -2, 2), 0, 1, 3, 3, [1, 1])

    def test_generated_sequences_satisfy_structured_geometry(self):
        for count in (5, 10, 20, 30, 40, 50):
            for _ in range(5):
                bounds, sampled = benchmark.sample_validated_corridors(self.rng, count, self.bicycle)
                self.assertEqual(len(bounds), count)
                self.assertEqual(len(sampled["headings"]), count)
                self.assertEqual(len(sampled["transition_types"]), count - 1)
                for j, rectangle in enumerate(bounds):
                    for earlier in bounds[:max(0, j - 1)]:
                        self.assertFalse(benchmark.rectangles_overlap(rectangle, earlier))
                for length in sampled["walk_lengths"][1:-1]:
                    self.assertGreaterEqual(length, benchmark.INTERIOR_WALK_LENGTH_RANGE[0])
                    self.assertLessEqual(length, benchmark.INTERIOR_WALK_LENGTH_RANGE[1])
                corridors = benchmark.corridor_worlds_from_bounds(bounds, sampled["headings"])
                for j in (0, count - 2):
                    self.assertIn(
                        (sampled["headings"][j + 1] - sampled["headings"][j]) % 4,
                        (1, 3),
                    )
                for corridor in (corridors[0], corridors[-1]):
                    self.assertGreaterEqual(
                        corridor.width,
                        2 * self.bicycle.max_radius + self.bicycle.width
                        + benchmark.ENDPOINT_SAMPLING_LENGTH,
                    )
                for initial in (True, False):
                    corridor, neighbor = (corridors[0], corridors[1]) if initial else (corridors[-1], corridors[-2])
                    pose = benchmark.sample_endpoint_pose(
                        self.rng, corridor, neighbor, self.bicycle, initial=initial
                    )
                    self.assertIsNotNone(pose)
                    self.assertGreaterEqual(pose["overlap_clearance"], 2 * self.bicycle.max_radius)
                self.assertEqual(sampled["transverse_offsets"], [0.0] * (count - 1))
                for j in range(count - 1):
                    change = (sampled["headings"][j + 1] - sampled["headings"][j]) % 4
                    self.assertNotEqual(change, 2)
                    change = -1 if change == 3 else change
                    self.assert_structured_connection(bounds[j], bounds[j + 1], sampled["headings"][j], change)

    def test_short_corridors_are_skipped_without_reducing_clearance(self):
        corridor = CorridorWorld(4, 4, [0, 0], 0)
        neighbor = CorridorWorld(4, 2, [2, 0], 0)
        pose = benchmark.sample_endpoint_pose(
            self.rng, corridor, neighbor, self.bicycle, initial=True
        )
        self.assertIsNone(pose)
        self.assertIsNone(benchmark.sample_endpoint_pose(
            self.rng, corridor, corridor, self.bicycle, initial=True
        ))

    def test_rectangle_overlap_detects_containment_and_crossings_but_allows_touching(self):
        rectangle = (0, 4, 0, 4)
        for other in ((1, 2, 1, 2), (-1, 5, 1, 2), (3, 5, 3, 5), rectangle):
            self.assertTrue(benchmark.rectangles_overlap(rectangle, other))
            self.assertTrue(benchmark.rectangles_overlap(other, rectangle))
        for other in ((4, 6, 0, 4), (4, 6, 4, 6), (5, 6, 0, 4)):
            self.assertFalse(benchmark.rectangles_overlap(rectangle, other))

    def test_initial_failure_does_not_claim_final_connection_was_attempted(self):
        baseline = SimpleNamespace(fillets=[object(), object()])
        with patch.object(benchmark, "compute_baseline_boundary_connections", return_value=(None, None)):
            result = benchmark.attach_boundary_connections([], baseline, self.bicycle, [0, 0, 0], [0, 0, 0])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["initial_status"], "failed")
        self.assertEqual(result["final_status"], "not_attempted")

    def test_boundary_errors_are_recorded_without_raising(self):
        baseline = SimpleNamespace(fillets=[object(), object()])
        with patch.object(benchmark, "compute_baseline_boundary_connections", side_effect=ValueError("bad tangent")):
            result = benchmark.attach_boundary_connections([], baseline, self.bicycle, [0, 0, 0], [0, 0, 0])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "ValueError: bad tangent")
        self.assertEqual(result["failure_reason"], "connection_error")

    def test_final_failure_keeps_successful_initial_maneuvers_for_plotting(self):
        baseline = SimpleNamespace(fillets=[object(), object()])
        initial = [object(), object()]
        with patch.object(benchmark, "compute_baseline_boundary_connections", return_value=(initial, None)):
            result = benchmark.attach_boundary_connections([], baseline, self.bicycle, [0, 0, 0], [0, 0, 0])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["initial_status"], "success")
        self.assertEqual(result["final_status"], "failed")
        self.assertIs(baseline.initial_maneuvers, initial)
        self.assertIsNone(baseline.final_maneuvers)

    def test_boundary_failure_preserves_baseline_success_in_report(self):
        failure = {"status": "failed", "failure_reason": "test_boundary_failure"}
        with patch.object(benchmark, "attach_boundary_connections", return_value=failure), patch.object(
            benchmark, "evaluate_refinement", return_value={
                "status": "failed", "computation_ms": 1.0, "failure_reason": "test_refinement_failure"
            },
        ) as refinement:
            group = benchmark.run_group(5, 5, 100, self.rng, self.bicycle)
        successful = [record for record in group["records"] if record["baseline_success"]]
        self.assertTrue(successful)
        for record in successful:
            self.assertEqual(record["status"], "success")
            self.assertFalse(record["boundary_success"])
            self.assertTrue(record["baseline_success_boundary_failure"])
        self.assertEqual(group["baseline_success_boundary_failure_cases"], len(successful))
        self.assertEqual(refinement.call_count, len(successful))

    def test_computation_comparison_does_not_double_count_baseline_attachment(self):
        boundary = {"status": "success", "traversal_time": 100.0}
        refinement = {"status": "success", "computation_ms": 8.0, "traversal_time": 75.0}
        comparison = benchmark.compare_solutions(2.0, 3.0, boundary, refinement)
        self.assertTrue(comparison["paired_success"])
        self.assertEqual(comparison["baseline_computation_ms"], 5.0)
        self.assertEqual(comparison["refined_computation_ms"], 10.0)
        self.assertEqual(comparison["traversal_time_reduction_percent"], 25.0)

    def test_failed_solutions_are_excluded_from_paired_statistics(self):
        boundary = {"status": "success", "traversal_time": 100.0}
        refined = {"status": "success", "computation_ms": 8.0, "traversal_time": 75.0}
        record = {
            "boundary": boundary, "refinement": refined,
            "comparison": benchmark.compare_solutions(2.0, 3.0, boundary, refined),
        }
        summary = benchmark.summarize_comparison([record, {"comparison": {"paired_success": False}}])
        self.assertEqual(summary["paired_cases"], 1)
        self.assertEqual(summary["refined_shorter"], 1)
        self.assertEqual(summary["baseline_traversal_time"]["median"], 100.0)
        self.assertEqual(summary["refined_traversal_time"]["median"], 75.0)

    def test_refinement_receives_explicit_poses_and_records_boundary_errors(self):
        baseline = SimpleNamespace()
        start, end = [0, 0, 0], [1, 1, 1]
        with patch.object(benchmark, "refine_bicycle_baseline", side_effect=ValueError("invalid connection")) as refine:
            outcome = benchmark.evaluate_refinement([], baseline, self.bicycle, start, end)
        self.assertIs(refine.call_args.kwargs["initial_pose"], start)
        self.assertIs(refine.call_args.kwargs["final_pose"], end)
        self.assertEqual(outcome["status"], "error")
        self.assertEqual(outcome["error"], "ValueError: invalid connection")
        self.assertIsNotNone(outcome["computation_ms"])

    def test_unsuccessful_refinement_retains_complete_baseline(self):
        trajectory = [SimpleNamespace(maneuver_time=4.0), SimpleNamespace(maneuver_time=6.0)]
        baseline = SimpleNamespace(trajectory=trajectory)
        for result, error in ((None, None), (None, ValueError("invalid tangent"))):
            with self.subTest(error=error), patch.object(
                benchmark, "refine_bicycle_baseline", return_value=result, side_effect=error
            ):
                outcome = benchmark.evaluate_refinement([], baseline, self.bicycle, [0, 0, 0], [1, 1, 1])
            self.assertEqual(outcome["status"], "success")
            self.assertEqual(outcome["solution_source"], "baseline_fallback")
            self.assertEqual(outcome["attempt_status"], "error" if error else "failed")
            self.assertEqual(outcome["traversal_time"], 10.0)
            self.assertIs(baseline.trajectory, trajectory)

    def test_baseline_fallback_counts_attempt_and_attachment_time(self):
        boundary = {"status": "success", "traversal_time": 100.0}
        refinement = {
            "status": "success", "solution_source": "baseline_fallback",
            "computation_ms": 8.0, "traversal_time": 100.0,
        }
        comparison = benchmark.compare_solutions(2.0, 3.0, boundary, refinement)
        self.assertTrue(comparison["paired_success"])
        self.assertEqual(comparison["baseline_computation_ms"], 5.0)
        self.assertEqual(comparison["refined_computation_ms"], 13.0)
        self.assertEqual(comparison["traversal_time_reduction_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
