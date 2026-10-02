"""Evaluate all corridor maps with the joint solver explicitly disabled.

Run directly in VS Code. Timings exclude map setup and use median repeated calls.
First rejected waypoint uses one-based p_j numbering in the CSV and console.
"""

import csv
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from examples_maps_polyline import EXAMPLE_NUMBERS, example_corridor_sequence
from kappa_planner.baseline_construction import compute_filleted_baseline

REPETITIONS = 10
MAX_ATTEMPTS = 128
OUTPUT = Path(__file__).parent / 'results' / 'baseline_heuristic.csv'


def main():
    rows = []
    for number in EXAMPLE_NUMBERS:
        corridors, _, _, robot = example_corridor_sequence(number)
        samples = []
        for _ in range(REPETITIONS):
            start = perf_counter_ns()
            result = compute_filleted_baseline(
                corridors, robot, use_joint_solver=False,
                max_backtracking_attempts=MAX_ATTEMPTS,
            )
            samples.append((perf_counter_ns() - start) / 1e6)
        row = dict(example=number, status=result.status, feasible=result.feasible,
                   midpoint_only=result.midpoint_only_success,
                   attempts=result.backtracking_attempts,
                   first_rejected_waypoint=('' if result.first_rejected_waypoint is None
                                            else result.first_rejected_waypoint + 1),
                   median_ms=round(median(samples), 6), use_joint_solver=False)
        rows.append(row)
        print(row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUTPUT}")
    print(f"{sum(row['feasible'] for row in rows)}/{len(rows)} constructed; "
          f"{sum(row['midpoint_only'] for row in rows)} with first midpoints only.")


if __name__ == '__main__':
    main()
