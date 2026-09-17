"""Print discrepancy statistics for saved sweeps, without generating plots.

Run with no arguments for the Sobol TST N100 sweep, or pass JSON paths.
"""

import argparse
import json
from pathlib import Path

import numpy as np


DIRECTORY = Path(__file__).resolve().parent.parent
RESULTS_FILENAME = "sweep4/sobol_sweep_OCP_TST_initial_guess_N100_M4.json"


def load_metrics(path):
    with Path(path).open(encoding="utf-8") as source:
        data = json.load(source)
    metrics = {
        "Absolute time error [s]": [],
        "Relative time error [%]": [],
        "Signed time difference [s]": [],
        "Hausdorff distance [m]": [],
    }
    successful = 0
    case_metrics = []
    faster_case_ids = set()
    most_negative_difference = 0.0
    for case in data["results"]:
        if not (case.get("success") and case.get("ocp_success")):
            continue
        analytical = case.get("best_analytical_time")
        ocp = case.get("ocp_time")
        distance = case.get("hausdorff_distance")
        if analytical is None or ocp is None or distance is None:
            continue
        analytical, ocp, distance = map(float, (analytical, ocp, distance))
        if not (np.isfinite(analytical) and analytical > 0
                and np.isfinite(ocp) and np.isfinite(distance) and distance >= 0):
            continue
        successful += 1
        difference = ocp - analytical
        absolute_error = abs(difference)
        if difference < 0:
            faster_case_ids.add(case["case_id"])
            most_negative_difference = min(most_negative_difference, difference)
        metrics["Absolute time error [s]"].append(absolute_error)
        metrics["Relative time error [%]"].append(
            100 * absolute_error / analytical
        )
        metrics["Signed time difference [s]"].append(difference)
        metrics["Hausdorff distance [m]"].append(distance)
        case_metrics.append((case["case_id"], difference, distance))
    return (data.get("metadata", {}), len(data["results"]), successful,
            metrics, faster_case_ids, most_negative_difference, case_metrics)


def print_summary(path, top_cases=0):
    (metadata, total, successful, metrics, faster_case_ids, most_negative,
     case_metrics) = load_metrics(path)
    print(f"\nInput: {path}")
    print(f"N={metadata.get('N', '?')}, M={metadata.get('M', '?')}")
    print(f"Cases: {total:,}; successful comparisons: {successful:,}")
    print(f"OCP faster: {len(faster_case_ids):,} cases "
          f"({len(faster_case_ids) / successful:.2%})")
    if faster_case_ids:
        print(f"Largest OCP time reduction: {-most_negative:.6e} s")
    print("Absolute error: |T_OCP - T_analytical|")
    print("Relative error: 100 * |T_OCP - T_analytical| / T_analytical")
    print("Signed difference: T_OCP - T_analytical")
    print("Hausdorff distances use the saved measurements.")
    print(f"\n{'Metric':32s} {'Count':>7s} {'Mean':>14s} {'Median':>14s} "
          f"{'95th percentile':>16s} {'Maximum':>14s}")
    for name, values in metrics.items():
        if not values:
            print(f"{name:32s} {0:7d}  No valid saved measurements")
            continue
        print(f"{name:32s} {len(values):7,d} {np.mean(values):14.6e} "
              f"{np.median(values):14.6e} {np.percentile(values, 95):16.6e} "
              f"{np.max(values):14.6e}")
    if top_cases:
        by_time = sorted(case_metrics, key=lambda item: abs(item[1]), reverse=True)[:top_cases]
        by_distance = sorted(case_metrics, key=lambda item: item[2], reverse=True)[:top_cases]
        for title, cases in (("Absolute time error", by_time),
                             ("Hausdorff distance", by_distance)):
            print(f"\nTop {top_cases} by {title}:")
            print(f"{'Rank':>4s} {'Case ID':>8s} {'Delta T [s]':>16s} {'Hausdorff [m]':>16s}")
            for rank, (case_id, difference, distance) in enumerate(cases, 1):
                print(f"{rank:4d} {case_id:8d} {difference:16.6e} {distance:16.6e}")
        shared_ids = {item[0] for item in by_time} & {item[0] for item in by_distance}
        print(f"Shared top-{top_cases} case IDs: "
              f"{', '.join(map(str, sorted(shared_ids))) if shared_ids else 'none'}")
    return faster_case_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path,
                        default=[DIRECTORY / "results" / RESULTS_FILENAME])
    parser.add_argument("--top-cases", type=int, default=0,
                        help="Print the largest absolute time errors and Hausdorff distances.")
    args = parser.parse_args()
    if args.top_cases < 0:
        parser.error("--top-cases must be nonnegative")
    faster_case_sets = []
    for input_path in args.inputs:
        path = input_path.expanduser()
        if not path.is_absolute() and not path.is_file():
            path = DIRECTORY / "results" / path
        faster_case_sets.append(print_summary(path, args.top_cases))
    if len(faster_case_sets) > 1:
        print(f"\nOCP faster in every input: "
              f"{len(set.intersection(*faster_case_sets)):,} shared case IDs")


if __name__ == "__main__":
    main()
