import json
from pathlib import Path


def print_case(case):

    print(
        f"\nCase {case['case_id']:02d}: "
        f"theta0={case['theta0_deg']:6.1f} deg, "
        f"thetaf={case['thetaf_deg']:6.1f} deg"
    )

    print(f"  Best analytical : {case['best_analytical_name']}")
    print(f"  Analytical time : {case['best_analytical_time']:.6f} s")

    print(f"  OCP time        : {case['ocp_time']:.6f} s")
    print(f"  Difference      : {case['time_difference']:.6f} s")
    print(f"  OCP solve time  : {case['ocp_solve_time']:.3f} s")

    print(
        f"  OCP sequence    : "
        f"{' - '.join(case['ocp_sequence'])}"
    )

    print(f"  OCP success     : {case['ocp_success']}")


if __name__ == "__main__":

    load_path = Path(
        "/home/sonia/Projects/kappa-motion-planner/experiments/"
        "pose2pose_unicycle/pose_to_pose_sweep_results.json"
    )

    with open(load_path, "r") as f:
        results = json.load(f)

    print("\n" + "=" * 80)
    print("LOADED SWEEP RESULTS")
    print("=" * 80)

    for case in results:
        print_case(case)

    print("\n" + "=" * 80)
    print(f"TOTAL CASES LOADED: {len(results)}")
    print("=" * 80)

    successful_cases = [
        case for case in results
        if case.get("ocp_success", False)
    ]

    if successful_cases:

        worst_case = max(
            successful_cases,
            key=lambda c: abs(c["time_difference"])
        )

        print("\n" + "=" * 80)
        print("MAXIMUM TIME DIFFERENCE")
        print("=" * 80)

        print(
            f"Case {worst_case['case_id']:02d}: "
            f"theta0={worst_case['theta0_deg']:.1f} deg, "
            f"thetaf={worst_case['thetaf_deg']:.1f} deg"
        )

        print(
            f"Analytical time : "
            f"{worst_case['best_analytical_time']:.6f} s"
        )

        print(
            f"OCP time        : "
            f"{worst_case['ocp_time']:.6f} s"
        )

        print(
            f"Difference      : "
            f"{worst_case['time_difference']:.6f} s"
        )

        print(
            f"OCP sequence    : "
            f"{' - '.join(worst_case['ocp_sequence'])}"
        )