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


def direction_to_suffix(direction):
    if direction == "left":
        return "L"
    elif direction == "right":
        return "R"
    raise ValueError(f"Unknown direction: {direction}")


def analytical_name_to_sequence(best_name):
    """
    Example:
        'TCSCT left-right'
    becomes:
        ['SpinL', 'ArcL', 'Straight', 'ArcR', 'SpinR']
    """

    family, directions = best_name.split(" ", 1)
    direction0, directionf = directions.split("-")

    suffix0 = direction_to_suffix(direction0)
    suffixf = direction_to_suffix(directionf)

    sequence = []

    for symbol in family:
        if symbol == "T":
            # First T uses initial direction, final T uses final direction
            if len(sequence) == 0:
                sequence.append(f"Spin{suffix0}")
            else:
                sequence.append(f"Spin{suffixf}")

        elif symbol == "C":
            # First C uses initial direction, final C uses final direction
            if "Straight" not in sequence:
                sequence.append(f"Arc{suffix0}")
            else:
                sequence.append(f"Arc{suffixf}")

        elif symbol == "S":
            sequence.append("Straight")

    return sequence


def sequences_match(analytical_sequence, ocp_sequence):
    return analytical_sequence == ocp_sequence


if __name__ == "__main__":

    load_path = Path(
        "/home/sonia/Projects/kappa-motion-planner/experiments/"
        "pose2pose_unicycle/pose_to_pose_sweep_results_90x90.json"
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


    # -------------------------------------------------------------------------
    # Largest negative time difference
    # (OCP better than analytical)
    # -------------------------------------------------------------------------

    negative_cases = [
        case for case in successful_cases
        if case["time_difference"] < 0.0
    ]

    if negative_cases:

        best_ocp_case = min(
            negative_cases,
            key=lambda c: c["time_difference"]
        )

        print("\n" + "=" * 80)
        print("LARGEST NEGATIVE TIME DIFFERENCE")
        print("=" * 80)

        print(
            f"Case {best_ocp_case['case_id']:02d}: "
            f"theta0={best_ocp_case['theta0_deg']:.1f} deg, "
            f"thetaf={best_ocp_case['thetaf_deg']:.1f} deg"
        )

        print(
            f"Analytical time : "
            f"{best_ocp_case['best_analytical_time']:.6f} s"
        )

        print(
            f"OCP time        : "
            f"{best_ocp_case['ocp_time']:.6f} s"
        )

        print(
            f"Difference      : "
            f"{best_ocp_case['time_difference']:.6f} s"
        )

        print(
            f"OCP sequence    : "
            f"{' - '.join(best_ocp_case['ocp_sequence'])}"
        )

    else:

        print("\nNo negative time differences found.")


    # -------------------------------------------------------------------------
    # Failed cases
    # -------------------------------------------------------------------------

    failed_cases = [
        case for case in results
        if not case.get("ocp_success", False)
    ]

    print("\n" + "=" * 80)
    print("FAILED CASES")
    print("=" * 80)

    if len(failed_cases) == 0:

        print("No failed cases found.")

    else:

        print(f"Number of failed cases: {len(failed_cases)}")

        for case in failed_cases:

            print(
                f"\nCase {case['case_id']:02d}: "
                f"theta0={case['theta0_deg']:.1f} deg, "
                f"thetaf={case['thetaf_deg']:.1f} deg"
            )

            if "error" in case:
                print(f"  Error: {case['error']}")


    # -------------------------------------------------------------------------
    # Structure comparison between analytical and OCP sequences
    # -------------------------------------------------------------------------

    nonmatching_cases = []

    for case in successful_cases:

        analytical_sequence = analytical_name_to_sequence(
            case["best_analytical_name"]
        )

        ocp_sequence = case["ocp_sequence"]

        if analytical_sequence != ocp_sequence:

            case_with_sequences = case.copy()
            case_with_sequences["analytical_sequence"] = analytical_sequence

            nonmatching_cases.append(case_with_sequences)


    print("\n" + "=" * 80)
    print("STRUCTURE COMPARISON")
    print("=" * 80)

    n_matching = len(successful_cases) - len(nonmatching_cases)
    n_nonmatching = len(nonmatching_cases)

    print(f"Matching structures    : {n_matching}")
    print(f"Nonmatching structures : {n_nonmatching}")


    # -------------------------------------------------------------------------
    # Optional detailed print
    # -------------------------------------------------------------------------

    if n_nonmatching > 0:

        user_input = input(
            "\nPrint nonmatching cases? [y/n]: "
        ).strip().lower()

        if user_input == "y":

            for case in nonmatching_cases:

                print("\n" + "-" * 80)

                print(
                    f"Case {case['case_id']:02d}: "
                    f"theta0={case['theta0_deg']:.1f} deg, "
                    f"thetaf={case['thetaf_deg']:.1f} deg"
                )

                print(
                    f"  Best analytical : "
                    f"{case['best_analytical_name']}"
                )

                print(
                    f"  Analytical seq  : "
                    f"{' - '.join(case['analytical_sequence'])}"
                )

                print(
                    f"  OCP sequence    : "
                    f"{' - '.join(case['ocp_sequence'])}"
                )

                print(
                    f"  Time difference : "
                    f"{case['time_difference']:.6f} s"
                )

    else:

        print("\nAll structures match.")