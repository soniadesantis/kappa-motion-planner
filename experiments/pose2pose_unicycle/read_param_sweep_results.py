import json
from pathlib import Path
import numpy as np


def print_case(case):

    print(
        f"\nCase {case['case_id']:02d}: "
        f"theta0={case['theta0_deg']:6.1f} deg, "
        f"thetaf={case['thetaf_deg']:6.1f} deg"
    )

    print("-" * 60)

    # ------------------------------------------------------------------
    # Analytical solution
    # ------------------------------------------------------------------

    print(f"  Best analytical : {case['best_analytical_name']}")
    print(f"  Analytical time : {case['best_analytical_time']:.6f} s")

    if case.get("analytical_solve_time") is not None:
        print(
            f"  Analytical solve: "
            f"{case['analytical_solve_time']:.6f} s"
        )

    if case.get("best_analytical_sequence") is not None:

        print(
            f"  Analytical seq  : "
            f"{' - '.join(case['best_analytical_sequence'])}"
        )

    # ------------------------------------------------------------------
    # Arc / segment statistics
    # ------------------------------------------------------------------

    arc1 = case.get("arc1_length")
    arc2 = case.get("arc2_length")
    segment = case.get("segment_length")

    r1 = case.get("arc_segment_ratio_1")
    r2 = case.get("arc_segment_ratio_2")

    if (
        arc1 is not None
        and arc2 is not None
        and segment is not None
    ):

        print(
            f"  Arc lengths     : "
            f"{arc1:.3f}, {arc2:.3f}"
        )

        print(
            f"  Segment length  : "
            f"{segment:.3f}"
        )

        if r1 is not None:
            print(
                f"  Segment/Arc r1  : "
                f"{r1:.3f}"
            )
        else:
            print(
                "  Segment/Arc r1  : None"
            )

        if r2 is not None:
            print(
                f"  Segment/Arc r2  : "
                f"{r2:.3f}"
            )
        else:
            print(
                "  Segment/Arc r2  : None"
            )

    elif (
        r1 is not None
        or r2 is not None
    ):

        if r1 is not None:
            print(
                f"  Segment/Arc r1  : "
                f"{r1:.3f}"
            )

        if r2 is not None:
            print(
                f"  Segment/Arc r2  : "
                f"{r2:.3f}"
            )

    # ------------------------------------------------------------------
    # Primitive details
    # ------------------------------------------------------------------

    primitives = case.get("best_analytical_primitives")

    if primitives is not None:

        print(
            f"  # primitives    : "
            f"{len(primitives)}"
        )

        for primitive in primitives:

            print(
                f"      "
                f"{primitive['label']:15s}"
                f"  t={primitive['maneuver_time']:.3f}"
                f"  l={primitive['path_length']:.3f}"
            )

    # ------------------------------------------------------------------
    # OCP solution
    # ------------------------------------------------------------------

    print(f"  OCP time        : {case['ocp_time']:.6f} s")
    print(f"  Difference      : {case['time_difference']:.6f} s")
    print(f"  OCP solve time  : {case['ocp_solve_time']:.3f} s")

    print(
        f"  OCP sequence    : "
        f"{' - '.join(case['ocp_sequence'])}"
    )

    print(
        f"  OCP success     : "
        f"{case['ocp_success']}"
    )


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


def analytical_primitives_to_effective_sequence(
    primitives,
    time_tol=1e-6,
):
    sequence = []

    for primitive in primitives:

        if primitive["maneuver_time"] <= time_tol:
            continue

        label = primitive["label"]

        if label == "turn on-the-spot":

            turn_direction = primitive.get("turn_direction")

            if turn_direction == 1:
                sequence.append("SpinL")
            elif turn_direction == -1:
                sequence.append("SpinR")
            else:
                sequence.append("Spin")

        elif label == "arc":

            turn_direction = primitive.get("turn_direction")

            if turn_direction == 1:
                sequence.append("ArcL")
            elif turn_direction == -1:
                sequence.append("ArcR")
            else:
                sequence.append("Arc")

        elif label == "segment":

            sequence.append("Straight")

        else:
            sequence.append(label)

    return sequence


if __name__ == "__main__":

    RESULTS_FILENAME = "orientation_sweep_N100_OCPpath.json"

    current_dir = Path(__file__).resolve().parent

    load_path = current_dir / "results" / RESULTS_FILENAME

    with open(load_path, "r") as f:
        data = json.load(f)

    metadata = data["metadata"]
    results = data["results"]

    print("\n" + "=" * 80)
    print("LOADED SWEEP RESULTS")
    print("=" * 80)

    print("\nMETADATA")
    print("-" * 80)
    for key, value in metadata.items():
        print(f"{key}: {value}")

    # -------------------------------------------------------------------------
    # Check consistency of discretization settings
    # -------------------------------------------------------------------------

    case_N_values = sorted({
        case.get("N")
        for case in results
        if case.get("N") is not None
    })

    case_M_values = sorted({
        case.get("M")
        for case in results
        if case.get("M") is not None
    })

    print("\n" + "=" * 80)
    print("DISCRETIZATION CONSISTENCY CHECK")
    print("=" * 80)

    print(f"N values found in cases: {case_N_values}")
    print(f"M values found in cases: {case_M_values}")

    metadata_N = metadata.get("N")
    metadata_M = metadata.get("M")

    if len(case_N_values) == 1 and case_N_values[0] == metadata_N:
        print("N consistency           : OK")
    else:
        print("N consistency           : WARNING")

    if len(case_M_values) == 1 and case_M_values[0] == metadata_M:
        print("M consistency           : OK")
    else:
        print("M consistency           : WARNING")

    # for case in results:
    #     print_case(case)

    print("\n" + "=" * 80)
    print(f"TOTAL CASES LOADED: {len(results)}")
    print("=" * 80)

    successful_cases = [
        case for case in results
        if case.get("success", False) and case.get("ocp_success", False)
    ]

    # -------------------------------------------------------------------------
    # Maximum absolute time difference
    # -------------------------------------------------------------------------

    if successful_cases:

        worst_case = max(
            successful_cases,
            key=lambda c: abs(c["time_difference"]),
        )

        print("\n" + "=" * 80)
        print("MAXIMUM TIME DIFFERENCE")
        print("=" * 80)

        print(
            f"Case {worst_case['case_id']:02d}: "
            f"theta0={worst_case['theta0_deg']:.1f} deg, "
            f"thetaf={worst_case['thetaf_deg']:.1f} deg"
        )

        print(f"Analytical time : {worst_case['best_analytical_time']:.6f} s")
        print(f"OCP time        : {worst_case['ocp_time']:.6f} s")
        print(f"Difference      : {worst_case['time_difference']:.6f} s")
        print(f"OCP sequence    : {' - '.join(worst_case['ocp_sequence'])}")

    else:
        print("\nNo successful cases found.")

    # -------------------------------------------------------------------------
    # Arc/segment ratio statistics
    # -------------------------------------------------------------------------

    ratio_values = []

    for case in successful_cases:

        r1 = case.get("arc_segment_ratio_1")
        r2 = case.get("arc_segment_ratio_2")

        if r1 is not None:
            ratio_values.append(
                ("r1", r1, case)
            )

        if r2 is not None:
            ratio_values.append(
                ("r2", r2, case)
            )

    # -------------------------------------------------------------------------
    # Largest negative time difference
    # OCP better than analytical
    # -------------------------------------------------------------------------

    negative_cases = [
        case for case in successful_cases
        if case["time_difference"] < 0.0
    ]

    if negative_cases:

        best_ocp_case = min(
            negative_cases,
            key=lambda c: c["time_difference"],
        )

        print("\n" + "=" * 80)
        print("LARGEST NEGATIVE TIME DIFFERENCE")
        print("=" * 80)

        print(
            f"Case {best_ocp_case['case_id']:02d}: "
            f"theta0={best_ocp_case['theta0_deg']:.1f} deg, "
            f"thetaf={best_ocp_case['thetaf_deg']:.1f} deg"
        )

        print(f"Analytical time : {best_ocp_case['best_analytical_time']:.6f} s")
        print(f"OCP time        : {best_ocp_case['ocp_time']:.6f} s")
        print(f"Difference      : {best_ocp_case['time_difference']:.6f} s")
        print(f"OCP sequence    : {' - '.join(best_ocp_case['ocp_sequence'])}")

    else:
        print("\nNo negative time differences found.")


    print("\n" + "=" * 80)
    print("ARC / SEGMENT RATIO STATISTICS")
    print("=" * 80)

    if len(ratio_values) == 0:

        print("No valid ratios found.")

    else:

        min_ratio_type, min_ratio, min_case = min(
            ratio_values,
            key=lambda x: x[1]
        )

        max_ratio_type, max_ratio, max_case = max(
            ratio_values,
            key=lambda x: x[1]
        )

        print(
            f"Minimum ratio ({min_ratio_type}) : "
            f"{min_ratio:.6f}"
        )

        print(
            f"  Case {min_case['case_id']:02d}: "
            f"theta0={min_case['theta0_deg']:.1f} deg, "
            f"thetaf={min_case['thetaf_deg']:.1f} deg"
        )

        print()

        print(
            f"Maximum ratio ({max_ratio_type}) : "
            f"{max_ratio:.6f}"
        )

        print(
            f"  Case {max_case['case_id']:02d}: "
            f"theta0={max_case['theta0_deg']:.1f} deg, "
            f"thetaf={max_case['thetaf_deg']:.1f} deg"
        )


    # -------------------------------------------------------------------------
    # Failed cases
    # -------------------------------------------------------------------------

    failed_cases = [
        case for case in results
        if not case.get("success", False) or not case.get("ocp_success", False)
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

            print(f"  success     : {case.get('success', None)}")
            print(f"  ocp_success : {case.get('ocp_success', None)}")

            if "error" in case:
                print(f"  Error: {case['error']}")


    # -------------------------------------------------------------------------
    # Structure comparison between analytical and OCP sequences
    # -------------------------------------------------------------------------

    nonmatching_cases = []

    time_tol = 1e-6

    for case in successful_cases:

        analytical_sequence = analytical_primitives_to_effective_sequence(
            case["best_analytical_primitives"],
            time_tol=time_tol,
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

    print(f"Time tolerance         : {time_tol:.1e} s")
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

                print(f"  Best analytical : {case['best_analytical_name']}")

                print(
                    f"  Analytical seq  : "
                    f"{' - '.join(case['analytical_sequence'])}"
                )

                print(
                    f"  OCP sequence    : "
                    f"{' - '.join(case['ocp_sequence'])}"
                )

                print(f"  Time difference : {case['time_difference']:.6f} s")

    else:

        print("\nAll structures match.")

    # -------------------------------------------------------------------------
    # Relative time and time difference statistics
    # -------------------------------------------------------------------------

    relative_differences = [
        100.0 * abs(case["time_difference"]) / case["ocp_time"]
        for case in successful_cases
        if case["ocp_time"] > 1e-12
    ]
    abs_differences = [
        abs(case["time_difference"])
        for case in successful_cases
    ]

    relative_differences = [
        100.0 * abs(case["time_difference"]) / case["ocp_time"]
        for case in successful_cases
        if case["ocp_time"] > 1e-12
    ]

    if len(abs_differences) > 0:

        mean_diff = sum(abs_differences) / len(abs_differences)
        median_diff = float(np.median(abs_differences))
        max_diff = max(abs_differences)

        mean_rel_diff = float(np.mean(relative_differences))
        median_rel_diff = float(np.median(relative_differences))
        max_rel_diff = float(np.max(relative_differences))

        print("\n" + "=" * 80)
        print("TIME DIFFERENCE STATISTICS")
        print("=" * 80)

        print(f"Mean   |ΔT| : {mean_diff:.6e} s")
        print(f"Median |ΔT| : {median_diff:.6e} s")
        print(f"Maximum|ΔT| : {max_diff:.6e} s")

        print()

        print(f"Mean   relative error : {mean_rel_diff:.6e} %")
        print(f"Median relative error : {median_rel_diff:.6e} %")
        print(f"Maximum relative error: {max_rel_diff:.6e} %")


    # -------------------------------------------------------------------------
    # Computation time statistics
    # -------------------------------------------------------------------------

    analytical_solve_times = [
        case["analytical_solve_time"]
        for case in successful_cases
        if case.get("analytical_solve_time") is not None
    ]

    ocp_solve_times = [
        case["ocp_solve_time"]
        for case in successful_cases
        if case.get("ocp_solve_time") is not None
    ]

    if len(analytical_solve_times) > 0 and len(ocp_solve_times) > 0:

        analytical_solve_times = np.array(analytical_solve_times)
        ocp_solve_times = np.array(ocp_solve_times)

        print("\n" + "=" * 80)
        print("COMPUTATION TIME STATISTICS")
        print("=" * 80)

        print("\nAnalytical planner")
        print("-" * 80)
        print(f"Mean solve time   : {np.mean(analytical_solve_times):.6e} s")
        print(f"Median solve time : {np.median(analytical_solve_times):.6e} s")
        print(f"Maximum solve time: {np.max(analytical_solve_times):.6e} s")

        print("\nOptimal control solver")
        print("-" * 80)
        print(f"Mean solve time   : {np.mean(ocp_solve_times):.6e} s")
        print(f"Median solve time : {np.median(ocp_solve_times):.6e} s")
        print(f"Maximum solve time: {np.max(ocp_solve_times):.6e} s")

        speedup_mean = np.mean(ocp_solve_times) / np.mean(analytical_solve_times)
        speedup_median = np.median(ocp_solve_times) / np.median(analytical_solve_times)

        print("\nRelative speedup")
        print("-" * 80)
        print(f"Mean OCP / analytical   : {speedup_mean:.3e}")
        print(f"Median OCP / analytical : {speedup_median:.3e}")