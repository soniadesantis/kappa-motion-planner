import json
from pathlib import Path

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

RESULTS_FILENAME = "orientation_sweep_all_candidates.json"

TOL_TIME = 1e-10
TOL_LENGTH = 1e-10
TOL_ANGLE = 1e-9

MAX_EXAMPLES_PER_CATEGORY = 8


# =============================================================================
# Helper functions
# =============================================================================

def classify_delta_gamma(delta_gamma, tol=TOL_ANGLE):
    """
    Classify Delta Gamma relative to integer multiples of 2*pi.

    Returns
    -------
    dict with:
        multiple_float
        multiple_int
        residual
        label
    """

    two_pi = 2.0 * np.pi

    multiple_float = delta_gamma / two_pi
    multiple_int = int(np.rint(multiple_float))

    residual = delta_gamma - multiple_int * two_pi

    if abs(residual) <= tol:
        if multiple_int == 0:
            label = "0"
        elif multiple_int > 0:
            label = f"+{multiple_int}*2pi"
        else:
            label = f"{multiple_int}*2pi"
    else:
        label = "non_multiple"

    return {
        "multiple_float": multiple_float,
        "multiple_int": multiple_int,
        "residual": residual,
        "label": label,
    }


def direction_code(candidate_name):
    """
    Example:
        'TCSC left-right' -> 'LR'
    """

    _, directions = candidate_name.split(" ", 1)
    d0, df = directions.split("-")

    code = {
        "left": "L",
        "right": "R",
    }

    return code[d0] + code[df]


def candidate_name(family, direction_pair):
    """
    Build the dictionary key used in all_candidates.

    Example:
        family='TCSC'
        direction_pair='LR'

        -> 'TCSC left-right'
    """

    decode = {
        "L": "left",
        "R": "right",
    }

    return (
        f"{family} "
        f"{decode[direction_pair[0]]}-"
        f"{decode[direction_pair[1]]}"
    )


def get_candidate(case, family, direction_pair):
    """
    Return one candidate from case['all_candidates'].
    """

    name = candidate_name(
        family,
        direction_pair,
    )

    return case["all_candidates"].get(name)


def make_comparison_record(
    case,
    family_a,
    family_b,
    direction_pair,
):
    """
    Compare candidate B against candidate A.

        Delta d     = d_B - d_A
        Delta Gamma = Gamma_B - Gamma_A
        Delta T     = T_B - T_A

    Negative Delta T means candidate B is faster.
    """

    candidate_a = get_candidate(
        case,
        family_a,
        direction_pair,
    )

    candidate_b = get_candidate(
        case,
        family_b,
        direction_pair,
    )

    if candidate_a is None or candidate_b is None:
        return None

    d_a = float(candidate_a["d"])
    d_b = float(candidate_b["d"])

    gamma_a = float(candidate_a["Gamma_rad"])
    gamma_b = float(candidate_b["Gamma_rad"])

    time_a = float(candidate_a["time"])
    time_b = float(candidate_b["time"])

    delta_d = d_b - d_a
    delta_gamma = gamma_b - gamma_a
    delta_time = time_b - time_a

    gamma_class = classify_delta_gamma(
        delta_gamma
    )

    return {
        "case_id": case["case_id"],

        "theta0_deg": case["theta0_deg"],
        "thetaf_deg": case["thetaf_deg"],

        "family_a": family_a,
        "family_b": family_b,

        "direction_pair": direction_pair,

        "candidate_a_name": candidate_name(
            family_a,
            direction_pair,
        ),

        "candidate_b_name": candidate_name(
            family_b,
            direction_pair,
        ),

        "d_a": d_a,
        "d_b": d_b,
        "delta_d": delta_d,

        "gamma_a": gamma_a,
        "gamma_b": gamma_b,
        "delta_gamma": delta_gamma,

        "gamma_a_deg": np.degrees(gamma_a),
        "gamma_b_deg": np.degrees(gamma_b),
        "delta_gamma_deg": np.degrees(delta_gamma),

        "time_a": time_a,
        "time_b": time_b,
        "delta_time": delta_time,

        "gamma_multiple_float": gamma_class[
            "multiple_float"
        ],

        "gamma_multiple_int": gamma_class[
            "multiple_int"
        ],

        "gamma_multiple_residual": gamma_class[
            "residual"
        ],

        "gamma_class": gamma_class[
            "label"
        ],

        "b_faster": (
            delta_time < -TOL_TIME
        ),

        "same_time": (
            abs(delta_time) <= TOL_TIME
        ),

        "b_shorter": (
            delta_d < -TOL_LENGTH
        ),

        "same_d": (
            abs(delta_d) <= TOL_LENGTH
        ),

        "gamma_reduced": (
            delta_gamma < -TOL_ANGLE
        ),

        "same_gamma": (
            abs(delta_gamma) <= TOL_ANGLE
        ),

        "gamma_increased": (
            delta_gamma > TOL_ANGLE
        ),

        "global_winner": case[
            "best_analytical_name"
        ],
    }


def print_examples(
    records,
    title,
    max_examples=MAX_EXAMPLES_PER_CATEGORY,
):
    """
    Print representative examples.
    """

    print("\n" + title)
    print("-" * 100)

    if not records:
        print("None")
        return

    for record in records[:max_examples]:

        print(
            f"case {record['case_id']:6d} | "
            f"theta0={record['theta0_deg']:7.2f} deg | "
            f"thetaf={record['thetaf_deg']:7.2f} deg | "
            f"{record['family_a']} -> "
            f"{record['family_b']} "
            f"{record['direction_pair']} | "
            f"Delta d={record['delta_d']:+.6f} | "
            f"Delta Gamma="
            f"{record['delta_gamma_deg']:+8.3f} deg | "
            f"Delta T={record['delta_time']:+.6f} | "
            f"class={record['gamma_class']}"
        )

    if len(records) > max_examples:
        print(
            f"... {len(records) - max_examples} "
            f"additional cases not printed."
        )


def print_comparison_summary(
    records,
    family_a,
    family_b,
    direction_pair,
):
    """
    Print statistics for one comparison.
    """

    print("\n" + "=" * 100)

    print(
        f"{family_a} -> {family_b}, "
        f"{direction_pair}"
    )

    print("=" * 100)

    n_total = len(records)

    if n_total == 0:
        print("No valid comparisons.")
        return

    faster = [
        r
        for r in records
        if r["b_faster"]
    ]

    slower = [
        r
        for r in records
        if r["delta_time"] > TOL_TIME
    ]

    equal_time = [
        r
        for r in records
        if r["same_time"]
    ]

    print(
        f"Total comparisons       : "
        f"{n_total}"
    )

    print(
        f"{family_b} faster          : "
        f"{len(faster):7d} "
        f"({100.0 * len(faster) / n_total:6.2f} %)"
    )

    print(
        f"{family_b} slower          : "
        f"{len(slower):7d} "
        f"({100.0 * len(slower) / n_total:6.2f} %)"
    )

    print(
        f"Equal traversal time    : "
        f"{len(equal_time):7d} "
        f"({100.0 * len(equal_time) / n_total:6.2f} %)"
    )

    # -------------------------------------------------------------------------
    # Delta Gamma distribution over all comparisons
    # -------------------------------------------------------------------------

    print("\nDelta Gamma classification: all comparisons")
    print("-" * 100)

    gamma_classes = {}

    for record in records:

        label = record["gamma_class"]

        gamma_classes[label] = (
            gamma_classes.get(label, 0)
            + 1
        )

    for label, count in sorted(
        gamma_classes.items(),
        key=lambda item: item[0],
    ):

        print(
            f"{label:15s}: "
            f"{count:7d} "
            f"({100.0 * count / n_total:6.2f} %)"
        )

    # -------------------------------------------------------------------------
    # Mechanisms only where B wins
    # -------------------------------------------------------------------------

    print(
        f"\nMechanism classification where "
        f"{family_b} is faster"
    )

    print("-" * 100)

    if not faster:
        print("No faster cases.")
        return

    n_faster = len(faster)

    same_gamma_shorter = [
        r
        for r in faster
        if (
            r["same_gamma"]
            and r["b_shorter"]
        )
    ]

    gamma_reduced = [
        r
        for r in faster
        if r["gamma_reduced"]
    ]

    gamma_reduced_and_shorter = [
        r
        for r in faster
        if (
            r["gamma_reduced"]
            and r["b_shorter"]
        )
    ]

    gamma_reduced_but_longer = [
        r
        for r in faster
        if (
            r["gamma_reduced"]
            and r["delta_d"] > TOL_LENGTH
        )
    ]

    gamma_increased_but_faster = [
        r
        for r in faster
        if r["gamma_increased"]
    ]

    gamma_increased_and_shorter = [
        r
        for r in faster
        if (
            r["gamma_increased"]
            and r["b_shorter"]
        )
    ]

    both_reduced = [
        r
        for r in faster
        if (
            r["gamma_reduced"]
            and r["b_shorter"]
        )
    ]

    print(
        f"Delta Gamma = 0 and Delta d < 0 : "
        f"{len(same_gamma_shorter):7d} "
        f"({100.0 * len(same_gamma_shorter) / n_faster:6.2f} %)"
    )

    print(
        f"Delta Gamma < 0                 : "
        f"{len(gamma_reduced):7d} "
        f"({100.0 * len(gamma_reduced) / n_faster:6.2f} %)"
    )

    print(
        f"Delta Gamma < 0 and Delta d < 0: "
        f"{len(gamma_reduced_and_shorter):7d} "
        f"({100.0 * len(gamma_reduced_and_shorter) / n_faster:6.2f} %)"
    )

    print(
        f"Delta Gamma < 0 and Delta d > 0: "
        f"{len(gamma_reduced_but_longer):7d} "
        f"({100.0 * len(gamma_reduced_but_longer) / n_faster:6.2f} %)"
    )

    print(
        f"Delta Gamma > 0 but Delta T < 0: "
        f"{len(gamma_increased_but_faster):7d} "
        f"({100.0 * len(gamma_increased_but_faster) / n_faster:6.2f} %)"
    )

    print(
        f"Delta Gamma > 0 and Delta d < 0: "
        f"{len(gamma_increased_and_shorter):7d} "
        f"({100.0 * len(gamma_increased_and_shorter) / n_faster:6.2f} %)"
    )

    print(
        f"Both d and Gamma reduced        : "
        f"{len(both_reduced):7d} "
        f"({100.0 * len(both_reduced) / n_faster:6.2f} %)"
    )

    # -------------------------------------------------------------------------
    # Delta Gamma distribution where B wins
    # -------------------------------------------------------------------------

    print(
        f"\nDelta Gamma classification where "
        f"{family_b} is faster"
    )

    print("-" * 100)

    faster_gamma_classes = {}

    for record in faster:

        label = record["gamma_class"]

        faster_gamma_classes[label] = (
            faster_gamma_classes.get(
                label,
                0,
            )
            + 1
        )

    for label, count in sorted(
        faster_gamma_classes.items(),
        key=lambda item: item[0],
    ):

        print(
            f"{label:15s}: "
            f"{count:7d} "
            f"({100.0 * count / n_faster:6.2f} %)"
        )

    # -------------------------------------------------------------------------
    # Useful extrema
    # -------------------------------------------------------------------------

    delta_d_values = np.array([
        r["delta_d"]
        for r in faster
    ])

    delta_gamma_values = np.array([
        r["delta_gamma"]
        for r in faster
    ])

    delta_time_values = np.array([
        r["delta_time"]
        for r in faster
    ])

    print("\nStatistics over faster cases")
    print("-" * 100)

    print(
        f"Delta d:"
        f" min={np.min(delta_d_values):+.6f},"
        f" mean={np.mean(delta_d_values):+.6f},"
        f" max={np.max(delta_d_values):+.6f}"
    )

    print(
        f"Delta Gamma [deg]:"
        f" min={np.degrees(np.min(delta_gamma_values)):+.3f},"
        f" mean={np.degrees(np.mean(delta_gamma_values)):+.3f},"
        f" max={np.degrees(np.max(delta_gamma_values)):+.3f}"
    )

    print(
        f"Delta T:"
        f" min={np.min(delta_time_values):+.6f},"
        f" mean={np.mean(delta_time_values):+.6f},"
        f" max={np.max(delta_time_values):+.6f}"
    )

    # -------------------------------------------------------------------------
    # Representative examples
    # -------------------------------------------------------------------------

    print_examples(
        same_gamma_shorter,
        (
            f"Examples: {family_b} faster with "
            f"Delta Gamma = 0 and Delta d < 0"
        ),
    )

    print_examples(
        gamma_reduced_and_shorter,
        (
            f"Examples: {family_b} faster with "
            f"both Gamma and d reduced"
        ),
    )

    print_examples(
        gamma_reduced_but_longer,
        (
            f"Examples: {family_b} faster with "
            f"Gamma reduced but d increased"
        ),
    )

    print_examples(
        gamma_increased_but_faster,
        (
            f"Examples: {family_b} faster despite "
            f"Gamma increasing"
        ),
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    current_dir = Path(__file__).resolve().parent

    results_path = (
        current_dir
        / "results"
        / RESULTS_FILENAME
    )

    print("\nLoading:")
    print(results_path)

    with open(
        results_path,
        "r",
    ) as f:

        data = json.load(f)

    metadata = data.get(
        "metadata",
        {},
    )

    results = data["results"]

    successful_cases = [
        case
        for case in results
        if (
            case.get("success", False)
            and case.get(
                "all_candidates"
            ) is not None
        )
    ]

    print("\n" + "=" * 100)
    print("ALL-CANDIDATE GEOMETRIC COMPARISON")
    print("=" * 100)

    print(
        f"Sweep name       : "
        f"{metadata.get('sweep_name', 'unknown')}"
    )

    print(
        f"Successful cases : "
        f"{len(successful_cases)}"
    )

    print(
        f"D/R              : "
        f"{metadata.get('D_over_R', 'not fixed')}"
    )


    # =========================================================================
    # Candidate comparisons
    # =========================================================================

    comparisons = [
        ("CSC", "TCSC"),
        ("CSC", "CSCT"),
        ("CSC", "TCSCT"),
    ]

    direction_pairs = [
        "LL",
        "LR",
        "RL",
        "RR",
    ]

    all_comparison_results = {}

    for family_a, family_b in comparisons:

        for direction_pair in direction_pairs:

            records = []

            for case in successful_cases:

                record = make_comparison_record(
                    case,
                    family_a,
                    family_b,
                    direction_pair,
                )

                if record is not None:
                    records.append(record)

            key = (
                family_a,
                family_b,
                direction_pair,
            )

            all_comparison_results[
                key
            ] = records

            print_comparison_summary(
                records,
                family_a,
                family_b,
                direction_pair,
            )


    # =========================================================================
    # Equal-turn aggregate analysis
    # =========================================================================

    print("\n" + "=" * 100)
    print("AGGREGATED EQUAL-TURN COMPARISONS")
    print("=" * 100)

    for family_a, family_b in comparisons:

        equal_turn_records = []

        for direction_pair in [
            "LL",
            "RR",
        ]:

            equal_turn_records.extend(
                all_comparison_results[
                    (
                        family_a,
                        family_b,
                        direction_pair,
                    )
                ]
            )

        print_comparison_summary(
            equal_turn_records,
            family_a,
            family_b,
            "LL + RR",
        )


    # =========================================================================
    # Globally winning candidate analysis
    # =========================================================================

    print("\n" + "=" * 100)
    print("GLOBAL WINNER MECHANISMS")
    print("=" * 100)

    for family_a, family_b in comparisons:

        for direction_pair in direction_pairs:

            records = all_comparison_results[
                (
                    family_a,
                    family_b,
                    direction_pair,
                )
            ]

            winning_b_records = [
                r
                for r in records
                if (
                    r["b_faster"]
                    and r["global_winner"]
                    == r["candidate_b_name"]
                )
            ]

            if not winning_b_records:
                continue

            n = len(winning_b_records)

            same_gamma_shorter = sum(
                1
                for r in winning_b_records
                if (
                    r["same_gamma"]
                    and r["b_shorter"]
                )
            )

            gamma_reduced = sum(
                1
                for r in winning_b_records
                if r["gamma_reduced"]
            )

            gamma_increased = sum(
                1
                for r in winning_b_records
                if r["gamma_increased"]
            )

            print(
                f"\n{family_a} -> "
                f"{family_b} "
                f"{direction_pair}"
            )

            print(
                f"  Global wins of {family_b}: "
                f"{n}"
            )

            print(
                f"  Same Gamma, shorter d : "
                f"{same_gamma_shorter} "
                f"("
                f"{100.0 * same_gamma_shorter / n:.2f} %"
                f")"
            )

            print(
                f"  Gamma reduced         : "
                f"{gamma_reduced} "
                f"("
                f"{100.0 * gamma_reduced / n:.2f} %"
                f")"
            )

            print(
                f"  Gamma increased       : "
                f"{gamma_increased} "
                f"("
                f"{100.0 * gamma_increased / n:.2f} %"
                f")"
            )


    # =========================================================================
    # Overall residual check for multiples of 2*pi
    # =========================================================================

    print("\n" + "=" * 100)
    print("DELTA-GAMMA MULTIPLE-OF-2PI CONSISTENCY")
    print("=" * 100)

    residuals = []

    for records in all_comparison_results.values():

        for record in records:

            if record[
                "gamma_class"
            ] != "non_multiple":

                residuals.append(
                    abs(
                        record[
                            "gamma_multiple_residual"
                        ]
                    )
                )

    if residuals:

        residuals = np.array(
            residuals
        )

        print(
            f"Maximum residual : "
            f"{np.max(residuals):.3e} rad"
        )

        print(
            f"Mean residual    : "
            f"{np.mean(residuals):.3e} rad"
        )

    print("\nAnalysis completed.")