import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches


plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# =============================================================================
# Helper functions
# =============================================================================

def get_best_name(case):
    """
    Return the name of the best analytical trajectory.
    """
    return (
        case.get("best_analytical_name")
        or case.get("best_name")
    )


def short_label(best_name):
    """
    Example:
        'TCSCT left-right' -> 'TCSCT LR'
    """
    family, directions = best_name.split(" ", 1)
    d0, df = directions.split("-")

    direction_code = {
        "left": "L",
        "right": "R",
    }

    return f"{family} {direction_code[d0]}{direction_code[df]}"


def get_family(best_name):
    """
    Example:
        'TCSCT left-right' -> 'TCSCT'
    """
    return best_name.split(" ", 1)[0]


def get_direction_pair(best_name):
    """
    Example:
        'TCSCT left-right' -> (+1, -1)

    Convention:
        +1 = left / counterclockwise
        -1 = right / clockwise
    """
    _, directions = best_name.split(" ", 1)
    d0, df = directions.split("-")

    direction_to_tau = {
        "left": +1,
        "right": -1,
    }

    return direction_to_tau[d0], direction_to_tau[df]


def directed_angular_displacement(theta_from, theta_to, tau):
    """
    Magnitude of the directed angular displacement from theta_from
    to theta_to in the prescribed direction tau.

    tau = +1:
        counterclockwise / left

    tau = -1:
        clockwise / right

    Returns
    -------
    float
        Directed angular displacement in [0, 2*pi).
    """
    if tau == +1:
        return (theta_to - theta_from) % (2.0 * np.pi)

    if tau == -1:
        return (theta_from - theta_to) % (2.0 * np.pi)

    raise ValueError(f"Invalid turn direction tau={tau}")


def compute_trajectory_quantities(case):
    """
    Compute the main geometric quantities of the stored best trajectory.

    Gamma
    -----
    Sum of the absolute angular amplitudes of all rotational primitives.

        Gamma = |phi_0| + |iota_0| + |iota_f| + |phi_f|

    d
    -
    Length of the straight primitive.

    The current CSC/TCSC/CSCT/TCSCT families contain exactly one straight
    segment. To remain slightly more general, the code sums the path lengths
    of all primitives that do not contain an angular-amplitude field.
    """
    trajectory = case["best_trajectory"]

    gamma = 0.0
    d = 0.0

    angular_components = []
    straight_components = []

    for primitive in trajectory:

        if "angular_amplitude_rad" in primitive:
            angle = float(primitive["angular_amplitude_rad"])

            gamma += abs(angle)

            angular_components.append({
                "label": primitive["label"],
                "angle_rad": angle,
                "angle_deg": np.degrees(angle),
            })

        else:
            length = float(primitive["path_length"])
            d += length

            straight_components.append({
                "label": primitive["label"],
                "length": length,
            })

    return d, gamma, angular_components, straight_components


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Load results
    # -------------------------------------------------------------------------

    RESULTS_FILENAME = "sweep1/orientation_sweep_analytical.json"

    current_dir = Path(__file__).resolve().parent

    results_path = current_dir / "results" / RESULTS_FILENAME

    figures_dir = current_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    with open(results_path, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    results = data["results"]

    successful_cases = [
        case
        for case in results
        if (
            case.get("success", False)
            and get_best_name(case) is not None
            and case.get("best_trajectory") is not None
        )
    ]

    if not successful_cases:
        raise RuntimeError("No successful analytical cases found.")


    # =========================================================================
    # Angular-displacement analysis
    # =========================================================================

    angular_results = []

    for case in successful_cases:

        best_name = get_best_name(case)
        family = get_family(best_name)

        tau0, tauf = get_direction_pair(best_name)

        (
            d,
            gamma,
            angular_components,
            straight_components,
        ) = compute_trajectory_quantities(case)

        result = {
            "case_id": case["case_id"],

            "theta0_rad": case["theta0_rad"],
            "thetaf_rad": case["thetaf_rad"],

            "theta0_deg": case["theta0_deg"],
            "thetaf_deg": case["thetaf_deg"],

            "best_name": best_name,
            "family": family,

            "tau0": tau0,
            "tauf": tauf,

            "d": d,

            "gamma": gamma,
            "gamma_deg": np.degrees(gamma),

            "angular_components": angular_components,
            "straight_components": straight_components,
        }

        # ---------------------------------------------------------------------
        # Equal-turn case:
        #
        # Gamma = Gamma_min^tau + 2 k pi
        #
        # This relation is checked only for LL and RR candidates.
        # ---------------------------------------------------------------------

        if tau0 == tauf:

            tau = tau0

            theta0 = case["theta0_rad"]
            thetaf = case["thetaf_rad"]

            gamma_min = directed_angular_displacement(
                theta0,
                thetaf,
                tau,
            )

            gamma_excess = gamma - gamma_min

            k_float = gamma_excess / (2.0 * np.pi)
            k_nearest = int(np.rint(k_float))

            residual = (
                gamma
                - gamma_min
                - 2.0 * np.pi * k_nearest
            )

            result.update({
                "gamma_min": gamma_min,
                "gamma_min_deg": np.degrees(gamma_min),

                "gamma_excess": gamma_excess,
                "gamma_excess_deg": np.degrees(gamma_excess),

                "k_float": k_float,
                "k": k_nearest,

                "residual": residual,
                "residual_deg": np.degrees(residual),
            })

        angular_results.append(result)


    # =========================================================================
    # Sequence-map data
    # =========================================================================

    theta0_values = sorted(
        set(case["theta0_deg"] for case in successful_cases)
    )

    thetaf_values = sorted(
        set(case["thetaf_deg"] for case in successful_cases)
    )

    n_theta0 = len(theta0_values)
    n_thetaf = len(thetaf_values)

    theta0_to_idx = {
        theta: i
        for i, theta in enumerate(theta0_values)
    }

    thetaf_to_idx = {
        theta: i
        for i, theta in enumerate(thetaf_values)
    }

    labels = sorted(
        set(
            short_label(get_best_name(case))
            for case in successful_cases
        )
    )

    label_to_id = {
        label: i
        for i, label in enumerate(labels)
    }

    sequence_grid = np.full(
        (n_thetaf, n_theta0),
        np.nan,
    )

    label_counts = {
        label: 0
        for label in labels
    }

    n_valid_cases = 0

    for case in successful_cases:

        theta0 = case["theta0_deg"]
        thetaf = case["thetaf_deg"]

        i = thetaf_to_idx[thetaf]
        j = theta0_to_idx[theta0]

        best_name = get_best_name(case)
        label = short_label(best_name)

        sequence_grid[i, j] = label_to_id[label]

        label_counts[label] += 1
        n_valid_cases += 1


    # =========================================================================
    # Sequence-map figure
    # =========================================================================

    n_labels = len(labels)

    sequence_colors = {

        # CSC: blue shades
        "CSC LL": "#08306B",
        "CSC LR": "#2171B5",
        "CSC RL": "#6BAED6",
        "CSC RR": "#C6DBEF",

        # TCSC: orange shades
        "TCSC LL": "#7F2704",
        "TCSC LR": "#D94801",
        "TCSC RL": "#FD8D3C",
        "TCSC RR": "#FDD0A2",

        # CSCT: green shades
        "CSCT LL": "#00441B",
        "CSCT LR": "#238B45",
        "CSCT RL": "#74C476",
        "CSCT RR": "#C7E9C0",

        # TCSCT: purple shades
        "TCSCT LL": "#4A1486",
        "TCSCT LR": "#807DBA",
        "TCSCT RL": "#BCBDDC",
        "TCSCT RR": "#EFEDF5",
    }

    missing_colors = [
        label
        for label in labels
        if label not in sequence_colors
    ]

    if missing_colors:
        raise ValueError(
            "Missing colors for labels: "
            + ", ".join(missing_colors)
        )

    colors = [
        sequence_colors[label]
        for label in labels
    ]

    listed_cmap = ListedColormap(colors)

    norm = BoundaryNorm(
        boundaries=np.arange(
            -0.5,
            n_labels + 0.5,
            1,
        ),
        ncolors=n_labels,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.imshow(
        sequence_grid,
        origin="lower",
        cmap=listed_cmap,
        norm=norm,
        extent=[
            min(theta0_values),
            max(theta0_values) + 360 / n_theta0,
            min(thetaf_values),
            max(thetaf_values) + 360 / n_thetaf,
        ],
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xlabel(
        r"Initial orientation $\theta_0$ [deg]"
    )

    ax.set_ylabel(
        r"Final orientation $\theta_f$ [deg]"
    )

    ax.set_title(
        r"Optimal analytical sequence map"
    )

    tick_values = np.arange(0, 361, 45)

    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)

    ax.grid(False)

    legend_patches = [
        mpatches.Patch(
            color=colors[label_to_id[label]],
            label=label,
        )
        for label in labels
    ]

    ax.legend(
        handles=legend_patches,
        title="Sequence",
        bbox_to_anchor=(1.04, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
        ncol=1,
    )

    fig.tight_layout()

    pdf_path = (
        figures_dir
        / "sweep1_sequence_map.pdf"
    )

    png_path = (
        figures_dir
        / "sweep1_sequence_map.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )


    # =========================================================================
    # Sequence statistics
    # =========================================================================

    print("\nSequence labels")
    print("-" * 60)

    for label, idx in label_to_id.items():
        print(f"{idx:02d}: {label}")

    print("\nSequence occurrence statistics")
    print("-" * 60)

    print(
        f"Total valid cases: "
        f"{n_valid_cases}"
    )

    for label in labels:

        count = label_counts[label]

        percentage = (
            100.0
            * count
            / n_valid_cases
        )

        print(
            f"{label:9s}: "
            f"{count:7d} cases "
            f"({percentage:6.2f} %)"
        )

    print("\nLaTeX table rows")
    print("-" * 60)

    for label in labels:

        count = label_counts[label]

        percentage = (
            100.0
            * count
            / n_valid_cases
        )

        print(
            f"{label} & "
            f"{count} & "
            f"{percentage:.2f} \\\\"
        )


    # =========================================================================
    # Gamma statistics: all winning trajectories
    # =========================================================================

    print("\n" + "=" * 80)
    print("ANGULAR-DISPLACEMENT ANALYSIS")
    print("=" * 80)

    gamma_values = np.array([
        r["gamma"]
        for r in angular_results
    ])

    print("\nGamma statistics over all winning trajectories")
    print("-" * 80)

    print(
        f"Number of winning trajectories : "
        f"{len(gamma_values)}"
    )

    print(
        f"Minimum Gamma                  : "
        f"{np.min(gamma_values):.8f} rad "
        f"({np.degrees(np.min(gamma_values)):.3f} deg)"
    )

    print(
        f"Maximum Gamma                  : "
        f"{np.max(gamma_values):.8f} rad "
        f"({np.degrees(np.max(gamma_values)):.3f} deg)"
    )

    print(
        f"Mean Gamma                     : "
        f"{np.mean(gamma_values):.8f} rad "
        f"({np.degrees(np.mean(gamma_values)):.3f} deg)"
    )


    # =========================================================================
    # Equal-turn vs opposite-turn winners
    # =========================================================================

    equal_turn_results = [
        r
        for r in angular_results
        if r["tau0"] == r["tauf"]
    ]

    opposite_turn_results = [
        r
        for r in angular_results
        if r["tau0"] != r["tauf"]
    ]

    print("\nWinning turn-direction classes")
    print("-" * 80)

    print(
        f"Total successful cases : "
        f"{len(angular_results)}"
    )

    print(
        f"Equal-turn winners     : "
        f"{len(equal_turn_results)} "
        f"("
        f"{100.0 * len(equal_turn_results) / len(angular_results):.2f} %"
        f")"
    )

    print(
        f"Opposite-turn winners  : "
        f"{len(opposite_turn_results)} "
        f"("
        f"{100.0 * len(opposite_turn_results) / len(angular_results):.2f} %"
        f")"
    )


    # =========================================================================
    # Verify Gamma = Gamma_min + 2 k pi for equal-turn winners
    # =========================================================================

    if equal_turn_results:

        residuals = np.array([
            abs(r["residual"])
            for r in equal_turn_results
        ])

        k_float_values = np.array([
            r["k_float"]
            for r in equal_turn_results
        ])

        k_values = np.array([
            r["k"]
            for r in equal_turn_results
        ])

        print("\nEqual-turn consistency check")
        print("-" * 80)

        print(
            "Testing:"
        )

        print(
            "    Gamma = Gamma_min + 2 k pi"
        )

        print(
            f"Maximum residual : "
            f"{np.max(residuals):.3e} rad"
        )

        print(
            f"Mean residual    : "
            f"{np.mean(residuals):.3e} rad"
        )

        print(
            f"Maximum distance of k_float "
            f"from nearest integer: "
            f"{np.max(np.abs(k_float_values - k_values)):.3e}"
        )


        # =====================================================================
        # Distribution of k
        # =====================================================================

        print("\nk distribution for equal-turn winners")
        print("-" * 80)

        unique_k, counts = np.unique(
            k_values,
            return_counts=True,
        )

        for k, count in zip(
            unique_k,
            counts,
        ):

            percentage = (
                100.0
                * count
                / len(equal_turn_results)
            )

            print(
                f"k = {k:2d}: "
                f"{count:7d} cases "
                f"({percentage:6.2f} %)"
            )


        # =====================================================================
        # Gamma-minimum statistics
        # =====================================================================

        n_k0 = np.sum(k_values == 0)

        print("\nMinimum-angular-displacement winners")
        print("-" * 80)

        print(
            f"Gamma = Gamma_min : "
            f"{n_k0} / "
            f"{len(equal_turn_results)} "
            f"("
            f"{100.0 * n_k0 / len(equal_turn_results):.2f} %"
            f")"
        )

        print(
            f"Gamma > Gamma_min : "
            f"{len(equal_turn_results) - n_k0} / "
            f"{len(equal_turn_results)} "
            f"("
            f"{100.0 * (len(equal_turn_results) - n_k0) / len(equal_turn_results):.2f} %"
            f")"
        )


        # =====================================================================
        # Breakdown by family
        # =====================================================================

        print("\nEqual-turn winners by family")
        print("-" * 80)

        families = [
            "CSC",
            "TCSC",
            "CSCT",
            "TCSCT",
        ]

        for family in families:

            family_results = [
                r
                for r in equal_turn_results
                if r["family"] == family
            ]

            if not family_results:
                continue

            family_k = np.array([
                r["k"]
                for r in family_results
            ])

            n_family = len(family_results)
            n_family_k0 = np.sum(family_k == 0)

            print(
                f"\n{family}"
            )

            print(
                f"  Number of winners : "
                f"{n_family}"
            )

            print(
                f"  k = 0             : "
                f"{n_family_k0} "
                f"("
                f"{100.0 * n_family_k0 / n_family:.2f} %"
                f")"
            )

            unique_family_k, family_counts = np.unique(
                family_k,
                return_counts=True,
            )

            distribution = ", ".join(
                f"k={k}: {count}"
                for k, count in zip(
                    unique_family_k,
                    family_counts,
                )
            )

            print(
                f"  k distribution    : "
                f"{distribution}"
            )


        # =====================================================================
        # Breakdown by LL and RR
        # =====================================================================

        print("\nEqual-turn winners by direction")
        print("-" * 80)

        for tau, direction_name in [
            (+1, "LL"),
            (-1, "RR"),
        ]:

            direction_results = [
                r
                for r in equal_turn_results
                if r["tau0"] == tau
            ]

            if not direction_results:
                continue

            direction_k = np.array([
                r["k"]
                for r in direction_results
            ])

            n_direction = len(direction_results)
            n_direction_k0 = np.sum(direction_k == 0)

            print(
                f"{direction_name}: "
                f"{n_direction:7d} cases, "
                f"k=0: {n_direction_k0:7d} "
                f"("
                f"{100.0 * n_direction_k0 / n_direction:6.2f} %"
                f")"
            )


        # =====================================================================
        # Inspect all non-minimum equal-turn winners
        # =====================================================================

        nonminimum_cases = [
            r
            for r in equal_turn_results
            if r["k"] != 0
        ]

        print("\nEqual-turn winners with Gamma > Gamma_min")
        print("-" * 80)

        print(
            f"Number of cases: "
            f"{len(nonminimum_cases)}"
        )

        # Print only a limited number so that the terminal is not flooded.
        MAX_CASES_TO_PRINT = 30

        for r in nonminimum_cases[:MAX_CASES_TO_PRINT]:

            direction = (
                "LL"
                if r["tau0"] == +1
                else "RR"
            )

            print(
                f"case {r['case_id']:6d} | "
                f"theta0={r['theta0_deg']:7.2f} deg | "
                f"thetaf={r['thetaf_deg']:7.2f} deg | "
                f"{r['family']:5s} {direction} | "
                f"Gamma={r['gamma_deg']:8.2f} deg | "
                f"Gamma_min={r['gamma_min_deg']:8.2f} deg | "
                f"k={r['k']:2d} | "
                f"res={r['residual']:.2e}"
            )

        if len(nonminimum_cases) > MAX_CASES_TO_PRINT:

            print(
                f"... "
                f"{len(nonminimum_cases) - MAX_CASES_TO_PRINT} "
                f"additional cases not printed."
            )


        # =====================================================================
        # Largest numerical residuals
        # =====================================================================

        print("\nLargest consistency residuals")
        print("-" * 80)

        worst_residual_cases = sorted(
            equal_turn_results,
            key=lambda r: abs(r["residual"]),
            reverse=True,
        )

        for r in worst_residual_cases[:10]:

            print(
                f"case {r['case_id']:6d} | "
                f"{r['best_name']:18s} | "
                f"theta0={r['theta0_deg']:7.2f} deg | "
                f"thetaf={r['thetaf_deg']:7.2f} deg | "
                f"k_float={r['k_float']: .12f} | "
                f"k={r['k']:2d} | "
                f"residual={r['residual']:+.3e} rad"
            )


    # =========================================================================
    # Summary by complete optimal sequence
    # =========================================================================

    print("\nGamma statistics by winning sequence")
    print("-" * 80)

    for label in labels:

        label_results = [
            r
            for r in angular_results
            if short_label(r["best_name"]) == label
        ]

        if not label_results:
            continue

        values = np.array([
            r["gamma"]
            for r in label_results
        ])

        print(
            f"{label:9s}: "
            f"N={len(values):7d}, "
            f"Gamma_min={np.degrees(np.min(values)):8.2f} deg, "
            f"Gamma_mean={np.degrees(np.mean(values)):8.2f} deg, "
            f"Gamma_max={np.degrees(np.max(values)):8.2f} deg"
        )


    # =========================================================================
    # Saved figures
    # =========================================================================

    print("\nSaved figures:")
    print(pdf_path)
    print(png_path)

    plt.show()