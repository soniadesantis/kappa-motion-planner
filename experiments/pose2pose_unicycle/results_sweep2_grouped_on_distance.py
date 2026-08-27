import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


# =============================================================================
# MATPLOTLIB TYPOGRAPHY
# =============================================================================

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": [
        "Computer Modern Roman",
        "CMU Serif",
        "DejaVu Serif",
    ],
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.unicode_minus": True,
})


# =============================================================================
# USER CONFIGURATION
# =============================================================================

RESULTS_FILES = {
    30: Path(
        "experiments/pose2pose_unicycle/results/"
        "sweep2/distance_over_radius_sweep_OCP_path_saved_N30_M4.json"
    ),
    100: Path(
        "experiments/pose2pose_unicycle/results/"
        "sweep2/distance_over_radius_sweep_OCP_path_saved_N100_M4.json"
    ),
}

EXPECTED_D_OVER_R_VALUES = (
    10.0,
    15.0,
    20.0,
)

N_COMPARISON_POINTS = 1001

SAVE_FIGURES = True
SHOW_FIGURES = True

FIGURES_DIRECTORY = Path(
    "experiments/pose2pose_unicycle/figures"
)


GRID_COLOR = "0.75"
GRID_LINESTYLE = ":"
GRID_LINEWIDTH = 0.7
GRID_ALPHA = 0.7


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def compute_statistics(values):
    """Return the summary statistics used in the thesis tables."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.size == 0:
        return None

    return {
        "count": int(values.size),
        "mean": float(
            np.mean(values)
        ),
        "median": float(
            np.median(values)
        ),
        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),
        "maximum": float(
            np.max(values)
        ),
    }


def get_case_d_over_r(case):
    """
    Return D/R for one saved case.

    The saved value is used when available. Otherwise, D/R is reconstructed
    from the boundary positions and the saved turning radius.
    """

    if case.get("D_over_R") is not None:
        return float(
            case["D_over_R"]
        )

    x0 = float(
        case["x0"]
    )

    y0 = float(
        case["y0"]
    )

    xf = float(
        case["xf"]
    )

    yf = float(
        case["yf"]
    )

    distance = np.hypot(
        xf - x0,
        yf - y0,
    )

    if case.get("R") is not None:
        radius = float(
            case["R"]
        )

    else:
        v_max = float(
            case["v_max"]
        )

        omega_max = float(
            case["omega_max"]
        )

        radius = (
            v_max
            / omega_max
        )

    if radius <= 0.0:
        raise ValueError(
            f"Invalid radius for case {case.get('case_id')}."
        )

    return float(
        distance / radius
    )


def match_expected_distance(
    value,
    expected_values=EXPECTED_D_OVER_R_VALUES,
    tolerance=1.0e-8,
):
    """
    Match a numerical D/R value to one of the expected sweep values.
    """

    for expected_value in expected_values:
        if np.isclose(
            value,
            expected_value,
            atol=tolerance,
            rtol=0.0,
        ):
            return float(
                expected_value
            )

    raise ValueError(
        f"Unexpected D/R value: {value:.12g}. "
        f"Expected one of {expected_values}."
    )


# =============================================================================
# INTERPOLATION
# =============================================================================

def interpolate_pose(
    times,
    xs,
    ys,
    thetas,
    normalized_grid,
):
    """Interpolate a pose trajectory on a common normalized-time grid."""

    times = np.asarray(
        times,
        dtype=float,
    ).reshape(-1)

    xs = np.asarray(
        xs,
        dtype=float,
    ).reshape(-1)

    ys = np.asarray(
        ys,
        dtype=float,
    ).reshape(-1)

    thetas = np.asarray(
        thetas,
        dtype=float,
    ).reshape(-1)

    if not (
        len(times)
        == len(xs)
        == len(ys)
        == len(thetas)
    ):
        raise ValueError(
            "Time and state arrays must have equal lengths."
        )

    if len(times) < 2:
        raise ValueError(
            "At least two trajectory samples are required."
        )

    duration = (
        times[-1]
        - times[0]
    )

    if duration <= 0.0:
        raise ValueError(
            "Trajectory duration must be positive."
        )

    normalized_times = (
        times - times[0]
    ) / duration

    (
        normalized_times_unique,
        unique_indices,
    ) = np.unique(
        normalized_times,
        return_index=True,
    )

    xs = xs[
        unique_indices
    ]

    ys = ys[
        unique_indices
    ]

    thetas = thetas[
        unique_indices
    ]

    thetas_unwrapped = np.unwrap(
        thetas
    )

    x_interp = np.interp(
        normalized_grid,
        normalized_times_unique,
        xs,
    )

    y_interp = np.interp(
        normalized_grid,
        normalized_times_unique,
        ys,
    )

    theta_interp = np.interp(
        normalized_grid,
        normalized_times_unique,
        thetas_unwrapped,
    )

    return (
        x_interp,
        y_interp,
        theta_interp,
    )


# =============================================================================
# ANALYTICAL TRAJECTORY RECONSTRUCTION
# =============================================================================

def exact_unicycle_step(
    x,
    y,
    theta,
    v,
    omega,
    dt,
):
    """Exact state update for constant unicycle controls."""

    tolerance = 1.0e-12

    if abs(omega) < tolerance:
        x_new = (
            x
            + v
            * np.cos(theta)
            * dt
        )

        y_new = (
            y
            + v
            * np.sin(theta)
            * dt
        )

        theta_new = theta

    elif abs(v) < tolerance:
        x_new = x
        y_new = y

        theta_new = (
            theta
            + omega * dt
        )

    else:
        theta_new = (
            theta
            + omega * dt
        )

        x_new = (
            x
            + (v / omega)
            * (
                np.sin(theta_new)
                - np.sin(theta)
            )
        )

        y_new = (
            y
            - (v / omega)
            * (
                np.cos(theta_new)
                - np.cos(theta)
            )
        )

    return (
        x_new,
        y_new,
        theta_new,
    )


def primitive_controls(
    primitive,
    v_max,
    omega_max,
):
    """Return the constant controls associated with a saved primitive."""

    label = primitive[
        "label"
    ].strip().lower()

    if label == "segment":
        return (
            v_max,
            0.0,
        )

    if label == "arc":
        turn_direction = int(
            primitive[
                "turn_direction"
            ]
        )

        return (
            v_max,
            turn_direction
            * omega_max,
        )

    if label == "turn on-the-spot":
        turn_direction = int(
            primitive[
                "turn_direction"
            ]
        )

        return (
            0.0,
            turn_direction
            * omega_max,
        )

    raise ValueError(
        f"Unknown analytical primitive label: {label!r}"
    )


def reconstruct_analytical_trajectory(
    case,
    samples_per_primitive=300,
):
    """
    Reconstruct the analytical pose trajectory from the saved primitives.
    """

    primitives = case[
        "best_analytical_primitives"
    ]

    if not primitives:
        raise ValueError(
            f"No analytical primitives saved for case "
            f"{case.get('case_id')}."
        )

    v_max = float(
        case["v_max"]
    )

    omega_max = float(
        case["omega_max"]
    )

    initial_pose = primitives[
        0
    ][
        "start_pose"
    ]

    x = float(
        initial_pose[0]
    )

    y = float(
        initial_pose[1]
    )

    theta = float(
        initial_pose[2]
    )

    times = [
        0.0
    ]

    xs = [
        x
    ]

    ys = [
        y
    ]

    thetas = [
        theta
    ]

    current_time = 0.0

    for primitive in primitives:
        duration = float(
            primitive[
                "maneuver_time"
            ]
        )

        if duration <= 1.0e-12:
            continue

        v, omega = primitive_controls(
            primitive,
            v_max,
            omega_max,
        )

        n_steps = max(
            2,
            int(
                samples_per_primitive
            ),
        )

        dt = (
            duration
            / n_steps
        )

        for _ in range(
            n_steps
        ):
            (
                x,
                y,
                theta,
            ) = exact_unicycle_step(
                x,
                y,
                theta,
                v,
                omega,
                dt,
            )

            current_time += dt

            times.append(
                current_time
            )

            xs.append(
                x
            )

            ys.append(
                y
            )

            thetas.append(
                theta
            )

    return (
        np.asarray(
            times,
            dtype=float,
        ),
        np.asarray(
            xs,
            dtype=float,
        ),
        np.asarray(
            ys,
            dtype=float,
        ),
        np.asarray(
            thetas,
            dtype=float,
        ),
    )


# =============================================================================
# HAUSDORFF DISTANCE
# =============================================================================

def discrete_symmetric_hausdorff_distance(
    analytical_x,
    analytical_y,
    ocp_x,
    ocp_y,
):
    """
    Compute the discrete symmetric Hausdorff distance between sampled paths.
    """

    analytical_points = np.column_stack(
        (
            analytical_x,
            analytical_y,
        )
    )

    ocp_points = np.column_stack(
        (
            ocp_x,
            ocp_y,
        )
    )

    analytical_tree = cKDTree(
        analytical_points
    )

    ocp_tree = cKDTree(
        ocp_points
    )

    analytical_to_ocp, _ = ocp_tree.query(
        analytical_points,
        k=1,
    )

    ocp_to_analytical, _ = analytical_tree.query(
        ocp_points,
        k=1,
    )

    return max(
        float(
            np.max(
                analytical_to_ocp
            )
        ),
        float(
            np.max(
                ocp_to_analytical
            )
        ),
    )


# =============================================================================
# CASE PROCESSING
# =============================================================================

def compare_case(
    case,
    n_comparison_points,
):
    """
    Compute the temporal and geometric discrepancies for one saved case.
    """

    analytical_time = float(
        case[
            "best_analytical_time"
        ]
    )

    ocp_time = float(
        case[
            "ocp_time"
        ]
    )

    absolute_time_error = abs(
        ocp_time
        - analytical_time
    )

    if analytical_time <= 0.0:
        raise ValueError(
            f"Nonpositive analytical time in case "
            f"{case.get('case_id')}."
        )

    relative_time_error_percent = (
        100.0
        * absolute_time_error
        / analytical_time
    )

    (
        analytical_times,
        analytical_x,
        analytical_y,
        analytical_theta,
    ) = reconstruct_analytical_trajectory(
        case
    )

    ocp_times = np.asarray(
        case[
            "ocp_time_grid"
        ],
        dtype=float,
    )

    ocp_x = np.asarray(
        case[
            "ocp_x"
        ],
        dtype=float,
    )

    ocp_y = np.asarray(
        case[
            "ocp_y"
        ],
        dtype=float,
    )

    ocp_theta = np.asarray(
        case[
            "ocp_theta"
        ],
        dtype=float,
    )

    normalized_grid = np.linspace(
        0.0,
        1.0,
        n_comparison_points,
    )

    (
        analytical_x_interp,
        analytical_y_interp,
        analytical_theta_interp,
    ) = interpolate_pose(
        analytical_times,
        analytical_x,
        analytical_y,
        analytical_theta,
        normalized_grid,
    )

    (
        ocp_x_interp,
        ocp_y_interp,
        ocp_theta_interp,
    ) = interpolate_pose(
        ocp_times,
        ocp_x,
        ocp_y,
        ocp_theta,
        normalized_grid,
    )

    hausdorff_distance = (
        discrete_symmetric_hausdorff_distance(
            analytical_x_interp,
            analytical_y_interp,
            ocp_x_interp,
            ocp_y_interp,
        )
    )

    d_over_r = match_expected_distance(
        get_case_d_over_r(
            case
        )
    )

    return {
        "case_id": int(
            case[
                "case_id"
            ]
        ),
        "d_over_r": d_over_r,
        "theta0_deg": float(
            case[
                "theta0_deg"
            ]
        ),
        "thetaf_deg": float(
            case[
                "thetaf_deg"
            ]
        ),
        "analytical_name": case.get(
            "best_analytical_name"
        ),
        "absolute_time_error": float(
            absolute_time_error
        ),
        "relative_time_error_percent": float(
            relative_time_error_percent
        ),
        "hausdorff_distance": float(
            hausdorff_distance
        ),
    }


def load_and_process_results(
    results_file,
    n_comparison_points,
):
    """Load and process every valid case in one Sweep 2 file."""

    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found:\n"
            f"{results_file.resolve()}"
        )

    with open(
        results_file,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    metadata = data[
        "metadata"
    ]

    cases = data[
        "results"
    ]

    required_fields = (
        "best_analytical_time",
        "best_analytical_primitives",
        "ocp_time",
        "ocp_time_grid",
        "ocp_x",
        "ocp_y",
        "ocp_theta",
    )

    results = []
    skipped_case_ids = []

    for case in cases:
        if not case.get(
            "success",
            False,
        ):
            skipped_case_ids.append(
                case.get(
                    "case_id"
                )
            )
            continue

        if not case.get(
            "ocp_success",
            False,
        ):
            skipped_case_ids.append(
                case.get(
                    "case_id"
                )
            )
            continue

        if any(
            case.get(
                field
            ) is None
            for field in required_fields
        ):
            skipped_case_ids.append(
                case.get(
                    "case_id"
                )
            )
            continue

        result = compare_case(
            case=case,
            n_comparison_points=(
                n_comparison_points
            ),
        )

        results.append(
            result
        )

    print("\n" + "=" * 80)
    print("LOADED SWEEP 2 FILE")
    print("=" * 80)

    print(
        f"Results file  : "
        f"{results_file}"
    )

    print(
        f"N             : "
        f"{metadata.get('N')}"
    )

    print(
        f"M             : "
        f"{metadata.get('M')}"
    )

    print(
        f"Cases loaded  : "
        f"{len(cases)}"
    )

    print(
        f"Cases compared: "
        f"{len(results)}"
    )

    print(
        f"Cases skipped : "
        f"{len(skipped_case_ids)}"
    )

    if skipped_case_ids:
        print(
            f"Skipped IDs   : "
            f"{skipped_case_ids[:20]}"
        )

    return (
        metadata,
        results,
    )


# =============================================================================
# GROUPED STATISTICS
# =============================================================================

def compute_grouped_statistics(
    comparison_results,
):
    """
    Group the metrics by D/R and compute statistics for each group.
    """

    grouped_statistics = {}

    for d_over_r in EXPECTED_D_OVER_R_VALUES:
        distance_results = [
            result
            for result in comparison_results
            if np.isclose(
                result[
                    "d_over_r"
                ],
                d_over_r,
            )
        ]

        absolute_time_errors = [
            result[
                "absolute_time_error"
            ]
            for result in distance_results
        ]

        relative_time_errors = [
            result[
                "relative_time_error_percent"
            ]
            for result in distance_results
        ]

        hausdorff_distances = [
            result[
                "hausdorff_distance"
            ]
            for result in distance_results
        ]

        grouped_statistics[
            float(
                d_over_r
            )
        ] = {
            "absolute_time_error": compute_statistics(
                absolute_time_errors
            ),
            "relative_time_error_percent": compute_statistics(
                relative_time_errors
            ),
            "hausdorff_distance": compute_statistics(
                hausdorff_distances
            ),
        }

    return grouped_statistics


def print_grouped_statistics(
    transcription_resolution,
    grouped_statistics,
):
    """Print all grouped statistics for one transcription resolution."""

    print("\n" + "=" * 120)

    print(
        f"SWEEP 2 STATISTICS BY D/R, "
        f"N={transcription_resolution}"
    )

    print("=" * 120)

    for metric_key, metric_label, unit in (
        (
            "absolute_time_error",
            "ABSOLUTE TRAVERSAL-TIME DISCREPANCY",
            "s",
        ),
        (
            "relative_time_error_percent",
            "RELATIVE TRAVERSAL-TIME DISCREPANCY",
            "%",
        ),
        (
            "hausdorff_distance",
            "SYMMETRIC HAUSDORFF DISTANCE",
            "m",
        ),
    ):
        print(f"\n{metric_label}")
        print("-" * 120)

        header = (
            f"{'D/R':>8}"
            f"{'Cases':>10}"
            f"{'Mean':>22}"
            f"{'Median':>22}"
            f"{'95th percentile':>22}"
            f"{'Maximum':>22}"
        )

        print(
            header
        )

        print(
            "-" * len(
                header
            )
        )

        for d_over_r in EXPECTED_D_OVER_R_VALUES:
            statistics = grouped_statistics[
                d_over_r
            ][
                metric_key
            ]

            if statistics is None:
                print(
                    f"{d_over_r:>8.1f}"
                    f"{0:>10d}"
                    f"{'--':>22}"
                    f"{'--':>22}"
                    f"{'--':>22}"
                    f"{'--':>22}"
                )

                continue

            print(
                f"{d_over_r:>8.1f}"
                f"{statistics['count']:>10d}"
                f"{statistics['mean']:>22.6e}"
                f"{statistics['median']:>22.6e}"
                f"{statistics['p95']:>22.6e}"
                f"{statistics['maximum']:>22.6e}"
            )

        print(
            f"\nUnits: {unit}"
        )


def print_latex_rows(
    all_grouped_statistics,
):
    """
    Print compact LaTeX rows grouped by N and D/R.

    The columns are:
    N, D/R,
    mean absolute time error,
    95th absolute time error,
    mean relative time error,
    95th relative time error,
    mean Hausdorff distance,
    95th Hausdorff distance.
    """

    print("\n" + "=" * 140)
    print("LATEX ROWS FOR SWEEP 2 DISTANCE BREAKDOWN")
    print("=" * 140)

    print(
        "Columns: N, D/R, mean |Delta T|, 95th |Delta T|, "
        "mean relative error, 95th relative error, "
        "mean Hausdorff distance, 95th Hausdorff distance"
    )

    print("-" * 140)

    for transcription_resolution in (
        30,
        100,
    ):
        grouped_statistics = all_grouped_statistics[
            transcription_resolution
        ]

        for d_over_r in EXPECTED_D_OVER_R_VALUES:
            absolute_statistics = grouped_statistics[
                d_over_r
            ][
                "absolute_time_error"
            ]

            relative_statistics = grouped_statistics[
                d_over_r
            ][
                "relative_time_error_percent"
            ]

            hausdorff_statistics = grouped_statistics[
                d_over_r
            ][
                "hausdorff_distance"
            ]

            print(
                rf"\({transcription_resolution}\)"
                rf" & \({d_over_r:.0f}\)"
                rf" & \({absolute_statistics['mean']:.2e}\)"
                rf" & \({absolute_statistics['p95']:.2e}\)"
                rf" & \({relative_statistics['mean']:.2e}\)"
                rf" & \({relative_statistics['p95']:.2e}\)"
                rf" & \({hausdorff_statistics['mean']:.2e}\)"
                rf" & \({hausdorff_statistics['p95']:.2e}\)"
                rf" \\"
            )


# =============================================================================
# TREND REPORTING
# =============================================================================

def print_monotonicity_report(
    all_grouped_statistics,
):
    """
    Check whether the mean and 95th-percentile errors increase with D/R.
    """

    print("\n" + "=" * 100)
    print("MONOTONICITY CHECK WITH INCREASING D/R")
    print("=" * 100)

    for transcription_resolution in (
        30,
        100,
    ):
        print(
            f"\nN={transcription_resolution}"
        )

        grouped_statistics = all_grouped_statistics[
            transcription_resolution
        ]

        for metric_key, metric_label in (
            (
                "absolute_time_error",
                "Absolute time error",
            ),
            (
                "relative_time_error_percent",
                "Relative time error",
            ),
            (
                "hausdorff_distance",
                "Hausdorff distance",
            ),
        ):
            mean_values = np.asarray(
                [
                    grouped_statistics[
                        d_over_r
                    ][
                        metric_key
                    ][
                        "mean"
                    ]
                    for d_over_r
                    in EXPECTED_D_OVER_R_VALUES
                ],
                dtype=float,
            )

            p95_values = np.asarray(
                [
                    grouped_statistics[
                        d_over_r
                    ][
                        metric_key
                    ][
                        "p95"
                    ]
                    for d_over_r
                    in EXPECTED_D_OVER_R_VALUES
                ],
                dtype=float,
            )

            mean_increasing = bool(
                np.all(
                    np.diff(
                        mean_values
                    ) >= 0.0
                )
            )

            p95_increasing = bool(
                np.all(
                    np.diff(
                        p95_values
                    ) >= 0.0
                )
            )

            print(
                f"  {metric_label:<26} "
                f"mean increasing: {mean_increasing!s:<5} | "
                f"95th increasing: {p95_increasing!s:<5}"
            )


# =============================================================================
# PLOTTING
# =============================================================================

def plot_sweep2_trends(
    all_grouped_statistics,
):
    """
    Plot the Sweep 2 trends versus D/R.

    Left panel:
        Mean relative traversal-time discrepancy.

    Right panel:
        Mean symmetric Hausdorff distance.
    """

    d_over_r_values = np.asarray(
        EXPECTED_D_OVER_R_VALUES,
        dtype=float,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.3),
        sharex=True,
    )

    relative_axis = axes[0]
    hausdorff_axis = axes[1]

    for transcription_resolution in (
        30,
        100,
    ):
        grouped_statistics = all_grouped_statistics[
            transcription_resolution
        ]

        mean_relative_errors = np.asarray(
            [
                grouped_statistics[
                    d_over_r
                ][
                    "relative_time_error_percent"
                ][
                    "mean"
                ]
                for d_over_r in EXPECTED_D_OVER_R_VALUES
            ],
            dtype=float,
        )

        mean_hausdorff_distances = np.asarray(
            [
                grouped_statistics[
                    d_over_r
                ][
                    "hausdorff_distance"
                ][
                    "mean"
                ]
                for d_over_r in EXPECTED_D_OVER_R_VALUES
            ],
            dtype=float,
        )

        relative_axis.plot(
            d_over_r_values,
            mean_relative_errors,
            marker="o",
            markersize=6,
            linewidth=2.0,
            label=rf"$N={transcription_resolution}$",
        )

        hausdorff_axis.plot(
            d_over_r_values,
            mean_hausdorff_distances,
            marker="o",
            markersize=6,
            linewidth=2.0,
            label=rf"$N={transcription_resolution}$",
        )

    relative_axis.set_xlabel(
        r"$D/R$"
    )

    relative_axis.set_ylabel(
        r"Mean relative traversal-time discrepancy [\%]"
    )

    hausdorff_axis.set_xlabel(
        r"$D/R$"
    )

    hausdorff_axis.set_ylabel(
        r"Mean symmetric Hausdorff distance [m]"
    )

    for axis in axes:
        axis.set_xticks(
            d_over_r_values
        )

        axis.grid(
            True,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDTH,
            alpha=GRID_ALPHA,
        )

        axis.legend(
            frameon=True
        )

        axis.tick_params(
            direction="out"
        )

    relative_axis.text(
        -0.14,
        1.02,
        "(a)",
        transform=relative_axis.transAxes,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    hausdorff_axis.text(
        -0.14,
        1.02,
        "(b)",
        transform=hausdorff_axis.transAxes,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    figure.tight_layout(
        w_pad=2.4
    )

    if SAVE_FIGURES:
        FIGURES_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        pdf_path = (
            FIGURES_DIRECTORY
            / "sweep2_trends_by_distance.pdf"
        )

        png_path = (
            FIGURES_DIRECTORY
            / "sweep2_trends_by_distance.png"
        )

        figure.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        figure.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"\nSaved: {pdf_path}"
        )

        print(
            f"Saved: {png_path}"
        )

    return figure


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    all_comparison_results = {}
    all_grouped_statistics = {}

    for transcription_resolution in (
        30,
        100,
    ):
        (
            metadata,
            comparison_results,
        ) = load_and_process_results(
            results_file=RESULTS_FILES[
                transcription_resolution
            ],
            n_comparison_points=(
                N_COMPARISON_POINTS
            ),
        )

        grouped_statistics = (
            compute_grouped_statistics(
                comparison_results
            )
        )

        all_grouped_statistics[
            transcription_resolution
        ] = grouped_statistics

        print_grouped_statistics(
            transcription_resolution=(
                transcription_resolution
            ),
            grouped_statistics=(
                grouped_statistics
            ),
        )

    print_latex_rows(
        all_grouped_statistics
    )

    print_monotonicity_report(
        all_grouped_statistics
    )

    # -------------------------------------------------------------------------
    # Figures: mean metrics versus D/R
    # -------------------------------------------------------------------------

    trend_figure = plot_sweep2_trends(
        all_grouped_statistics
    )

    if SHOW_FIGURES:
        plt.show(
            block=True
        )

    else:
        plt.close(
            trend_figure
        )