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
        "sweep4/sobol_sweep_N30_M4.json"
    ),
    100: Path(
        "experiments/pose2pose_unicycle/results/"
        "sweep4/sobol_sweep_N100_M4.json"
    ),
}

N_COMPARISON_POINTS = 1001

SAVE_FIGURE = True
SHOW_FIGURE = True

FIGURES_DIRECTORY = Path(
    "experiments/pose2pose_unicycle/figures"
)

OUTPUT_BASENAME = (
    "sweep4_validation_ecdf_N30_N100"
)

GRID_COLOR = "0.75"
GRID_LINESTYLE = ":"
GRID_LINEWIDTH = 0.7
GRID_ALPHA = 0.7

LINEWIDTH = 2.2


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
            + omega
            * dt
        )

    else:
        theta_new = (
            theta
            + omega
            * dt
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
    """Return the constant controls associated with one primitive."""

    label = primitive[
        "label"
    ].strip().lower()

    if label == "segment":
        return (
            v_max,
            0.0,
        )

    if label == "arc":
        direction = int(
            primitive[
                "turn_direction"
            ]
        )

        return (
            v_max,
            direction
            * omega_max,
        )

    if label == "turn on-the-spot":
        direction = int(
            primitive[
                "turn_direction"
            ]
        )

        return (
            0.0,
            direction
            * omega_max,
        )

    raise ValueError(
        f"Unknown primitive label: {label!r}"
    )


def reconstruct_analytical_trajectory(
    case,
    samples_per_primitive=300,
):
    """Reconstruct the analytical pose trajectory."""

    primitives = case[
        "best_analytical_primitives"
    ]

    if not primitives:
        raise ValueError(
            f"No analytical primitives saved for case "
            f"{case.get('case_id')}."
        )

    v_max = float(
        case[
            "v_max"
        ]
    )

    omega_max = float(
        case[
            "omega_max"
        ]
    )

    initial_pose = primitives[
        0
    ][
        "start_pose"
    ]

    x = float(
        initial_pose[
            0
        ]
    )

    y = float(
        initial_pose[
            1
        ]
    )

    theta = float(
        initial_pose[
            2
        ]
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
            primitive=primitive,
            v_max=v_max,
            omega_max=omega_max,
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
                x=x,
                y=y,
                theta=theta,
                v=v,
                omega=omega,
                dt=dt,
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
    """Compute the discrete symmetric Hausdorff distance."""

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
# CASE COMPARISON
# =============================================================================

def compare_case(
    case,
    n_comparison_points,
):
    """Compute the two metrics used in the Sweep 4 ECDF figure."""

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

    if analytical_time <= 0.0:
        raise ValueError(
            f"Nonpositive analytical time in case "
            f"{case.get('case_id')}."
        )

    relative_time_error_percent = (
        100.0
        * abs(
            ocp_time
            - analytical_time
        )
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
            analytical_x=(
                analytical_x_interp
            ),
            analytical_y=(
                analytical_y_interp
            ),
            ocp_x=ocp_x_interp,
            ocp_y=ocp_y_interp,
        )
    )

    return {
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
    """Load all valid cases and compute the ECDF metrics."""

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

    comparison_results = []
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

        comparison_results.append(
            compare_case(
                case=case,
                n_comparison_points=(
                    n_comparison_points
                ),
            )
        )

    print("\n" + "=" * 80)
    print("LOADED SOBOL SWEEP FILE")
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
        f"{len(comparison_results)}"
    )

    print(
        f"Cases skipped : "
        f"{len(skipped_case_ids)}"
    )

    return comparison_results


# =============================================================================
# ECDF HELPERS
# =============================================================================

def compute_ecdf(values):
    """Return sorted values and empirical cumulative probabilities."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.size == 0:
        raise ValueError(
            "Cannot compute an ECDF from an empty array."
        )

    sorted_values = np.sort(
        values
    )

    cumulative_probabilities = (
        np.arange(
            1,
            sorted_values.size + 1,
            dtype=float,
        )
        / sorted_values.size
    )

    return (
        sorted_values,
        cumulative_probabilities,
    )


def print_metric_statistics(
    transcription_resolution,
    relative_errors,
    hausdorff_distances,
):
    """Print the main percentile values used to interpret the ECDF."""

    print(
        f"\nN={transcription_resolution}"
    )

    print(
        "-" * 70
    )

    print(
        "Relative traversal-time discrepancy"
    )

    print(
        f"  Median : "
        f"{np.median(relative_errors):.6e} %"
    )

    print(
        f"  95th   : "
        f"{np.percentile(relative_errors, 95):.6e} %"
    )

    print(
        f"  Maximum: "
        f"{np.max(relative_errors):.6e} %"
    )

    print(
        "Symmetric Hausdorff distance"
    )

    print(
        f"  Median : "
        f"{np.median(hausdorff_distances):.6e} m"
    )

    print(
        f"  95th   : "
        f"{np.percentile(hausdorff_distances, 95):.6e} m"
    )

    print(
        f"  Maximum: "
        f"{np.max(hausdorff_distances):.6e} m"
    )


# =============================================================================
# PLOTTING
# =============================================================================

def plot_sobol_ecdfs(
    all_metrics,
):
    """
    Plot ECDFs of the relative time discrepancy and Hausdorff distance.
    """

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            10.8,
            4.4,
        ),
        sharey=True,
    )

    time_axis = axes[
        0
    ]

    hausdorff_axis = axes[
        1
    ]

    for transcription_resolution in (
        30,
        100,
    ):
        relative_errors = all_metrics[
            transcription_resolution
        ][
            "relative_time_error_percent"
        ]

        hausdorff_distances = all_metrics[
            transcription_resolution
        ][
            "hausdorff_distance"
        ]

        (
            sorted_relative_errors,
            relative_probabilities,
        ) = compute_ecdf(
            relative_errors
        )

        (
            sorted_hausdorff_distances,
            hausdorff_probabilities,
        ) = compute_ecdf(
            hausdorff_distances
        )

        curve_label = (
            rf"$N={transcription_resolution}$"
        )

        time_axis.step(
            sorted_relative_errors,
            relative_probabilities,
            where="post",
            linewidth=LINEWIDTH,
            label=curve_label,
        )

        hausdorff_axis.step(
            sorted_hausdorff_distances,
            hausdorff_probabilities,
            where="post",
            linewidth=LINEWIDTH,
            label=curve_label,
        )

    # Mark the 95% cumulative probability.
    for axis in axes:
        axis.axhline(
            0.95,
            color="0.35",
            linestyle=":",
            linewidth=1.3,
            zorder=0,
        )

        axis.text(
            0.98,
            0.955,
            r"$95\%$",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=10,
        )

        axis.set_ylim(
            0.0,
            1.01,
        )

    time_axis.set_xlabel(
        r"Relative traversal-time discrepancy [\%]"
    )

    time_axis.set_ylabel(
        "Fraction of cases"
    )

    hausdorff_axis.set_xlabel(
        r"Symmetric Hausdorff distance [m]"
    )

    time_axis.text(
        -0.14,
        1.02,
        "(a)",
        transform=time_axis.transAxes,
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
        w_pad=2.5
    )

    if SAVE_FIGURE:
        FIGURES_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        pdf_path = (
            FIGURES_DIRECTORY
            / f"{OUTPUT_BASENAME}.pdf"
        )

        png_path = (
            FIGURES_DIRECTORY
            / f"{OUTPUT_BASENAME}.png"
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

    all_metrics = {}

    for transcription_resolution in (
        30,
        100,
    ):
        comparison_results = (
            load_and_process_results(
                results_file=RESULTS_FILES[
                    transcription_resolution
                ],
                n_comparison_points=(
                    N_COMPARISON_POINTS
                ),
            )
        )

        relative_errors = np.asarray(
            [
                result[
                    "relative_time_error_percent"
                ]
                for result in comparison_results
            ],
            dtype=float,
        )

        hausdorff_distances = np.asarray(
            [
                result[
                    "hausdorff_distance"
                ]
                for result in comparison_results
            ],
            dtype=float,
        )

        all_metrics[
            transcription_resolution
        ] = {
            "relative_time_error_percent": (
                relative_errors
            ),
            "hausdorff_distance": (
                hausdorff_distances
            ),
        }

        print_metric_statistics(
            transcription_resolution=(
                transcription_resolution
            ),
            relative_errors=(
                relative_errors
            ),
            hausdorff_distances=(
                hausdorff_distances
            ),
        )

    figure = plot_sobol_ecdfs(
        all_metrics
    )

    if SHOW_FIGURE:
        plt.show(
            block=True
        )

    else:
        plt.close(
            figure
        )