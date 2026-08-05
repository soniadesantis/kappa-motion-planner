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
    "legend.fontsize": 9,
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
        "sweep3/polar_goal_position_sweep_OCP_path_saved_N30_M4.json"
    ),
    100: Path(
        "experiments/pose2pose_unicycle/results/"
        "sweep3/polar_goal_position_sweep_OCP_path_saved_N100_M4.json"
    ),
}

EXPECTED_D_OVER_R_VALUES = (
    5.0,
    10.0,
    15.0,
    20.0,
)

EXPECTED_GOAL_ANGLES_DEG = tuple(
    np.arange(
        0.0,
        360.0,
        22.5,
    )
)

N_COMPARISON_POINTS = 1001

# Use the finer transcription for the direction-dependence figure.
FIGURE_RESOLUTION = 100

SAVE_FIGURE = True
SHOW_FIGURE = True

FIGURES_DIRECTORY = Path(
    "experiments/pose2pose_unicycle/figures"
)

OUTPUT_BASENAME = (
    "sweep3_trends_by_goal_direction"
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
        "count": int(
            values.size
        ),
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


def get_position_component(
    case,
    scalar_key,
    vector_keys,
    component,
):
    """
    Read one position component from either a scalar field or a saved vector.
    """

    if case.get(
        scalar_key
    ) is not None:
        return float(
            case[
                scalar_key
            ]
        )

    for vector_key in vector_keys:
        value = case.get(
            vector_key
        )

        if value is not None:
            return float(
                value[
                    component
                ]
            )

    raise KeyError(
        f"Could not recover position component '{scalar_key}' "
        f"for case {case.get('case_id')}."
    )


def get_case_positions(case):
    """Recover the initial and final planar positions."""

    x0 = get_position_component(
        case=case,
        scalar_key="x0",
        vector_keys=(
            "p0",
            "initial_position",
            "start_position",
        ),
        component=0,
    )

    y0 = get_position_component(
        case=case,
        scalar_key="y0",
        vector_keys=(
            "p0",
            "initial_position",
            "start_position",
        ),
        component=1,
    )

    xf = get_position_component(
        case=case,
        scalar_key="xf",
        vector_keys=(
            "pf",
            "final_position",
            "goal_position",
        ),
        component=0,
    )

    yf = get_position_component(
        case=case,
        scalar_key="yf",
        vector_keys=(
            "pf",
            "final_position",
            "goal_position",
        ),
        component=1,
    )

    return (
        x0,
        y0,
        xf,
        yf,
    )


def get_case_d_over_r(case):
    """
    Recover D/R from a saved field or reconstruct it from the positions.
    """

    for key in (
        "D_over_R",
        "d_over_r",
        "distance_over_radius",
    ):
        if case.get(
            key
        ) is not None:
            return float(
                case[
                    key
                ]
            )

    (
        x0,
        y0,
        xf,
        yf,
    ) = get_case_positions(
        case
    )

    distance = float(
        np.hypot(
            xf - x0,
            yf - y0,
        )
    )

    if case.get(
        "R"
    ) is not None:
        radius = float(
            case[
                "R"
            ]
        )

    else:
        radius = (
            float(
                case[
                    "v_max"
                ]
            )
            / float(
                case[
                    "omega_max"
                ]
            )
        )

    if radius <= 0.0:
        raise ValueError(
            f"Invalid radius in case {case.get('case_id')}."
        )

    return (
        distance
        / radius
    )


def get_case_goal_angle_deg(case):
    """
    Recover the goal-position direction phi in degrees.

    Several possible saved field names are checked first. If none is present,
    the angle is reconstructed from the initial and final positions.
    """

    degree_keys = (
        "goal_angle_deg",
        "phi_deg",
        "goal_direction_deg",
        "goal_position_angle_deg",
    )

    for key in degree_keys:
        if case.get(
            key
        ) is not None:
            angle_deg = float(
                case[
                    key
                ]
            )

            return (
                angle_deg
                % 360.0
            )

    radian_keys = (
        "goal_angle",
        "phi",
        "goal_direction",
        "goal_position_angle",
    )

    for key in radian_keys:
        if case.get(
            key
        ) is not None:
            angle_deg = float(
                np.degrees(
                    float(
                        case[
                            key
                        ]
                    )
                )
            )

            return (
                angle_deg
                % 360.0
            )

    (
        x0,
        y0,
        xf,
        yf,
    ) = get_case_positions(
        case
    )

    angle_deg = float(
        np.degrees(
            np.arctan2(
                yf - y0,
                xf - x0,
            )
        )
    )

    return (
        angle_deg
        % 360.0
    )


def match_expected_value(
    value,
    expected_values,
    name,
    tolerance=1.0e-7,
):
    """Match a numerical value to one of the expected sweep values."""

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
        f"Unexpected {name} value {value:.12g} in the results. "
        f"Expected one of {expected_values}."
    )


# =============================================================================
# TRAJECTORY INTERPOLATION
# =============================================================================

def interpolate_pose(
    times,
    xs,
    ys,
    thetas,
    normalized_grid,
):
    """Interpolate a pose trajectory on a normalized-time grid."""

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

    unwrapped_thetas = np.unwrap(
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
        unwrapped_thetas,
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

    if abs(
        omega
    ) < tolerance:
        x_new = (
            x
            + v
            * np.cos(
                theta
            )
            * dt
        )

        y_new = (
            y
            + v
            * np.sin(
                theta
            )
            * dt
        )

        theta_new = theta

    elif abs(
        v
    ) < tolerance:
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
            + (
                v
                / omega
            )
            * (
                np.sin(
                    theta_new
                )
                - np.sin(
                    theta
                )
            )
        )

        y_new = (
            y
            - (
                v
                / omega
            )
            * (
                np.cos(
                    theta_new
                )
                - np.cos(
                    theta
                )
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
    """Return the controls corresponding to a saved analytical primitive."""

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
        f"Unknown analytical primitive label: {label!r}"
    )


def reconstruct_analytical_trajectory(
    case,
    samples_per_primitive=300,
):
    """Reconstruct the analytical trajectory from the saved primitives."""

    primitives = case[
        "best_analytical_primitives"
    ]

    if not primitives:
        raise ValueError(
            f"No analytical primitives in case "
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

        (
            v,
            omega,
        ) = primitive_controls(
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
# CASE PROCESSING
# =============================================================================

def compare_case(
    case,
    n_comparison_points,
):
    """Compute the metrics required for the Sweep 3 grouped analysis."""

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
        analytical_x,
        analytical_y,
        analytical_theta,
    ) = interpolate_pose(
        analytical_times,
        analytical_x,
        analytical_y,
        analytical_theta,
        normalized_grid,
    )

    (
        ocp_x,
        ocp_y,
        ocp_theta,
    ) = interpolate_pose(
        ocp_times,
        ocp_x,
        ocp_y,
        ocp_theta,
        normalized_grid,
    )

    hausdorff_distance = (
        discrete_symmetric_hausdorff_distance(
            analytical_x=analytical_x,
            analytical_y=analytical_y,
            ocp_x=ocp_x,
            ocp_y=ocp_y,
        )
    )

    d_over_r = match_expected_value(
        value=get_case_d_over_r(
            case
        ),
        expected_values=(
            EXPECTED_D_OVER_R_VALUES
        ),
        name="D/R",
    )

    goal_angle_deg = match_expected_value(
        value=get_case_goal_angle_deg(
            case
        ),
        expected_values=(
            EXPECTED_GOAL_ANGLES_DEG
        ),
        name="goal angle",
    )

    return {
        "case_id": int(
            case[
                "case_id"
            ]
        ),
        "d_over_r": d_over_r,
        "goal_angle_deg": goal_angle_deg,
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
    """Load and process all valid cases in one Sweep 3 result file."""

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
    print("LOADED SWEEP 3 FILE")
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

    return (
        metadata,
        comparison_results,
    )


# =============================================================================
# GROUPING BY NORMALIZED SEPARATION
# =============================================================================

def compute_statistics_by_distance(
    comparison_results,
):
    """Compute grouped statistics for each D/R value."""

    grouped_statistics = {}

    for d_over_r in EXPECTED_D_OVER_R_VALUES:
        selected_results = [
            result
            for result in comparison_results
            if np.isclose(
                result[
                    "d_over_r"
                ],
                d_over_r,
            )
        ]

        grouped_statistics[
            d_over_r
        ] = {
            "relative_time_error": compute_statistics(
                [
                    result[
                        "relative_time_error_percent"
                    ]
                    for result in selected_results
                ]
            ),
            "hausdorff_distance": compute_statistics(
                [
                    result[
                        "hausdorff_distance"
                    ]
                    for result in selected_results
                ]
            ),
        }

    return grouped_statistics


def print_distance_table_rows(
    all_distance_statistics,
):
    """
    Print compact LaTeX rows for a Sweep 3 distance table.

    Columns:
        N, D/R,
        mean and 95th relative time discrepancy,
        mean and 95th Hausdorff distance.
    """

    print("\n" + "=" * 120)
    print("LATEX ROWS: SWEEP 3 GROUPED BY D/R")
    print("=" * 120)

    for transcription_resolution in (
        30,
        100,
    ):
        grouped_statistics = all_distance_statistics[
            transcription_resolution
        ]

        for d_over_r in EXPECTED_D_OVER_R_VALUES:
            relative_statistics = grouped_statistics[
                d_over_r
            ][
                "relative_time_error"
            ]

            hausdorff_statistics = grouped_statistics[
                d_over_r
            ][
                "hausdorff_distance"
            ]

            print(
                rf"\({transcription_resolution}\)"
                rf" & \({d_over_r:.0f}\)"
                rf" & \({relative_statistics['mean']:.2e}\)"
                rf" & \({relative_statistics['p95']:.2e}\)"
                rf" & \({hausdorff_statistics['mean']:.2e}\)"
                rf" & \({hausdorff_statistics['p95']:.2e}\)"
                rf" \\"
            )


# =============================================================================
# GROUPING BY DISTANCE AND GOAL DIRECTION
# =============================================================================

def compute_statistics_by_distance_and_angle(
    comparison_results,
):
    """
    Compute the mean metrics for every pair (D/R, phi).
    """

    grouped_statistics = {}

    for d_over_r in EXPECTED_D_OVER_R_VALUES:
        grouped_statistics[
            d_over_r
        ] = {}

        for goal_angle_deg in EXPECTED_GOAL_ANGLES_DEG:
            selected_results = [
                result
                for result in comparison_results
                if np.isclose(
                    result[
                        "d_over_r"
                    ],
                    d_over_r,
                )
                and np.isclose(
                    result[
                        "goal_angle_deg"
                    ],
                    goal_angle_deg,
                )
            ]

            expected_cases = (
                12
                * 12
            )

            if len(
                selected_results
            ) != expected_cases:
                raise ValueError(
                    f"Expected {expected_cases} cases for "
                    f"D/R={d_over_r:g}, phi={goal_angle_deg:g} deg, "
                    f"but found {len(selected_results)}."
                )

            relative_statistics = compute_statistics(
                [
                    result[
                        "relative_time_error_percent"
                    ]
                    for result in selected_results
                ]
            )

            hausdorff_statistics = compute_statistics(
                [
                    result[
                        "hausdorff_distance"
                    ]
                    for result in selected_results
                ]
            )

            grouped_statistics[
                d_over_r
            ][
                goal_angle_deg
            ] = {
                "relative_time_error": relative_statistics,
                "hausdorff_distance": hausdorff_statistics,
            }

    return grouped_statistics


# =============================================================================
# DIRECTION-INVARIANCE REPORT
# =============================================================================

def print_direction_variation_report(
    angle_statistics,
):
    """
    Report the variation of the mean metrics across phi for each D/R.

    Relative spread is defined as

        (maximum - minimum) / overall mean * 100.
    """

    print("\n" + "=" * 112)
    print("VARIATION OF THE MEAN METRICS WITH GOAL DIRECTION")
    print("=" * 112)

    header = (
        f"{'D/R':>8}"
        f"{'Metric':>30}"
        f"{'Minimum':>18}"
        f"{'Maximum':>18}"
        f"{'Mean':>18}"
        f"{'Rel. spread [%]':>18}"
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
        for metric_key, metric_label in (
            (
                "relative_time_error",
                "Relative time discrepancy",
            ),
            (
                "hausdorff_distance",
                "Hausdorff distance",
            ),
        ):
            values = np.asarray(
                [
                    angle_statistics[
                        d_over_r
                    ][
                        goal_angle_deg
                    ][
                        metric_key
                    ][
                        "mean"
                    ]
                    for goal_angle_deg
                    in EXPECTED_GOAL_ANGLES_DEG
                ],
                dtype=float,
            )

            minimum = float(
                np.min(
                    values
                )
            )

            maximum = float(
                np.max(
                    values
                )
            )

            mean = float(
                np.mean(
                    values
                )
            )

            relative_spread = (
                100.0
                * (
                    maximum
                    - minimum
                )
                / mean
                if mean > 0.0
                else 0.0
            )

            print(
                f"{d_over_r:>8.1f}"
                f"{metric_label:>30}"
                f"{minimum:>18.6e}"
                f"{maximum:>18.6e}"
                f"{mean:>18.6e}"
                f"{relative_spread:>18.3f}"
            )


# =============================================================================
# PLOTTING
# =============================================================================

def plot_goal_direction_trends(
    angle_statistics,
    transcription_resolution,
):
    """
    Plot the mean discrepancies versus goal direction.

    One curve is shown for each normalized separation.
    """

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            11.0,
            4.4,
        ),
        sharex=True,
    )

    relative_axis = axes[
        0
    ]

    hausdorff_axis = axes[
        1
    ]

    original_angles = np.asarray(
        EXPECTED_GOAL_ANGLES_DEG,
        dtype=float,
    )

    # Repeat the first point at 360 deg to emphasize periodicity.
    plot_angles = np.append(
        original_angles,
        360.0,
    )

    for d_over_r in EXPECTED_D_OVER_R_VALUES:
        mean_relative_errors = np.asarray(
            [
                angle_statistics[
                    d_over_r
                ][
                    goal_angle_deg
                ][
                    "relative_time_error"
                ][
                    "mean"
                ]
                for goal_angle_deg
                in EXPECTED_GOAL_ANGLES_DEG
            ],
            dtype=float,
        )

        mean_hausdorff_distances = np.asarray(
            [
                angle_statistics[
                    d_over_r
                ][
                    goal_angle_deg
                ][
                    "hausdorff_distance"
                ][
                    "mean"
                ]
                for goal_angle_deg
                in EXPECTED_GOAL_ANGLES_DEG
            ],
            dtype=float,
        )

        mean_relative_errors = np.append(
            mean_relative_errors,
            mean_relative_errors[
                0
            ],
        )

        mean_hausdorff_distances = np.append(
            mean_hausdorff_distances,
            mean_hausdorff_distances[
                0
            ],
        )

        curve_label = (
            rf"$D/R={d_over_r:g}$"
        )

        relative_axis.plot(
            plot_angles,
            mean_relative_errors,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            label=curve_label,
        )

        hausdorff_axis.plot(
            plot_angles,
            mean_hausdorff_distances,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            label=curve_label,
        )

    relative_axis.set_xlabel(
        r"Goal direction $\varphi$ [deg]"
    )

    relative_axis.set_ylabel(
        r"Mean relative traversal-time discrepancy [\%]"
    )

    hausdorff_axis.set_xlabel(
        r"Goal direction $\varphi$ [deg]"
    )

    hausdorff_axis.set_ylabel(
        r"Mean symmetric Hausdorff distance [m]"
    )

    for axis in axes:
        axis.set_xlim(
            0.0,
            360.0,
        )

        axis.set_xticks(
            np.arange(
                0.0,
                361.0,
                45.0,
            )
        )

        axis.grid(
            True,
            color=GRID_COLOR,
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDTH,
            alpha=GRID_ALPHA,
        )

        axis.tick_params(
            direction="out"
        )

    relative_axis.text(
        -0.13,
        1.02,
        "(a)",
        transform=relative_axis.transAxes,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    hausdorff_axis.text(
        -0.13,
        1.02,
        "(b)",
        transform=hausdorff_axis.transAxes,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    handles, labels = (
        relative_axis.get_legend_handles_labels()
    )

    figure.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.02,
        ),
        ncol=4,
        frameon=True,
    )

    figure.tight_layout(
        rect=[
            0.0,
            0.11,
            1.0,
            1.0,
        ],
        w_pad=2.5,
    )

    if SAVE_FIGURE:
        FIGURES_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        pdf_path = (
            FIGURES_DIRECTORY
            / f"{OUTPUT_BASENAME}_N{transcription_resolution}.pdf"
        )

        png_path = (
            FIGURES_DIRECTORY
            / f"{OUTPUT_BASENAME}_N{transcription_resolution}.png"
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

    all_results = {}
    all_distance_statistics = {}

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

        all_results[
            transcription_resolution
        ] = comparison_results

        all_distance_statistics[
            transcription_resolution
        ] = compute_statistics_by_distance(
            comparison_results
        )

    # -------------------------------------------------------------------------
    # Print compact D/R table rows for both resolutions
    # -------------------------------------------------------------------------

    print_distance_table_rows(
        all_distance_statistics
    )

    # -------------------------------------------------------------------------
    # Direction-dependent analysis for N=100
    # -------------------------------------------------------------------------

    selected_results = all_results[
        FIGURE_RESOLUTION
    ]

    angle_statistics = (
        compute_statistics_by_distance_and_angle(
            selected_results
        )
    )

    print_direction_variation_report(
        angle_statistics
    )

    trend_figure = plot_goal_direction_trends(
        angle_statistics=angle_statistics,
        transcription_resolution=(
            FIGURE_RESOLUTION
        ),
    )

    if SHOW_FIGURE:
        plt.show(
            block=True
        )

    else:
        plt.close(
            trend_figure
        )