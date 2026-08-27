import json
from math import cos, sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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
# CONFIGURATION
# =============================================================================
# Sweep 1
# RESULTS_FILES = {
#     30: Path(
#         "experiments/pose2pose_unicycle/results/"
#         "sweep1/orientation_sweep_OCP_path_saved_N30_M4.json"
#     ),
#     100: Path(
#         "experiments/pose2pose_unicycle/results/"
#         "sweep1/orientation_sweep_N100_OCPpath.json"
#     ),
# }

# # Sweep 2
# RESULTS_FILES = {
#     30: Path(
#         "experiments/pose2pose_unicycle/results/"
#         "sweep2/distance_over_radius_sweep_OCP_path_saved_N30_M4.json"
#     ),
#     100: Path(
#         "experiments/pose2pose_unicycle/results/"
#         "sweep2/distance_over_radius_sweep_OCP_path_saved_N100_M4.json"
#     ),
# }

# # Sweep 3
# RESULTS_FILES = {
#     30: Path(
#         "experiments/pose2pose_unicycle/results/"
#         "sweep3/polar_goal_position_sweep_OCP_path_saved_N30_M4.json"
#     ),
#     100: Path(
#         "experiments/pose2pose_unicycle/results/"
#         "sweep3/polar_goal_position_sweep_OCP_path_saved_N100_M4.json"
#     ),
# }

# Sweep 4
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

# Print detailed information for the worst cases of each resolution.
N_WORST_CASES_TO_PRINT = 10

FAMILY_ORDER = (
    "CSC",
    "TCSC",
    "CSCT",
    "TCSCT",
)


# =============================================================================
# WORST HAUSDORFF CASE PLOT CONFIGURATION
# =============================================================================

PLOT_WORST_HAUSDORFF_CASE = True
SAVE_WORST_HAUSDORFF_FIGURE = True
SHOW_WORST_HAUSDORFF_FIGURE = True

# Use the finer transcription for the thesis figure.
WORST_HAUSDORFF_RESOLUTION = 100

WORST_HAUSDORFF_OUTPUT_BASENAME = (
    "sweep4_worst_hausdorff_distance"
)

ANALYTICAL_COLOR = "tab:blue"
ANALYTICAL_LINESTYLE = "-"
ANALYTICAL_LINEWIDTH = 2.5

OCP_COLOR = "black"
OCP_LINESTYLE = "--"
OCP_LINEWIDTH = 2.2

INITIAL_POSE_COLOR = "tab:green"
FINAL_POSE_COLOR = "tab:red"

HAUSDORFF_COLOR = "tab:red"
HAUSDORFF_LINESTYLE = "-"
HAUSDORFF_LINEWIDTH = 1.5

GRID_COLOR = "0.75"
GRID_LINESTYLE = ":"
GRID_LINEWIDTH = 0.7
GRID_ALPHA = 0.7

ZOOM_RECTANGLE_COLOR = "0.45"
ZOOM_RECTANGLE_LINEWIDTH = 1.0

POSE_ARROW_LENGTH = 0.42
POSE_MARKER_SIZE = 7

# The zoom half-width is selected automatically as the larger of:
#   1. this minimum value;
#   2. this scale factor multiplied by the Hausdorff distance.
MINIMUM_ZOOM_HALF_WIDTH = 0.12
ZOOM_SCALE_FACTOR = 3.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def wrap_to_pi(angle):
    """Wrap an angle or array of angles to [-pi, pi)."""
    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


def get_analytical_family(
    analytical_name,
):
    """
    Extract the trajectory-family name from a saved analytical candidate name.

    Examples
    --------
    ``TCSCT left-right`` -> ``TCSCT``
    ``CSC right-right``  -> ``CSC``
    """

    if analytical_name is None:
        return None

    family = analytical_name.split(
        " ",
        1,
    )[0]

    if family not in FAMILY_ORDER:
        raise ValueError(
            f"Unknown analytical family '{family}' in "
            f"analytical_name='{analytical_name}'."
        )

    return family


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
    )

    xs = np.asarray(
        xs,
        dtype=float,
    )

    ys = np.asarray(
        ys,
        dtype=float,
    )

    thetas = np.asarray(
        thetas,
        dtype=float,
    )

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

    tau = (
        times - times[0]
    ) / duration

    # Remove repeated normalized-time samples.
    tau_unique, unique_indices = np.unique(
        tau,
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

    # Remove artificial jumps at +/- pi before interpolation.
    thetas_unwrapped = np.unwrap(
        thetas
    )

    x_interp = np.interp(
        normalized_grid,
        tau_unique,
        xs,
    )

    y_interp = np.interp(
        normalized_grid,
        tau_unique,
        ys,
    )

    theta_interp = np.interp(
        normalized_grid,
        tau_unique,
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
        direction = int(
            primitive[
                "turn_direction"
            ]
        )

        return (
            v_max,
            direction * omega_max,
        )

    if label == "turn on-the-spot":
        direction = int(
            primitive[
                "turn_direction"
            ]
        )

        return (
            0.0,
            direction * omega_max,
        )

    raise ValueError(
        f"Unknown analytical primitive label: {label!r}"
    )


def reconstruct_analytical_trajectory(
    case,
    samples_per_primitive=300,
):
    """
    Reconstruct the analytical pose trajectory from the saved primitive data.

    Each primitive is sampled using the exact constant-control unicycle
    solution.
    """

    primitives = case[
        "best_analytical_primitives"
    ]

    v_max = float(
        case["v_max"]
    )

    omega_max = float(
        case["omega_max"]
    )

    initial_pose = primitives[
        0
    ]["start_pose"]

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
            samples_per_primitive,
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
# GEOMETRIC DISTANCE
# =============================================================================

def discrete_symmetric_hausdorff_distance(
    analytical_x,
    analytical_y,
    ocp_x,
    ocp_y,
):
    """
    Compute the discrete symmetric Hausdorff distance between two sampled
    planar paths.
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

    directed_analytical_to_ocp = float(
        np.max(
            analytical_to_ocp
        )
    )

    directed_ocp_to_analytical = float(
        np.max(
            ocp_to_analytical
        )
    )

    symmetric_distance = max(
        directed_analytical_to_ocp,
        directed_ocp_to_analytical,
    )

    return (
        symmetric_distance,
        directed_analytical_to_ocp,
        directed_ocp_to_analytical,
    )


def symmetric_hausdorff_witness(
    analytical_x,
    analytical_y,
    ocp_x,
    ocp_y,
):
    """
    Identify the sampled point pair realizing the discrete symmetric
    Hausdorff distance.

    Returns
    -------
    dict
        Dictionary containing the distance, the source point, the closest
        target point, and the active directed-distance orientation.
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

    (
        analytical_to_ocp_distances,
        analytical_to_ocp_indices,
    ) = ocp_tree.query(
        analytical_points,
        k=1,
    )

    (
        ocp_to_analytical_distances,
        ocp_to_analytical_indices,
    ) = analytical_tree.query(
        ocp_points,
        k=1,
    )

    analytical_source_index = int(
        np.argmax(
            analytical_to_ocp_distances
        )
    )

    ocp_source_index = int(
        np.argmax(
            ocp_to_analytical_distances
        )
    )

    analytical_to_ocp_maximum = float(
        analytical_to_ocp_distances[
            analytical_source_index
        ]
    )

    ocp_to_analytical_maximum = float(
        ocp_to_analytical_distances[
            ocp_source_index
        ]
    )

    if (
        analytical_to_ocp_maximum
        >= ocp_to_analytical_maximum
    ):
        target_index = int(
            analytical_to_ocp_indices[
                analytical_source_index
            ]
        )

        distance = (
            analytical_to_ocp_maximum
        )

        source_point = analytical_points[
            analytical_source_index
        ]

        target_point = ocp_points[
            target_index
        ]

        active_direction = (
            "analytical_to_ocp"
        )

    else:
        target_index = int(
            ocp_to_analytical_indices[
                ocp_source_index
            ]
        )

        distance = (
            ocp_to_analytical_maximum
        )

        source_point = ocp_points[
            ocp_source_index
        ]

        target_point = analytical_points[
            target_index
        ]

        active_direction = (
            "ocp_to_analytical"
        )

    return {
        "distance": float(
            distance
        ),
        "source_point": np.asarray(
            source_point,
            dtype=float,
        ),
        "target_point": np.asarray(
            target_point,
            dtype=float,
        ),
        "active_direction": active_direction,
    }


# =============================================================================
# TRAJECTORY COMPARISON
# =============================================================================

def compare_case(
    case,
    n_comparison_points,
):
    """Compare one analytical trajectory with its saved OCP trajectory."""

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

    tau = np.linspace(
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
        tau,
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
        tau,
    )

    (
        hausdorff_distance,
        hausdorff_analytical_to_ocp,
        hausdorff_ocp_to_analytical,
    ) = discrete_symmetric_hausdorff_distance(
        analytical_x,
        analytical_y,
        ocp_x,
        ocp_y,
    )

    position_error = np.hypot(
        ocp_x
        - analytical_x,
        ocp_y
        - analytical_y,
    )

    heading_error = wrap_to_pi(
        ocp_theta
        - analytical_theta
    )

    absolute_heading_error = np.abs(
        heading_error
    )

    analytical_name = case[
        "best_analytical_name"
    ]

    return {
        "case_id": int(
            case[
                "case_id"
            ]
        ),
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
        "analytical_name": analytical_name,
        "analytical_family": get_analytical_family(
            analytical_name
        ),
        "ocp_sequence": case[
            "ocp_sequence"
        ],

        "hausdorff_distance": float(
            hausdorff_distance
        ),
        "hausdorff_analytical_to_ocp": float(
            hausdorff_analytical_to_ocp
        ),
        "hausdorff_ocp_to_analytical": float(
            hausdorff_ocp_to_analytical
        ),

        "position_mean_error": float(
            np.mean(
                position_error
            )
        ),
        "position_rms_error": float(
            np.sqrt(
                np.mean(
                    position_error**2
                )
            )
        ),
        "position_max_error": float(
            np.max(
                position_error
            )
        ),

        "heading_mean_abs_error_rad": float(
            np.mean(
                absolute_heading_error
            )
        ),
        "heading_rms_error_rad": float(
            np.sqrt(
                np.mean(
                    heading_error**2
                )
            )
        ),
        "heading_max_error_rad": float(
            np.max(
                absolute_heading_error
            )
        ),

        "heading_mean_abs_error_deg": float(
            np.degrees(
                np.mean(
                    absolute_heading_error
                )
            )
        ),
        "heading_rms_error_deg": float(
            np.degrees(
                np.sqrt(
                    np.mean(
                        heading_error**2
                    )
                )
            )
        ),
        "heading_max_error_deg": float(
            np.degrees(
                np.max(
                    absolute_heading_error
                )
            )
        ),
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_pose(
    axis,
    pose,
    marker_color,
    label,
):
    """Plot a planar pose using a point and a heading arrow."""

    x = float(
        pose[0]
    )

    y = float(
        pose[1]
    )

    theta = float(
        pose[2]
    )

    axis.plot(
        x,
        y,
        marker="o",
        markersize=POSE_MARKER_SIZE,
        markerfacecolor=marker_color,
        markeredgecolor=marker_color,
        linestyle="None",
        label=label,
        zorder=20,
    )

    axis.arrow(
        x,
        y,
        POSE_ARROW_LENGTH
        * cos(theta),
        POSE_ARROW_LENGTH
        * sin(theta),
        width=0.012,
        head_width=0.12,
        head_length=0.16,
        length_includes_head=True,
        color=marker_color,
        zorder=19,
    )


def plot_worst_hausdorff_case(
    case,
    transcription_resolution,
    figures_directory,
    n_comparison_points=N_COMPARISON_POINTS,
):
    """
    Plot the analytical and OCP paths for the case with the largest symmetric
    Hausdorff distance.

    The left panel shows the complete trajectories. The right panel shows an
    enlarged view of the region containing the sampled point pair that realizes
    the discrete symmetric Hausdorff distance.
    """

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

    witness = symmetric_hausdorff_witness(
        analytical_x_interp,
        analytical_y_interp,
        ocp_x_interp,
        ocp_y_interp,
    )

    hausdorff_distance = witness[
        "distance"
    ]

    source_point = witness[
        "source_point"
    ]

    target_point = witness[
        "target_point"
    ]

    midpoint = 0.5 * (
        source_point
        + target_point
    )

    zoom_half_width = max(
        MINIMUM_ZOOM_HALF_WIDTH,
        ZOOM_SCALE_FACTOR
        * hausdorff_distance,
    )

    zoom_x_min = (
        midpoint[0]
        - zoom_half_width
    )

    zoom_x_max = (
        midpoint[0]
        + zoom_half_width
    )

    zoom_y_min = (
        midpoint[1]
        - zoom_half_width
    )

    zoom_y_max = (
        midpoint[1]
        + zoom_half_width
    )

    start_pose = case[
        "best_analytical_primitives"
    ][0][
        "start_pose"
    ]

    end_pose = case[
        "best_analytical_primitives"
    ][-1][
        "end_pose"
    ]

    # =========================================================================
    # FIGURE LAYOUT
    # =========================================================================

    figure = plt.figure(
        figsize=(8.2, 6.2),
        constrained_layout=True,
    )

    grid = figure.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=(
            0.72,
            1.0,
        ),
        wspace=0.12,
    )

    path_axis = figure.add_subplot(
        grid[
            0,
            0,
        ]
    )

    zoom_axis = figure.add_subplot(
        grid[
            0,
            1,
        ]
    )

    # =========================================================================
    # LEFT PANEL: COMPLETE TRAJECTORIES
    # =========================================================================

    path_axis.plot(
        analytical_x_interp,
        analytical_y_interp,
        color=ANALYTICAL_COLOR,
        linestyle=ANALYTICAL_LINESTYLE,
        linewidth=ANALYTICAL_LINEWIDTH,
        label="Analytical",
        zorder=4,
    )

    path_axis.plot(
        ocp_x_interp,
        ocp_y_interp,
        color=OCP_COLOR,
        linestyle=OCP_LINESTYLE,
        linewidth=OCP_LINEWIDTH,
        label="OCP",
        zorder=5,
    )

    plot_pose(
        axis=path_axis,
        pose=start_pose,
        marker_color=INITIAL_POSE_COLOR,
        label="Initial pose",
    )

    plot_pose(
        axis=path_axis,
        pose=end_pose,
        marker_color=FINAL_POSE_COLOR,
        label="Final pose",
    )

    path_axis.set_xlabel(
        r"$x$ [m]"
    )

    path_axis.set_ylabel(
        r"$y$ [m]"
    )

    path_axis.set_aspect(
        "equal",
        adjustable="box",
    )

    path_axis.grid(
        True,
        color=GRID_COLOR,
        linestyle=GRID_LINESTYLE,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
        zorder=0,
    )

    path_axis.tick_params(
        direction="out"
    )

    zoom_rectangle = Rectangle(
        (
            zoom_x_min,
            zoom_y_min,
        ),
        2.0
        * zoom_half_width,
        2.0
        * zoom_half_width,
        fill=False,
        edgecolor=ZOOM_RECTANGLE_COLOR,
        linewidth=ZOOM_RECTANGLE_LINEWIDTH,
        linestyle="-",
        zorder=8,
    )

    path_axis.add_patch(
        zoom_rectangle
    )

    # =========================================================================
    # RIGHT PANEL: ENLARGED HAUSDORFF REGION
    # =========================================================================

    zoom_axis.plot(
        analytical_x_interp,
        analytical_y_interp,
        color=ANALYTICAL_COLOR,
        linestyle=ANALYTICAL_LINESTYLE,
        linewidth=ANALYTICAL_LINEWIDTH,
        label="Analytical",
        zorder=4,
    )

    zoom_axis.plot(
        ocp_x_interp,
        ocp_y_interp,
        color=OCP_COLOR,
        linestyle=OCP_LINESTYLE,
        linewidth=OCP_LINEWIDTH,
        label="OCP",
        zorder=5,
    )

    # Segment realizing the discrete symmetric Hausdorff distance.
    zoom_axis.plot(
        [
            source_point[0],
            target_point[0],
        ],
        [
            source_point[1],
            target_point[1],
        ],
        color=HAUSDORFF_COLOR,
        linestyle=HAUSDORFF_LINESTYLE,
        linewidth=HAUSDORFF_LINEWIDTH,
        zorder=8,
    )

    # Filled marker: source point of the active directed distance.
    zoom_axis.plot(
        source_point[0],
        source_point[1],
        marker="o",
        markersize=5.5,
        markerfacecolor=HAUSDORFF_COLOR,
        markeredgecolor=HAUSDORFF_COLOR,
        linestyle="None",
        zorder=9,
    )

    # Open marker: nearest point on the other path.
    zoom_axis.plot(
        target_point[0],
        target_point[1],
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor=HAUSDORFF_COLOR,
        markeredgewidth=1.3,
        linestyle="None",
        zorder=9,
    )

    zoom_axis.set_xlim(
        zoom_x_min,
        zoom_x_max,
    )

    zoom_axis.set_ylim(
        zoom_y_min,
        zoom_y_max,
    )

    zoom_axis.set_aspect(
        "equal",
        adjustable="box",
    )

    zoom_axis.set_xlabel(
        r"$x$ [m]"
    )

    zoom_axis.set_ylabel(
        r"$y$ [m]"
    )

    zoom_axis.grid(
        True,
        color=GRID_COLOR,
        linestyle=GRID_LINESTYLE,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
        zorder=0,
    )

    zoom_axis.tick_params(
        direction="out"
    )

    # =========================================================================
    # PANEL LABELS
    # =========================================================================

    path_axis.text(
        -0.18,
        1.01,
        "(a)",
        transform=path_axis.transAxes,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    zoom_axis.text(
        -0.12,
        1.01,
        "(b)",
        transform=zoom_axis.transAxes,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # =========================================================================
    # SHARED LEGEND
    # =========================================================================

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=ANALYTICAL_COLOR,
            linestyle=ANALYTICAL_LINESTYLE,
            linewidth=ANALYTICAL_LINEWIDTH,
            label="Analytical",
        ),
        Line2D(
            [0],
            [0],
            color=OCP_COLOR,
            linestyle=OCP_LINESTYLE,
            linewidth=OCP_LINEWIDTH,
            label="OCP",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor=INITIAL_POSE_COLOR,
            markeredgecolor=INITIAL_POSE_COLOR,
            linestyle="None",
            markersize=7,
            label="Initial pose",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor=FINAL_POSE_COLOR,
            markeredgecolor=FINAL_POSE_COLOR,
            linestyle="None",
            markersize=7,
            label="Final pose",
        ),
    ]

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.02,
        ),
        ncol=4,
        frameon=True,
        fontsize=9,
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_basename = (
        f"{WORST_HAUSDORFF_OUTPUT_BASENAME}"
        f"_N{transcription_resolution}"
    )

    pdf_path = (
        figures_directory
        / f"{output_basename}.pdf"
    )

    png_path = (
        figures_directory
        / f"{output_basename}.png"
    )

    if SAVE_WORST_HAUSDORFF_FIGURE:
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.05,
        )

        figure.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )

        print(
            "\nWorst Hausdorff-distance figure saved to:"
        )

        print(
            pdf_path
        )

        print(
            png_path
        )

    # =========================================================================
    # TERMINAL SUMMARY
    # =========================================================================

    print(
        "\nWorst Hausdorff-distance case"
    )

    print(
        "-" * 80
    )

    print(
        f"Case ID             : "
        f"{case['case_id']}"
    )

    print(
        f"theta0              : "
        f"{float(case['theta0_deg']):.1f} deg"
    )

    print(
        f"thetaf              : "
        f"{float(case['thetaf_deg']):.1f} deg"
    )

    print(
        f"Analytical solution : "
        f"{case['best_analytical_name']}"
    )

    print(
        f"Hausdorff distance  : "
        f"{hausdorff_distance:.9e} m"
    )

    print(
        f"Active direction    : "
        f"{witness['active_direction']}"
    )

    if SHOW_WORST_HAUSDORFF_FIGURE:
        plt.show(
            block=True
        )

    else:
        plt.close(
            figure
        )

    return figure


# =============================================================================
# STATISTICS
# =============================================================================

def compute_statistics(
    values,
):
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
            np.mean(
                values
            )
        ),
        "median": float(
            np.median(
                values
            )
        ),
        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),
        "maximum": float(
            np.max(
                values
            )
        ),
    }


def print_statistics(
    name,
    values,
    unit,
):
    """Print one set of summary statistics."""

    statistics = compute_statistics(
        values
    )

    print(
        f"\n{name}"
    )

    print(
        "-" * 80
    )

    if statistics is None:
        print(
            "No values available."
        )
        return

    print(
        f"Cases   : "
        f"{statistics['count']}"
    )

    print(
        f"Mean    : "
        f"{statistics['mean']:.6e} {unit}"
    )

    print(
        f"Median  : "
        f"{statistics['median']:.6e} {unit}"
    )

    print(
        f"95th pct: "
        f"{statistics['p95']:.6e} {unit}"
    )

    print(
        f"Maximum : "
        f"{statistics['maximum']:.6e} {unit}"
    )


def print_hausdorff_statistics_by_family(
    comparison_results,
):
    """
    Print symmetric Hausdorff-distance statistics grouped by analytical family.
    """

    print(
        "\n"
        + "=" * 108
    )

    print(
        "SYMMETRIC HAUSDORFF DISTANCE BY ANALYTICAL FAMILY"
    )

    print(
        "=" * 108
    )

    header = (
        f"{'Family':<8}"
        f"{'Cases':>8}"
        f"{'Mean [m]':>20}"
        f"{'Median [m]':>20}"
        f"{'95th pct [m]':>20}"
        f"{'Maximum [m]':>20}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    statistics_by_family = {}

    for family in FAMILY_ORDER:
        values = [
            result[
                "hausdorff_distance"
            ]
            for result in comparison_results
            if result[
                "analytical_family"
            ] == family
        ]

        statistics = compute_statistics(
            values
        )

        statistics_by_family[
            family
        ] = statistics

        if statistics is None:
            print(
                f"{family:<8}"
                f"{0:>8}"
                f"{'--':>20}"
                f"{'--':>20}"
                f"{'--':>20}"
                f"{'--':>20}"
            )

            continue

        print(
            f"{family:<8}"
            f"{statistics['count']:>8d}"
            f"{statistics['mean']:>20.6e}"
            f"{statistics['median']:>20.6e}"
            f"{statistics['p95']:>20.6e}"
            f"{statistics['maximum']:>20.6e}"
        )

    print(
        "\nLaTeX table rows"
    )

    print(
        "-" * 108
    )

    for family in FAMILY_ORDER:
        statistics = statistics_by_family[
            family
        ]

        if statistics is None:
            print(
                rf"\({family}\)"
                rf" & \(0\)"
                rf" & --"
                rf" & --"
                rf" & --"
                rf" & -- \\"
            )

            continue

        print(
            rf"\({family}\)"
            rf" & \({statistics['count']}\)"
            rf" & \({statistics['mean']:.2e}\)"
            rf" & \({statistics['median']:.2e}\)"
            rf" & \({statistics['p95']:.2e}\)"
            rf" & \({statistics['maximum']:.2e}\)"
            rf" \\"
        )

    return statistics_by_family


# =============================================================================
# FILE PROCESSING
# =============================================================================

def process_results_file(
    results_file,
    n_comparison_points,
):
    """Load and compare all valid cases from one result file."""

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

    print(
        "=" * 80
    )

    print(
        "ANALYTICAL / OCP POSE-TRAJECTORY COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        f"Results file : "
        f"{results_file}"
    )

    print(
        f"Sweep ID     : "
        f"{metadata.get('sweep_id')}"
    )

    print(
        f"N            : "
        f"{metadata.get('N')}"
    )

    print(
        f"M            : "
        f"{metadata.get('M')}"
    )

    print(
        f"Cases loaded : "
        f"{len(cases)}"
    )

    comparison_results = []
    skipped_cases = []
    valid_cases_by_id = {}

    required_fields = (
        "ocp_time_grid",
        "ocp_x",
        "ocp_y",
        "ocp_theta",
        "best_analytical_primitives",
    )

    for case in cases:
        if not case.get(
            "success",
            False,
        ):
            skipped_cases.append(
                case.get(
                    "case_id"
                )
            )
            continue

        if not case.get(
            "ocp_success",
            False,
        ):
            skipped_cases.append(
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
            skipped_cases.append(
                case.get(
                    "case_id"
                )
            )
            continue

        result = compare_case(
            case,
            n_comparison_points,
        )

        comparison_results.append(
            result
        )

        valid_cases_by_id[
            int(
                case[
                    "case_id"
                ]
            )
        ] = case

    print(
        f"Cases compared: "
        f"{len(comparison_results)}"
    )

    print(
        f"Cases skipped : "
        f"{len(skipped_cases)}"
    )

    if skipped_cases:
        print(
            f"Skipped IDs   : "
            f"{skipped_cases[:20]}"
        )

    return (
        metadata,
        comparison_results,
        skipped_cases,
        valid_cases_by_id,
    )


def print_overall_metric_statistics(
    comparison_results,
):
    """Print all overall path and heading metrics."""

    hausdorff_distances = [
        result[
            "hausdorff_distance"
        ]
        for result in comparison_results
    ]

    position_mean_errors = [
        result[
            "position_mean_error"
        ]
        for result in comparison_results
    ]

    position_rms_errors = [
        result[
            "position_rms_error"
        ]
        for result in comparison_results
    ]

    position_max_errors = [
        result[
            "position_max_error"
        ]
        for result in comparison_results
    ]

    heading_mean_errors = [
        result[
            "heading_mean_abs_error_deg"
        ]
        for result in comparison_results
    ]

    heading_rms_errors = [
        result[
            "heading_rms_error_deg"
        ]
        for result in comparison_results
    ]

    heading_max_errors = [
        result[
            "heading_max_error_deg"
        ]
        for result in comparison_results
    ]

    print_statistics(
        "SYMMETRIC HAUSDORFF DISTANCE",
        hausdorff_distances,
        "m",
    )

    print_statistics(
        "MEAN PLANAR DEVIATION",
        position_mean_errors,
        "m",
    )

    print_statistics(
        "RMS PLANAR DEVIATION",
        position_rms_errors,
        "m",
    )

    print_statistics(
        "MAXIMUM PLANAR DEVIATION",
        position_max_errors,
        "m",
    )

    print_statistics(
        "MEAN ABSOLUTE HEADING DEVIATION",
        heading_mean_errors,
        "deg",
    )

    print_statistics(
        "RMS HEADING DEVIATION",
        heading_rms_errors,
        "deg",
    )

    print_statistics(
        "MAXIMUM HEADING DEVIATION",
        heading_max_errors,
        "deg",
    )


def print_worst_cases(
    comparison_results,
):
    """Print the worst cases by symmetric Hausdorff distance."""

    worst_hausdorff_cases = sorted(
        comparison_results,
        key=lambda result: result[
            "hausdorff_distance"
        ],
        reverse=True,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "WORST CASES BY SYMMETRIC HAUSDORFF DISTANCE"
    )

    print(
        "=" * 80
    )

    for result in worst_hausdorff_cases[
        :N_WORST_CASES_TO_PRINT
    ]:
        sequence = " - ".join(
            result[
                "ocp_sequence"
            ]
        )

        print(
            f"\nCase {result['case_id']}: "
            f"theta0={result['theta0_deg']:.1f} deg, "
            f"thetaf={result['thetaf_deg']:.1f} deg"
        )

        print(
            f"  Analytical family : "
            f"{result['analytical_name']}"
        )

        print(
            f"  OCP sequence      : "
            f"{sequence}"
        )

        print(
            f"  Hausdorff distance: "
            f"{result['hausdorff_distance']:.6e} m"
        )

        print(
            f"  Analytical -> OCP : "
            f"{result['hausdorff_analytical_to_ocp']:.6e} m"
        )

        print(
            f"  OCP -> analytical : "
            f"{result['hausdorff_ocp_to_analytical']:.6e} m"
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    all_results = {}

    for transcription_resolution in (
        30,
        100,
    ):
        results_file = RESULTS_FILES[
            transcription_resolution
        ]

        print(
            "\n"
            + "#" * 90
        )

        print(
            f"PROCESSING TRANSCRIPTION RESOLUTION "
            f"N={transcription_resolution}"
        )

        print(
            "#" * 90
        )

        (
            metadata,
            comparison_results,
            skipped_cases,
            valid_cases_by_id,
        ) = process_results_file(
            results_file=results_file,
            n_comparison_points=N_COMPARISON_POINTS,
        )

        all_results[
            transcription_resolution
        ] = {
            "metadata": metadata,
            "comparison_results": comparison_results,
            "skipped_cases": skipped_cases,
            "valid_cases_by_id": valid_cases_by_id,
        }

        print_overall_metric_statistics(
            comparison_results=(
                comparison_results
            ),
        )

        print_hausdorff_statistics_by_family(
            comparison_results=(
                comparison_results
            ),
        )

        print_worst_cases(
            comparison_results=(
                comparison_results
            ),
        )

    # -------------------------------------------------------------------------
    # Compact cross-resolution summary for the thesis table
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 108
    )

    print(
        "SYMMETRIC HAUSDORFF DISTANCE: N=30 VS N=100"
    )

    print(
        "=" * 108
    )

    header = (
        f"{'N':<8}"
        f"{'Cases':>8}"
        f"{'Mean [m]':>20}"
        f"{'Median [m]':>20}"
        f"{'95th pct [m]':>20}"
        f"{'Maximum [m]':>20}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    overall_statistics = {}

    for transcription_resolution in (
        30,
        100,
    ):
        comparison_results = all_results[
            transcription_resolution
        ][
            "comparison_results"
        ]

        values = [
            result[
                "hausdorff_distance"
            ]
            for result in comparison_results
        ]

        statistics = compute_statistics(
            values
        )

        overall_statistics[
            transcription_resolution
        ] = statistics

        print(
            f"{transcription_resolution:<8d}"
            f"{statistics['count']:>8d}"
            f"{statistics['mean']:>20.6e}"
            f"{statistics['median']:>20.6e}"
            f"{statistics['p95']:>20.6e}"
            f"{statistics['maximum']:>20.6e}"
        )

    print(
        "\nLaTeX table rows"
    )

    print(
        "-" * 108
    )

    for transcription_resolution in (
        30,
        100,
    ):
        statistics = overall_statistics[
            transcription_resolution
        ]

        print(
            rf"\({transcription_resolution}\)"
            rf" & \({statistics['mean']:.2e}\)"
            rf" & \({statistics['median']:.2e}\)"
            rf" & \({statistics['p95']:.2e}\)"
            rf" & \({statistics['maximum']:.2e}\)"
            rf" \\"
        )

    # -------------------------------------------------------------------------
    # Plot the worst symmetric-Hausdorff case
    # -------------------------------------------------------------------------

    if PLOT_WORST_HAUSDORFF_CASE:
        transcription_resolution = (
            WORST_HAUSDORFF_RESOLUTION
        )

        selected_results = all_results[
            transcription_resolution
        ]

        worst_result = max(
            selected_results[
                "comparison_results"
            ],
            key=lambda result: result[
                "hausdorff_distance"
            ],
        )

        worst_case_id = int(
            worst_result[
                "case_id"
            ]
        )

        worst_case = selected_results[
            "valid_cases_by_id"
        ][
            worst_case_id
        ]

        figures_directory = Path(
            "experiments/pose2pose_unicycle/figures"
        )

        plot_worst_hausdorff_case(
            case=worst_case,
            transcription_resolution=(
                transcription_resolution
            ),
            figures_directory=(
                figures_directory
            ),
        )