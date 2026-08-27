import json
from math import degrees, pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from kappa_planner.geometry import Point, Pose
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.plot_helpers import (
    plot_analytical_trajectory,
    plot_velocity_profiles_comparison,
)
from kappa_planner.helpers.pose_to_pose_unicycle import (
    compute_all_pose_to_pose_trajectories,
)
from kappa_planner.helpers.ocp_pose_to_pose_unicycle import (
    compute_ocp_pose_to_pose_trajectory,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

THETA0_DEG = 288.0
THETAF_DEG = 252.0

TRANSCRIPTION_RESOLUTION = 100

RESULTS_FILENAME = (
    "experiments/pose2pose_unicycle/results/"
    "orientation_sweep_N100_M4.json"
)

SHOW_ALL_ANALYTICAL_CANDIDATES = False
SHOW_CONTROL_COMPARISON = False

# Optional narrow y-range used to inspect the approximately straight section.
STRAIGHT_SECTION_Y_MIN = 1.5
STRAIGHT_SECTION_Y_MAX = 3.5

ANALYTICAL_COLOR = "tab:blue"
SAVED_OCP_COLOR = "tab:orange"
RECOMPUTED_OCP_COLOR = "black"

ANALYTICAL_LINEWIDTH = 2.5
OCP_LINEWIDTH = 2.2


# =============================================================================
# TRAJECTORY INFORMATION
# =============================================================================

def get_trajectory_info(trajectory):
    """
    Return useful information about a trajectory composed of motion primitives.
    """

    info = {
        "primitive_sequence": [],
        "primitives": [],
        "total_duration": 0.0,
    }

    for index, primitive in enumerate(
        trajectory,
        start=1,
    ):
        primitive_info = {
            "index": index,
            "type": primitive.label,
            "duration": float(
                primitive.maneuver_time
            ),
        }

        info["primitive_sequence"].append(
            primitive.label
        )

        info["total_duration"] += float(
            primitive.maneuver_time
        )

        if primitive.label == "turn on-the-spot":
            primitive_info.update({
                "turn_angle_rad": float(
                    primitive.delta_angle
                ),
                "turn_angle_deg": degrees(
                    float(
                        primitive.delta_angle
                    )
                ),
                "turn_direction": primitive.turn_direction,
            })

        elif primitive.label == "arc":
            primitive_info.update({
                "arc_angle_rad": float(
                    primitive.iota
                ),
                "arc_angle_deg": degrees(
                    float(
                        primitive.iota
                    )
                ),
                "turn_direction": primitive.turn_direction,
                "radius": float(
                    primitive.radius
                ),
                "arc_length": float(
                    primitive.path_length
                ),
            })

        elif primitive.label == "segment":
            primitive_info.update({
                "length": float(
                    primitive.path_length
                ),
            })

        info["primitives"].append(
            primitive_info
        )

    return info


def print_trajectory_info(
    name,
    trajectory,
):
    """Print the primitive structure of one analytical trajectory."""

    info = get_trajectory_info(
        trajectory
    )

    print(f"\n{name}")
    print("-" * len(name))

    print(
        "Primitive sequence: "
        + " -> ".join(
            info["primitive_sequence"]
        )
    )

    print(
        f"Total duration: "
        f"{info['total_duration']:.9f} s"
    )

    for primitive in info["primitives"]:
        print(
            f"\nPrimitive {primitive['index']}: "
            f"{primitive['type']}"
        )

        print(
            f"  Duration       : "
            f"{primitive['duration']:.9f} s"
        )

        if primitive["type"] == "turn on-the-spot":
            print(
                f"  Turn angle     : "
                f"{primitive['turn_angle_rad']:.9f} rad "
                f"({primitive['turn_angle_deg']:.6f} deg)"
            )

            print(
                f"  Turn direction : "
                f"{primitive['turn_direction']}"
            )

        elif primitive["type"] == "arc":
            print(
                f"  Arc amplitude  : "
                f"{primitive['arc_angle_rad']:.9f} rad "
                f"({primitive['arc_angle_deg']:.6f} deg)"
            )

            print(
                f"  Turn direction : "
                f"{primitive['turn_direction']}"
            )

            print(
                f"  Radius         : "
                f"{primitive['radius']:.9f} m"
            )

            print(
                f"  Arc length     : "
                f"{primitive['arc_length']:.9f} m"
            )

        elif primitive["type"] == "segment":
            print(
                f"  Segment length : "
                f"{primitive['length']:.9f} m"
            )


def tau_to_string(tau):
    """Convert a turn direction to a short string."""

    if tau == 1:
        return "L"

    if tau == -1:
        return "R"

    return str(tau)


def get_primitive_amplitudes_string(
    trajectory,
):
    """Build a compact description of the primitive amplitudes."""

    parts = []

    for primitive in trajectory:
        if primitive.label == "turn on-the-spot":
            parts.append(
                f"T={degrees(abs(primitive.delta_angle)):.1f}°"
            )

        elif primitive.label == "arc":
            parts.append(
                f"C={degrees(abs(primitive.iota)):.1f}°"
            )

        elif primitive.label == "segment":
            parts.append(
                f"S={primitive.path_length:.2f} m"
            )

    return ", ".join(parts)


def plot_all_trajectories_grid(
    trajectories,
    rows=4,
    cols=4,
):
    """Plot all analytical candidates in a grid."""

    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(18, 16),
    )

    axes = axes.flatten()

    best_name = min(
        trajectories,
        key=lambda name: trajectories[name]["time"],
    )

    for axis, (
        name,
        data,
    ) in zip(
        axes,
        trajectories.items(),
    ):
        trajectory = data["trajectory"]
        total_time = float(
            data["time"]
        )

        trajectory_type = data["type"]
        tau0 = data["tau0"]
        tauf = data["tauf"]

        is_best = (
            name == best_name
        )

        color = (
            "tab:red"
            if is_best
            else "tab:blue"
        )

        linewidth = (
            3.0
            if is_best
            else 1.8
        )

        plot_analytical_trajectory(
            trajectory,
            figure=axis,
            plot_circles=True,
            color=color,
            linewidth=linewidth,
            plot_primitive_arrows=True,
            plot_turn_sectors=True,
        )

        amplitudes = (
            get_primitive_amplitudes_string(
                trajectory
            )
        )

        title = (
            f"{trajectory_type} "
            f"{tau_to_string(tau0)}-"
            f"{tau_to_string(tauf)}\n"
            f"time = {total_time:.3f} s\n"
            f"{amplitudes}"
        )

        if is_best:
            title = (
                "BEST\n"
                + title
            )

        axis.set_title(
            title,
            fontsize=9,
        )

        axis.set_aspect(
            "equal",
            adjustable="box",
        )

        axis.grid(
            True
        )

    for axis in axes[
        len(trajectories):
    ]:
        axis.axis(
            "off"
        )

    figure.tight_layout()

    return figure


# =============================================================================
# SAVED CASE
# =============================================================================

def load_saved_case(
    results_filename,
    theta0_deg,
    thetaf_deg,
    angle_tolerance=1.0e-9,
):
    """
    Load the sweep result matching a prescribed pair of boundary orientations.

    The case is identified by its saved orientation values, rather than by a
    case ID.
    """

    results_path = Path(
        results_filename
    )

    if not results_path.exists():
        raise FileNotFoundError(
            "The sweep result file does not exist:\n"
            f"{results_path.resolve()}"
        )

    with open(
        results_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    metadata = data[
        "metadata"
    ]

    results = data[
        "results"
    ]

    matching_cases = [
        case
        for case in results
        if np.isclose(
            float(
                case["theta0_deg"]
            ),
            theta0_deg,
            atol=angle_tolerance,
            rtol=0.0,
        )
        and np.isclose(
            float(
                case["thetaf_deg"]
            ),
            thetaf_deg,
            atol=angle_tolerance,
            rtol=0.0,
        )
    ]

    if len(matching_cases) == 0:
        raise RuntimeError(
            "No saved sweep case was found for "
            f"theta0={theta0_deg} deg and "
            f"thetaf={thetaf_deg} deg."
        )

    if len(matching_cases) > 1:
        case_ids = [
            case.get(
                "case_id"
            )
            for case in matching_cases
        ]

        raise RuntimeError(
            "More than one saved case matches the requested orientations. "
            f"Matching case IDs: {case_ids}"
        )

    saved_case = matching_cases[
        0
    ]

    print("\n" + "=" * 80)
    print("SAVED SWEEP CASE")
    print("=" * 80)

    print(
        f"Results file          : "
        f"{results_path}"
    )

    print(
        f"Metadata N            : "
        f"{metadata.get('N')}"
    )

    print(
        f"Metadata M            : "
        f"{metadata.get('M')}"
    )

    print(
        f"Case ID               : "
        f"{saved_case.get('case_id')}"
    )

    print(
        f"theta0                : "
        f"{float(saved_case['theta0_deg']):.9f} deg"
    )

    print(
        f"thetaf                : "
        f"{float(saved_case['thetaf_deg']):.9f} deg"
    )

    print(
        f"Saved analytical name : "
        f"{saved_case.get('best_analytical_name')}"
    )

    print(
        f"Saved analytical time : "
        f"{float(saved_case['best_analytical_time']):.9f} s"
    )

    print(
        f"Saved OCP time        : "
        f"{float(saved_case['ocp_time']):.9f} s"
    )

    print(
        f"Saved case N          : "
        f"{saved_case.get('N')}"
    )

    print(
        f"Saved case M          : "
        f"{saved_case.get('M')}"
    )

    print(
        f"Saved OCP sequence    : "
        f"{' - '.join(saved_case.get('ocp_sequence', []))}"
    )

    return (
        metadata,
        saved_case,
    )


# =============================================================================
# GEOMETRIC COMPARISON
# =============================================================================

def symmetric_hausdorff_distance(
    x_a,
    y_a,
    x_b,
    y_b,
):
    """Compute the discrete symmetric Hausdorff distance."""

    points_a = np.column_stack(
        (
            np.asarray(
                x_a,
                dtype=float,
            ),
            np.asarray(
                y_a,
                dtype=float,
            ),
        )
    )

    points_b = np.column_stack(
        (
            np.asarray(
                x_b,
                dtype=float,
            ),
            np.asarray(
                y_b,
                dtype=float,
            ),
        )
    )

    tree_a = cKDTree(
        points_a
    )

    tree_b = cKDTree(
        points_b
    )

    distances_a_to_b, _ = tree_b.query(
        points_a,
        k=1,
    )

    distances_b_to_a, _ = tree_a.query(
        points_b,
        k=1,
    )

    directed_a_to_b = float(
        np.max(
            distances_a_to_b
        )
    )

    directed_b_to_a = float(
        np.max(
            distances_b_to_a
        )
    )

    return {
        "symmetric": max(
            directed_a_to_b,
            directed_b_to_a,
        ),
        "a_to_b": directed_a_to_b,
        "b_to_a": directed_b_to_a,
    }


def interpolate_xy_on_normalized_time(
    times,
    xs,
    ys,
    normalized_grid,
):
    """
    Interpolate an x-y trajectory on a common normalized-time grid.
    """

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

    if not (
        len(times)
        == len(xs)
        == len(ys)
    ):
        raise ValueError(
            "Time, x, and y arrays must have equal lengths."
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

    normalized_times, unique_indices = np.unique(
        normalized_times,
        return_index=True,
    )

    xs = xs[
        unique_indices
    ]

    ys = ys[
        unique_indices
    ]

    x_interp = np.interp(
        normalized_grid,
        normalized_times,
        xs,
    )

    y_interp = np.interp(
        normalized_grid,
        normalized_times,
        ys,
    )

    return (
        x_interp,
        y_interp,
    )


def print_straight_section_statistics(
    name,
    xs,
    ys,
):
    """
    Print approximate x statistics over the selected central y-range.
    """

    xs = np.asarray(
        xs,
        dtype=float,
    )

    ys = np.asarray(
        ys,
        dtype=float,
    )

    mask = (
        (ys >= STRAIGHT_SECTION_Y_MIN)
        & (ys <= STRAIGHT_SECTION_Y_MAX)
    )

    print(f"\n{name}")
    print("-" * 80)

    if not np.any(
        mask
    ):
        print(
            "No trajectory samples lie in the selected y-range."
        )
        return

    selected_x = xs[
        mask
    ]

    print(
        f"Selected y-range : "
        f"[{STRAIGHT_SECTION_Y_MIN:.3f}, "
        f"{STRAIGHT_SECTION_Y_MAX:.3f}] m"
    )

    print(
        f"Samples          : "
        f"{selected_x.size}"
    )

    print(
        f"Mean x           : "
        f"{np.mean(selected_x):.9f} m"
    )

    print(
        f"Median x         : "
        f"{np.median(selected_x):.9f} m"
    )

    print(
        f"Minimum x        : "
        f"{np.min(selected_x):.9f} m"
    )

    print(
        f"Maximum x        : "
        f"{np.max(selected_x):.9f} m"
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Load the original saved sweep case
    # -------------------------------------------------------------------------

    metadata, saved_case = load_saved_case(
        results_filename=RESULTS_FILENAME,
        theta0_deg=THETA0_DEG,
        thetaf_deg=THETAF_DEG,
    )

    # -------------------------------------------------------------------------
    # Reconstruct the same boundary-value problem
    # -------------------------------------------------------------------------

    start_pose = Pose(
        Point(
            0.0,
            0.0,
        ),
        THETA0_DEG
        * pi
        / 180.0,
    )

    end_pose = Pose(
        Point(
            0.0,
            5.0,
        ),
        THETAF_DEG
        * pi
        / 180.0,
    )

    vehicle_width = 0.430
    vehicle_length = 0.430
    vehicle_vmax = 1.0
    vehicle_omegamax = 1.0

    unicycle = Unicycle(
        state=[
            0.0,
            0.0,
            0.0,
        ],
        width=vehicle_width,
        length=vehicle_length,
        v_max=vehicle_vmax,
        v_min=0.0,
        omega_max=vehicle_omegamax,
        omega_min=-vehicle_omegamax,
    )

    # -------------------------------------------------------------------------
    # Recompute all analytical candidates
    # -------------------------------------------------------------------------

    trajectories = compute_all_pose_to_pose_trajectories(
        start_pose,
        end_pose,
        unicycle,
    )

    if SHOW_ALL_ANALYTICAL_CANDIDATES:
        plot_all_trajectories_grid(
            trajectories
        )

    best_name, best_data = min(
        trajectories.items(),
        key=lambda item: item[1]["time"],
    )

    best_trajectory = best_data[
        "trajectory"
    ]

    best_time = float(
        best_data[
            "time"
        ]
    )

    print("\n" + "=" * 80)
    print("RECOMPUTED ANALYTICAL SOLUTION")
    print("=" * 80)

    print(
        f"Best analytical name : "
        f"{best_name}"
    )

    print(
        f"Best analytical time : "
        f"{best_time:.9f} s"
    )

    print(
        f"Saved analytical name: "
        f"{saved_case['best_analytical_name']}"
    )

    print(
        f"Saved analytical time: "
        f"{float(saved_case['best_analytical_time']):.9f} s"
    )

    print(
        f"Analytical time diff : "
        f"{best_time - float(saved_case['best_analytical_time']):.9e} s"
    )

    print_trajectory_info(
        best_name,
        best_trajectory,
    )

    # -------------------------------------------------------------------------
    # Recompute the OCP
    # -------------------------------------------------------------------------

    ocp_result = compute_ocp_pose_to_pose_trajectory(
        start_pose,
        end_pose,
        unicycle,
        analytical_initial_guess=best_trajectory,
        T_guess=best_time,
        N=TRANSCRIPTION_RESOLUTION,
        M = 4,
    )

    print("\n" + "=" * 80)
    print("RECOMPUTED OCP SOLUTION")
    print("=" * 80)

    print(
        f"Recomputed OCP time : "
        f"{float(ocp_result['time']):.9f} s"
    )

    print(
        f"Saved OCP time      : "
        f"{float(saved_case['ocp_time']):.9f} s"
    )

    print(
        f"OCP time difference : "
        f"{float(ocp_result['time']) - float(saved_case['ocp_time']):.9e} s"
    )

    print(
        f"Recomputed sequence : "
        f"{' - '.join(ocp_result['sequence'])}"
    )

    print(
        f"Saved sequence      : "
        f"{' - '.join(saved_case['ocp_sequence'])}"
    )

    print(
        f"Recomputed solve time: "
        f"{float(ocp_result['solve_time']):.9f} s"
    )

    # -------------------------------------------------------------------------
    # Extract paths
    # -------------------------------------------------------------------------

    saved_ocp_x = np.asarray(
        saved_case[
            "ocp_x"
        ],
        dtype=float,
    )

    saved_ocp_y = np.asarray(
        saved_case[
            "ocp_y"
        ],
        dtype=float,
    )

    saved_ocp_times = np.asarray(
        saved_case[
            "ocp_time_grid"
        ],
        dtype=float,
    )

    recomputed_ocp_x = np.asarray(
        ocp_result[
            "xs"
        ],
        dtype=float,
    )

    recomputed_ocp_y = np.asarray(
        ocp_result[
            "ys"
        ],
        dtype=float,
    )

    # The recomputed state grid may be saved with a key different from the
    # control grid. Prefer a state-time grid when available.
    if "ts" in ocp_result:
        recomputed_ocp_times = np.asarray(
            ocp_result[
                "ts"
            ],
            dtype=float,
        )

    elif "time_grid" in ocp_result:
        recomputed_ocp_times = np.asarray(
            ocp_result[
                "time_grid"
            ],
            dtype=float,
        )

    elif len(
        ocp_result.get(
            "ts_ctrl",
            [],
        )
    ) == len(
        recomputed_ocp_x
    ):
        recomputed_ocp_times = np.asarray(
            ocp_result[
                "ts_ctrl"
            ],
            dtype=float,
        )

    else:
        recomputed_ocp_times = np.linspace(
            0.0,
            float(
                ocp_result[
                    "time"
                ]
            ),
            len(
                recomputed_ocp_x
            ),
        )

        print(
            "\nWarning: no explicit recomputed OCP state-time grid was found. "
            "A uniform state-time grid was constructed for comparison."
        )

    # -------------------------------------------------------------------------
    # Compare the saved and recomputed OCP paths
    # -------------------------------------------------------------------------

    normalized_grid = np.linspace(
        0.0,
        1.0,
        2001,
    )

    (
        saved_ocp_x_interp,
        saved_ocp_y_interp,
    ) = interpolate_xy_on_normalized_time(
        saved_ocp_times,
        saved_ocp_x,
        saved_ocp_y,
        normalized_grid,
    )

    (
        recomputed_ocp_x_interp,
        recomputed_ocp_y_interp,
    ) = interpolate_xy_on_normalized_time(
        recomputed_ocp_times,
        recomputed_ocp_x,
        recomputed_ocp_y,
        normalized_grid,
    )

    saved_vs_recomputed_pointwise_error = np.hypot(
        saved_ocp_x_interp
        - recomputed_ocp_x_interp,
        saved_ocp_y_interp
        - recomputed_ocp_y_interp,
    )

    saved_vs_recomputed_hausdorff = (
        symmetric_hausdorff_distance(
            saved_ocp_x_interp,
            saved_ocp_y_interp,
            recomputed_ocp_x_interp,
            recomputed_ocp_y_interp,
        )
    )

    print("\n" + "=" * 80)
    print("SAVED OCP VS RECOMPUTED OCP")
    print("=" * 80)

    print(
        f"Mean normalized-time planar difference : "
        f"{np.mean(saved_vs_recomputed_pointwise_error):.9e} m"
    )

    print(
        f"RMS normalized-time planar difference  : "
        f"{np.sqrt(np.mean(saved_vs_recomputed_pointwise_error**2)):.9e} m"
    )

    print(
        f"Max normalized-time planar difference  : "
        f"{np.max(saved_vs_recomputed_pointwise_error):.9e} m"
    )

    print(
        f"Symmetric Hausdorff distance            : "
        f"{saved_vs_recomputed_hausdorff['symmetric']:.9e} m"
    )

    print(
        f"Saved -> recomputed directed distance   : "
        f"{saved_vs_recomputed_hausdorff['a_to_b']:.9e} m"
    )

    print(
        f"Recomputed -> saved directed distance   : "
        f"{saved_vs_recomputed_hausdorff['b_to_a']:.9e} m"
    )

    # -------------------------------------------------------------------------
    # Inspect the central approximately straight portion
    # -------------------------------------------------------------------------

    print_straight_section_statistics(
        "SAVED OCP: CENTRAL PATH SECTION",
        saved_ocp_x,
        saved_ocp_y,
    )

    print_straight_section_statistics(
        "RECOMPUTED OCP: CENTRAL PATH SECTION",
        recomputed_ocp_x,
        recomputed_ocp_y,
    )

    # -------------------------------------------------------------------------
    # Plot analytical, saved OCP, and recomputed OCP paths together
    # -------------------------------------------------------------------------

    comparison_axis = plot_analytical_trajectory(
        best_trajectory,
        plot_circles=False,
        color=ANALYTICAL_COLOR,
        linewidth=ANALYTICAL_LINEWIDTH,
        plot_primitive_arrows=False,
        plot_turn_sectors=False,
    )

    comparison_axis.plot(
        [],
        [],
        color=ANALYTICAL_COLOR,
        linestyle="-",
        linewidth=ANALYTICAL_LINEWIDTH,
        label="Analytical",
    )

    comparison_axis.plot(
        saved_ocp_x,
        saved_ocp_y,
        color=SAVED_OCP_COLOR,
        linestyle="--",
        linewidth=OCP_LINEWIDTH,
        label="Saved OCP",
        zorder=8,
    )

    comparison_axis.plot(
        recomputed_ocp_x,
        recomputed_ocp_y,
        color=RECOMPUTED_OCP_COLOR,
        linestyle=":",
        linewidth=OCP_LINEWIDTH,
        label="Recomputed OCP",
        zorder=9,
    )

    comparison_axis.set_xlabel(
        r"$x$ [m]"
    )

    comparison_axis.set_ylabel(
        r"$y$ [m]"
    )

    comparison_axis.set_aspect(
        "equal",
        adjustable="box",
    )

    comparison_axis.grid(
        True,
        linestyle=":",
        alpha=0.6,
    )

    comparison_axis.legend(
        frameon=True
    )

    comparison_axis.set_title(
        "Analytical, saved OCP, and recomputed OCP paths"
    )

    # -------------------------------------------------------------------------
    # Zoomed comparison of the region used in the Hausdorff figure
    # -------------------------------------------------------------------------

    zoom_figure, zoom_axis = plt.subplots(
        figsize=(8.5, 5.5)
    )

    zoom_axis.plot(
        saved_ocp_x,
        saved_ocp_y,
        color=SAVED_OCP_COLOR,
        linestyle="--",
        linewidth=OCP_LINEWIDTH,
        label="Saved OCP",
    )

    zoom_axis.plot(
        recomputed_ocp_x,
        recomputed_ocp_y,
        color=RECOMPUTED_OCP_COLOR,
        linestyle=":",
        linewidth=OCP_LINEWIDTH,
        label="Recomputed OCP",
    )

    # The analytical straight section is at approximately x = 1 m.
    zoom_axis.axvline(
        1.0,
        color=ANALYTICAL_COLOR,
        linestyle="-",
        linewidth=ANALYTICAL_LINEWIDTH,
        label="Analytical segment",
    )

    zoom_axis.set_xlim(
        0.82,
        1.16,
    )

    zoom_axis.set_ylim(
        3.77,
        4.03,
    )

    zoom_axis.set_xlabel(
        r"$x$ [m]"
    )

    zoom_axis.set_ylabel(
        r"$y$ [m]"
    )

    zoom_axis.grid(
        True,
        linestyle=":",
        alpha=0.6,
    )

    zoom_axis.legend(
        frameon=True
    )

    zoom_axis.set_title(
        "Zoomed comparison of saved and recomputed OCP paths"
    )

    zoom_figure.tight_layout()

    # -------------------------------------------------------------------------
    # Control comparison for the recomputed solution
    # -------------------------------------------------------------------------

    if SHOW_CONTROL_COMPARISON:
        plot_velocity_profiles_comparison(
            best_trajectory,
            ocp_result,
            vehicle=unicycle,
            analytical_label="Analytical",
            ocp_label="Recomputed OCP",
            analytical_color=ANALYTICAL_COLOR,
            analytical_linestyle="-",
            analytical_linewidth=2.5,
            ocp_color=RECOMPUTED_OCP_COLOR,
            ocp_linestyle="--",
            ocp_linewidth=2.2,
            bounds_color="tab:red",
            bounds_linestyle=":",
            bounds_linewidth=1.0,
            bounds_alpha=0.8,
            grid_color="0.75",
            grid_linestyle=":",
            grid_linewidth=0.7,
            grid_alpha=0.7,
            show_legend=True,
        )

    plt.show(
        block=True
    )