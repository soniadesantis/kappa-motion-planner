import json
from math import cos, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kappa_planner.geometry import Point, Pose
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.plot_helpers import (
    get_analytical_control_profiles,
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
# WORST-CASE PLOT CONFIGURATION
# =============================================================================

SAVE_WORST_CASE_FIGURE = True
SHOW_WORST_CASE_FIGURE = True

# These names are generated automatically from the value of N.
WORST_CASE_OUTPUT_BASENAME = "sweep1_worst_time_difference"
WORST_CASE_CONTROLS_OUTPUT_BASENAME = (
    "sweep1_worst_time_difference_controls"
)
WORST_CASE_COMBINED_OUTPUT_BASENAME = (
    "sweep1_worst_time_difference_combined"
)
# Common analytical/OCP visual convention.
ANALYTICAL_COLOR = "tab:blue"
ANALYTICAL_LINESTYLE = "-"
ANALYTICAL_LINEWIDTH = 2.5

OCP_COLOR = "black"
OCP_LINESTYLE = "--"
OCP_LINEWIDTH = 2.2

BOUNDS_COLOR = "tab:red"
BOUNDS_LINESTYLE = ":"
BOUNDS_LINEWIDTH = 1.0
BOUNDS_ALPHA = 0.8

GRID_COLOR = "0.75"
GRID_LINESTYLE = ":"
GRID_LINEWIDTH = 0.7
GRID_ALPHA = 0.7

INITIAL_POSE_COLOR = "tab:green"
FINAL_POSE_COLOR = "tab:red"

# Used only if the corresponding values are not saved in the result file.
DEFAULT_START_POSITION = (0.0, 0.0)
DEFAULT_END_POSITION = (0.0, 5.0)
DEFAULT_VEHICLE_WIDTH = 0.430
DEFAULT_VEHICLE_LENGTH = 0.430
DEFAULT_V_MAX = 1.0
DEFAULT_OMEGA_MAX = 1.0

POSE_ARROW_LENGTH = 0.42
POSE_MARKER_SIZE = 7

def ask_yes_no(question, default="n"):
    """
    Ask a yes/no question in the terminal.

    Pressing Enter selects ``default``.
    """

    default = default.lower()

    if default not in {"y", "n"}:
        raise ValueError("default must be either 'y' or 'n'.")

    prompt = "[Y/n]" if default == "y" else "[y/N]"

    while True:
        answer = input(
            f"\n{question} {prompt}: "
        ).strip().lower()

        if not answer:
            return default == "y"

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no"}:
            return False

        print("Please answer 'y' or 'n'.")


def extract_case_boundary_poses(case):
    """
    Reconstruct the prescribed boundary poses for one saved sweep case.

    The saved analytical primitive poses are used when available. Otherwise,
    the fixed positions of Sweep 1 and the saved orientation angles are used.
    """

    primitives = case.get(
        "best_analytical_primitives",
        [],
    )

    start_data = None
    end_data = None

    if primitives:
        start_data = primitives[0].get(
            "start_pose"
        )

        end_data = primitives[-1].get(
            "end_pose"
        )

    if start_data is not None:
        start_x = float(start_data[0])
        start_y = float(start_data[1])
        start_theta = float(start_data[2])

    else:
        start_x = float(
            case.get(
                "x0",
                DEFAULT_START_POSITION[0],
            )
        )
        start_y = float(
            case.get(
                "y0",
                DEFAULT_START_POSITION[1],
            )
        )
        start_theta = (
            float(case["theta0_deg"])
            * pi
            / 180.0
        )

    if end_data is not None:
        end_x = float(end_data[0])
        end_y = float(end_data[1])
        end_theta = float(end_data[2])

    else:
        end_x = float(
            case.get(
                "xf",
                DEFAULT_END_POSITION[0],
            )
        )
        end_y = float(
            case.get(
                "yf",
                DEFAULT_END_POSITION[1],
            )
        )
        end_theta = (
            float(case["thetaf_deg"])
            * pi
            / 180.0
        )

    start_pose = Pose(
        Point(start_x, start_y),
        start_theta,
    )

    end_pose = Pose(
        Point(end_x, end_y),
        end_theta,
    )

    return start_pose, end_pose


def build_unicycle_from_case(case):
    """Build the unicycle used for the saved test case."""

    v_max = float(
        case.get(
            "v_max",
            DEFAULT_V_MAX,
        )
    )

    omega_max = float(
        case.get(
            "omega_max",
            DEFAULT_OMEGA_MAX,
        )
    )

    width = float(
        case.get(
            "vehicle_width",
            DEFAULT_VEHICLE_WIDTH,
        )
    )

    length = float(
        case.get(
            "vehicle_length",
            DEFAULT_VEHICLE_LENGTH,
        )
    )

    return Unicycle(
        state=[0.0, 0.0, 0.0],
        width=width,
        length=length,
        v_max=v_max,
        v_min=0.0,
        omega_max=omega_max,
        omega_min=-omega_max,
    )


def plot_pose(
    axis,
    pose,
    marker_color,
    label,
):
    """Plot one planar pose using a point and a heading arrow."""

    x = float(pose.x)
    y = float(pose.y)
    theta = float(pose.theta)

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
        POSE_ARROW_LENGTH * cos(theta),
        POSE_ARROW_LENGTH * sin(theta),
        width=0.012,
        head_width=0.12,
        head_length=0.16,
        length_includes_head=True,
        color=marker_color,
        zorder=19,
    )


def plot_and_save_worst_time_case(
    case,
    figures_directory,
):
    """
    Recompute and plot the analytical and OCP solutions for the case with the
    largest absolute traversal-time discrepancy.

    Two figures are produced:

    1. analytical and OCP planar paths;
    2. analytical and OCP control profiles.

    The same colors and line styles are used in both figures.
    """

    required_ocp_fields = (
        "ocp_x",
        "ocp_y",
    )

    missing_fields = [
        field
        for field in required_ocp_fields
        if case.get(field) is None
    ]

    if missing_fields:
        raise ValueError(
            "Cannot plot the saved OCP path. Missing fields: "
            + ", ".join(missing_fields)
        )

    start_pose, end_pose = (
        extract_case_boundary_poses(case)
    )

    unicycle = build_unicycle_from_case(
        case
    )

    trajectories = (
        compute_all_pose_to_pose_trajectories(
            start_pose,
            end_pose,
            unicycle,
        )
    )

    saved_best_name = case.get(
        "best_analytical_name"
    )

    if saved_best_name in trajectories:
        best_name = saved_best_name
        best_data = trajectories[
            saved_best_name
        ]

    else:
        best_name, best_data = min(
            trajectories.items(),
            key=lambda item: item[1]["time"],
        )

        if saved_best_name is not None:
            print(
                "\nWarning: the saved analytical name "
                f"'{saved_best_name}' was not found among the "
                "recomputed candidates."
            )

            print(
                f"Using the recomputed minimum '{best_name}'."
            )

    best_trajectory = best_data[
        "trajectory"
    ]

    recomputed_analytical_time = float(
        best_data["time"]
    )

    transcription_resolution = int(
        case.get(
            "N",
            100,
        )
    )

    print(
        "\nRerunning the OCP for the worst case "
        f"with N={transcription_resolution} "
        "to recover the control inputs..."
    )

    ocp_result = compute_ocp_pose_to_pose_trajectory(
        start_pose,
        end_pose,
        unicycle,
        analytical_initial_guess=best_trajectory,
        T_guess=recomputed_analytical_time,
        N=transcription_resolution,
    )

     # =========================================================================
    # COMBINED PATH AND CONTROL COMPARISON
    # =========================================================================

    combined_figure = plt.figure(
        figsize=(10.5, 5.8),
        constrained_layout=True,
    )

    grid = combined_figure.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=(0.43, 1.0),
        height_ratios=(1.0, 1.0),
        wspace=0.18,
        hspace=0.10,
    )

    path_axis = combined_figure.add_subplot(
        grid[:, 0]
    )

    velocity_axis = combined_figure.add_subplot(
        grid[0, 1]
    )

    angular_axis = combined_figure.add_subplot(
        grid[1, 1],
        sharex=velocity_axis,
    )

    # =========================================================================
    # LEFT PANEL: PLANAR PATH
    # =========================================================================

    plot_analytical_trajectory(
        best_trajectory,
        figure=path_axis,
        plot_circles=False,
        color=ANALYTICAL_COLOR,
        linewidth=ANALYTICAL_LINEWIDTH,
        plot_primitive_arrows=False,
        plot_turn_sectors=False,
    )

    # Controlled legend entry for the analytical trajectory.
    path_axis.plot(
        [],
        [],
        color=ANALYTICAL_COLOR,
        linestyle=ANALYTICAL_LINESTYLE,
        linewidth=ANALYTICAL_LINEWIDTH,
        label="Analytical",
    )

    path_axis.plot(
        np.asarray(
            case["ocp_x"],
            dtype=float,
        ),
        np.asarray(
            case["ocp_y"],
            dtype=float,
        ),
        color=OCP_COLOR,
        linestyle=OCP_LINESTYLE,
        linewidth=OCP_LINEWIDTH,
        label="OCP",
        zorder=10,
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

    # Slightly enlarge the visible horizontal range so that the panel itself
    # does not look excessively narrow.
    path_x_values = np.concatenate(
        (
            np.asarray(
                case["ocp_x"],
                dtype=float,
            ),
            np.asarray(
                [
                    float(start_pose.x),
                    float(end_pose.x),
                ]
            ),
        )
    )

    path_x_min = float(
        np.min(path_x_values)
    )

    path_x_max = float(
        np.max(path_x_values)
    )

    path_x_range = max(
        path_x_max - path_x_min,
        1.0,
    )

    path_x_margin = max(
        0.25,
        0.18 * path_x_range,
    )

    path_axis.set_xlim(
        path_x_min - path_x_margin,
        path_x_max + path_x_margin,
    )

    # Remove duplicate legend entries potentially generated by the analytical
    # trajectory helper.
    handles, labels = (
        path_axis.get_legend_handles_labels()
    )

    desired_order = (
        "Analytical",
        "OCP",
        "Initial pose",
        "Final pose",
    )

    ordered_handles = []
    ordered_labels = []

    for desired_label in desired_order:
        for handle, label in zip(
            handles,
            labels,
        ):
            if label == desired_label:
                ordered_handles.append(
                    handle
                )

                ordered_labels.append(
                    label
                )

                break

    path_axis.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=2,
        frameon=True,
        borderaxespad=0.0,
        fontsize=9,
    )

    # =========================================================================
    # RIGHT PANELS: CONTROL INPUTS
    # =========================================================================

    (
        analytical_time_grid,
        analytical_v,
        analytical_omega,
    ) = get_analytical_control_profiles(
        best_trajectory
    )

    ocp_time_grid = np.asarray(
        ocp_result["ts_ctrl"],
        dtype=float,
    )

    ocp_v = np.asarray(
        ocp_result["vs"],
        dtype=float,
    )

    ocp_omega = np.asarray(
        ocp_result["omegas"],
        dtype=float,
    )

    # -------------------------------------------------------------------------
    # Forward velocity
    # -------------------------------------------------------------------------

    velocity_axis.step(
        analytical_time_grid,
        analytical_v,
        where="post",
        color=ANALYTICAL_COLOR,
        linestyle=ANALYTICAL_LINESTYLE,
        linewidth=ANALYTICAL_LINEWIDTH,
        label="Analytical",
        zorder=4,
    )

    velocity_axis.step(
        ocp_time_grid,
        ocp_v,
        where="post",
        color=OCP_COLOR,
        linestyle=OCP_LINESTYLE,
        linewidth=OCP_LINEWIDTH,
        label="OCP",
        zorder=5,
    )

    velocity_axis.axhline(
        unicycle.v_max,
        color=BOUNDS_COLOR,
        linestyle=BOUNDS_LINESTYLE,
        linewidth=BOUNDS_LINEWIDTH,
        alpha=BOUNDS_ALPHA,
        zorder=1,
    )

    velocity_axis.axhline(
        unicycle.v_min,
        color=BOUNDS_COLOR,
        linestyle=BOUNDS_LINESTYLE,
        linewidth=BOUNDS_LINEWIDTH,
        alpha=BOUNDS_ALPHA,
        zorder=1,
    )

    velocity_axis.set_ylabel(
        r"$v(t)$ [m/s]"
    )

    velocity_axis.grid(
        True,
        color=GRID_COLOR,
        linestyle=GRID_LINESTYLE,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
        zorder=0,
    )

    velocity_axis.legend(
        loc="center right",
        frameon=True,
        fontsize=9,
    )

    # Hide the upper x-axis tick labels because the lower panel already shows
    # the common time axis.
    velocity_axis.tick_params(
        axis="x",
        labelbottom=False,
    )

    # -------------------------------------------------------------------------
    # Angular velocity
    # -------------------------------------------------------------------------

    angular_axis.step(
        analytical_time_grid,
        analytical_omega,
        where="post",
        color=ANALYTICAL_COLOR,
        linestyle=ANALYTICAL_LINESTYLE,
        linewidth=ANALYTICAL_LINEWIDTH,
        zorder=4,
    )

    angular_axis.step(
        ocp_time_grid,
        ocp_omega,
        where="post",
        color=OCP_COLOR,
        linestyle=OCP_LINESTYLE,
        linewidth=OCP_LINEWIDTH,
        zorder=5,
    )

    angular_axis.axhline(
        unicycle.omega_max,
        color=BOUNDS_COLOR,
        linestyle=BOUNDS_LINESTYLE,
        linewidth=BOUNDS_LINEWIDTH,
        alpha=BOUNDS_ALPHA,
        zorder=1,
    )

    angular_axis.axhline(
        unicycle.omega_min,
        color=BOUNDS_COLOR,
        linestyle=BOUNDS_LINESTYLE,
        linewidth=BOUNDS_LINEWIDTH,
        alpha=BOUNDS_ALPHA,
        zorder=1,
    )

    angular_axis.axhline(
        0.0,
        color="0.45",
        linestyle=":",
        linewidth=0.9,
        alpha=0.8,
        zorder=1,
    )

    angular_axis.set_ylabel(
        r"$\omega(t)$ [rad/s]"
    )

    angular_axis.set_xlabel(
        r"$t$ [s]"
    )

    angular_axis.grid(
        True,
        color=GRID_COLOR,
        linestyle=GRID_LINESTYLE,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
        zorder=0,
    )

    final_time = max(
        float(analytical_time_grid[-1]),
        float(ocp_result["time"]),
    )

    velocity_axis.set_xlim(
        0.0,
        final_time,
    )

    angular_axis.set_xlim(
        0.0,
        final_time,
    )

    # =========================================================================
    # PANEL LABELS
    # =========================================================================

    path_axis.text(
        -0.18,
        1.01,
        "(a)",
        transform=path_axis.transAxes,
        ha="left",
        va="bottom",
    )

    velocity_axis.text(
        -0.10,
        1.02,
        "(b)",
        transform=velocity_axis.transAxes,
        ha="left",
        va="bottom",
    )

    angular_axis.text(
        -0.10,
        1.02,
        "(c)",
        transform=angular_axis.transAxes,
        ha="left",
        va="bottom",
    )

        # =========================================================================
    # SAVE COMBINED FIGURE
    # =========================================================================

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_basename = (
        f"{WORST_CASE_COMBINED_OUTPUT_BASENAME}"
        f"_N{transcription_resolution}"
    )

    combined_pdf_path = (
        figures_directory
        / f"{combined_basename}.pdf"
    )

    combined_png_path = (
        figures_directory
        / f"{combined_basename}.png"
    )

    if SAVE_WORST_CASE_FIGURE:
        combined_figure.savefig(
            combined_pdf_path,
            bbox_inches="tight",
        )

        combined_figure.savefig(
            combined_png_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            "\nCombined worst-case figure saved to:"
        )

        print(
            combined_pdf_path
        )

        print(
            combined_png_path
        )

    # =========================================================================
    # TERMINAL SUMMARY
    # =========================================================================

    print(
        "\nWorst-case analytical reconstruction"
    )

    print(
        "-" * 80
    )

    print(
        f"Saved analytical name : "
        f"{saved_best_name}"
    )

    print(
        f"Recomputed best name  : "
        f"{best_name}"
    )

    print(
        f"Saved analytical time : "
        f"{float(case['best_analytical_time']):.9f} s"
    )

    print(
        f"Recomputed time       : "
        f"{recomputed_analytical_time:.9f} s"
    )

    print(
        f"Saved OCP time        : "
        f"{float(case['ocp_time']):.9f} s"
    )

    print(
        f"Recomputed OCP time   : "
        f"{float(ocp_result['time']):.9f} s"
    )

    print(
        f"Recomputed OCP seq.   : "
        f"{' - '.join(ocp_result['sequence'])}"
    )

    if SHOW_WORST_CASE_FIGURE:
        plt.show(
            block=True
        )

    else:
        plt.close(
            combined_figure
        )

    return combined_figure


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



def get_analytical_family(case):
    """
    Extract the analytical trajectory family from ``best_analytical_name``.

    Examples
    --------
    ``TCSCT left-right`` -> ``TCSCT``
    ``CSC right-right``  -> ``CSC``
    """

    best_name = case.get("best_analytical_name")

    if not best_name:
        return None

    family = best_name.split(" ", 1)[0]

    valid_families = {
        "CSC",
        "TCSC",
        "CSCT",
        "TCSCT",
    }

    if family not in valid_families:
        raise ValueError(
            f"Unknown analytical family '{family}' "
            f"in best_analytical_name='{best_name}'."
        )

    return family


def compute_time_statistics(cases):
    """
    Compute absolute and relative traversal-time discrepancy statistics.

    The relative discrepancy is expressed as a percentage with respect to
    the OCP traversal time.
    """

    absolute_differences = np.array(
        [
            abs(float(case["time_difference"]))
            for case in cases
        ],
        dtype=float,
    )

    relative_differences = np.array(
        [
            100.0
            * abs(float(case["time_difference"]))
            / float(case["ocp_time"])
            for case in cases
            if float(case["ocp_time"]) > 1e-12
        ],
        dtype=float,
    )

    if absolute_differences.size == 0:
        return None

    if relative_differences.size == 0:
        mean_relative = np.nan
        median_relative = np.nan
        percentile_95_relative = np.nan
        maximum_relative = np.nan

    else:
        mean_relative = float(
            np.mean(relative_differences)
        )

        median_relative = float(
            np.median(relative_differences)
        )

        percentile_95_relative = float(
            np.percentile(
                relative_differences,
                95,
            )
        )

        maximum_relative = float(
            np.max(relative_differences)
        )

    return {
        "cases": len(cases),

        "mean_abs": float(
            np.mean(absolute_differences)
        ),

        "median_abs": float(
            np.median(absolute_differences)
        ),

        "p95_abs": float(
            np.percentile(
                absolute_differences,
                95,
            )
        ),

        "max_abs": float(
            np.max(absolute_differences)
        ),

        "mean_rel": mean_relative,
        "median_rel": median_relative,
        "p95_rel": percentile_95_relative,
        "max_rel": maximum_relative,
    }


def print_time_statistics_by_family(successful_cases):
    """
    Print traversal-time discrepancy statistics grouped by analytical family.
    """

    family_order = (
        "CSC",
        "TCSC",
        "CSCT",
        "TCSCT",
    )

    cases_by_family = {
        family: []
        for family in family_order
    }

    for case in successful_cases:
        family = get_analytical_family(case)

        if family is not None:
            cases_by_family[family].append(
                case
            )

    print("\n" + "=" * 146)
    print(
        "TIME DIFFERENCE STATISTICS "
        "BY ANALYTICAL FAMILY"
    )
    print("=" * 146)

    header = (
        f"{'Family':<8}"
        f"{'Cases':>8}"
        f"{'Mean |ΔT| [s]':>18}"
        f"{'Median |ΔT| [s]':>20}"
        f"{'95th |ΔT| [s]':>18}"
        f"{'Max |ΔT| [s]':>18}"
        f"{'Mean rel. [%]':>18}"
        f"{'95th rel. [%]':>18}"
        f"{'Max rel. [%]':>18}"
    )

    print(header)
    print("-" * len(header))

    statistics_by_family = {}

    for family in family_order:
        family_cases = cases_by_family[
            family
        ]

        statistics = compute_time_statistics(
            family_cases
        )

        statistics_by_family[
            family
        ] = statistics

        if statistics is None:
            print(
                f"{family:<8}"
                f"{0:>8}"
                f"{'--':>18}"
                f"{'--':>20}"
                f"{'--':>18}"
                f"{'--':>18}"
                f"{'--':>18}"
                f"{'--':>18}"
                f"{'--':>18}"
            )
            continue

        print(
            f"{family:<8}"
            f"{statistics['cases']:>8d}"
            f"{statistics['mean_abs']:>18.6e}"
            f"{statistics['median_abs']:>20.6e}"
            f"{statistics['p95_abs']:>18.6e}"
            f"{statistics['max_abs']:>18.6e}"
            f"{statistics['mean_rel']:>18.6e}"
            f"{statistics['p95_rel']:>18.6e}"
            f"{statistics['max_rel']:>18.6e}"
        )

    print("\nLaTeX table rows")
    print("-" * 146)

    for family in family_order:
        statistics = statistics_by_family[
            family
        ]

        if statistics is None:
            print(
                rf"\({family}\)"
                rf" & 0"
                rf" & --"
                rf" & --"
                rf" & --"
                rf" & --"
                rf" & --"
                rf" & --"
                rf" & -- \\"
            )
            continue

        print(
            rf"\({family}\)"
            rf" & \({statistics['cases']}\)"
            rf" & \({statistics['mean_abs']:.2e}\)"
            rf" & \({statistics['median_abs']:.2e}\)"
            rf" & \({statistics['p95_abs']:.2e}\)"
            rf" & \({statistics['max_abs']:.2e}\)"
            rf" & \({statistics['mean_rel']:.2e}\)"
            rf" & \({statistics['p95_rel']:.2e}\)"
            rf" & \({statistics['max_rel']:.2e}\)"
            rf" \\"
        )

    return statistics_by_family


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

    # Sweep 1
    # RESULTS_FILENAME = "sweep1/orientation_sweep_N100_OCPpath.json"
    # RESULTS_FILENAME = "sweep1/orientation_sweep_OCP_path_saved_N30_M4.json"
    
    # Sweep 2
    # RESULTS_FILENAME = "sweep2/distance_over_radius_sweep_OCP_path_saved_N30_M4.json"
    # RESULTS_FILENAME = "sweep2/distance_over_radius_sweep_OCP_path_saved_N100_M4.json"
    # RESULTS_FILENAME = "sweep2/distance_over_radius_sweep_OCP_path_saved_N200_M4.json"


    # Sweep 3
    # RESULTS_FILENAME = "sweep3/polar_goal_position_sweep_OCP_path_saved_N30_M4.json"
    # RESULTS_FILENAME = "sweep3/polar_goal_position_sweep_OCP_path_saved_N100_M4.json"

    # Sweep 4
    # RESULTS_FILENAME = "sweep4/sobol_sweep_N30_M4.json"
    # RESULTS_FILENAME = "sweep4/sobol_sweep_N100_M4.json"
    RESULTS_FILENAME = "sweep4/sobol_sweep_N200_M4.json"


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

        plot_worst_case = ask_yes_no(
            "Plot and save the path and control inputs for the case "
            "with the largest absolute time difference?",
            default="n",
        )

        if plot_worst_case:
            figures_directory = (
                current_dir
                / "figures"
            )

            plot_and_save_worst_time_case(
                case=worst_case,
                figures_directory=figures_directory,
            )

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
    # Overall traversal-time discrepancy statistics
    # -------------------------------------------------------------------------

    overall_time_statistics = compute_time_statistics(
        successful_cases
    )

    if overall_time_statistics is not None:

        print("\n" + "=" * 80)
        print("TIME DIFFERENCE STATISTICS")
        print("=" * 80)

        print("\nAbsolute traversal-time discrepancy")
        print("-" * 80)

        print(
            "Mean              : "
            f"{overall_time_statistics['mean_abs']:.6e} s"
        )

        print(
            "Median            : "
            f"{overall_time_statistics['median_abs']:.6e} s"
        )

        print(
            "95th percentile   : "
            f"{overall_time_statistics['p95_abs']:.6e} s"
        )

        print(
            "Maximum           : "
            f"{overall_time_statistics['max_abs']:.6e} s"
        )

        print("\nRelative traversal-time discrepancy")
        print("-" * 80)

        print(
            "Mean              : "
            f"{overall_time_statistics['mean_rel']:.6e} %"
        )

        print(
            "Median            : "
            f"{overall_time_statistics['median_rel']:.6e} %"
        )

        print(
            "95th percentile   : "
            f"{overall_time_statistics['p95_rel']:.6e} %"
        )

        print(
            "Maximum           : "
            f"{overall_time_statistics['max_rel']:.6e} %"
        )

        print("\nLaTeX rows for the overall validation table")
        print("-" * 80)

        print(
            r"\(e_T^{\mathrm{abs}}\,[\mathrm{s}]\)"
            rf" & \({metadata_N}\)"
            rf" & \({overall_time_statistics['mean_abs']:.2e}\)"
            rf" & \({overall_time_statistics['median_abs']:.2e}\)"
            rf" & \({overall_time_statistics['p95_abs']:.2e}\)"
            rf" & \({overall_time_statistics['max_abs']:.2e}\)"
            rf" \\"
        )

        print(
            r"\(e_T^{\mathrm{rel}}\,[\%]\)"
            rf" & \({metadata_N}\)"
            rf" & \({overall_time_statistics['mean_rel']:.2e}\)"
            rf" & \({overall_time_statistics['median_rel']:.2e}\)"
            rf" & \({overall_time_statistics['p95_rel']:.2e}\)"
            rf" & \({overall_time_statistics['max_rel']:.2e}\)"
            rf" \\"
        )

    else:
        print(
            "\nNo successful cases are available "
            "for the time-discrepancy statistics."
        )


    # -------------------------------------------------------------------------
    # Traversal-time discrepancies grouped by analytical family
    # -------------------------------------------------------------------------

    family_time_statistics = print_time_statistics_by_family(
        successful_cases=successful_cases,
    )


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