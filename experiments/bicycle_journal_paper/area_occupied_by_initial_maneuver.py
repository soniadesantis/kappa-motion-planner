from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from kappa_planner.geometry import IntermediateCircle, Point, Pose
from kappa_planner.helpers.pose_to_circle_bicycle import (
    compute_traj_to_circle_free_space_bicycle,
)
from kappa_planner.trajectory import (
    BackwardArc,
    CurvilinearArcUnicycle,
    LinearSegmentUnicycle,
)
from kappa_planner.vehicle import Bicycle


# ================================================================
# Plot style
# ================================================================

plt.rcParams["font.family"] = "serif"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)


# ================================================================
# Generic plotting helpers
# ================================================================

def draw_circle(
    ax,
    center,
    radius,
    *,
    color="black",
    linestyle="--",
    linewidth=1.2,
    alpha=1.0,
    label=None,
    zorder=5,
):
    """Draw a circle using line coordinates."""
    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        500,
    )

    center = np.asarray(
        center,
        dtype=float,
    )

    ax.plot(
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )


def plot_path_coordinates(
    ax,
    coordinates,
    *,
    color,
    linewidth=0.40,
    alpha=0.05,
    zorder=3,
):
    """Plot a sequence of planar path coordinates."""
    coordinates = np.asarray(
        coordinates,
        dtype=float,
    )

    ax.plot(
        coordinates[:, 0],
        coordinates[:, 1],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )


# ================================================================
# Initial-maneuver extraction
# ================================================================

def extract_selected_start_maneuver(
    trajectory,
):
    """
    Extract the planner-selected initial circular maneuver.

    Expected structures are

        C S ...
        C_b C S ...

    where tau0 is the turn direction of the first forward arc.

    Returns
    -------
    dict
        tau0
            Turn direction of the first forward arc.

        initial_primitives
            Circular primitives before the first linear segment.

        tangent_segment
            First linear segment after the initial circular maneuver.

        family
            LaTeX-formatted family label.

        has_backward_arc
            True for C_b C, False for C.
    """
    if trajectory is None or len(trajectory) == 0:
        raise ValueError(
            "The planner returned an empty trajectory."
        )

    initial_primitives = []
    tangent_segment = None
    first_forward_arc = None
    has_backward_arc = False

    for primitive in trajectory:
        if isinstance(
            primitive,
            LinearSegmentUnicycle,
        ):
            tangent_segment = primitive
            break

        if isinstance(
            primitive,
            BackwardArc,
        ):
            has_backward_arc = True

            initial_primitives.append(
                primitive
            )

            continue

        if isinstance(
            primitive,
            CurvilinearArcUnicycle,
        ):
            initial_primitives.append(
                primitive
            )

            if first_forward_arc is None:
                first_forward_arc = primitive

            continue

        raise RuntimeError(
            "Unexpected primitive before the first tangent segment: "
            f"{type(primitive).__name__}."
        )

    if first_forward_arc is None:
        raise RuntimeError(
            "No forward circular arc was found before the tangent segment."
        )

    if tangent_segment is None:
        raise RuntimeError(
            "No tangent segment was found after the initial circular maneuver."
        )

    tau0 = int(
        first_forward_arc.turn_direction
    )

    if tau0 not in (-1, +1):
        raise RuntimeError(
            f"Expected tau0 in {{-1,+1}}, received {tau0}."
        )

    family = (
        r"$C_{\mathrm{b}}C$"
        if has_backward_arc
        else r"$C$"
    )

    return {
        "tau0": tau0,
        "initial_primitives": initial_primitives,
        "tangent_segment": tangent_segment,
        "family": family,
        "has_backward_arc": has_backward_arc,
    }


def concatenate_primitive_coordinates(
    primitives,
):
    """
    Concatenate the sampled coordinates of consecutive primitives.

    The first point of each primitive after the first one is removed
    to avoid duplicate connection points.
    """
    if not primitives:
        raise ValueError(
            "No initial primitives were provided."
        )

    coordinate_blocks = []

    for primitive_index, primitive in enumerate(
        primitives
    ):
        coordinates = np.asarray(
            primitive.path_coordinates,
            dtype=float,
        )

        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 2
        ):
            raise ValueError(
                "Each primitive must provide N-by-2 path coordinates."
            )

        if primitive_index > 0:
            coordinates = coordinates[1:]

        coordinate_blocks.append(
            coordinates
        )

    return np.vstack(
        coordinate_blocks
    )


# ================================================================
# Outgoing-tangent frame
# ================================================================

def compute_outgoing_tangent_frame(
    tangent_segment,
):
    """
    Compute the local frame aligned with the outgoing tangent.

    The longitudinal unit vector t follows the direction in which the
    tangent segment is traversed. The normal vector n is obtained by
    rotating t counterclockwise by pi/2.
    """
    segment_start = np.array(
        [
            tangent_segment.x0,
            tangent_segment.y0,
        ],
        dtype=float,
    )

    segment_end = np.array(
        [
            tangent_segment.xf,
            tangent_segment.yf,
        ],
        dtype=float,
    )

    tangent_direction = (
        segment_end - segment_start
    )

    tangent_norm = np.linalg.norm(
        tangent_direction
    )

    if tangent_norm <= 1e-12:
        raise ValueError(
            "The outgoing tangent segment has zero length."
        )

    tangent_direction /= tangent_norm

    # Preserve the direction of travel if the segment can be backward.
    segment_velocity = getattr(
        tangent_segment,
        "v",
        1.0,
    )

    if segment_velocity < 0.0:
        tangent_direction *= -1.0

    normal_direction = np.array(
        [
            -tangent_direction[1],
            tangent_direction[0],
        ],
        dtype=float,
    )

    return tangent_direction, normal_direction


def world_to_tangent_coordinates(
    world_coordinates,
    start_position,
    tangent_direction,
    normal_direction,
):
    """
    Express world-frame points in tangent-aligned coordinates.

    A point is represented as

        p = p0 + s t + q n,

    where s is longitudinal and q is lateral.
    """
    world_coordinates = np.asarray(
        world_coordinates,
        dtype=float,
    )

    start_position = np.asarray(
        start_position,
        dtype=float,
    )

    relative_coordinates = (
        world_coordinates
        - start_position[None, :]
    )

    s_coordinates = (
        relative_coordinates
        @ tangent_direction
    )

    q_coordinates = (
        relative_coordinates
        @ normal_direction
    )

    return np.column_stack(
        (
            s_coordinates,
            q_coordinates,
        )
    )


# ================================================================
# Candidate-disk analysis
# ================================================================

def compute_spatial_metrics(
    world_coordinates,
    tangent_coordinates,
    start_position,
    R,
):
    """
    Compute spatial quantities relevant to the safe-start conjecture.

    The candidate disk in tangent coordinates is

        (s - R)^2 + q^2 <= R^2.

    Equivalently, in the world frame it is centered at

        p0 + R t

    and has radius R.
    """
    world_coordinates = np.asarray(
        world_coordinates,
        dtype=float,
    )

    tangent_coordinates = np.asarray(
        tangent_coordinates,
        dtype=float,
    )

    start_position = np.asarray(
        start_position,
        dtype=float,
    )

    radial_distances = np.linalg.norm(
        world_coordinates
        - start_position[None, :],
        axis=1,
    )

    s_coordinates = tangent_coordinates[:, 0]
    q_coordinates = tangent_coordinates[:, 1]

    candidate_disk_distances = np.sqrt(
        (s_coordinates - R) ** 2
        + q_coordinates ** 2
    )

    candidate_disk_residuals = (
        candidate_disk_distances - R
    )

    return {
        "d_max": float(
            np.max(radial_distances)
        ),
        "s_min": float(
            np.min(s_coordinates)
        ),
        "s_max": float(
            np.max(s_coordinates)
        ),
        "q_min": float(
            np.min(q_coordinates)
        ),
        "q_max": float(
            np.max(q_coordinates)
        ),
        "q_abs_max": float(
            np.max(
                np.abs(q_coordinates)
            )
        ),
        "candidate_disk_distance_max": float(
            np.max(candidate_disk_distances)
        ),
        "candidate_disk_residual_max": float(
            np.max(candidate_disk_residuals)
        ),
    }


def draw_candidate_maneuver_disk(
    ax,
    R,
):
    """
    Draw the candidate disk

        B((R,0), R)

    in the outgoing-tangent frame.
    """
    candidate_disk = Circle(
        xy=(
            R,
            0.0,
        ),
        radius=R,
        facecolor="0.92",
        edgecolor="tab:green",
        linewidth=1.8,
        linestyle="-",
        alpha=0.70,
        zorder=0,
        label=(
            r"Candidate disk "
            r"$\mathcal{B}((R,0),R)$"
        ),
    )

    ax.add_patch(
        candidate_disk
    )

    ax.plot(
        R,
        0.0,
        marker="x",
        markersize=7,
        markeredgewidth=1.5,
        color="tab:green",
        zorder=9,
    )

    ax.annotate(
        r"$(R,0)$",
        xy=(
            R,
            0.0,
        ),
        xytext=(
            R,
            0.12 * R,
        ),
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=9,
        color="tab:green",
        zorder=10,
    )

    ax.axvline(
        0.0,
        color="0.35",
        linestyle=":",
        linewidth=1.0,
        zorder=6,
    )

    ax.axhline(
        0.0,
        color="0.35",
        linestyle=":",
        linewidth=1.0,
        zorder=6,
    )


# ================================================================
# Main analysis
# ================================================================

def plot_candidate_disk_analysis(
    *,
    bicycle,
    target_circle,
    x0=0.0,
    y0=0.0,
    footprint_radius=0.0,
    angle_step_degrees=1,
    trajectory_alpha=0.045,
    trajectory_linewidth=0.35,
    containment_tolerance=1e-9,
    output_path=None,
):
    """
    Test whether the complete initial circular maneuver lies in a disk
    of radius R centered one radius ahead of the initial position.

    For every initial heading theta0:

        1. the planner chooses tau0 internally;
        2. the initial maneuver C or C_b C is extracted;
        3. tau0 is recovered from the first forward arc;
        4. the trajectory is transformed into its outgoing-tangent frame;
        5. containment in B((R,0), R) is tested.

    Rows
    ----
    selected tau0 = +1
    selected tau0 = -1

    Columns
    -------
    world frame
    outgoing-tangent frame
    """
    if angle_step_degrees <= 0:
        raise ValueError(
            "angle_step_degrees must be strictly positive."
        )

    if footprint_radius < 0.0:
        raise ValueError(
            "footprint_radius must be nonnegative."
        )

    if containment_tolerance < 0.0:
        raise ValueError(
            "containment_tolerance must be nonnegative."
        )

    R = bicycle.max_radius

    start_position = np.array(
        [
            x0,
            y0,
        ],
        dtype=float,
    )

    theta_degrees_array = np.arange(
        0.0,
        360.0,
        angle_step_degrees,
    )

    theta_array = np.deg2rad(
        theta_degrees_array
    )

    colormap = mpl.colormaps["hsv"]

    normalization = mpl.colors.Normalize(
        vmin=0.0,
        vmax=360.0,
    )

    tau0_values = [
        +1,
        -1,
    ]

    tau0_to_row = {
        +1: 0,
        -1: 1,
    }

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.8, 8.8),
        sharex="col",
        sharey="col",
    )

    results_by_tau0 = {
        +1: [],
        -1: [],
    }

    failures = []

    # ------------------------------------------------------------
    # Compute one planner-selected trajectory per initial heading
    # ------------------------------------------------------------

    for theta_degrees, theta0 in zip(
        theta_degrees_array,
        theta_array,
    ):
        start_pose = Pose(
            position=Point(
                x0,
                y0,
            ),
            theta=theta0,
        )

        try:
            # tau0 is deliberately omitted.
            # The planner chooses it internally.
            trajectory = (
                compute_traj_to_circle_free_space_bicycle(
                    start_pose,
                    bicycle,
                    target_circle,
                )
            )

            extracted = (
                extract_selected_start_maneuver(
                    trajectory
                )
            )

            world_coordinates = (
                concatenate_primitive_coordinates(
                    extracted[
                        "initial_primitives"
                    ]
                )
            )

            (
                tangent_direction,
                normal_direction,
            ) = compute_outgoing_tangent_frame(
                extracted[
                    "tangent_segment"
                ]
            )

            tangent_coordinates = (
                world_to_tangent_coordinates(
                    world_coordinates=world_coordinates,
                    start_position=start_position,
                    tangent_direction=tangent_direction,
                    normal_direction=normal_direction,
                )
            )

            metrics = compute_spatial_metrics(
                world_coordinates=world_coordinates,
                tangent_coordinates=tangent_coordinates,
                start_position=start_position,
                R=R,
            )

        except Exception as error:
            failures.append(
                {
                    "theta_degrees": theta_degrees,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

            continue

        tau0 = extracted["tau0"]
        row_index = tau0_to_row[tau0]

        trajectory_color = colormap(
            normalization(
                theta_degrees
            )
        )

        plot_path_coordinates(
            ax=axes[row_index, 0],
            coordinates=world_coordinates,
            color=trajectory_color,
            linewidth=trajectory_linewidth,
            alpha=trajectory_alpha,
        )

        plot_path_coordinates(
            ax=axes[row_index, 1],
            coordinates=tangent_coordinates,
            color=trajectory_color,
            linewidth=trajectory_linewidth,
            alpha=trajectory_alpha,
        )

        result = {
            "theta_degrees": float(
                theta_degrees
            ),
            "tau0": tau0,
            "family": extracted["family"],
            "has_backward_arc": extracted[
                "has_backward_arc"
            ],
            "world_coordinates": world_coordinates,
            "tangent_coordinates": tangent_coordinates,
            "tangent_direction": tangent_direction,
            "normal_direction": normal_direction,
            "disk_contained": (
                metrics[
                    "candidate_disk_residual_max"
                ]
                <= containment_tolerance
            ),
            **metrics,
        }

        results_by_tau0[tau0].append(
            result
        )

    # ------------------------------------------------------------
    # World-frame reference geometry
    # ------------------------------------------------------------

    for row_index, tau0 in enumerate(
        tau0_values
    ):
        world_ax = axes[
            row_index,
            0,
        ]

        draw_circle(
            ax=world_ax,
            center=(
                target_circle.xc,
                target_circle.yc,
            ),
            radius=target_circle.radius,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label="Target circle",
            zorder=8,
        )

        draw_circle(
            ax=world_ax,
            center=start_position,
            radius=R,
            color="0.40",
            linestyle=":",
            linewidth=1.1,
            label=r"Radius $R$ about $\mathbf{p}_0$",
            zorder=7,
        )

        draw_circle(
            ax=world_ax,
            center=start_position,
            radius=2.0 * R,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label=r"Radius $2R$ about $\mathbf{p}_0$",
            zorder=7,
        )

        world_ax.plot(
            x0,
            y0,
            marker="o",
            markersize=4,
            color="black",
            zorder=20,
        )

        world_ax.set_ylabel(
            (
                rf"Selected $\tau_0={tau0:+d}$"
                "\n"
                r"$y$ [m]"
            )
        )

    # ------------------------------------------------------------
    # Tangent-frame candidate disks
    # ------------------------------------------------------------

    for row_index, _tau0 in enumerate(
        tau0_values
    ):
        tangent_ax = axes[
            row_index,
            1,
        ]

        draw_candidate_maneuver_disk(
            ax=tangent_ax,
            R=R,
        )

        tangent_ax.plot(
            0.0,
            0.0,
            marker="o",
            markersize=4,
            color="black",
            zorder=20,
            label=r"Initial position $\mathbf{p}_0$",
        )

        tangent_ax.set_ylabel(
            r"Lateral coordinate $q$ [m]"
        )

    # ------------------------------------------------------------
    # Titles and labels
    # ------------------------------------------------------------

    axes[0, 0].set_title(
        "Planner-selected initial maneuvers\nin the world frame"
    )

    axes[0, 1].set_title(
        "Initial maneuvers in their\noutgoing-tangent frame"
    )

    axes[1, 0].set_xlabel(
        r"$x$ [m]"
    )

    axes[1, 1].set_xlabel(
        r"Longitudinal coordinate $s$ [m]"
    )

    # ------------------------------------------------------------
    # Highlight extremal and worst-containment trajectories
    # ------------------------------------------------------------

    for row_index, tau0 in enumerate(
        tau0_values
    ):
        results = results_by_tau0[tau0]

        if not results:
            axes[row_index, 0].text(
                0.5,
                0.5,
                "No selected trajectories",
                transform=axes[row_index, 0].transAxes,
                horizontalalignment="center",
                verticalalignment="center",
            )

            axes[row_index, 1].text(
                0.5,
                0.5,
                "No selected trajectories",
                transform=axes[row_index, 1].transAxes,
                horizontalalignment="center",
                verticalalignment="center",
            )

            continue

        worst_disk_result = max(
            results,
            key=lambda result: result[
                "candidate_disk_residual_max"
            ],
        )

        worst_radial_result = max(
            results,
            key=lambda result: result[
                "d_max"
            ],
        )

        worst_s_result = max(
            results,
            key=lambda result: result[
                "s_max"
            ],
        )

        worst_q_result = max(
            results,
            key=lambda result: result[
                "q_abs_max"
            ],
        )

        highlighted_results = []

        for candidate in (
            worst_disk_result,
            worst_radial_result,
            worst_s_result,
            worst_q_result,
        ):
            if all(
                candidate is not existing
                for existing in highlighted_results
            ):
                highlighted_results.append(
                    candidate
                )

        for highlighted_result in highlighted_results:
            plot_path_coordinates(
                ax=axes[row_index, 0],
                coordinates=highlighted_result[
                    "world_coordinates"
                ],
                color="black",
                linewidth=1.8,
                alpha=0.95,
                zorder=15,
            )

            plot_path_coordinates(
                ax=axes[row_index, 1],
                coordinates=highlighted_result[
                    "tangent_coordinates"
                ],
                color="black",
                linewidth=1.8,
                alpha=0.95,
                zorder=15,
            )

        annotation = (
            rf"$\max\rho/R="
            rf"{worst_disk_result['candidate_disk_distance_max']/R:.6f}$"
            "\n"
            rf"$\max(\rho-R)/R="
            rf"{worst_disk_result['candidate_disk_residual_max']/R:.2e}$"
            "\n"
            rf"$s_{{\min}}/R="
            rf"{min(result['s_min'] for result in results)/R:.3f}$"
            "\n"
            rf"$s_{{\max}}/R="
            rf"{max(result['s_max'] for result in results)/R:.3f}$"
            "\n"
            rf"$|q|_{{\max}}/R="
            rf"{max(result['q_abs_max'] for result in results)/R:.3f}$"
        )

        axes[row_index, 1].text(
            0.03,
            0.97,
            annotation,
            transform=axes[row_index, 1].transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.90,
                "boxstyle": "round,pad=0.25",
            },
            zorder=30,
        )

    # ------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------

    for ax in axes.flat:
        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.grid(
            True,
            linestyle=":",
            linewidth=0.4,
            alpha=0.28,
        )

    # Candidate disk occupies 0 <= s <= 2R and |q| <= R.
    for row_index in range(2):
        axes[row_index, 1].set_xlim(
            -0.20 * R,
            2.20 * R,
        )

        axes[row_index, 1].set_ylim(
            -1.25 * R,
            1.25 * R,
        )

    # ------------------------------------------------------------
    # Figure legend
    # ------------------------------------------------------------

    handles = []
    labels = []

    for ax in axes.flat:
        ax_handles, ax_labels = (
            ax.get_legend_handles_labels()
        )

        handles.extend(
            ax_handles
        )

        labels.extend(
            ax_labels
        )

    unique_items = {}

    for handle, label in zip(
        handles,
        labels,
    ):
        if (
            label
            and not label.startswith("_")
            and label not in unique_items
        ):
            unique_items[label] = handle

    figure.legend(
        unique_items.values(),
        unique_items.keys(),
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.995,
        ),
        ncol=4,
        frameon=False,
    )

    # ------------------------------------------------------------
    # Heading colorbar
    # ------------------------------------------------------------

    scalar_mappable = mpl.cm.ScalarMappable(
        norm=normalization,
        cmap=colormap,
    )

    scalar_mappable.set_array(
        []
    )

    colorbar = figure.colorbar(
        scalar_mappable,
        ax=axes,
        orientation="horizontal",
        fraction=0.035,
        pad=0.075,
        aspect=45,
    )

    colorbar.set_label(
        r"Initial orientation $\theta_0$ [deg]"
    )

    colorbar.set_ticks(
        [
            0,
            90,
            180,
            270,
            360,
        ]
    )

    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.13,
        top=0.88,
        wspace=0.12,
        hspace=0.10,
    )

    # ------------------------------------------------------------
    # Numerical summary
    # ------------------------------------------------------------

    print()
    print("CANDIDATE SAFE-START DISK ANALYSIS")
    print("=" * 80)

    print()
    print(
        "Candidate reference-path disk:"
    )

    print(
        "    center = p0 + R t"
    )

    print(
        "    radius = R"
    )

    print(
        "    tangent-frame equation: (s - R)^2 + q^2 <= R^2"
    )

    for tau0 in tau0_values:
        results = results_by_tau0[tau0]

        print()
        print(
            f"Selected tau0 = {tau0:+d}"
        )

        if not results:
            print(
                "  No selected trajectories."
            )

            continue

        nominal_results = [
            result
            for result in results
            if not result["has_backward_arc"]
        ]

        backward_results = [
            result
            for result in results
            if result["has_backward_arc"]
        ]

        worst_disk_result = max(
            results,
            key=lambda result: result[
                "candidate_disk_residual_max"
            ],
        )

        number_outside = sum(
            not result["disk_contained"]
            for result in results
        )

        print(
            f"  selected orientations            : {len(results)}"
        )

        print(
            f"  nominal C maneuvers              : "
            f"{len(nominal_results)}"
        )

        print(
            f"  C_b C maneuvers                  : "
            f"{len(backward_results)}"
        )

        print(
            f"  orientations outside disk        : "
            f"{number_outside}"
        )

        print(
            f"  worst initial heading            : "
            f"{worst_disk_result['theta_degrees']:.1f} deg"
        )

        print(
            f"  worst family                     : "
            f"{worst_disk_result['family']}"
        )

        print(
            f"  max distance from disk center    : "
            f"{worst_disk_result['candidate_disk_distance_max']:.12f} m"
        )

        print(
            f"  max distance from disk center / R: "
            f"{worst_disk_result['candidate_disk_distance_max']/R:.12f}"
        )

        print(
            f"  maximum disk residual            : "
            f"{worst_disk_result['candidate_disk_residual_max']:.12e} m"
        )

        print(
            f"  maximum disk residual / R        : "
            f"{worst_disk_result['candidate_disk_residual_max']/R:.12e}"
        )

        print(
            f"  minimum longitudinal coordinate  : "
            f"{min(result['s_min'] for result in results)/R:.9f} R"
        )

        print(
            f"  maximum longitudinal coordinate  : "
            f"{max(result['s_max'] for result in results)/R:.9f} R"
        )

        print(
            f"  maximum lateral magnitude        : "
            f"{max(result['q_abs_max'] for result in results)/R:.9f} R"
        )

    all_results = (
        results_by_tau0[+1]
        + results_by_tau0[-1]
    )

    if all_results:
        global_worst = max(
            all_results,
            key=lambda result: result[
                "candidate_disk_residual_max"
            ],
        )

        globally_contained = all(
            result["disk_contained"]
            for result in all_results
        )

        print()
        print("GLOBAL RESULT")
        print("-" * 80)

        print(
            f"All sampled maneuvers contained: "
            f"{globally_contained}"
        )

        print(
            f"Worst normalized residual: "
            f"{global_worst['candidate_disk_residual_max']/R:.12e}"
        )

        print(
            f"Worst heading: "
            f"{global_worst['theta_degrees']:.1f} deg"
        )

        print(
            f"Worst selected tau0: "
            f"{global_worst['tau0']:+d}"
        )

        print(
            f"Worst family: "
            f"{global_worst['family']}"
        )

    print()
    print("FOOTPRINT-AWARE INTERPRETATION")
    print("-" * 80)

    print(
        "If the reference path is contained in B(p0 + R t, R),"
    )

    print(
        "then a disk-shaped footprint of radius r is contained in"
    )

    print(
        "    B(p0 + R t, R + r)."
    )

    print(
        f"For r = {footprint_radius:.6f} m:"
    )

    print(
        f"    enlarged safe-start radius = "
        f"{R + footprint_radius:.6f} m"
    )

    print()
    print(
        f"Failed cases: {len(failures)}"
    )

    for failure in failures:
        print(
            f"  theta0={failure['theta_degrees']:.1f} deg: "
            f"{failure['error_type']}: "
            f"{failure['message']}"
        )

    # ------------------------------------------------------------
    # Save and display
    # ------------------------------------------------------------

    if output_path is not None:
        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0.04,
        )

    plt.show()

    return (
        figure,
        axes,
        results_by_tau0,
        failures,
    )


# ================================================================
# Example
# ================================================================

if __name__ == "__main__":
    bicycle = Bicycle(
        model="Bicycle circular"
    )

    R = bicycle.max_radius

    x0 = 0.0
    y0 = 0.0

    target_circle = IntermediateCircle(
        center=Point(
            0.0,
            5.0,
        ),
        radius=R,
        turn_direction=+1,
        corner_point=Point(
            0.5,
            0.5,
        ),
    )

    output_directory = (
        Path(__file__).resolve().parent
        / "saved_figures"
    )

    plot_candidate_disk_analysis(
        bicycle=bicycle,
        target_circle=target_circle,
        x0=x0,
        y0=y0,
        footprint_radius=0.5,
        angle_step_degrees=1,
        trajectory_alpha=0.045,
        trajectory_linewidth=0.35,
        containment_tolerance=1e-9,
        output_path=(
            output_directory
            / "candidate_safe_start_disk.pdf"
        ),
    )