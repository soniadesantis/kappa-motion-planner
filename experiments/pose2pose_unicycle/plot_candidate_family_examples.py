from math import pi, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from kappa_planner.geometry import Point, Pose
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.pose_to_pose_unicycle import (
    compute_CSC_trajectory,
    compute_TCSC_trajectory,
    compute_CSCT_trajectory,
    compute_TCSCT_trajectory,
)
from kappa_planner.helpers.plot_helpers import plot_analytical_trajectory


# =============================================================================
# PLOT STYLE
# =============================================================================
# Temporary check: circle of radius 2R centered at the final trajectory point.
FINAL_POINT_CHECK_CIRCLE_COLOR = "tab:purple"
FINAL_POINT_CHECK_CIRCLE_LINEWIDTH = 1.2

ANNOTATION_FONTSIZE = 18
PANEL_TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 13

# Prescribed boundary poses.
START_POSE_COLOR = "tab:green"
END_POSE_COLOR = "tab:red"
BOUNDARY_POSE_MARKERSIZE = 9

# Tangency points.
GEOMETRIC_POINT_COLOR = "k"
GEOMETRIC_POINT_MARKERSIZE = 6

# Supporting-circle center markers.
CIRCLE_CENTER_COLOR = "k"
CIRCLE_CENTER_MARKERSIZE = 0.2

# Auxiliary constructions.
AUXILIARY_GEOMETRY_COLOR = "0.55"
AUXILIARY_TANGENT_COLOR = "0.35"
AUXILIARY_LINEWIDTH = 1.0
AUXILIARY_CENTER_MARKERSIZE = 0.2
AUXILIARY_TANGENCY_MARKERSIZE = 6

# Figure layout.
PANEL_HORIZONTAL_SPACE = -0.08
COMMON_AXIS_PADDING = 0.03
PANEL_TITLE_VERTICAL_POSITION = 0.98

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.titlesize": PANEL_TITLE_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
})


# =============================================================================
# USER CONFIGURATION
# =============================================================================

START_POSE = Pose(
    Point(0.0, 0.0),
    90 * pi / 180.0,
)

END_POSE = Pose(
    Point(5.0, 0.0),
    90 * pi / 180.0,
)

VEHICLE_WIDTH = 0.430
VEHICLE_LENGTH = 0.430
V_MAX = 1.0
OMEGA_MAX = 1.0

SAVE_FIGURES = True
SHOW_FIGURES = True

OUTPUT_DIRECTORY = "candidate_family_figures"

# Representative direction cases:
#
# (+1,+1): equal turn directions
# (+1,-1): opposite turn directions
#
# The remaining cases follow by reflection symmetry.
TURN_DIRECTION_CASES = [
    (+1, +1),
    (+1, -1),
]


# =============================================================================
# BASIC HELPERS
# =============================================================================

def tau_symbol(tau):
    """Return the symbolic sign associated with a turn direction."""
    return "+" if tau == 1 else "-"


def get_arc_primitives(trajectory):
    """Return all circular-arc primitives in a trajectory."""
    return [
        primitive
        for primitive in trajectory
        if primitive.label == "arc"
    ]


def get_segment_primitive(trajectory):
    """Return the unique straight-segment primitive, when available."""
    segments = [
        primitive
        for primitive in trajectory
        if primitive.label == "segment"
    ]

    if len(segments) != 1:
        return None

    return segments[0]


def get_circle_center(arc):
    """Extract the supporting-circle center from an arc primitive."""
    if not hasattr(arc, "xc") or not hasattr(arc, "yc"):
        return None

    return float(arc.xc), float(arc.yc)


def reflect_point_across_line(
    point,
    line_point_1,
    line_point_2,
):
    """
    Reflect a point across the infinite line passing through two points.

    Returns ``None`` when the two line points coincide.
    """

    px, py = point
    x1, y1 = line_point_1
    x2, y2 = line_point_2

    dx = x2 - x1
    dy = y2 - y1

    squared_length = dx**2 + dy**2

    if squared_length <= 1e-12:
        return None

    projection_parameter = (
        (px - x1) * dx
        + (py - y1) * dy
    ) / squared_length

    projection_x = x1 + projection_parameter * dx
    projection_y = y1 + projection_parameter * dy

    reflected_x = 2.0 * projection_x - px
    reflected_y = 2.0 * projection_y - py

    return reflected_x, reflected_y


# =============================================================================
# GEOMETRIC ANNOTATIONS
# =============================================================================

def annotate_boundary_poses(
    ax,
    start_pose,
    end_pose,
):
    """Mark the prescribed initial and final positions."""

    ax.plot(
        start_pose.x,
        start_pose.y,
        marker="o",
        markersize=BOUNDARY_POSE_MARKERSIZE,
        markerfacecolor=START_POSE_COLOR,
        markeredgecolor=START_POSE_COLOR,
        linestyle="None",
        zorder=100,
    )

    ax.plot(
        end_pose.x,
        end_pose.y,
        marker="o",
        markersize=BOUNDARY_POSE_MARKERSIZE,
        markerfacecolor=END_POSE_COLOR,
        markeredgecolor=END_POSE_COLOR,
        linestyle="None",
        zorder=100,
    )


def annotate_supporting_circles(
    ax,
    trajectory,
):
    """Mark the initial and final supporting-circle centers."""

    arcs = get_arc_primitives(trajectory)

    if len(arcs) < 2:
        return

    initial_center = get_circle_center(arcs[0])
    final_center = get_circle_center(arcs[-1])

    for center in (initial_center, final_center):
        if center is None:
            continue

        ax.plot(
            center[0],
            center[1],
            marker="o",
            markersize=CIRCLE_CENTER_MARKERSIZE,
            markerfacecolor=CIRCLE_CENTER_COLOR,
            markeredgecolor=CIRCLE_CENTER_COLOR,
            linestyle="None",
            zorder=20,
        )


def annotate_tangency_points(
    ax,
    trajectory,
):
    """Mark the endpoints of the trajectory's straight segment."""

    segment = get_segment_primitive(trajectory)

    if segment is None:
        return

    required_attributes = (
        "x0",
        "y0",
        "xf",
        "yf",
    )

    if not all(
        hasattr(segment, name)
        for name in required_attributes
    ):
        return

    tangency_points = (
        (
            float(segment.x0),
            float(segment.y0),
        ),
        (
            float(segment.xf),
            float(segment.yf),
        ),
    )

    for point in tangency_points:
        ax.plot(
            point[0],
            point[1],
            marker="o",
            markersize=GEOMETRIC_POINT_MARKERSIZE,
            markerfacecolor=GEOMETRIC_POINT_COLOR,
            markeredgecolor=GEOMETRIC_POINT_COLOR,
            linestyle="None",
            zorder=30,
        )


# =============================================================================
# TCSC AND CSCT REFLECTED-CIRCLE GEOMETRY
# =============================================================================

def plot_final_point_check_circle(
    ax,
    family_name,
    end_pose,
    radius,
):
    """
    Temporarily draw a circle of radius 2R centered at the prescribed final
    trajectory position p_f.

    This construction is plotted only for TCSC candidates and is intended
    purely as a geometric check.
    """

    if family_name != "TCSC":
        return

    final_point_circle = Circle(
        (
            float(end_pose.x),
            float(end_pose.y),
        ),
        radius=2.0 * radius,
        fill=False,
        linestyle=(0, (6, 3)),
        linewidth=FINAL_POINT_CHECK_CIRCLE_LINEWIDTH,
        edgecolor=FINAL_POINT_CHECK_CIRCLE_COLOR,
        zorder=3,
    )

    ax.add_patch(final_point_circle)

    # Optional center marker at p_f. This will eventually be covered by the
    # red final-pose marker, which is drawn later.
    ax.plot(
        float(end_pose.x),
        float(end_pose.y),
        marker="x",
        markersize=7,
        markeredgewidth=1.2,
        color=FINAL_POINT_CHECK_CIRCLE_COLOR,
        linestyle="None",
        zorder=4,
    )

    # Temporary radius annotation.
    ax.annotate(
        r"$2R$",
        (
            float(end_pose.x) + 2.0 * radius,
            float(end_pose.y),
        ),
        xytext=(7, -14),
        textcoords="offset points",
        fontsize=ANNOTATION_FONTSIZE,
        color=FINAL_POINT_CHECK_CIRCLE_COLOR,
        zorder=5,
    )


def plot_reflected_geometry(
    ax,
    family_name,
    trajectory,
    start_pose,
    end_pose,
    radius,
    tau0,
    tauf,
):
    """
    Plot the reflected-circle construction for opposite-direction TCSC and
    CSCT candidates.
    """

    if family_name not in {"TCSC", "CSCT"}:
        return

    if tau0 == tauf:
        return

    arcs = get_arc_primitives(trajectory)

    if len(arcs) < 2:
        return

    segment = get_segment_primitive(trajectory)

    if segment is None:
        return

    required_segment_attributes = (
        "x0",
        "y0",
        "xf",
        "yf",
    )

    if not all(
        hasattr(segment, attribute)
        for attribute in required_segment_attributes
    ):
        return

    segment_start = (
        float(segment.x0),
        float(segment.y0),
    )

    segment_end = (
        float(segment.xf),
        float(segment.yf),
    )

    initial_center = get_circle_center(arcs[0])
    final_center = get_circle_center(arcs[-1])

    if initial_center is None or final_center is None:
        return

    if family_name == "TCSC":
        # The final supporting circle is fixed.
        fixed_center = final_center

        # The auxiliary tangent starts at the initial boundary position.
        boundary_point = (
            float(start_pose.x),
            float(start_pose.y),
        )

        auxiliary_label = (
            r"Auxiliary circle "
            r"$\mathcal{O}_f^{\mathrm{aux}}$"
        )

        reflected_circle_label = (
            r"Reflected circle "
            r"$\mathcal{O}_f^{\mathrm{ref}}$"
        )

        construction_line_label = (
            r"Tangent from $\mathbf{p}_0$"
        )

    else:
        # The initial supporting circle is fixed.
        fixed_center = initial_center

        # The auxiliary tangent starts at the final boundary position.
        boundary_point = (
            float(end_pose.x),
            float(end_pose.y),
        )

        auxiliary_label = (
            r"Auxiliary circle "
            r"$\mathcal{O}_0^{\mathrm{aux}}$"
        )

        reflected_circle_label = (
            r"Reflected circle "
            r"$\mathcal{O}_0^{\mathrm{ref}}$"
        )

        construction_line_label = (
            r"Tangent from $\mathbf{p}_f$"
        )

    reflected_center = reflect_point_across_line(
        point=fixed_center,
        line_point_1=segment_start,
        line_point_2=segment_end,
    )

    if reflected_center is None:
        return

    fixed_x, fixed_y = fixed_center
    boundary_x, boundary_y = boundary_point

    auxiliary_circle = Circle(
        fixed_center,
        radius=2.0 * radius,
        fill=False,
        linestyle="--",
        linewidth=AUXILIARY_LINEWIDTH,
        edgecolor=AUXILIARY_GEOMETRY_COLOR,
        label=auxiliary_label,
        zorder=1,
    )

    ax.add_patch(auxiliary_circle)

    reflected_circle = Circle(
        reflected_center,
        radius=radius,
        fill=False,
        linestyle="-.",
        linewidth=AUXILIARY_LINEWIDTH,
        edgecolor=AUXILIARY_GEOMETRY_COLOR,
        label=reflected_circle_label,
        zorder=2,
    )

    ax.add_patch(reflected_circle)

    ax.plot(
        [
            boundary_x,
            reflected_center[0],
        ],
        [
            boundary_y,
            reflected_center[1],
        ],
        linestyle=":",
        linewidth=AUXILIARY_LINEWIDTH,
        color=AUXILIARY_TANGENT_COLOR,
        label=construction_line_label,
        zorder=2,
    )

    ax.plot(
        reflected_center[0],
        reflected_center[1],
        marker="o",
        markersize=AUXILIARY_CENTER_MARKERSIZE,
        markerfacecolor=AUXILIARY_GEOMETRY_COLOR,
        markeredgecolor=AUXILIARY_GEOMETRY_COLOR,
        linestyle="None",
        zorder=20,
    )

    ax.annotate(
        r"$2R$",
        (
            fixed_x + 2.0 * radius,
            fixed_y,
        ),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=ANNOTATION_FONTSIZE,
        color=AUXILIARY_GEOMETRY_COLOR,
        zorder=21,
    )


# =============================================================================
# TCSCT AUXILIARY-CIRCLE GEOMETRY
# =============================================================================

def plot_tcsct_auxiliary_geometry(
    ax,
    family_name,
    trajectory,
    start_pose,
    end_pose,
    radius,
    tau0,
    tauf,
):
    """
    Plot the auxiliary reflected-circle construction for an
    opposite-direction TCSCT candidate.

    The construction includes:

        - the auxiliary circle centered at p_f, with radius 2R;
        - the tangent drawn from p_0 to the auxiliary circle;
        - its tangency point q;
        - the reflected final supporting circle.

    The reflected circle and the actual final supporting circle are tangent
    at the final point of the straight trajectory segment.
    """

    if family_name != "TCSCT":
        return

    if tau0 == tauf:
        return

    arcs = get_arc_primitives(trajectory)

    if len(arcs) < 2:
        return

    initial_center = get_circle_center(arcs[0])
    final_center = get_circle_center(arcs[-1])

    if initial_center is None or final_center is None:
        return

    segment = get_segment_primitive(trajectory)

    if segment is None:
        return

    required_segment_attributes = (
        "x0",
        "y0",
        "xf",
        "yf",
    )

    if not all(
        hasattr(segment, attribute)
        for attribute in required_segment_attributes
    ):
        return

    x0 = float(start_pose.x)
    y0 = float(start_pose.y)

    xf = float(end_pose.x)
    yf = float(end_pose.y)

    dx = xf - x0
    dy = yf - y0

    distance = sqrt(
        dx**2 + dy**2
    )

    if distance < 2.0 * radius:
        return

    # -------------------------------------------------------------------------
    # Tangency point q on the auxiliary circle
    # -------------------------------------------------------------------------

    # The direction from p_0 to o_0 identifies the selected tangent branch.
    tangent_direction_x = initial_center[0] - x0
    tangent_direction_y = initial_center[1] - y0

    tangent_direction_norm = sqrt(
        tangent_direction_x**2
        + tangent_direction_y**2
    )

    if tangent_direction_norm <= 1e-12:
        return

    unit_tangent_x = (
        tangent_direction_x
        / tangent_direction_norm
    )

    unit_tangent_y = (
        tangent_direction_y
        / tangent_direction_norm
    )

    tangent_length = sqrt(
        max(
            distance**2 - (2.0 * radius)**2,
            0.0,
        )
    )

    auxiliary_tangency_point = (
        x0 + tangent_length * unit_tangent_x,
        y0 + tangent_length * unit_tangent_y,
    )

    # -------------------------------------------------------------------------
    # Reflected final-circle center
    # -------------------------------------------------------------------------

    # This is the final point of the actual straight segment, i.e. the
    # tangency point between the straight segment and the final circular arc.
    segment_final_point = (
        float(segment.xf),
        float(segment.yf),
    )

    # The actual and reflected final circles are tangent at this point.
    reflected_center = (
        2.0 * segment_final_point[0] - final_center[0],
        2.0 * segment_final_point[1] - final_center[1],
    )

    # -------------------------------------------------------------------------
    # Auxiliary circle centered at p_f, with radius 2R
    # -------------------------------------------------------------------------

    auxiliary_circle = Circle(
        (xf, yf),
        radius=2.0 * radius,
        fill=False,
        linestyle="--",
        linewidth=AUXILIARY_LINEWIDTH,
        edgecolor=AUXILIARY_GEOMETRY_COLOR,
        label=r"Auxiliary circle $\mathcal{A}_f$",
        zorder=1,
    )

    ax.add_patch(auxiliary_circle)

    # -------------------------------------------------------------------------
    # Tangent from p_0 to the auxiliary circle
    # -------------------------------------------------------------------------

    ax.plot(
        [
            x0,
            auxiliary_tangency_point[0],
        ],
        [
            y0,
            auxiliary_tangency_point[1],
        ],
        linestyle=":",
        linewidth=AUXILIARY_LINEWIDTH,
        color=AUXILIARY_TANGENT_COLOR,
        label=r"Tangent from $\mathbf{p}_0$",
        zorder=2,
    )

    # Tangency point on the auxiliary circle.
    ax.plot(
        auxiliary_tangency_point[0],
        auxiliary_tangency_point[1],
        marker="o",
        markersize=AUXILIARY_TANGENCY_MARKERSIZE,
        markerfacecolor=AUXILIARY_GEOMETRY_COLOR,
        markeredgecolor=AUXILIARY_GEOMETRY_COLOR,
        linestyle="None",
        zorder=20,
    )

    # -------------------------------------------------------------------------
    # Reflected final supporting circle
    # -------------------------------------------------------------------------

    reflected_circle = Circle(
        reflected_center,
        radius=radius,
        fill=False,
        linestyle="-.",
        linewidth=AUXILIARY_LINEWIDTH,
        edgecolor=AUXILIARY_GEOMETRY_COLOR,
        label=r"Reflected circle $\mathcal{O}_f^{\mathrm{ref}}$",
        zorder=2,
    )

    ax.add_patch(reflected_circle)

    ax.plot(
        reflected_center[0],
        reflected_center[1],
        marker="o",
        markersize=AUXILIARY_CENTER_MARKERSIZE,
        markerfacecolor=AUXILIARY_GEOMETRY_COLOR,
        markeredgecolor=AUXILIARY_GEOMETRY_COLOR,
        linestyle="None",
        zorder=20,
    )

    # -------------------------------------------------------------------------
    # Auxiliary-circle radius label
    # -------------------------------------------------------------------------

    ax.annotate(
        r"$2R$",
        (
            xf + 2.0 * radius,
            yf,
        ),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=ANNOTATION_FONTSIZE,
        color=AUXILIARY_GEOMETRY_COLOR,
        zorder=21,
    )

# =============================================================================
# INDIVIDUAL CANDIDATE PLOT
# =============================================================================

def plot_candidate_on_axis(
    ax,
    family_name,
    trajectory,
    tau0,
    tauf,
    start_pose,
    end_pose,
    radius,
):
    """Plot one candidate and its relevant geometric construction."""

    plot_analytical_trajectory(
        trajectory,
        figure=ax,
        plot_circles=True,
        linewidth=2.2,
        plot_primitive_arrows=True,
        plot_turn_sectors=True,
    )

    annotate_supporting_circles(
        ax=ax,
        trajectory=trajectory,
    )

    annotate_tangency_points(
        ax=ax,
        trajectory=trajectory,
    )

    # Reflected-circle constructions for TCSC and CSCT.
    plot_reflected_geometry(
        ax=ax,
        family_name=family_name,
        trajectory=trajectory,
        start_pose=start_pose,
        end_pose=end_pose,
        radius=radius,
        tau0=tau0,
        tauf=tauf,
    )


    # Temporary geometric check: radius-2R circle centered at the prescribed
    # final trajectory position p_f. This is drawn only for TCSC.
    # plot_final_point_check_circle(
    #     ax=ax,
    #     family_name=family_name,
    #     end_pose=end_pose,
    #     radius=radius,
    # )

    # Auxiliary-circle construction for TCSCT.
    plot_tcsct_auxiliary_geometry(
        ax=ax,
        family_name=family_name,
        trajectory=trajectory,
        start_pose=start_pose,
        end_pose=end_pose,
        radius=radius,
        tau0=tau0,
        tauf=tauf,
    )

    # Draw the prescribed boundary poses last.
    annotate_boundary_poses(
        ax=ax,
        start_pose=start_pose,
        end_pose=end_pose,
    )

    ax.set_title(
        rf"$(\tau_0,\tau_f)="
        rf"({tau_symbol(tau0)},{tau_symbol(tauf)})$",
        fontsize=PANEL_TITLE_FONTSIZE,
        pad=2,
        y=PANEL_TITLE_VERTICAL_POSITION,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_axis_off()


# =============================================================================
# COMMON AXIS LIMITS
# =============================================================================

def apply_common_axis_limits(
    axes,
    padding=COMMON_AXIS_PADDING,
):
    """
    Apply identical x and y limits to all axes containing plotted data.

    This guarantees the same geometric scale in every panel, so that a circle
    of radius R has the same displayed size in each subplot.
    """

    data_axes = [
        ax
        for ax in axes
        if ax.has_data()
    ]

    if not data_axes:
        return

    x_limits = [
        ax.get_xlim()
        for ax in data_axes
    ]

    y_limits = [
        ax.get_ylim()
        for ax in data_axes
    ]

    x_min = min(
        limits[0]
        for limits in x_limits
    )

    x_max = max(
        limits[1]
        for limits in x_limits
    )

    y_min = min(
        limits[0]
        for limits in y_limits
    )

    y_max = max(
        limits[1]
        for limits in y_limits
    )

    x_span = x_max - x_min
    y_span = y_max - y_min

    if x_span <= 0.0:
        x_span = 1.0

    if y_span <= 0.0:
        y_span = 1.0

    x_min -= padding * x_span
    x_max += padding * x_span

    y_min -= padding * y_span
    y_max += padding * y_span

    for ax in data_axes:
        ax.set_xlim(
            x_min,
            x_max,
        )

        ax.set_ylim(
            y_min,
            y_max,
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )


# =============================================================================
# ONE TWO-PANEL FIGURE FOR EACH FAMILY
# =============================================================================

def create_family_figure(
    family_name,
    compute_function,
    start_pose,
    end_pose,
    unicycle,
):
    """Create one two-panel figure for a candidate trajectory family."""

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.3),
        gridspec_kw={
            "wspace": PANEL_HORIZONTAL_SPACE,
        },
    )

    axes = axes.flatten()

    for ax, (tau0, tauf) in zip(
        axes,
        TURN_DIRECTION_CASES,
    ):
        try:
            trajectory, total_time = compute_function(
                start_pose,
                end_pose,
                tau0,
                tauf,
                unicycle,
            )

        except ValueError as error:
            ax.text(
                0.5,
                0.5,
                "Candidate unavailable\n"
                f"{error}",
                ha="center",
                va="center",
                fontsize=ANNOTATION_FONTSIZE,
                transform=ax.transAxes,
            )

            ax.set_title(
                rf"$(\tau_0,\tau_f)="
                rf"({tau_symbol(tau0)},{tau_symbol(tauf)})$",
                fontsize=PANEL_TITLE_FONTSIZE,
                pad=2,
                y=PANEL_TITLE_VERTICAL_POSITION,
            )

            ax.set_axis_off()
            continue

        plot_candidate_on_axis(
            ax=ax,
            family_name=family_name,
            trajectory=trajectory,
            tau0=tau0,
            tauf=tauf,
            start_pose=start_pose,
            end_pose=end_pose,
            radius=unicycle.max_radius,
        )

        print(
            f"{family_name:5s} | "
            f"({tau0:+d},{tauf:+d}) | "
            f"T = {total_time:.6f} s"
        )

    apply_common_axis_limits(
        axes=axes,
        padding=COMMON_AXIS_PADDING,
    )

    # Main figure title intentionally omitted.
    # Legend intentionally omitted so that labels can be added in Inkscape.

    if family_name in {
        "TCSC",
        "CSCT",
        "TCSCT",
    }:
        # handles, labels = axes[1].get_legend_handles_labels()
        #
        # if handles:
        #     fig.legend(
        #         handles,
        #         labels,
        #         loc="lower center",
        #         bbox_to_anchor=(0.5, 0.01),
        #         ncol=3,
        #         frameon=False,
        #         fontsize=LEGEND_FONTSIZE,
        #         handlelength=2.8,
        #         columnspacing=1.8,
        #     )

        fig.subplots_adjust(
            left=0.01,
            right=0.99,
            bottom=0.08,
            top=0.93,
            wspace=PANEL_HORIZONTAL_SPACE,
        )

    else:
        fig.subplots_adjust(
            left=0.01,
            right=0.99,
            bottom=0.01,
            top=0.93,
            wspace=PANEL_HORIZONTAL_SPACE,
        )

    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    unicycle = Unicycle(
        state=[0.0, 0.0, 0.0],
        width=VEHICLE_WIDTH,
        length=VEHICLE_LENGTH,
        v_max=V_MAX,
        v_min=0.0,
        omega_max=OMEGA_MAX,
        omega_min=-OMEGA_MAX,
    )

    family_functions = {
        "CSC": compute_CSC_trajectory,
        "TCSC": compute_TCSC_trajectory,
        "CSCT": compute_CSCT_trajectory,
        "TCSCT": compute_TCSCT_trajectory,
    }

    output_directory = (
        Path(__file__).resolve().parent
        / OUTPUT_DIRECTORY
    )

    if SAVE_FIGURES:
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    figures = []

    for family_name, compute_function in family_functions.items():
        figure = create_family_figure(
            family_name=family_name,
            compute_function=compute_function,
            start_pose=START_POSE,
            end_pose=END_POSE,
            unicycle=unicycle,
        )

        figures.append(figure)

        if SAVE_FIGURES:
            filename = (
                f"{family_name.lower()}"
                "_candidate_trajectories"
            )

            figure.savefig(
                output_directory / f"{filename}.pdf",
                bbox_inches="tight",
            )

            figure.savefig(
                output_directory / f"{filename}.png",
                dpi=300,
                bbox_inches="tight",
            )

    print(
        f"\nCreated {len(figures)} figures."
    )

    if SAVE_FIGURES:
        print(
            "Figures saved in: "
            f"{output_directory}"
        )

    if SHOW_FIGURES:
        plt.show(block=True)

    else:
        for figure in figures:
            plt.close(figure)