from fractions import Fraction
from math import cos, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle, Wedge
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker

from kappa_planner.geometry import Point, Pose
from kappa_planner.vehicle import Unicycle
from kappa_planner.helpers.plot_helpers import (
    plot_analytical_trajectory,
)
from kappa_planner.helpers.pose_to_pose_unicycle import (
    compute_all_pose_to_pose_trajectories,
)


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Boundary poses
# -----------------------------------------------------------------------------

START_X = 0.0
START_Y = 0.0
START_THETA_DEG = 135.0

END_X = 5.0
END_Y = 0.0
END_THETA_DEG = 135.0


# -----------------------------------------------------------------------------
# Vehicle
# -----------------------------------------------------------------------------

VEHICLE_WIDTH = 0.430
VEHICLE_LENGTH = 0.430

V_MAX = 1.0
OMEGA_MAX = 1.0


# -----------------------------------------------------------------------------
# Figure content
# -----------------------------------------------------------------------------

PLOT_SUPPORTING_CIRCLES = False

# Show the pose at the end of each primitive.
PLOT_PRIMITIVE_ARROWS = True

PLOT_TURN_SECTORS = True

SHOW_TRAVERSAL_TIME = True

# Useful while searching for a good boundary-pose configuration.
SHOW_TURN_AMPLITUDES = False


# -----------------------------------------------------------------------------
# Typography
# -----------------------------------------------------------------------------
#
# These are the main values to change when tuning the thesis figure.
#

FONT_SIZE_BASE = 15
FONT_SIZE_DIRECTION = 17
FONT_SIZE_FAMILY = 17
FONT_SIZE_TIME = 16
FONT_SIZE_TURN_AMPLITUDE = 11
FONT_SIZE_PROBLEM_PARAMETERS = 15
FONT_SIZE_POSE_LEGEND = 14
FONT_SIZE_BEST_LABEL = 12

SHOW_PROBLEM_PARAMETERS = True
# Gap above the column headings, in points.
PROBLEM_PARAMETERS_PADDING = 12

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": FONT_SIZE_BASE,
})


# -----------------------------------------------------------------------------
# Figure layout
# -----------------------------------------------------------------------------

FIGURE_WIDTH = 11.0
FIGURE_HEIGHT = 6.0

# Negative values deliberately bring panels closer together.
PANEL_HORIZONTAL_SPACE = -0.12
PANEL_VERTICAL_SPACE = 0.08

FIGURE_LEFT = 0.055
FIGURE_RIGHT = 0.995
FIGURE_BOTTOM = 0.025
FIGURE_TOP = 0.985

AXIS_PADDING_FRACTION = 0.035

GRID_COLOR = "0.80"
GRID_LINEWIDTH = 0.4


# -----------------------------------------------------------------------------
# Text placement
# -----------------------------------------------------------------------------

# Turn-direction heading above each column (relative to the top-row axes).
DIRECTION_LABEL_X = 0.50
DIRECTION_LABEL_Y = 1.40

# Traversal time below the trajectory geometry.
TIME_LABEL_X = 0.50
TIME_LABEL_Y = -0.055

# Optional turn-amplitude information.
TURN_AMPLITUDE_X = 0.50
TURN_AMPLITUDE_Y = 0.86

# Family label on left-hand side of each row.
FAMILY_LABEL_X = -0.07
FAMILY_LABEL_Y = 0.50


# -----------------------------------------------------------------------------
# Trajectory appearance
# -----------------------------------------------------------------------------

NORMAL_TRAJECTORY_COLOR = "tab:blue"
BEST_TRAJECTORY_COLOR = NORMAL_TRAJECTORY_COLOR

NORMAL_LINEWIDTH = 1.8
BEST_LINEWIDTH = NORMAL_LINEWIDTH

# Turn-on-the-spot sector.
TURN_SECTOR_ALPHA = 0.16
TURN_SECTOR_EDGE_ALPHA = 0.45


# -----------------------------------------------------------------------------
# Boundary-pose appearance
# -----------------------------------------------------------------------------

START_POSE_COLOR = "tab:green"
END_POSE_COLOR = "tab:red"

BOUNDARY_POSITION_MARKERSIZE = 4

# Length of the prescribed-heading arrows, in world coordinates.
BOUNDARY_HEADING_LENGTH = 0.8

BOUNDARY_HEADING_LINEWIDTH = 2.0

# Mutation scale controls the arrowhead size.
BOUNDARY_HEADING_ARROWHEAD_SIZE = 7

# Shared arrow shape: a longer, narrower head with crisp corners.
POSE_ARROW_STYLE = "-|>,head_length=0.6,head_width=0.2"

# Primitive-end poses, colored to match the trajectory.
PRIMITIVE_POSITION_MARKERSIZE = 4
PRIMITIVE_HEADING_LENGTH = 0.8
PRIMITIVE_HEADING_LINEWIDTH = 2.0
PRIMITIVE_HEADING_ARROWHEAD_SIZE = 7


# -----------------------------------------------------------------------------
# Best-candidate highlight
# -----------------------------------------------------------------------------

BEST_BOX_LINEWIDTH = 1.2
BEST_BOX_COLOR = "tab:green"
BEST_BOX_LINESTYLE = "--"

BEST_BOX_LEFT = 0.085
BEST_BOX_BOTTOM = -0.35
BEST_BOX_WIDTH = 0.830
BEST_BOX_HEIGHT = 1.335


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

SAVE_FIGURE = True
SHOW_FIGURE = True

# Save in the same directory as this script.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent

OUTPUT_FILENAME = "all_candidate_trajectories"


# =============================================================================
# CANDIDATE ORDER
# =============================================================================

FAMILY_ORDER = [
    "CSC",
    "TCSC",
    "CSCT",
    "TCSCT",
]

TURN_DIRECTION_ORDER = [
    (+1, +1),
    (+1, -1),
    (-1, +1),
    (-1, -1),
]


# =============================================================================
# BASIC HELPERS
# =============================================================================

def degrees_to_radians(angle_deg):
    """Convert degrees to radians."""

    return angle_deg * pi / 180.0


def format_radians(angle):
    """Format simple multiples of pi exactly, otherwise use numeric radians."""
    multiple = Fraction(float(angle) / pi).limit_denominator(24)
    if abs(float(multiple) * pi - float(angle)) > 1e-10:
        return f"{float(angle):.4g}"
    numerator, denominator = multiple.numerator, multiple.denominator
    if numerator == 0:
        return "0"
    sign = "-" if numerator < 0 else ""
    coefficient = "" if abs(numerator) == 1 else str(abs(numerator))
    expression = rf"{coefficient}\pi"
    if denominator != 1:
        expression = rf"{{{expression}}}/{{{denominator}}}"
    return sign + expression


def tau_symbol(tau):
    """Return the symbolic sign of a turn direction."""

    if tau == +1:
        return "+"

    if tau == -1:
        return "-"

    raise ValueError(
        f"Unexpected turn direction: {tau}"
    )


def get_candidate_lookup(trajectories):
    """
    Reorganize the dictionary returned by
    compute_all_pose_to_pose_trajectories() according to

        (family, tau0, tauf).

    Each stored entry preserves the original candidate dictionary and adds
    its dictionary name.
    """

    lookup = {}

    for name, data in trajectories.items():

        family = data["type"]
        tau0 = int(data["tau0"])
        tauf = int(data["tauf"])

        key = (
            family,
            tau0,
            tauf,
        )

        if key in lookup:
            raise RuntimeError(
                "More than one analytical candidate was found for "
                f"{key}."
            )

        lookup[key] = {
            **data,
            "name": name,
        }

    return lookup


def get_best_candidate(trajectories):
    """Return the minimum-time analytical candidate."""

    return min(
        trajectories.items(),
        key=lambda item: float(
            item[1]["time"]
        ),
    )


# =============================================================================
# TURN INFORMATION
# =============================================================================

def get_turn_amplitudes_string(
    trajectory,
):
    """
    Return a compact description of the turn-on-the-spot amplitudes.

    This option is primarily intended while searching for boundary poses that
    do not produce visually excessive boundary rotations.
    """

    from math import degrees

    amplitudes = []

    for primitive in trajectory:

        if primitive.label != "turn on-the-spot":
            continue

        angle_deg = degrees(
            abs(
                float(
                    primitive.delta_angle
                )
            )
        )

        amplitudes.append(
            angle_deg
        )

    if not amplitudes:
        return ""

    return ", ".join(
        rf"$\phi={angle:.0f}^\circ$"
        for angle in amplitudes
    )


# =============================================================================
# BOUNDARY POSES
# =============================================================================

def annotate_boundary_pose(
    ax,
    pose,
    color,
    marker_size=BOUNDARY_POSITION_MARKERSIZE,
    heading_length=BOUNDARY_HEADING_LENGTH,
    heading_linewidth=BOUNDARY_HEADING_LINEWIDTH,
    arrowhead_size=BOUNDARY_HEADING_ARROWHEAD_SIZE,
):
    """
    Draw a prescribed boundary position and its heading.

    These arrows are intentionally generated here rather than relying on
    primitive arrows from the trajectory plotting helper.
    """

    x = float(pose.x)
    y = float(pose.y)

    ax.plot(
        x,
        y,
        marker="o",
        markersize=marker_size,
        markerfacecolor=color,
        markeredgecolor=color,
        linestyle="None",
        zorder=100,
    )

    dx = (
        heading_length
        * cos(pose.theta)
    )

    dy = (
        heading_length
        * sin(pose.theta)
    )

    ax.annotate(
        "",
        xy=(
            x + dx,
            y + dy,
        ),
        xytext=(
            x,
            y,
        ),
        arrowprops={
            "arrowstyle": POSE_ARROW_STYLE,
            "joinstyle": "miter",
            "capstyle": "butt",
            "linewidth": heading_linewidth,
            "color": color,
            "mutation_scale": arrowhead_size,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        zorder=101,
    )


# =============================================================================
# TURN-SECTOR COLOR
# =============================================================================

def recolor_new_turn_sectors(
    ax,
    existing_patch_count,
    trajectory_color,
):
    """
    Recolor turn-on-the-spot sectors generated by
    plot_analytical_trajectory().

    The plotting helper may use its own default sector color. For the thesis
    overview, the sector is instead matched to the color of the corresponding
    analytical candidate.

    Only Wedge patches added during the current trajectory plot are modified.
    """

    new_patches = ax.patches[
        existing_patch_count:
    ]

    for patch in new_patches:

        if not isinstance(
            patch,
            Wedge,
        ):
            continue

        patch.set_facecolor(
            trajectory_color
        )

        patch.set_alpha(
            TURN_SECTOR_ALPHA
        )

        patch.set_edgecolor(
            trajectory_color
        )

        # Edge visibility is controlled independently.
        patch.set_linewidth(
            0.8
        )


# =============================================================================
# BEST-CANDIDATE BOX
# =============================================================================

def add_best_candidate_box(
    ax,
):
    """
    Draw a box around the minimum-time candidate using axis coordinates.
    """

    rectangle = Rectangle(
        (
            BEST_BOX_LEFT,
            BEST_BOX_BOTTOM,
        ),
        BEST_BOX_WIDTH,
        BEST_BOX_HEIGHT,
        transform=ax.transAxes,
        fill=False,
        linewidth=BEST_BOX_LINEWIDTH,
        linestyle=BEST_BOX_LINESTYLE,
        edgecolor=BEST_BOX_COLOR,
        clip_on=False,
        zorder=200,
    )

    ax.add_patch(
        rectangle
    )


# =============================================================================
# COMMON AXIS LIMITS
# =============================================================================

def apply_common_axis_limits(
    axes,
    padding_fraction=AXIS_PADDING_FRACTION,
):
    """
    Give every candidate panel identical geometric limits.

    This guarantees that a supporting circle of radius R has the same
    displayed size in every subplot.
    """

    active_axes = [
        ax
        for ax in axes.flat
        if ax.has_data()
    ]

    if not active_axes:
        return

    x_limits = [
        ax.get_xlim()
        for ax in active_axes
    ]

    y_limits = [
        ax.get_ylim()
        for ax in active_axes
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

    x_min -= padding_fraction * x_span
    x_max += padding_fraction * x_span

    y_min -= padding_fraction * y_span
    y_max += padding_fraction * y_span

    for ax in active_axes:

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
# PLOT ONE CANDIDATE
# =============================================================================

def plot_candidate(
    ax,
    candidate,
    start_pose,
    end_pose,
    is_best,
):
    """Plot one of the sixteen analytical candidates."""

    trajectory = candidate[
        "trajectory"
    ]

    total_time = float(
        candidate[
            "time"
        ]
    )

    trajectory_color = (
        BEST_TRAJECTORY_COLOR
        if is_best
        else NORMAL_TRAJECTORY_COLOR
    )

    trajectory_linewidth = (
        BEST_LINEWIDTH
        if is_best
        else NORMAL_LINEWIDTH
    )

    # Record the pre-existing patches so that any newly generated turn
    # sectors can subsequently be recolored.
    existing_patch_count = len(
        ax.patches
    )

    plot_analytical_trajectory(
        trajectory,
        figure=ax,
        plot_circles=PLOT_SUPPORTING_CIRCLES,
        color=trajectory_color,
        linewidth=trajectory_linewidth,
        plot_primitive_arrows=False,
        plot_turn_sectors=True,
        turn_sector_color=trajectory_color,
    )

    if PLOT_TURN_SECTORS:
        recolor_new_turn_sectors(
            ax=ax,
            existing_patch_count=existing_patch_count,
            trajectory_color=trajectory_color,
        )

    if PLOT_PRIMITIVE_ARROWS:
        for primitive in trajectory:
            annotate_boundary_pose(
                ax=ax,
                pose=Pose(Point(primitive.xf, primitive.yf), primitive.thetaf),
                color=trajectory_color,
                marker_size=PRIMITIVE_POSITION_MARKERSIZE,
                heading_length=PRIMITIVE_HEADING_LENGTH,
                heading_linewidth=PRIMITIVE_HEADING_LINEWIDTH,
                arrowhead_size=PRIMITIVE_HEADING_ARROWHEAD_SIZE,
            )

    # Draw prescribed boundary poses after the analytical trajectory so that
    # they remain visible in every panel.
    annotate_boundary_pose(
        ax=ax,
        pose=start_pose,
        color=START_POSE_COLOR,
    )

    annotate_boundary_pose(
        ax=ax,
        pose=end_pose,
        color=END_POSE_COLOR,
    )

    # -------------------------------------------------------------------------
    # Traversal time
    # -------------------------------------------------------------------------

    if SHOW_TRAVERSAL_TIME:

        ax.text(
            TIME_LABEL_X,
            TIME_LABEL_Y,
            rf"$\mathcal{{T}}={total_time:.3f}\,\mathrm{{s}}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=FONT_SIZE_TIME,
        )

    # -------------------------------------------------------------------------
    # Optional turn-angle diagnostic
    # -------------------------------------------------------------------------

    if SHOW_TURN_AMPLITUDES:

        turn_text = (
            get_turn_amplitudes_string(
                trajectory
            )
        )

        if turn_text:

            ax.text(
                TURN_AMPLITUDE_X,
                TURN_AMPLITUDE_Y,
                turn_text,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=FONT_SIZE_TURN_AMPLITUDE,
            )

    # -------------------------------------------------------------------------
    # Minimum-time candidate
    # -------------------------------------------------------------------------

    if is_best:

        add_best_candidate_box(
            ax
        )
        ax.annotate(
            "minimum-time candidate",
            xy=(TIME_LABEL_X, TIME_LABEL_Y),
            xycoords="axes fraction",
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=BEST_BOX_COLOR,
            fontsize=FONT_SIZE_BEST_LABEL,
            annotation_clip=False,
        )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_axis_off()


# =============================================================================
# COMPLETE 4 x 4 FIGURE
# =============================================================================

def create_all_candidates_figure(
    trajectories,
    start_pose,
    end_pose,
):
    """Create the complete sixteen-candidate overview."""

    candidate_lookup = (
        get_candidate_lookup(
            trajectories
        )
    )

    best_name, best_data = (
        get_best_candidate(
            trajectories
        )
    )

    figure, axes = plt.subplots(
        4,
        4,
        figsize=(
            FIGURE_WIDTH,
            FIGURE_HEIGHT,
        ),
        gridspec_kw={
            "wspace": PANEL_HORIZONTAL_SPACE,
            "hspace": PANEL_VERTICAL_SPACE,
        },
    )

    # -------------------------------------------------------------------------
    # Candidate panels
    # -------------------------------------------------------------------------

    for row_index, family in enumerate(
        FAMILY_ORDER
    ):

        for column_index, (
            tau0,
            tauf,
        ) in enumerate(
            TURN_DIRECTION_ORDER
        ):

            ax = axes[
                row_index,
                column_index,
            ]

            key = (
                family,
                tau0,
                tauf,
            )

            if key not in candidate_lookup:

                ax.text(
                    0.5,
                    0.5,
                    "Candidate\nunavailable",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE_TIME,
                )

                ax.set_axis_off()

                continue

            candidate = candidate_lookup[
                key
            ]

            is_best = (
                candidate["name"]
                == best_name
            )

            plot_candidate(
                ax=ax,
                candidate=candidate,
                start_pose=start_pose,
                end_pose=end_pose,
                is_best=is_best,
            )

    # -------------------------------------------------------------------------
    # Column headings
    # -------------------------------------------------------------------------

    for column_index, (tau0, tauf) in enumerate(TURN_DIRECTION_ORDER):
        ax = axes[0, column_index]
        ax.text(
            DIRECTION_LABEL_X,
            DIRECTION_LABEL_Y,
            rf"$(\tau_0,\tau_f)=({tau_symbol(tau0)},{tau_symbol(tauf)})$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_DIRECTION,
            clip_on=False,
        )

    # -------------------------------------------------------------------------
    # Family labels
    # -------------------------------------------------------------------------

    for row_index, family in enumerate(
        FAMILY_ORDER
    ):

        axes[
            row_index,
            0,
        ].text(
            FAMILY_LABEL_X,
            FAMILY_LABEL_Y,
            rf"${family}$",
            transform=axes[
                row_index,
                0,
            ].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=FONT_SIZE_FAMILY,
        )

    # -------------------------------------------------------------------------
    # Common geometric scale
    # -------------------------------------------------------------------------

    apply_common_axis_limits(
        axes
    )

    # -------------------------------------------------------------------------
    # Final layout
    # -------------------------------------------------------------------------

    figure.subplots_adjust(
        left=FIGURE_LEFT,
        right=FIGURE_RIGHT,
        bottom=FIGURE_BOTTOM,
        top=FIGURE_TOP,
        wspace=PANEL_HORIZONTAL_SPACE,
        hspace=PANEL_VERTICAL_SPACE,
    )

    # Subtle table separators, placed between the rendered panel contents.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    positions = [ax.get_position() for ax in axes[0]]
    row_bottoms = [
        min(
            [ax.get_position().y0 for ax in row]
            + [
                text.get_window_extent(renderer).y0 / figure.bbox.height
                for ax in row
                for text in ax.texts
                if text.get_text().startswith(r"$\mathcal{T}=")
            ]
        )
        for row in axes
    ]
    grid_top = max(
        text.get_window_extent(renderer).y1 / figure.bbox.height
        for ax in axes[0]
        for text in ax.texts
    )
    grid_bottom = row_bottoms[-1] - 0.015
    # Internal column separators only; leave both outer edges open.
    for left, right in zip(positions, positions[1:]):
        x = (left.x1 + right.x0) / 2
        figure.add_artist(Line2D(
            [x, x], [grid_bottom, grid_top],
            transform=figure.transFigure, color=GRID_COLOR,
            linewidth=GRID_LINEWIDTH, zorder=0, clip_on=False,
        ))
    for row_index in range(len(axes) - 1):
        y = (row_bottoms[row_index] + axes[row_index + 1, 0].get_position().y1) / 2
        figure.add_artist(Line2D(
            [positions[0].x0, positions[-1].x1], [y, y],
            transform=figure.transFigure, color=GRID_COLOR,
            linewidth=GRID_LINEWIDTH, zorder=0, clip_on=False,
        ))

    if SHOW_PROBLEM_PARAMETERS:
        # Place the parameter line above the rendered headings without
        # changing the compact panel layout. The tight export includes it.
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        heading_top = max(
            text.get_window_extent(renderer).y1
            for ax in axes[0]
            for text in ax.texts
        )
        parameters_y = (
            heading_top + PROBLEM_PARAMETERS_PADDING * figure.dpi / 72
        ) / figure.bbox.height
        parameter_values = TextArea(
            (
                rf"$\boldsymbol{{x}}_0=[{float(start_pose.x):g},\,{float(start_pose.y):g},\,"
                rf"{format_radians(start_pose.theta)}]^{{\top}}$"
                rf"$\qquad \boldsymbol{{x}}_f=[{float(end_pose.x):g},\,{float(end_pose.y):g},\,"
                rf"{format_radians(end_pose.theta)}]^{{\top}}$"
                rf"$\qquad v_{{\max}}={V_MAX:g}\,\mathrm{{m/s}}$"
                rf"$\qquad \omega_{{\max}}={OMEGA_MAX:g}\,\mathrm{{rad/s}}$"
                rf"$\qquad R={V_MAX / OMEGA_MAX:g}\,\mathrm{{m}}$"
            ),
            textprops={"fontsize": FONT_SIZE_PROBLEM_PARAMETERS},
        )
        parameter_row = HPacker(
            children=[
                TextArea("Problem parameters:", textprops={
                    "fontsize": FONT_SIZE_PROBLEM_PARAMETERS,
                    "fontweight": "bold",
                }),
                parameter_values,
            ],
            align="center", pad=0, sep=10,
        )
        legend_items = []
        for label, color, pose in [
            ("Initial pose:", START_POSE_COLOR, start_pose),
            ("Final pose:", END_POSE_COLOR, end_pose),
        ]:
            symbol = DrawingArea(36, 26, 0, 0)
            origin = (18, 13)
            tip = (
                origin[0] + 16 * cos(pose.theta),
                origin[1] + 16 * sin(pose.theta),
            )
            symbol.add_artist(Line2D(
                [origin[0]], [origin[1]], marker="o", linestyle="None",
                markersize=BOUNDARY_POSITION_MARKERSIZE, color=color,
            ))
            symbol.add_artist(FancyArrowPatch(
                origin, tip, arrowstyle=POSE_ARROW_STYLE,
                mutation_scale=BOUNDARY_HEADING_ARROWHEAD_SIZE,
                linewidth=BOUNDARY_HEADING_LINEWIDTH, color=color,
                joinstyle="miter", capstyle="butt", shrinkA=0, shrinkB=0,
            ))
            legend_items.append(HPacker(
                children=[TextArea(label, textprops={
                    "fontsize": FONT_SIZE_POSE_LEGEND,
                }), symbol],
                align="center", pad=0, sep=4,
            ))
        legend_row = HPacker(
            children=legend_items, align="center", pad=0, sep=28,
        )
        header = AnchoredOffsetbox(
            loc="lower center",
            child=VPacker(
                children=[parameter_row, legend_row],
                align="center", pad=0, sep=7,
            ),
            bbox_to_anchor=((FIGURE_LEFT + FIGURE_RIGHT) / 2, parameters_y),
            bbox_transform=figure.transFigure,
            frameon=False, pad=0, borderpad=0,
        )
        figure.add_artist(header)

    return (
        figure,
        best_name,
        best_data,
    )


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def print_candidate_summary(
    trajectories,
):
    """Print all candidate traversal times."""

    candidate_lookup = (
        get_candidate_lookup(
            trajectories
        )
    )

    best_name, best_data = (
        get_best_candidate(
            trajectories
        )
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "ANALYTICAL CANDIDATES"
    )

    print(
        "=" * 78
    )

    for family in FAMILY_ORDER:

        print(
            f"\n{family}"
        )

        for tau0, tauf in (
            TURN_DIRECTION_ORDER
        ):

            key = (
                family,
                tau0,
                tauf,
            )

            if key not in candidate_lookup:

                print(
                    f"  ({tau0:+d},{tauf:+d}) : unavailable"
                )

                continue

            candidate = candidate_lookup[
                key
            ]

            total_time = float(
                candidate[
                    "time"
                ]
            )

            marker = (
                "  <-- minimum"
                if candidate["name"]
                == best_name
                else ""
            )

            print(
                f"  ({tau0:+d},{tauf:+d}) "
                f": T = {total_time:.6f} s"
                f"{marker}"
            )

    print(
        "\n"
        + "-" * 78
    )

    print(
        f"Minimum-time candidate : "
        f"{best_name}"
    )

    print(
        f"Minimum traversal time : "
        f"{float(best_data['time']):.9f} s"
    )

    print(
        "=" * 78
        + "\n"
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Boundary poses
    # -------------------------------------------------------------------------

    start_pose = Pose(
        Point(
            START_X,
            START_Y,
        ),
        degrees_to_radians(
            START_THETA_DEG
        ),
    )

    end_pose = Pose(
        Point(
            END_X,
            END_Y,
        ),
        degrees_to_radians(
            END_THETA_DEG
        ),
    )

    # -------------------------------------------------------------------------
    # Vehicle
    # -------------------------------------------------------------------------

    unicycle = Unicycle(
        state=[
            0.0,
            0.0,
            0.0,
        ],
        width=VEHICLE_WIDTH,
        length=VEHICLE_LENGTH,
        v_max=V_MAX,
        v_min=0.0,
        omega_max=OMEGA_MAX,
        omega_min=-OMEGA_MAX,
    )

    # -------------------------------------------------------------------------
    # Compute all analytical candidates
    # -------------------------------------------------------------------------

    trajectories = (
        compute_all_pose_to_pose_trajectories(
            start_pose,
            end_pose,
            unicycle,
        )
    )

    print_candidate_summary(
        trajectories
    )

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------

    (
        figure,
        best_name,
        best_data,
    ) = create_all_candidates_figure(
        trajectories=trajectories,
        start_pose=start_pose,
        end_pose=end_pose,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    if SAVE_FIGURE:

        OUTPUT_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        pdf_path = (
            OUTPUT_DIRECTORY
            / f"{OUTPUT_FILENAME}.pdf"
        )

        png_path = (
            OUTPUT_DIRECTORY
            / f"{OUTPUT_FILENAME}.png"
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
            "\nFigure saved to:"
        )

        print(
            pdf_path.resolve()
        )

        print(
            png_path.resolve()
        )

    # -------------------------------------------------------------------------
    # Show
    # -------------------------------------------------------------------------

    if SHOW_FIGURE:

        plt.show(
            block=True
        )

    else:

        plt.close(
            figure
        )
