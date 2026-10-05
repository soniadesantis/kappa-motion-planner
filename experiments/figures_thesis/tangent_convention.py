"""Two-panel thesis tangent construction; run with --no-show for export."""

import argparse
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, ArrowStyle, Circle, FancyArrowPatch, Wedge


# =============================================================================
# CONFIGURATION
# =============================================================================
SYMBOL_FONT_SIZE = 22
TICK_LABEL_FONT_SIZE = 14
REFERENCE_LABEL_FONT_SIZE = SYMBOL_FONT_SIZE
POINT_LABEL_FONT_SIZE = SYMBOL_FONT_SIZE
SET_LABEL_FONT_SIZE = SYMBOL_FONT_SIZE
SEGMENT_LABEL_FONT_SIZE = SYMBOL_FONT_SIZE
TANGENCY_LABEL_FONT_SIZE = SYMBOL_FONT_SIZE
TANGENT_LABEL_FONT_SIZE = SYMBOL_FONT_SIZE
ANGLE_LABEL_FONT_SIZE = SYMBOL_FONT_SIZE
REFERENCE_ARROW_COLOR = "0.45"
REFERENCE_LABEL_COLOR = "0.45"
CIRCLE_EDGE_COLOR = "black"
POINT_MARKER_SIZE = 5.8
TANGENCY_MARKER_SIZE = 5.5

SAVE_FIGURE = True
SHOW_FIGURE = True

OUTPUT_FILENAME = "geometric_tangent_constructions_clean"

# The same side convention is used in both panels:
#
#   + : tangency point on the right-hand side of the oriented reference line
#   - : tangency point on the left-hand side of the oriented reference line
#
# For the point-circle construction, the reference line is oriented from
# p to o. For the circle-circle construction, it is oriented from o_1 to o_2.

RADIUS = 1.15

POINT = np.array([-2.55, -0.15])
POINT_CIRCLE_CENTER = np.array([1.15, 0.35])

CIRCLE_1_CENTER = np.array([-2.1, -0.35])
CIRCLE_2_CENTER = np.array([2.15, 1.25])

FIGURE_SIZE = (14.5, 5.8)
JOURNAL = runpy.run_path(str(Path(__file__).resolve().parents[1]
                           / "unicycle_journal_paper" / "tangent_convention_journal.py"))


# Redundant visual encoding for grayscale and color-vision accessibility.
TANGENT_LINESTYLES = {
    ("-", "-"): (0, (7, 3)),             # Long dashes.
    ("-", "+"): (0, (1, 2)),             # Dots.
    ("+", "-"): (0, (6, 2, 1, 2, 1, 2)), # Dash-dot-dot.
    ("+", "+"): (0, (6, 2, 1, 2)),       # Dash-dot.
}


THESIS_STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "text.latex.preamble": r"\usepackage{amsmath}",
}
# This is a private runpy namespace: only the thesis rendering changes.
# The reused drawing function enters its own STYLE context, so override that too.
JOURNAL["STYLE"].update(THESIS_STYLE)


plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": [
        "Computer Modern Roman",
        "CMU Serif",
        "DejaVu Serif",
    ],
    "font.size": SYMBOL_FONT_SIZE,
    "axes.labelsize": SYMBOL_FONT_SIZE,
    "axes.titlesize": SYMBOL_FONT_SIZE,
    "xtick.labelsize": TICK_LABEL_FONT_SIZE,
    "ytick.labelsize": TICK_LABEL_FONT_SIZE,
})


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def perpendicular(vector):
    """Return the vector rotated counterclockwise by pi/2."""
    return np.array([-vector[1], vector[0]], dtype=float)


def cross_2d(vector_1, vector_2):
    """Return the scalar 2-D cross product."""
    return (
        vector_1[0] * vector_2[1]
        - vector_1[1] * vector_2[0]
    )


def side_symbol(oriented_direction, offset):
    """
    Classify ``offset`` relative to ``oriented_direction``.

    The convention used in this figure is

        + : right-hand side
        - : left-hand side.
    """
    cross_value = cross_2d(
        oriented_direction,
        offset,
    )

    return "+" if cross_value < 0.0 else "-"


def angle_of(vector):
    """Return the orientation of a vector in [0, 2*pi)."""
    angle = np.arctan2(
        vector[1],
        vector[0],
    )

    return angle % (2.0 * np.pi)


def point_circle_tangents(point, center, radius):
    """
    Compute the two tangency points from an external point to a circle.

    Results are returned in a dictionary indexed by the side symbol.
    """
    center_to_point = point - center
    h_squared = float(
        np.dot(center_to_point, center_to_point)
    )

    if h_squared <= radius**2:
        raise ValueError(
            "The point must lie strictly outside the circle."
        )

    h = np.sqrt(h_squared)
    tangent_length = np.sqrt(
        h_squared - radius**2
    )

    base = (
        center
        + (radius**2 / h_squared)
        * center_to_point
    )

    offset = (
        radius
        * tangent_length
        / h_squared
        * perpendicular(center_to_point)
    )

    oriented_direction = center - point

    candidates = (
        base + offset,
        base - offset,
    )

    tangents = {}

    for tangency_point in candidates:
        symbol = side_symbol(
            oriented_direction,
            tangency_point - point,
        )

        tangents[symbol] = {
            "q": tangency_point,
            "alpha": angle_of(
                tangency_point - point
            ),
            "length": float(
                np.linalg.norm(
                    tangency_point - point
                )
            ),
        }

    return tangents


def equal_circle_common_tangents(
    center_1,
    center_2,
    radius,
):
    """
    Compute all common tangents of two disjoint equal-radius circles.

    The tangent points are classified using their side relative to the
    oriented center line from center_1 to center_2.
    """
    displacement = center_2 - center_1
    distance_squared = float(
        np.dot(displacement, displacement)
    )

    if distance_squared <= (2.0 * radius)**2:
        raise ValueError(
            "The circles must satisfy c > 2R."
        )

    tangents = {}

    # signed_second_radius = +R gives external tangents;
    # signed_second_radius = -R gives internal tangents.
    for signed_second_radius in (
        radius,
        -radius,
    ):
        radius_difference = (
            radius - signed_second_radius
        )

        height_squared = (
            distance_squared
            - radius_difference**2
        )

        root = np.sqrt(
            max(height_squared, 0.0)
        )

        for branch in (
            -1.0,
            1.0,
        ):
            normal = (
                displacement
                * radius_difference
                + perpendicular(displacement)
                * root
                * branch
            ) / distance_squared

            q_1 = center_1 + radius * normal
            q_2 = (
                center_2
                + signed_second_radius
                * normal
            )

            sigma_1 = side_symbol(
                displacement,
                q_1 - center_1,
            )

            sigma_2 = side_symbol(
                displacement,
                q_2 - center_2,
            )

            tangents[
                (sigma_1, sigma_2)
            ] = {
                "q1": q_1,
                "q2": q_2,
                "alpha": angle_of(q_2 - q_1),
                "length": float(
                    np.linalg.norm(q_2 - q_1)
                ),
            }

    return tangents


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def draw_circle_label(axis, center, radius, label, angle_degrees):
    """Place a circle symbol outside its boundary with a short leader."""
    angle = np.radians(angle_degrees)
    direction = np.array([np.cos(angle), np.sin(angle)])
    axis.annotate(
        label, xy=center + radius * direction,
        xytext=center + (radius + 0.55) * direction,
        fontsize=SET_LABEL_FONT_SIZE, ha="center", va="center",
        arrowprops=dict(arrowstyle="-", color="0.35", lw=1,
                        shrinkA=2, shrinkB=0),
        zorder=10,
    )


def draw_dimension(axis, start, end, label, color, offset, label_fraction=0.5):
    """Show length directly on the segment, with arrowheads at both endpoints."""
    direction = (end - start) / np.linalg.norm(end - start)
    normal = perpendicular(direction)
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle="<|-|>", mutation_scale=15,
        linewidth=1.8, color=color, shrinkA=0, shrinkB=0, zorder=10,
    ))
    position = start + label_fraction * (end - start) + np.sign(offset) * 0.28 * normal
    axis.text(*position, label, fontsize=SEGMENT_LABEL_FONT_SIZE,
              color=color, ha="center", va="center", zorder=10,
              rotation=np.degrees(np.arctan2(direction[1], direction[0])),
              rotation_mode="anchor")


def draw_oriented_reference_line(
    axis,
    start,
    end,
    label,
    label_offset,
):
    """Match the journal center-to-center solid double-arrow segment."""
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="<|-|>",
            mutation_scale=11,
            linewidth=1.1,
            linestyle="-",
            color=REFERENCE_ARROW_COLOR,
            zorder=1,
        )
    )

    midpoint = 0.5 * (start + end)

    axis.text(
        midpoint[0] + label_offset[0],
        midpoint[1] + label_offset[1],
        label,
        fontsize=REFERENCE_LABEL_FONT_SIZE,
        color=REFERENCE_LABEL_COLOR,
        ha="center",
        va="center",
    )


def draw_tangent(
    axis,
    start,
    end,
    color,
    linewidth=2.0,
):
    """Draw an oriented tangent segment."""
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=linewidth,
            color=color,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=5,
        )
    )


def draw_extended_line(
    axis,
    point_1,
    point_2,
    extension_start=0.80,
    extension_end=1.25,
    color="0.60",
):
    """Draw the supporting tangent line as a light dashed line."""
    direction = point_2 - point_1
    direction = direction / np.linalg.norm(direction)

    start = point_1 - extension_start * direction
    end = point_2 + extension_end * direction

    axis.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        linestyle="--",
        linewidth=0.9,
        color=color,
        alpha=0.6,
        zorder=0,
    )


def draw_positive_tangent_angle(axis, vertex, direction, radius, color, show_reference=True):
    """Shade the counterclockwise angle in [0, 2pi) from the local +x axis."""
    radius *= 0.7
    angle = np.arctan2(direction[1], direction[0]) % (2 * np.pi)
    axis.add_patch(Wedge(vertex, radius, 0, np.degrees(angle),
                         facecolor=color, edgecolor="none", alpha=0.17, zorder=1))
    axis.add_patch(Arc(vertex, 2 * radius, 2 * radius,
                       theta1=0, theta2=np.degrees(angle), color=color,
                       linewidth=1.25, zorder=6))
    # A short arrow at the arc end makes the positive rotation explicit.
    arc_start = max(0, angle - min(0.18, angle / 3))
    axis.add_patch(FancyArrowPatch(
        vertex + radius * np.array([np.cos(arc_start), np.sin(arc_start)]),
        vertex + radius * np.array([np.cos(angle), np.sin(angle)]),
        arrowstyle="->", mutation_scale=9, color=color, lw=1.1,
        shrinkA=0, shrinkB=0, zorder=7))
    if show_reference:
        axis.plot([vertex[0], vertex[0] + radius + 0.08],
                  [vertex[1], vertex[1]], color="0.55", lw=0.85, zorder=4)


def draw_angle_arc(
    axis,
    vertex,
    reference_angle,
    target_angle,
    radius,
    label=None,
    label_radius=None,
    color="0.38",
):
    """Draw a small orientation arc and its label."""
    delta = (
        target_angle - reference_angle
    ) % (2.0 * np.pi)

    if delta > np.pi:
        delta -= 2.0 * np.pi

    theta_1 = np.degrees(reference_angle)
    theta_2 = np.degrees(
        reference_angle + delta
    )

    if theta_2 < theta_1:
        theta_1, theta_2 = theta_2, theta_1

    axis.add_patch(
        Arc(
            vertex,
            2.0 * radius,
            2.0 * radius,
            angle=0.0,
            theta1=theta_1,
            theta2=theta_2,
            linewidth=1.0,
            color=color,
            zorder=7,
        )
    )

    if label is None:
        return

    if label_radius is None:
        label_radius = 1.25 * radius

    middle_angle = (
        reference_angle
        + 0.5 * delta
    )

    label_position = (
        vertex
        + label_radius
        * np.array([
            np.cos(middle_angle),
            np.sin(middle_angle),
        ])
    )

    axis.text(
        label_position[0],
        label_position[1],
        label,
        fontsize=ANGLE_LABEL_FONT_SIZE,
        color=color,
        ha="center",
        va="center",
    )

def setup_axis(axis):
    axis.set_aspect(
        "equal",
        adjustable="box",
    )
    axis.axis("off")


# =============================================================================
# PANEL 1: POINT-CIRCLE TANGENTS
# =============================================================================

def plot_point_circle_panel(axis, distances=False, combined=False):
    point = POINT
    center = POINT_CIRCLE_CENTER
    radius = RADIUS

    tangents = point_circle_tangents(
        point,
        center,
        radius,
    )

    axis.add_patch(
        Circle(
            center,
            radius,
            fill=False,
            linewidth=1.8,
            edgecolor=CIRCLE_EDGE_COLOR,
            zorder=2,
        )
    )

    if distances or combined:
        draw_oriented_reference_line(
            axis,
            point,
            center,
            label=r"$h$",
            label_offset=np.array([0.05, -0.16]),
        )

    axis.plot(
        point[0],
        point[1],
        marker="o",
        markersize=POINT_MARKER_SIZE,
        color="black",
        zorder=8,
    )

    axis.plot(
        center[0],
        center[1],
        marker="o",
        markersize=POINT_MARKER_SIZE,
        color="black",
        zorder=8,
    )

    axis.text(
        point[0],
        point[1] + 0.25,
        r"$\mathbf{p}$",
        fontsize=POINT_LABEL_FONT_SIZE,
        ha="center",
        va="center",
    )

    axis.text(
        center[0],
        center[1] + 0.28,
        r"$\mathbf{o}$",
        fontsize=POINT_LABEL_FONT_SIZE,
        ha="center",
        va="center",
    )

    label_direction = np.array([np.cos(np.radians(25)), np.sin(np.radians(25))])
    axis.text(*(center + 0.62 * radius * label_direction), r"$\mathcal{O}$",
              fontsize=SET_LABEL_FONT_SIZE, ha="center", va="center")

    tangent_colors = {
        "+": JOURNAL["COLORS"][("+", "+")],  # Match the external tangent.
        "-": JOURNAL["COLORS"][("-", "-")],
    }

    label_offsets = {
        "+": np.array([-0.10, -0.20]),
        "-": np.array([-0.10, 0.16]),
    }

    # Equal outward normal offsets keep both labels outside the circle.
    q_offsets = {symbol: 0.65 * (tangent["q"]-center)/radius
                 for symbol, tangent in tangents.items()}

    for symbol in (
        "-",
        "+",
    ):
        tangent = tangents[symbol]
        q = tangent["q"]
        color = tangent_colors[symbol]
        tangent_direction = q - point
        tangent_direction = (
            tangent_direction
            / np.linalg.norm(tangent_direction)
        )
        tangent_normal = perpendicular(
            tangent_direction
        )

        if not distances:
            draw_extended_line(
                axis,
                point,
                q,
                color=color,
            )


        axis.plot(
            q[0],
            q[1],
            marker="o",
            markersize=TANGENCY_MARKER_SIZE if distances else 9,
            markeredgecolor=color if distances else "white",
            markeredgewidth=0 if distances else 1.1,
            color=color,
            zorder=9,
        )

        if distances or combined:
            draw_dimension(axis, point, q, rf"$d^{{{symbol}}}$", color,
                           0.22 if symbol == "-" else -0.22)

        if not distances:
            angle_fraction = 0.34 if symbol == "-" else 0.65
            angle_vertex = point + angle_fraction * (q - point)
            draw_positive_tangent_angle(axis, angle_vertex, tangent_direction,
                                        0.46 if symbol == "-" else 0.34, color)
            angle = angle_of(tangent_direction)
            label_angle = angle/2 if symbol == "-" else np.pi/2
            label_radius = 0.52 if symbol == "-" else 0.35
            label_position = angle_vertex + label_radius * np.array([
                np.cos(label_angle), np.sin(label_angle)])
            axis.text(*label_position, rf"$\alpha^{{{symbol}}}$", color=color,
                      fontsize=16, ha="center", va="center", zorder=8,
                      bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5))
            # Optional tangency labels and arrows: uncomment to restore.
            # axis.annotate(
            #     rf"$\mathbf{{q}}^{{{symbol}}}$", xy=q,
            #     xytext=q+q_offsets[symbol],
            #     fontsize=TANGENCY_LABEL_FONT_SIZE, color=color,
            #     ha="center", va="center", zorder=10,
            #     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.35",
            #                     color=color, lw=1.0, mutation_scale=10,
            #                     shrinkA=1, shrinkB=3),
            # )

    if not distances:
        legend = axis.legend(
            handles=[Line2D([], [], color=tangent_colors[symbol], lw=1.5,
                            linestyle="--", label=rf"$t^{{{symbol}}}$")
                     for symbol in ("-", "+")],
            loc="lower center", bbox_to_anchor=(0.5, 0.16), ncol=2,
            frameon=True, edgecolor="0.8", facecolor="white", framealpha=1,
            fancybox=False, columnspacing=1.2, handlelength=1.5,
            fontsize=SYMBOL_FONT_SIZE,
        )
        legend.get_frame().set_linewidth(0.6)

    axis.set_xlim(-3.45, 2.95)
    axis.set_ylim(-2.30, 2.35)
    setup_axis(axis)


# =============================================================================
# PANEL 2: TWO-CIRCLE COMMON TANGENTS
# =============================================================================

def plot_circle_circle_panel(axis, distances=False):
    center_1 = CIRCLE_1_CENTER
    center_2 = CIRCLE_2_CENTER
    radius = RADIUS

    tangents = equal_circle_common_tangents(
        center_1,
        center_2,
        radius,
    )

    for center in (
        center_1,
        center_2,
    ):
        axis.add_patch(
            Circle(
                center,
                radius,
                fill=False,
                linewidth=1.8,
                edgecolor=CIRCLE_EDGE_COLOR,
                zorder=2,
            )
        )

        axis.plot(
            center[0],
            center[1],
            marker="o",
            markersize=POINT_MARKER_SIZE,
            color="black",
            zorder=8,
        )

    if distances:
        axis.add_patch(FancyArrowPatch(
            center_1, center_2, arrowstyle="<|-|>", mutation_scale=15,
            linewidth=1.6, color="0.3", zorder=3))
        reference_direction = (center_2 - center_1) / np.linalg.norm(center_2 - center_1)
        reference_label = center_1 + 0.16 * (center_2 - center_1) - 0.25 * perpendicular(reference_direction)
        axis.text(*reference_label, r"$c$",
                  fontsize=SEGMENT_LABEL_FONT_SIZE, color="0.25",
                  ha="center", va="center", zorder=10)

    axis.text(
        center_1[0],
        center_1[1] + 0.40,
        r"$\mathbf{o}_1$",
        fontsize=POINT_LABEL_FONT_SIZE,
        ha="center",
        va="center",
    )

    axis.text(
        center_2[0],
        center_2[1] + 0.40,
        r"$\mathbf{o}_2$",
        fontsize=POINT_LABEL_FONT_SIZE,
        ha="center",
        va="center",
    )

    draw_circle_label(axis, center_1, radius, r"$\mathcal{O}_1$", 200)
    draw_circle_label(axis, center_2, radius, r"$\mathcal{O}_2$", 25)

    tangent_colors = {
        ("+", "+"): "#0072B2",
        ("-", "-"): "#0072B2",
        ("+", "-"): "#CC79A7",
        ("-", "+"): "#CC79A7",
    }

    dimension_offsets = {
        ("+", "+"): -0.24, ("-", "-"): 0.24,
        ("+", "-"): -0.18, ("-", "+"): 0.18,
    }
    label_fractions = {
        ("+", "+"): 0.5, ("-", "-"): 0.5,
        ("+", "-"): 0.30, ("-", "+"): 0.30,
    }

    for key in (
        ("-", "-"),
        ("-", "+"),
        ("+", "-"),
        ("+", "+"),
    ):
        tangent = tangents[key]
        q_1 = tangent["q1"]
        q_2 = tangent["q2"]
        color = tangent_colors[key]

        if not distances:
            draw_extended_line(
                axis,
                q_1,
                q_2,
                extension_start=1.10,
                extension_end=1.30,
            )


        axis.plot(
            [q_1[0], q_2[0]],
            [q_1[1], q_2[1]],
            linestyle="None",
            marker="o",
            markersize=TANGENCY_MARKER_SIZE if distances else 9,
            markeredgecolor=color if distances else "white",
            markeredgewidth=0 if distances else 1.1,
            color=color,
            zorder=9,
        )

        sigma_1, sigma_2 = key
        if distances:
            draw_dimension(axis, q_1, q_2, rf"$d^{{{sigma_1},{sigma_2}}}$",
                           color, dimension_offsets[key], label_fractions[key])

        if not distances:
            angle_fractions = {
                ("-", "-"): 0.48, ("+", "+"): 0.58,
                ("-", "+"): 0.26, ("+", "-"): 0.70,
            }
            angle_vertex = q_1 + angle_fractions[key] * (q_2 - q_1)
            draw_positive_tangent_angle(axis, angle_vertex, q_2 - q_1,
                                        0.34 if key[0] != key[1] else 0.40, color)
            label_position = (
                q_2
                + 0.92
                * (
                    q_2 - q_1
                )
                / np.linalg.norm(q_2 - q_1)
            )

            axis.text(
                label_position[0],
                label_position[1],
                rf"$t^{{{sigma_1},{sigma_2}}}$",
                fontsize=REFERENCE_LABEL_FONT_SIZE,
                color=REFERENCE_ARROW_COLOR,
                ha="center",
                va="center",
            )

    axis.set_xlim(-3.90, 4.00)
    axis.set_ylim(-2.05, 3.00)
    setup_axis(axis)


# =============================================================================
# MAIN
# =============================================================================

def create_figure():
    # Reuse the original drawing without modifying its source or exported files.
    journal = JOURNAL
    with plt.rc_context(THESIS_STYLE):
        figure = journal["create_figure"]()
        figure.set_size_inches(*FIGURE_SIZE)
        right = figure.axes[0]
        grid = figure.add_gridspec(1, 2, left=0.025, right=0.975,
                                  bottom=0.04, top=0.93, wspace=0.025,
                                  width_ratios=(6.4, 8.15))
        right.set_subplotspec(grid[0, 1])
        left = figure.add_subplot(grid[0, 0])
        plot_point_circle_panel(left, combined=True)
        right.set_ylim(-2.10, 3.15)
        left.set_ylim(right.get_ylim())  # Equal scale avoids aspect-ratio padding between panels.
        center_labels = {
            r"$\mathbf{p}^{o}_1$": r"$\mathbf{o}_1$",
            r"$\mathbf{p}^{o}_2$": r"$\mathbf{o}_2$",
        }
        center_positions = {
            r"$\mathbf{o}$": POINT_CIRCLE_CENTER,
            r"$\mathbf{o}_1$": journal["CENTER_1"],
            r"$\mathbf{o}_2$": journal["CENTER_2"],
        }
        patterns_by_color = {journal["COLORS"][signs]: pattern
                             for signs, pattern in TANGENT_LINESTYLES.items()}
        for axis in (left, right):
            for line in axis.lines:
                if line.get_marker() == "o" and line.get_color() in patterns_by_color:
                    line.set_markersize(TANGENCY_MARKER_SIZE)
                    line.set_markeredgecolor("white")
                    line.set_markeredgewidth(0.8)
                if line.is_dashed() and line.get_color() in patterns_by_color:
                    if axis is right:
                        endpoints = line.get_xydata()
                        direction = endpoints[-1]-endpoints[0]
                        direction /= np.linalg.norm(direction)
                        # The journal already extends by 1; add another 0.4 per end.
                        endpoints = np.array([endpoints[0]-0.4*direction,
                                              endpoints[-1]+0.4*direction])
                        line.set_data(endpoints[:, 0], endpoints[:, 1])
                    line.set_linestyle(patterns_by_color[line.get_color()])
                    line.set_alpha(0.8)
            legend = axis.get_legend()
            if legend is not None:
                for handle in legend.get_lines():
                    if handle.get_color() in patterns_by_color:
                        handle.set_linestyle(patterns_by_color[handle.get_color()])
            # Larger distance arrowheads; angle arrows stay intact.
            for patch in axis.patches:
                if (isinstance(patch, FancyArrowPatch)
                        and isinstance(patch.get_arrowstyle(), ArrowStyle.CurveFilledAB)):
                    patch.set_mutation_scale(20)
            for text in axis.texts:
                text.set_text(center_labels.get(text.get_text(), text.get_text()))
                text.set_fontsize(SYMBOL_FONT_SIZE)
                if text.get_text() in center_positions:
                    center = center_positions[text.get_text()]
                    offset = 0.30 if text.get_text() == r"$\mathbf{o}_1$" else 0.20
                    text.set_position((center[0], center[1]+offset))
            if axis.get_legend() is not None:
                for text in axis.get_legend().get_texts():
                    text.set_fontsize(SYMBOL_FONT_SIZE)
        # Preserve the journal's alpha positions, alignment, and label backgrounds.
        # Its smaller alpha size keeps the original spacing around the sectors.
        for axis in (left, right):
            for text in axis.texts:
                if text.get_text().startswith(r"$\alpha^{"):
                    text.set_fontsize(16)
        for text in right.texts:
            if text.get_text() in {r"$\alpha^{-,-}$", r"$\alpha^{+,+}$"}:
                x, y = text.get_position()
                text.set_position((x+0.08, y))
            if text.get_text() in {r"$d^{+,+}$", r"$d^{+,-}$", r"$d^{-,+}$"}:
                x, y = text.get_position()
                text.set_position((x, y-0.07))
        for text in left.texts:
            if text.get_text() in {r"$d^{+}$", r"$d^{-}$"}:
                x, y = text.get_position()
                shift = 0.08 if text.get_text() == r"$d^{+}$" else -0.08
                text.set_position((x, y+shift))
        # Optional right-panel tangency labels and arrows: uncomment to restore.
        # # Equal normal offsets make each pair symmetric about the circle midpoint.
        # for signs, contacts in journal["common_tangents"](
        #         journal["CENTER_1"], journal["CENTER_2"], journal["RADIUS"]).items():
        #     direction = contacts[1]-contacts[0]
        #     normal = perpendicular(direction)/np.linalg.norm(direction)
        #     external = signs[0] == signs[1]
        #     for index, contact in enumerate(contacts, 1):
        #         if external:
        #             side = 1 if signs == ("-", "-") else -1
        #             offset = side * 0.95 * normal
        #         else:
        #             # Reverse the inward normals to place both labels outside their circles.
        #             side = (1 if signs == ("+", "-") else -1) * (1 if index == 1 else -1)
        #             offset = -side * 1.00 * normal
        #         dx, dy = offset
        #         ha = va = "center"
        #         color = journal["COLORS"][signs]
        #         right.annotate(rf"$\mathbf{{q}}_{index}^{{{','.join(signs)}}}$",
        #                        xy=contact, xytext=(contact[0]+dx, contact[1]+dy),
        #                        color=color, fontsize=SYMBOL_FONT_SIZE,
        #                        ha=ha, va=va, zorder=10,
        #                        arrowprops=dict(arrowstyle="->",
        #                                        connectionstyle=f"arc3,rad={0.35 if external else -0.35}",
        #                                        color=color, lw=1.0, mutation_scale=10,
        #                                        shrinkA=1, shrinkB=3))
        # Align the centers of the one-row and two-row legend boxes.
        for axis, horizontal in ((left, 0.5), (right, 0.73)):
            legend = axis.get_legend()
            legend.set_loc("center")
            legend.set_bbox_to_anchor((horizontal, 0.12))
        left.set_title("a) Point–circle construction", fontsize=24, y=1.0, pad=6)
        right.set_title("b) Circle–circle construction", fontsize=24, y=1.0, pad=6)

    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Export without opening a window.")
    args = parser.parse_args()
    figure = create_figure()
    if SAVE_FIGURE:
        for extension in ("pdf", "png"):
            output = Path(__file__).resolve().parent / f"{OUTPUT_FILENAME}.{extension}"
            with plt.rc_context(THESIS_STYLE):
                figure.savefig(output, dpi=300, bbox_inches="tight")
            print(f"Saved {output}")
    if SHOW_FIGURE and not args.no_show:
        plt.show(block=True)
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
