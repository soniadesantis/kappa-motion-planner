from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch


# =============================================================================
# CONFIGURATION
# =============================================================================

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
CIRCLE_2_CENTER = np.array([2.15, 0.50])

FIGURE_SIZE = (15.5, 6.4)


plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": [
        "Computer Modern Roman",
        "CMU Serif",
        "DejaVu Serif",
    ],
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
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

def draw_oriented_reference_line(
    axis,
    start,
    end,
    label,
    label_offset,
):
    """Draw a thin ordered reference line with an arrow."""
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.25,
            linestyle="--",
            color="0.35",
            zorder=1,
        )
    )

    midpoint = 0.5 * (start + end)

    axis.text(
        midpoint[0] + label_offset[0],
        midpoint[1] + label_offset[1],
        label,
        fontsize=17,
        color="0.25",
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
    extension=0.55,
):
    """Draw the supporting tangent line as a light dashed line."""
    direction = point_2 - point_1
    direction = direction / np.linalg.norm(direction)

    start = point_1 - extension * direction
    end = point_2 + extension * direction

    axis.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        linestyle="--",
        linewidth=0.85,
        color="0.60",
        zorder=0,
    )


def draw_angle_arc(
    axis,
    vertex,
    reference_angle,
    target_angle,
    radius,
    label,
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
        fontsize=16,
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

def plot_point_circle_panel(axis):
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
            edgecolor="0.55",
            zorder=2,
        )
    )

    draw_oriented_reference_line(
        axis,
        point,
        center,
        label=r"$h$",
        label_offset=np.array([0.05, -0.28]),
    )

    axis.plot(
        point[0],
        point[1],
        marker="o",
        markersize=4.5,
        color="black",
        zorder=8,
    )

    axis.plot(
        center[0],
        center[1],
        marker="o",
        markersize=4.5,
        color="black",
        zorder=8,
    )

    axis.text(
        point[0] - 0.15,
        point[1] - 0.32,
        r"$\mathbf{p}$",
        fontsize=21,
        ha="center",
    )

    axis.text(
        center[0] + 0.06,
        center[1] - 0.31,
        r"$\mathbf{o}$",
        fontsize=21,
        ha="center",
    )

    axis.text(
        center[0] + 0.40,
        center[1] + 0.75,
        r"$\mathcal{O}$",
        fontsize=23,
    )

    reference_angle = angle_of(
        center - point
    )

    tangent_colors = {
        "+": "tab:blue",
        "-": "tab:orange",
    }

    label_offsets = {
        "+": np.array([0.00, -0.38]),
        "-": np.array([0.00, 0.40]),
    }

    q_offsets = {
        "+": np.array([0.18, -0.22]),
        "-": np.array([0.18, 0.22]),
    }

    for symbol in (
        "-",
        "+",
    ):
        tangent = tangents[symbol]
        q = tangent["q"]
        color = tangent_colors[symbol]

        draw_extended_line(
            axis,
            point,
            q,
        )

        draw_tangent(
            axis,
            point,
            q,
            color=color,
        )

        axis.plot(
            q[0],
            q[1],
            marker="o",
            markersize=4.2,
            color="black",
            zorder=9,
        )

        midpoint = 0.5 * (point + q)

        axis.text(
            midpoint[0] + label_offsets[symbol][0],
            midpoint[1] + label_offsets[symbol][1],
            rf"$d^{{{symbol}}}$",
            fontsize=19,
            color=color,
            ha="center",
            va="center",
        )

        axis.text(
            q[0] + q_offsets[symbol][0],
            q[1] + q_offsets[symbol][1],
            rf"$\mathbf{{q}}^{{{symbol}}}$",
            fontsize=18,
            ha="left",
            va="center",
        )

        axis.text(
            point[0] - 0.05,
            point[1]
            + (
                0.95
                if symbol == "-"
                else -0.95
            ),
            rf"$t^{{{symbol}}}$",
            fontsize=20,
            color=color,
            ha="center",
            va="center",
        )

        draw_angle_arc(
            axis,
            vertex=point,
            reference_angle=reference_angle,
            target_angle=tangent["alpha"],
            radius=0.42,
            label=rf"$\alpha^{{{symbol}}}$",
            label_radius=0.72,
        )

    axis.set_xlim(-3.45, 2.95)
    axis.set_ylim(-2.30, 2.35)
    setup_axis(axis)


# =============================================================================
# PANEL 2: TWO-CIRCLE COMMON TANGENTS
# =============================================================================

def plot_circle_circle_panel(axis):
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
                edgecolor="0.55",
                zorder=2,
            )
        )

        axis.plot(
            center[0],
            center[1],
            marker="o",
            markersize=4.5,
            color="black",
            zorder=8,
        )

    draw_oriented_reference_line(
        axis,
        center_1,
        center_2,
        label=r"$c$",
        label_offset=np.array([0.0, -0.34]),
    )

    axis.text(
        center_1[0] - 0.05,
        center_1[1] - 0.31,
        r"$\mathbf{o}_1$",
        fontsize=20,
        ha="center",
    )

    axis.text(
        center_2[0] + 0.05,
        center_2[1] - 0.31,
        r"$\mathbf{o}_2$",
        fontsize=20,
        ha="center",
    )

    axis.text(
        center_1[0] - 0.68,
        center_1[1] + 0.62,
        r"$\mathcal{O}_1$",
        fontsize=22,
    )

    axis.text(
        center_2[0] + 0.38,
        center_2[1] + 0.70,
        r"$\mathcal{O}_2$",
        fontsize=22,
    )

    tangent_colors = {
        ("+", "+"): "tab:blue",
        ("-", "-"): "tab:blue",
        ("+", "-"): "tab:orange",
        ("-", "+"): "tab:orange",
    }

    text_offsets = {
        ("+", "+"): np.array([0.00, -0.34]),
        ("-", "-"): np.array([0.00, 0.34]),
        ("+", "-"): np.array([0.16, -0.30]),
        ("-", "+"): np.array([-0.16, 0.30]),
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

        draw_extended_line(
            axis,
            q_1,
            q_2,
            extension=0.45,
        )

        draw_tangent(
            axis,
            q_1,
            q_2,
            color=color,
            linewidth=1.9,
        )

        axis.plot(
            [q_1[0], q_2[0]],
            [q_1[1], q_2[1]],
            linestyle="None",
            marker="o",
            markersize=3.7,
            color="black",
            zorder=9,
        )

        midpoint = 0.5 * (q_1 + q_2)
        sigma_1, sigma_2 = key

        axis.text(
            midpoint[0] + text_offsets[key][0],
            midpoint[1] + text_offsets[key][1],
            rf"$d^{{{sigma_1},{sigma_2}}}$",
            fontsize=17,
            color=color,
            ha="center",
            va="center",
        )

        # Tangency points are indicated by black markers. Their full
        # labels are omitted here to keep the four-tangent panel readable.

        label_position = (
            q_1
            - 0.62
            * (
                q_2 - q_1
            )
            / np.linalg.norm(q_2 - q_1)
        )

        axis.text(
            label_position[0],
            label_position[1],
            rf"$t^{{{sigma_1},{sigma_2}}}$",
            fontsize=17,
            color=color,
            ha="center",
            va="center",
        )

    # Show one representative tangent orientation to avoid overcrowding.
    representative_key = ("+", "+")
    representative = tangents[
        representative_key
    ]

    center_line_angle = angle_of(
        center_2 - center_1
    )

    draw_angle_arc(
        axis,
        vertex=representative["q1"],
        reference_angle=center_line_angle,
        target_angle=representative["alpha"],
        radius=0.42,
        label=r"$\alpha^{+,+}$",
        label_radius=0.72,
    )

    axis.set_xlim(-3.90, 4.00)
    axis.set_ylim(-2.45, 2.60)
    setup_axis(axis)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    figure, axes = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
    )

    plot_point_circle_panel(
        axes[0]
    )

    plot_circle_circle_panel(
        axes[1]
    )

    figure.tight_layout(
        w_pad=1.5,
    )

    current_directory = (
        Path(__file__).resolve().parent
    )

    pdf_path = (
        current_directory
        / f"{OUTPUT_FILENAME}.pdf"
    )

    png_path = (
        current_directory
        / f"{OUTPUT_FILENAME}.png"
    )

    if SAVE_FIGURE:
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        figure.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
        )

        print("Saved:")
        print(pdf_path)
        print(png_path)

    if SHOW_FIGURE:
        plt.show(
            block=True
        )

    else:
        plt.close(
            figure
        )