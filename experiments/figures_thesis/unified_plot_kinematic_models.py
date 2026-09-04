import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Affine2D
from pathlib import Path



# ---------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------
SYMBOL_FONT_SIZE = 20
PANEL_LABEL_FONT_SIZE = 20
AXIS_ARROWHEAD_SIZE = 18
VELOCITY_ARROW_LINEWIDTH = 1.5
VELOCITY_ARROWHEAD_SIZE = 15
OMEGA_ARROW_LINEWIDTH = 1.5
OMEGA_ARROWHEAD_SIZE = 15
OMEGA_ARROW_HEAD_BACKOFF = 10

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": SYMBOL_FONT_SIZE,
    "axes.linewidth": 0.8,
})


# ---------------------------------------------------------------------
# Colours and common dimensions
# ---------------------------------------------------------------------
geometry_color = "0.45"
wheel_fill_color = "0.90"

# Common limits guarantee the same geometric scale in all panels
X_LIMITS = (-1.72, 2.62)
Y_LIMITS = (-1.35, 2.65)

# Common coordinate axes
AXIS_ORIGIN = np.array([-1.55, -1.05])
X_AXIS_END = AXIS_ORIGIN + np.array([3.85, 0.0])
Y_AXIS_END = AXIS_ORIGIN + np.array([0.0, 3.30])


# ---------------------------------------------------------------------
# General helper functions
# ---------------------------------------------------------------------
def arrow(
    ax,
    start,
    end,
    *,
    arrowstyle="->",
    linewidth=1.2,
    mutation_scale=11,
    color="black",
    zorder=6,
):
    """Draw an arrow."""
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=arrowstyle,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def curved_arrow(
    ax,
    center,
    radius,
    theta1,
    theta2,
    *,
    linewidth=1.1,
    mutation_scale=9,
    color="black",
    zorder=7,
):
    """Draw a curved arrow."""
    angles = np.linspace(theta1, theta2, 80)

    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    head_start_idx = -OMEGA_ARROW_HEAD_BACKOFF

    ax.plot(
        x[:head_start_idx + 1],
        y[:head_start_idx + 1],
        color=color,
        linewidth=linewidth,
        zorder=zorder,
    )

    arrow(
        ax,
        np.array([x[head_start_idx], y[head_start_idx]]),
        np.array([x[-1], y[-1]]),
        arrowstyle="-|>",
        linewidth=linewidth,
        mutation_scale=mutation_scale,
        color=color,
        zorder=zorder + 1,
    )


def angle_arc(
    ax,
    center,
    radius,
    theta1,
    theta2,
    *,
    label=None,
    label_radius=None,
    label_offset=(0.0, 0.0),
    color="black",
    zorder=7,
):
    """Draw an angular arc and its label."""
    patch = Arc(
        center,
        2 * radius,
        2 * radius,
        theta1=np.degrees(theta1),
        theta2=np.degrees(theta2),
        linewidth=1.0,
        color=color,
        zorder=zorder,
    )
    ax.add_patch(patch)

    if label is None:
        return

    if label_radius is None:
        label_radius = 1.20 * radius

    theta_mid = 0.5 * (theta1 + theta2)

    label_position = (
        center
        + label_radius * np.array([
            np.cos(theta_mid),
            np.sin(theta_mid),
        ])
        + np.asarray(label_offset)
    )

    ax.text(
        label_position[0],
        label_position[1],
        label,
        ha="center",
        va="center",
        color=color,
        zorder=zorder + 1,
    )


def add_rounded_wheel(
    ax,
    center,
    orientation,
    length,
    width,
    *,
    linewidth=1.5,
    edgecolor=geometry_color,
    facecolor=wheel_fill_color,
    rounding_size=0.05,
):
    """Draw a rounded wheel on the lowest visual layer."""
    transform = (
        Affine2D()
        .rotate(orientation)
        .translate(center[0], center[1])
        + ax.transData
    )

    wheel = FancyBboxPatch(
        (-length / 2, -width / 2),
        length,
        width,
        boxstyle=(
            f"round,pad=0,"
            f"rounding_size={rounding_size}"
        ),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=transform,
        zorder=1,
    )

    ax.add_patch(wheel)


def right_angle_marker(
    ax,
    vertex,
    direction_1,
    direction_2,
    *,
    size=0.10,
):
    """Draw a right-angle marker."""
    p1 = vertex + size * direction_1
    p2 = p1 + size * direction_2
    p3 = vertex + size * direction_2

    ax.plot(
        [p1[0], p2[0], p3[0]],
        [p1[1], p2[1], p3[1]],
        color="black",
        linewidth=0.8,
        zorder=8,
    )


def dimension_line(
    ax,
    start,
    end,
    *,
    label,
    tick_direction,
    tick_length=0.08,
    label_offset=(0.0, 0.0),
):
    """Draw a dimension line with terminal ticks."""
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color="black",
        linewidth=0.9,
        zorder=6,
    )

    tick_vector = tick_length * tick_direction

    ax.plot(
        [start[0] - tick_vector[0], start[0] + tick_vector[0]],
        [start[1] - tick_vector[1], start[1] + tick_vector[1]],
        color="black",
        linewidth=0.9,
        zorder=6,
    )

    ax.plot(
        [end[0] - tick_vector[0], end[0] + tick_vector[0]],
        [end[1] - tick_vector[1], end[1] + tick_vector[1]],
        color="black",
        linewidth=0.9,
        zorder=6,
    )

    label_position = (
        0.5 * (start + end)
        + np.asarray(label_offset)
    )

    ax.text(
        label_position[0],
        label_position[1],
        label,
        ha="center",
        va="center",
        zorder=9,
    )


# ---------------------------------------------------------------------
# Common coordinate system
# ---------------------------------------------------------------------
def draw_coordinate_system(ax, p):
    """Draw the common world axes and coordinate projections."""
    arrow(
        ax,
        AXIS_ORIGIN,
        X_AXIS_END,
        arrowstyle="-|>",
        linewidth=1.0,
        mutation_scale=AXIS_ARROWHEAD_SIZE,
        zorder=9,
    )

    arrow(
        ax,
        AXIS_ORIGIN,
        Y_AXIS_END,
        arrowstyle="-|>",
        linewidth=1.0,
        mutation_scale=AXIS_ARROWHEAD_SIZE,
        zorder=9,
    )

    ax.text(
        X_AXIS_END[0] + 0.07,
        X_AXIS_END[1] - 0.02,
        r"$x$",
        ha="left",
        va="top",
        zorder=10,
    )

    ax.text(
        Y_AXIS_END[0] - 0.04,
        Y_AXIS_END[1] + 0.07,
        r"$y$",
        ha="right",
        va="bottom",
        zorder=10,
    )

    x_projection = np.array([p[0], AXIS_ORIGIN[1]])
    y_projection = np.array([AXIS_ORIGIN[0], p[1]])

    ax.plot(
        [p[0], p[0]],
        [AXIS_ORIGIN[1], p[1]],
        color="black",
        linewidth=0.75,
        linestyle="--",
        zorder=5,
    )

    ax.plot(
        [AXIS_ORIGIN[0], p[0]],
        [p[1], p[1]],
        color="black",
        linewidth=0.75,
        linestyle="--",
        zorder=5,
    )

    ax.text(
        x_projection[0],
        x_projection[1] - 0.09,
        r"$x$",
        ha="center",
        va="top",
        zorder=10,
    )

    ax.text(
        y_projection[0] - 0.09,
        y_projection[1],
        r"$y$",
        ha="right",
        va="center",
        zorder=10,
    )


def format_panel(ax, panel_label):
    """Apply common limits, aspect ratio, and panel label."""
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.axis("off")

    ax.text(
        0.5,
        -0.04,
        panel_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_LABEL_FONT_SIZE,
    )


# ---------------------------------------------------------------------
# Unicycle panel
# ---------------------------------------------------------------------
def draw_unicycle(ax):
    theta = np.deg2rad(35)
    p = np.array([0.0, 0.0])

    e_heading = np.array([
        np.cos(theta),
        np.sin(theta),
    ])

    e_lateral = np.array([
        -np.sin(theta),
        np.cos(theta),
    ])

    length = 1.55
    width = 0.48

    draw_coordinate_system(ax, p)

    add_rounded_wheel(
        ax,
        p,
        theta,
        length,
        width,
        rounding_size=0.09,
    )

    front = p + (length / 2) * e_heading
    back = p - (length / 2) * e_heading

    ax.plot(
        [back[0], front[0]],
        [back[1], front[1]],
        color=geometry_color,
        linewidth=0.8,
        zorder=3,
    )

    # Horizontal reference for theta
    ax.plot(
        [p[0], p[0] + 0.95],
        [p[1], p[1]],
        color="black",
        linewidth=0.75,
        linestyle="--",
        zorder=5,
    )

    # Linear velocity
    arrow(
        ax,
        p,
        p + 1.55 * e_heading,
        arrowstyle="-|>",
        linewidth=VELOCITY_ARROW_LINEWIDTH,
        mutation_scale=VELOCITY_ARROWHEAD_SIZE,
        zorder=8,
    )

    ax.text(
        *(p + 1.72 * e_heading),
        r"$v$",
        ha="center",
        va="center",
        zorder=10,
    )

    # Angular velocity
    omega_center = p + 0.04 * e_lateral

    curved_arrow(
        ax,
        omega_center,
        radius=0.30,
        theta1=theta + 0.30,
        theta2=theta + 1.55,
        linewidth=OMEGA_ARROW_LINEWIDTH,
        mutation_scale=OMEGA_ARROWHEAD_SIZE,
    )

    omega_label_position = (
        omega_center
        + 0.46 * np.array([
            np.cos(theta + 1.02),
            np.sin(theta + 1.02),
        ])
    )

    ax.text(
        omega_label_position[0],
        omega_label_position[1],
        r"$\omega$",
        ha="center",
        va="center",
        zorder=10,
    )

    angle_arc(
        ax,
        p,
        radius=0.58,
        theta1=0.0,
        theta2=theta,
        label=r"$\theta$",
        label_radius=0.88,
        label_offset=(0.02, -0.08),
    )

    ax.plot(
        p[0],
        p[1],
        marker="o",
        markersize=3.8,
        color="black",
        zorder=11,
    )

    format_panel(ax, r"(a) Unicycle")


# ---------------------------------------------------------------------
# Bicycle panel
# ---------------------------------------------------------------------
def draw_bicycle(ax):
    theta = np.deg2rad(28)
    delta = np.deg2rad(35)

    p_rear = np.array([0.0, 0.0])

    L = 1.55

    e_heading = np.array([
        np.cos(theta),
        np.sin(theta),
    ])

    e_lateral = np.array([
        -np.sin(theta),
        np.cos(theta),
    ])

    p_front = p_rear + L * e_heading

    front_angle = theta + delta

    e_front = np.array([
        np.cos(front_angle),
        np.sin(front_angle),
    ])

    # Enlarged bicycle wheels
    wheel_length = 0.72
    wheel_width = 0.24

    R = L / np.tan(delta)
    p_icr = p_rear + R * e_lateral

    draw_coordinate_system(ax, p_rear)

    # Wheels on the lowest layer
    add_rounded_wheel(
        ax,
        p_rear,
        theta,
        wheel_length,
        wheel_width,
        rounding_size=0.065,
    )

    add_rounded_wheel(
        ax,
        p_front,
        front_angle,
        wheel_length,
        wheel_width,
        rounding_size=0.065,
    )

    # Wheelbase
    ax.plot(
        [p_rear[0], p_front[0]],
        [p_rear[1], p_front[1]],
        color=geometry_color,
        linewidth=1.5,
        zorder=3,
    )

    # Three front reference directions
    reference_length = 0.90

    horizontal_end = (
        p_front
        + reference_length * np.array([1.0, 0.0])
    )

    heading_end = (
        p_front
        + reference_length * e_heading
    )

    front_direction_end = (
        p_front
        + reference_length * e_front
    )

    for end_point in [
        horizontal_end,
        heading_end,
        front_direction_end,
    ]:
        ax.plot(
            [p_front[0], end_point[0]],
            [p_front[1], end_point[1]],
            color="black",
            linewidth=0.75,
            linestyle="--",
            zorder=5,
        )

    # Theta and delta
    angle_arc(
        ax,
        p_front,
        radius=0.31,
        theta1=0.0,
        theta2=theta,
        label=r"$\theta$",
        label_radius=0.43,
        label_offset=(0.03, -0.01),
    )

    angle_arc(
        ax,
        p_front,
        radius=0.46,
        theta1=theta,
        theta2=front_angle,
        label=r"$\delta$",
        label_radius=0.61,
        label_offset=(0.03, 0.04),
    )

    # Velocity from rear-wheel midpoint
    arrow(
        ax,
        p_rear,
        p_rear + 1.05 * e_heading,
        arrowstyle="-|>",
        linewidth=VELOCITY_ARROW_LINEWIDTH,
        mutation_scale=VELOCITY_ARROWHEAD_SIZE,
        zorder=8,
    )

    velocity_label_position = (
        p_rear
        + 0.76 * e_heading
        + 0.13 * e_lateral
    )

    ax.text(
        velocity_label_position[0],
        velocity_label_position[1],
        r"$v$",
        ha="center",
        va="center",
        zorder=10,
    )

    # Wheelbase dimension
    offset = -0.30 * e_lateral
    L_start = p_rear + offset
    L_end = p_front + offset

    dimension_line(
        ax,
        L_start,
        L_end,
        label=r"$L$",
        tick_direction=e_lateral,
        label_offset=-0.10 * e_lateral,
    )

    # ICR construction
    ax.plot(
        [p_rear[0], p_icr[0]],
        [p_rear[1], p_icr[1]],
        color="black",
        linewidth=0.85,
        linestyle="--",
        zorder=5,
    )

    ax.plot(
        [p_front[0], p_icr[0]],
        [p_front[1], p_icr[1]],
        color="black",
        linewidth=0.85,
        linestyle="--",
        zorder=5,
    )

        # Angular velocity near the ICR
    omega_center = p_icr + np.array([0.10, -0.10])

    curved_arrow(
        ax,
        omega_center,
        radius=0.28,
        theta1=np.deg2rad(255),
        theta2=np.deg2rad(355),
        linewidth=OMEGA_ARROW_LINEWIDTH,
        mutation_scale=OMEGA_ARROWHEAD_SIZE,
        zorder=8,
    )

    ax.text(
        omega_center[0] + 0.24,
        omega_center[1] - 0.34,
        r"$\omega$",
        ha="center",
        va="center",
        zorder=10,
    )

    radius_midpoint = 0.5 * (p_rear + p_icr)

    ax.text(
        *(radius_midpoint - 0.12 * e_heading),
        r"$R$",
        ha="center",
        va="center",
        zorder=10,
    )

    right_angle_marker(
        ax,
        p_rear,
        e_lateral,
        e_heading,
    )

    front_to_icr = p_icr - p_front
    front_to_icr /= np.linalg.norm(front_to_icr)

    right_angle_marker(
        ax,
        p_front,
        front_to_icr,
        e_front,
    )

    ax.plot(
        p_rear[0],
        p_rear[1],
        marker="o",
        markersize=3.8,
        color="black",
        zorder=11,
    )

    ax.plot(
        p_front[0],
        p_front[1],
        marker="o",
        markersize=3.4,
        color="black",
        zorder=11,
    )

    ax.plot(
        p_icr[0],
        p_icr[1],
        marker="o",
        markersize=4.5,
        color="black",
        zorder=11,
    )

    ax.text(
        p_icr[0],
        p_icr[1] + 0.12,
        r"$\mathrm{ICR}$",
        ha="center",
        va="bottom",
        zorder=12,
    )

    format_panel(ax, r"(b) Bicycle")


# ---------------------------------------------------------------------
# Differential-drive panel
# ---------------------------------------------------------------------
def draw_differential_drive(ax):
    theta = np.deg2rad(35)

    p = np.array([0.0, 0.0])

    e_heading = np.array([
        np.cos(theta),
        np.sin(theta),
    ])

    e_lateral = np.array([
        -np.sin(theta),
        np.cos(theta),
    ])

    b = 1.45

    # Enlarged differential-drive wheels
    wheel_radius_visual = 0.50
    wheel_length = 2 * wheel_radius_visual
    wheel_width = 0.28

    left_wheel_center = (
        p + (b / 2.0) * e_lateral
    )

    right_wheel_center = (
        p - (b / 2.0) * e_lateral
    )

    draw_coordinate_system(ax, p)

    # Wheels on the lowest layer
    add_rounded_wheel(
        ax,
        left_wheel_center,
        theta,
        wheel_length,
        wheel_width,
        rounding_size=0.065,
    )

    add_rounded_wheel(
        ax,
        right_wheel_center,
        theta,
        wheel_length,
        wheel_width,
        rounding_size=0.065,
    )

    # Axle
    ax.plot(
        [left_wheel_center[0], right_wheel_center[0]],
        [left_wheel_center[1], right_wheel_center[1]],
        color=geometry_color,
        linewidth=1.5,
        zorder=3,
    )

    # Horizontal reference for theta
    ax.plot(
        [p[0], p[0] + 0.95],
        [p[1], p[1]],
        color="black",
        linewidth=0.75,
        linestyle="--",
        zorder=5,
    )

    angle_arc(
        ax,
        p,
        radius=0.56,
        theta1=0.0,
        theta2=theta,
        label=r"$\theta$",
        label_radius=0.70,
        label_offset=(0.02, -0.04),
    )

    # Linear velocity
    arrow(
        ax,
        p,
        p + 1.40 * e_heading,
        arrowstyle="-|>",
        linewidth=VELOCITY_ARROW_LINEWIDTH,
        mutation_scale=VELOCITY_ARROWHEAD_SIZE,
        zorder=8,
    )

    ax.text(
        *(p + 1.57 * e_heading),
        r"$v$",
        ha="center",
        va="center",
        zorder=10,
    )

    # Angular velocity
    omega_center = p + 0.04 * e_lateral

    curved_arrow(
        ax,
        omega_center,
        radius=0.34,
        theta1=theta + 0.30,
        theta2=theta + 1.55,
        linewidth=OMEGA_ARROW_LINEWIDTH,
        mutation_scale=OMEGA_ARROWHEAD_SIZE,
    )

    omega_label_position = (
        omega_center
        + 0.50 * np.array([
            np.cos(theta + 1.02),
            np.sin(theta + 1.02),
        ])
    )

    ax.text(
        omega_label_position[0],
        omega_label_position[1],
        r"$\omega$",
        ha="center",
        va="center",
        zorder=10,
    )

    # Wheel separation b
    dimension_offset = -0.58 * e_heading

    b_start = left_wheel_center + dimension_offset
    b_end = right_wheel_center + dimension_offset

    ax.plot(
        [left_wheel_center[0], b_start[0]],
        [left_wheel_center[1], b_start[1]],
        color="black",
        linewidth=0.75,
        zorder=5,
    )

    ax.plot(
        [right_wheel_center[0], b_end[0]],
        [right_wheel_center[1], b_end[1]],
        color="black",
        linewidth=0.75,
        zorder=5,
    )

    dimension_line(
        ax,
        b_start,
        b_end,
        label=r"$b$",
        tick_direction=e_heading,
        label_offset=-0.13 * e_heading,
    )

    # Wheel labels
    # ax.text(
    #     *(left_wheel_center + 0.33 * e_lateral),
    #     r"$L$",
    #     ha="center",
    #     va="center",
    #     zorder=10,
    # )

    # ax.text(
    #     *(right_wheel_center - 0.33 * e_lateral),
    #     r"$R$",
    #     ha="center",
    #     va="center",
    #     zorder=10,
    # )

    ax.plot(
        p[0],
        p[1],
        marker="o",
        markersize=3.8,
        color="black",
        zorder=11,
    )

    format_panel(ax, r"(c) Differential drive")


# ---------------------------------------------------------------------
# Combined figure
# ---------------------------------------------------------------------
fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.2, 4.8),
)

draw_unicycle(axes[0])
draw_bicycle(axes[1])
draw_differential_drive(axes[2])

fig.subplots_adjust(
    left=0.015,
    right=0.995,
    bottom=0.12,
    top=0.98,
    wspace=-0.08,
)


# ---------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------
fig_dir = Path(__file__).resolve().parent / "saved_figures"
fig_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(
    fig_dir / "kinematic_models.pdf",
    bbox_inches="tight",
)

fig.savefig(
    fig_dir / "kinematic_models.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
