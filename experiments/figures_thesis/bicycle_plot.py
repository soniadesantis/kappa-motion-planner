import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Affine2D
from pathlib import Path


# ---------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.linewidth": 0.8,
})


# ---------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------
geometry_color = "0.45"
wheel_fill_color = "0.90"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def arrow(
    ax,
    start,
    end,
    *,
    arrowstyle="->",
    linewidth=1.2,
    mutation_scale=12,
    color="black",
    zorder=6,
):
    """Draw an arrow."""
    arr = FancyArrowPatch(
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
    ax.add_patch(arr)
    return arr


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
    zorder=6,
):
    """Draw an angle arc."""
    arc = Arc(
        center,
        2 * radius,
        2 * radius,
        theta1=np.degrees(theta1),
        theta2=np.degrees(theta2),
        linewidth=1.0,
        color=color,
        zorder=zorder,
    )
    ax.add_patch(arc)

    if label is not None:
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
    linewidth=1.6,
    edgecolor="0.45",
    facecolor="0.90",
):
    """Draw a rounded equivalent wheel on the lowest layer."""
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
        boxstyle="round,pad=0,rounding_size=0.05",
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
    size=0.11,
    color="black",
):
    """Draw a right-angle marker."""
    p1 = vertex + size * direction_1
    p2 = p1 + size * direction_2
    p3 = vertex + size * direction_2

    ax.plot(
        [p1[0], p2[0], p3[0]],
        [p1[1], p2[1], p3[1]],
        color=color,
        linewidth=0.8,
        zorder=7,
    )


def dimension_line(
    ax,
    start,
    end,
    normal,
    *,
    offset,
    label,
    tick_length=0.10,
):
    """Draw a dimension line parallel to a segment."""
    start_dim = start + offset * normal
    end_dim = end + offset * normal

    ax.plot(
        [start_dim[0], end_dim[0]],
        [start_dim[1], end_dim[1]],
        color="black",
        linewidth=0.9,
        zorder=6,
    )

    tick_vector = tick_length * normal

    ax.plot(
        [start_dim[0] - tick_vector[0], start_dim[0] + tick_vector[0]],
        [start_dim[1] - tick_vector[1], start_dim[1] + tick_vector[1]],
        color="black",
        linewidth=0.9,
        zorder=6,
    )

    ax.plot(
        [end_dim[0] - tick_vector[0], end_dim[0] + tick_vector[0]],
        [end_dim[1] - tick_vector[1], end_dim[1] + tick_vector[1]],
        color="black",
        linewidth=0.9,
        zorder=6,
    )

    label_position = (
        0.5 * (start_dim + end_dim)
        + 0.10 * normal
    )

    ax.text(
        label_position[0],
        label_position[1],
        label,
        ha="center",
        va="center",
        zorder=8,
    )


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
theta = np.deg2rad(28)
delta = np.deg2rad(35)

# Rear-wheel midpoint and reference point
p_rear = np.array([0.0, 0.0])

# Wheelbase
L = 1.55

# Vehicle directions
e_heading = np.array([
    np.cos(theta),
    np.sin(theta),
])

e_lateral = np.array([
    -np.sin(theta),
    np.cos(theta),
])

# Front-wheel midpoint
p_front = p_rear + L * e_heading

# Front-wheel direction
front_angle = theta + delta

e_front = np.array([
    np.cos(front_angle),
    np.sin(front_angle),
])

# Equivalent-wheel dimensions
wheel_length = 0.58
wheel_width = 0.18


# ---------------------------------------------------------------------
# Instantaneous center of rotation
# ---------------------------------------------------------------------
R = L / np.tan(delta)
p_icr = p_rear + R * e_lateral


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.8))


# ---------------------------------------------------------------------
# Global coordinate axes
# ---------------------------------------------------------------------
axis_origin = np.array([-1.55, -1.05])

x_axis_end = axis_origin + np.array([3.85, 0.0])
y_axis_end = axis_origin + np.array([0.0, 3.30])

arrow(
    ax,
    axis_origin,
    x_axis_end,
    arrowstyle="-|>",
    linewidth=1.1,
    mutation_scale=11,
    zorder=9,
)

arrow(
    ax,
    axis_origin,
    y_axis_end,
    arrowstyle="-|>",
    linewidth=1.1,
    mutation_scale=11,
    zorder=9,
)

ax.text(
    x_axis_end[0] + 0.08,
    x_axis_end[1] - 0.02,
    r"$x$",
    ha="left",
    va="top",
    zorder=10,
)

ax.text(
    y_axis_end[0] - 0.04,
    y_axis_end[1] + 0.08,
    r"$y$",
    ha="right",
    va="bottom",
    zorder=10,
)


# ---------------------------------------------------------------------
# Coordinate projections
# ---------------------------------------------------------------------
x_projection = np.array([p_rear[0], axis_origin[1]])
y_projection = np.array([axis_origin[0], p_rear[1]])

ax.plot(
    [p_rear[0], p_rear[0]],
    [axis_origin[1], p_rear[1]],
    color="black",
    linewidth=0.8,
    linestyle="--",
    zorder=5,
)

ax.plot(
    [axis_origin[0], p_rear[0]],
    [p_rear[1], p_rear[1]],
    color="black",
    linewidth=0.8,
    linestyle="--",
    zorder=5,
)

ax.text(
    x_projection[0],
    x_projection[1] - 0.10,
    r"$x$",
    ha="center",
    va="top",
    zorder=10,
)

ax.text(
    y_projection[0] - 0.10,
    y_projection[1],
    r"$y$",
    ha="right",
    va="center",
    zorder=10,
)


# ---------------------------------------------------------------------
# Bicycle wheels: lowest layer
# ---------------------------------------------------------------------
add_rounded_wheel(
    ax,
    p_rear,
    theta,
    wheel_length,
    wheel_width,
    edgecolor=geometry_color,
    facecolor=wheel_fill_color,
)

add_rounded_wheel(
    ax,
    p_front,
    front_angle,
    wheel_length,
    wheel_width,
    edgecolor=geometry_color,
    facecolor=wheel_fill_color,
)


# ---------------------------------------------------------------------
# Wheelbase: above the wheels
# ---------------------------------------------------------------------
ax.plot(
    [p_rear[0], p_front[0]],
    [p_rear[1], p_front[1]],
    color=geometry_color,
    linewidth=1.6,
    zorder=3,
)


# ---------------------------------------------------------------------
# Reference directions at the front wheel
# ---------------------------------------------------------------------
reference_length = 0.90

# Horizontal world direction
horizontal_end = p_front + reference_length * np.array([1.0, 0.0])

ax.plot(
    [p_front[0], horizontal_end[0]],
    [p_front[1], horizontal_end[1]],
    color="black",
    linewidth=0.8,
    linestyle="--",
    zorder=5,
)

# Wheelbase direction
heading_extension_end = p_front + reference_length * e_heading

ax.plot(
    [p_front[0], heading_extension_end[0]],
    [p_front[1], heading_extension_end[1]],
    color="black",
    linewidth=0.8,
    linestyle="--",
    zorder=5,
)

# Front-wheel direction
front_direction_end = p_front + reference_length * e_front

ax.plot(
    [p_front[0], front_direction_end[0]],
    [p_front[1], front_direction_end[1]],
    color="black",
    linewidth=0.8,
    linestyle="--",
    zorder=5,
)


# ---------------------------------------------------------------------
# Heading and steering angles
# ---------------------------------------------------------------------
angle_arc(
    ax,
    p_front,
    radius=0.31,
    theta1=0.0,
    theta2=theta,
    label=r"$\theta$",
    label_radius=0.43,
    label_offset=(0.03, -0.01),
    zorder=7,
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
    zorder=7,
)


# ---------------------------------------------------------------------
# Longitudinal velocity: on top of the wheelbase and wheels
# ---------------------------------------------------------------------
velocity_end = p_rear + 1.05 * e_heading

arrow(
    ax,
    p_rear,
    velocity_end,
    arrowstyle="-|>",
    linewidth=1.25,
    mutation_scale=11,
    color="black",
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


# ---------------------------------------------------------------------
# Wheelbase dimension L
# ---------------------------------------------------------------------
dimension_line(
    ax,
    p_rear,
    p_front,
    -e_lateral,
    offset=0.30,
    label=r"$L$",
    tick_length=0.08,
)


# ---------------------------------------------------------------------
# ICR construction
# ---------------------------------------------------------------------
ax.plot(
    [p_rear[0], p_icr[0]],
    [p_rear[1], p_icr[1]],
    color="black",
    linewidth=0.9,
    linestyle="--",
    zorder=5,
)

ax.plot(
    [p_front[0], p_icr[0]],
    [p_front[1], p_icr[1]],
    color="black",
    linewidth=0.9,
    linestyle="--",
    zorder=5,
)

radius_midpoint = 0.5 * (p_rear + p_icr)

ax.text(
    *(radius_midpoint - 0.12 * e_heading),
    r"$R$",
    ha="center",
    va="center",
    zorder=10,
)


# ---------------------------------------------------------------------
# Right-angle markers
# ---------------------------------------------------------------------
right_angle_marker(
    ax,
    p_rear,
    e_lateral,
    e_heading,
    size=0.11,
)

front_to_icr = p_icr - p_front
front_to_icr /= np.linalg.norm(front_to_icr)

right_angle_marker(
    ax,
    p_front,
    front_to_icr,
    e_front,
    size=0.10,
)


# ---------------------------------------------------------------------
# Reference points and ICR: highest layer
# ---------------------------------------------------------------------
ax.plot(
    p_rear[0],
    p_rear[1],
    marker="o",
    markersize=4,
    color="black",
    zorder=11,
)

ax.plot(
    p_front[0],
    p_front[1],
    marker="o",
    markersize=3.5,
    color="black",
    zorder=11,
)

ax.plot(
    p_icr[0],
    p_icr[1],
    marker="o",
    markersize=5,
    color="black",
    zorder=11,
)

ax.text(
    p_icr[0] - 0.08,
    p_icr[1] + 0.10,
    r"$\mathrm{ICR}$",
    ha="right",
    va="bottom",
    zorder=12,
)


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-1.8, 2.75)
ax.set_ylim(-1.3, 2.65)
ax.axis("off")

fig.tight_layout()


# ---------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------
fig_dir = Path(__file__).resolve().parent / "saved_figures"
fig_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(
    fig_dir / "bicycle_model.pdf",
    bbox_inches="tight",
)

fig.savefig(
    fig_dir / "bicycle_model.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()