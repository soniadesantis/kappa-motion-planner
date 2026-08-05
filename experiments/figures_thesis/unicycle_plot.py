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
wheel_edge_color = "0.45"
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
    """Draw an arrow from start to end."""
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


def curved_arrow(
    ax,
    center,
    radius,
    theta1,
    theta2,
    *,
    linewidth=1.2,
    mutation_scale=10,
    color="black",
    zorder=6,
):
    """Draw a curved arrow from theta1 to theta2."""
    t = np.linspace(theta1, theta2, 80)

    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)

    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        zorder=zorder,
    )

    arrow(
        ax,
        np.array([x[-5], y[-5]]),
        np.array([x[-1], y[-1]]),
        arrowstyle="->",
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
    zorder=6,
):
    """Draw an angular arc."""
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
            label_radius = 1.25 * radius

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


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
theta = np.deg2rad(35)

# Robot reference point
p = np.array([0.0, 0.0])

# Body-frame unit vectors
e_heading = np.array([
    np.cos(theta),
    np.sin(theta),
])

e_lateral = np.array([
    -np.sin(theta),
    np.cos(theta),
])

# Wheel dimensions
length = 1.55
width = 0.48


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 4.2))


# ---------------------------------------------------------------------
# Global coordinate axes
# ---------------------------------------------------------------------
axis_origin = np.array([-1.65, -1.05])

x_axis_end = axis_origin + np.array([3.70, 0.0])
y_axis_end = axis_origin + np.array([0.0, 3.05])

arrow(
    ax,
    axis_origin,
    x_axis_end,
    arrowstyle="-|>",
    linewidth=1.1,
    mutation_scale=11,
    zorder=8,
)

arrow(
    ax,
    axis_origin,
    y_axis_end,
    arrowstyle="-|>",
    linewidth=1.1,
    mutation_scale=11,
    zorder=8,
)

ax.text(
    x_axis_end[0] + 0.08,
    x_axis_end[1] - 0.02,
    r"$x$",
    ha="left",
    va="top",
    zorder=9,
)

ax.text(
    y_axis_end[0] - 0.04,
    y_axis_end[1] + 0.08,
    r"$y$",
    ha="right",
    va="bottom",
    zorder=9,
)


# ---------------------------------------------------------------------
# Coordinate projections
# ---------------------------------------------------------------------
x_projection = np.array([p[0], axis_origin[1]])
y_projection = np.array([axis_origin[0], p[1]])

ax.plot(
    [p[0], p[0]],
    [axis_origin[1], p[1]],
    color="black",
    linewidth=0.8,
    linestyle="--",
    zorder=5,
)

ax.plot(
    [axis_origin[0], p[0]],
    [p[1], p[1]],
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
    zorder=9,
)

ax.text(
    y_projection[0] - 0.10,
    y_projection[1],
    r"$y$",
    ha="right",
    va="center",
    zorder=9,
)


# ---------------------------------------------------------------------
# Unicycle wheel: lowest layer
# ---------------------------------------------------------------------
wheel_transform = (
    Affine2D()
    .rotate(theta)
    .translate(p[0], p[1])
    + ax.transData
)

wheel = FancyBboxPatch(
    (-length / 2, -width / 2),
    length,
    width,
    boxstyle="round,pad=0,rounding_size=0.09",
    facecolor=wheel_fill_color,
    edgecolor=wheel_edge_color,
    linewidth=1.7,
    transform=wheel_transform,
    zorder=1,
)

ax.add_patch(wheel)


# ---------------------------------------------------------------------
# Wheel centerline: above the wheel
# ---------------------------------------------------------------------
front = p + (length / 2) * e_heading
back = p - (length / 2) * e_heading

ax.plot(
    [back[0], front[0]],
    [back[1], front[1]],
    color=wheel_edge_color,
    linewidth=0.8,
    zorder=3,
)


# ---------------------------------------------------------------------
# Dashed reference direction for theta: above the wheel
# ---------------------------------------------------------------------
heading_reference_length = 0.95

ax.plot(
    [p[0], p[0] + heading_reference_length],
    [p[1], p[1]],
    color="black",
    linewidth=0.8,
    linestyle="--",
    zorder=5,
)


# ---------------------------------------------------------------------
# Linear velocity: above the wheel
# ---------------------------------------------------------------------
velocity_end = p + 1.55 * e_heading

arrow(
    ax,
    p,
    velocity_end,
    arrowstyle="->",
    linewidth=1.3,
    mutation_scale=12,
    zorder=7,
)

ax.text(
    *(p + 1.72 * e_heading),
    r"$v$",
    ha="center",
    va="center",
    zorder=9,
)


# ---------------------------------------------------------------------
# Angular velocity
# ---------------------------------------------------------------------
omega_center = p + 0.04 * e_lateral

curved_arrow(
    ax,
    omega_center,
    radius=0.30,
    theta1=theta + 0.30,
    theta2=theta + 1.55,
    linewidth=1.15,
    mutation_scale=9,
    zorder=7,
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
    zorder=9,
)


# ---------------------------------------------------------------------
# Heading angle
# ---------------------------------------------------------------------
angle_arc(
    ax,
    p,
    radius=0.58,
    theta1=0.0,
    theta2=theta,
    label=r"$\theta$",
    label_radius=0.88,
    label_offset=(0.02, -0.08),
    zorder=7,
)


# ---------------------------------------------------------------------
# Reference point: highest layer
# ---------------------------------------------------------------------
ax.plot(
    p[0],
    p[1],
    marker="o",
    markersize=4,
    color="black",
    zorder=10,
)


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-1.90, 2.30)
ax.set_ylim(-1.30, 2.30)
ax.axis("off")

fig.tight_layout()


# ---------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------
fig_dir = Path(__file__).resolve().parent / "saved_figures"
fig_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(
    fig_dir / "unicycle_model.pdf",
    bbox_inches="tight",
)

fig.savefig(
    fig_dir / "unicycle_model.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()