import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Affine2D
from pathlib import Path


# ---------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------
SYMBOL_FONT_SIZE = 22
PANEL_LABEL_FONT_SIZE = 22
LEGEND_FONT_SCALE = 0.85
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
# Colours
# ---------------------------------------------------------------------
geometry_color = "0.45"
wheel_fill_color = "0.90"
parallel_color = "#1f77b4"
perpendicular_color = "#d55e00"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def arrow(
    ax,
    start,
    end,
    *,
    arrowstyle="-|>",
    linewidth=1.2,
    mutation_scale=11,
    color="black",
    zorder=6,
    **kwargs,
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
        **kwargs,
    )
    ax.add_patch(arr)
    return arr


def angle_arc(
    ax,
    center,
    radius,
    theta1,
    theta2,
    label=None,
    label_radius=None,
    label_offset=(0.0, 0.0),
):
    """Draw an angle arc from theta1 to theta2, with angles in radians."""
    arc = Arc(
        center,
        2 * radius,
        2 * radius,
        angle=0,
        theta1=np.degrees(theta1),
        theta2=np.degrees(theta2),
        linewidth=1.0,
        zorder=7,
    )
    ax.add_patch(arc)

    if label is not None:
        if label_radius is None:
            label_radius = 1.25 * radius

        theta_mid = 0.5 * (theta1 + theta2)
        label_position = np.array([
            center[0] + label_radius * np.cos(theta_mid),
            center[1] + label_radius * np.sin(theta_mid),
        ]) + np.asarray(label_offset)

        ax.text(
            label_position[0],
            label_position[1],
            label,
            ha="center",
            va="center",
            zorder=8,
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


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
theta = np.deg2rad(32)

# Contact point between wheel and plane
contact = np.array([0.0, 0.0])

# Unit vectors
e_parallel = np.array([np.cos(theta), np.sin(theta)])
e_lateral = np.array([-np.sin(theta), np.cos(theta)])

# Wheel visual geometry
wheel_length = 1.55
wheel_width = 0.48
wheel_center = contact


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.8, 5.4))


# ---------------------------------------------------------------------
# Global x-y axes
# ---------------------------------------------------------------------
axis_origin = np.array([-2.3, -1.25])

arrow(
    ax,
    axis_origin,
    axis_origin + np.array([5.2, 0.0]),
    mutation_scale=AXIS_ARROWHEAD_SIZE,
    zorder=9,
)
arrow(
    ax,
    axis_origin,
    axis_origin + np.array([0.0, 3.8]),
    mutation_scale=AXIS_ARROWHEAD_SIZE,
    zorder=9,
)

ax.text(
    axis_origin[0] + 5.35,
    axis_origin[1] - 0.03,
    r"$x$",
    ha="left",
    va="top",
)

ax.text(
    axis_origin[0] - 0.05,
    axis_origin[1] + 3.95,
    r"$y$",
    ha="right",
    va="bottom",
)


# ---------------------------------------------------------------------
# Wheel
# ---------------------------------------------------------------------
add_rounded_wheel(
    ax,
    wheel_center,
    theta,
    wheel_length,
    wheel_width,
    rounding_size=0.09,
)

# Contact point
ax.plot(contact[0], contact[1], "o", markersize=3.8, color="black", zorder=11)

x_projection = np.array([contact[0], axis_origin[1]])
y_projection = np.array([axis_origin[0], contact[1]])

ax.plot(
    [contact[0], contact[0]],
    [axis_origin[1], contact[1]],
    color="black",
    linewidth=0.75,
    linestyle="--",
    zorder=5,
)

ax.plot(
    [axis_origin[0], contact[0] + 0.92],
    [contact[1], contact[1]],
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


# ---------------------------------------------------------------------
# Wheel direction and lateral direction
# ---------------------------------------------------------------------
# Wheel rolling direction
arrow(
    ax,
    contact,
    contact + 1.60 * e_parallel,
    linewidth=VELOCITY_ARROW_LINEWIDTH,
    mutation_scale=VELOCITY_ARROWHEAD_SIZE,
    color=parallel_color,
    zorder=8,
)

ax.text(
    *(contact + 1.88 * e_parallel),
    r"$\mathbf{e}_{\parallel}(\theta)$",
    ha="center",
    va="center",
    color=parallel_color,
    zorder=10,
)

# Lateral direction
arrow(
    ax,
    contact,
    contact + 1.45 * e_lateral,
    linewidth=VELOCITY_ARROW_LINEWIDTH,
    mutation_scale=VELOCITY_ARROWHEAD_SIZE,
    color=perpendicular_color,
    zorder=8,
)

ax.text(
    *(contact + 1.72 * e_lateral),
    r"$\mathbf{e}_{\perp}(\theta)$",
    ha="center",
    va="center",
    color=perpendicular_color,
    zorder=10,
)


# ---------------------------------------------------------------------
# Angle annotation
# ---------------------------------------------------------------------
angle_arc(
    ax,
    contact,
    radius=0.62,
    theta1=0.0,
    theta2=np.arctan2(e_parallel[1], e_parallel[0]),
    label=r"$\theta$",
    label_radius=0.82,
    label_offset=(0.08, 0.0),
)

legend_handles = [
    Line2D(
        [0],
        [0],
        color=parallel_color,
        linewidth=2.0,
        label="Longitudinal direction\n(allowed velocity)",
    ),
    Line2D(
        [0],
        [0],
        color=perpendicular_color,
        linewidth=2.0,
        label="Lateral direction\n(no slip)",
    ),
]

ax.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(0.78, 0.76),
    frameon=True,
    fancybox=True,
    framealpha=1.0,
    edgecolor="0.70",
    facecolor="white",
    fontsize=SYMBOL_FONT_SIZE * LEGEND_FONT_SCALE,
    handlelength=1.6,
    handletextpad=0.6,
    borderpad=0.45,
    labelspacing=0.65,
)

# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-2.9, 3.8)
ax.set_ylim(-1.9, 3.5)
ax.axis("off")


# ---------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------
fig_dir = Path(__file__).resolve().parent / "saved_figures"
fig_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(fig_dir / "rolling_disk_no_lateral_slip.pdf", bbox_inches="tight")
plt.savefig(fig_dir / "rolling_disk_no_lateral_slip.png", dpi=300, bbox_inches="tight")

plt.show()
