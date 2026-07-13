import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch
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
# Helper functions
# ---------------------------------------------------------------------
def arrow(ax, start, end, **kwargs):
    """Draw an arrow from start to end."""
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        shrinkA=0,
        shrinkB=0,
        **kwargs,
    )
    ax.add_patch(arr)
    return arr


def angle_arc(ax, center, radius, theta1, theta2, label=None, label_radius=None):
    """Draw an angle arc from theta1 to theta2, with angles in radians."""
    arc = Arc(
        center,
        2 * radius,
        2 * radius,
        angle=0,
        theta1=np.degrees(theta1),
        theta2=np.degrees(theta2),
        linewidth=1.0,
    )
    ax.add_patch(arc)

    if label is not None:
        if label_radius is None:
            label_radius = 1.25 * radius

        theta_mid = 0.5 * (theta1 + theta2)
        label_position = np.array([
            center[0] + label_radius * np.cos(theta_mid),
            center[1] + label_radius * np.sin(theta_mid),
        ])

        ax.text(
            label_position[0],
            label_position[1],
            label,
            ha="center",
            va="center",
        )


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
theta = np.deg2rad(32)

# Contact point between wheel and plane
contact = np.array([0.0, 0.0])

# Unit vectors
e_parallel = np.array([np.cos(theta), np.sin(theta)])
e_lateral = np.array([-np.sin(theta), np.cos(theta)])

# Disk visual geometry.
# The long axis of the ellipse is aligned with the global y-axis.
disk_width = 1.15
disk_height = 2.05
disk_center = contact + np.array([0.0, disk_height / 2.0])


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))


# ---------------------------------------------------------------------
# Global x-y axes
# ---------------------------------------------------------------------
axis_origin = np.array([-2.3, -1.25])

arrow(ax, axis_origin, axis_origin + np.array([5.2, 0.0]))
arrow(ax, axis_origin, axis_origin + np.array([0.0, 3.8]))

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
# Wheel disk
# ---------------------------------------------------------------------
disk = Ellipse(
    disk_center,
    width=disk_width,
    height=disk_height,
    angle=0.0,
    fill=False,
    linewidth=1.7,
)
ax.add_patch(disk)

# Offset ellipse to suggest wheel thickness
disk_offset = Ellipse(
    disk_center + np.array([-0.10, 0.03]),
    width=disk_width,
    height=disk_height,
    angle=0.0,
    fill=False,
    linewidth=1.0,
    alpha=0.85,
)
ax.add_patch(disk_offset)

# Contact point
ax.plot(contact[0], contact[1], "o", markersize=3)

ax.text(
    contact[0] - 0.08,
    contact[1] - 0.12,
    r"$(x,y)$",
    ha="right",
    va="top",
)

# Radius-like visual line from contact point to disk center
ax.plot(
    [contact[0], disk_center[0]],
    [contact[1], disk_center[1]],
    linewidth=0.8,
    alpha=0.8,
)


# ---------------------------------------------------------------------
# Wheel direction and lateral direction
# ---------------------------------------------------------------------
# Wheel rolling direction
arrow(ax, contact, contact + 1.35 * e_parallel)

ax.text(
    *(contact + 1.55 * e_parallel),
    r"$[\cos\theta,\sin\theta]^\top$",
    ha="center",
    va="center",
)

# Lateral direction
arrow(ax, contact, contact + 1.15 * e_lateral)

ax.text(
    *(contact + 1.40 * e_lateral),
    r"$[-\sin\theta,\cos\theta]^\top$",
    ha="center",
    va="center",
)


# ---------------------------------------------------------------------
# Velocity vector aligned with wheel direction
# ---------------------------------------------------------------------
arrow(
    ax,
    contact,
    contact + 1.05 * e_parallel,
    linestyle="--",
)

ax.text(
    *(contact + 0.82 * e_parallel + np.array([0.05, -0.18])),
    r"$\dot{\mathbf{p}}$",
    ha="center",
    va="center",
)


# ---------------------------------------------------------------------
# World velocity components
# ---------------------------------------------------------------------
arrow(ax, contact, contact + np.array([1.0, 0.0]))
arrow(ax, contact, contact + np.array([0.0, 0.95]))

ax.text(1.07, -0.07, r"$\dot{x}$", ha="left", va="top")
ax.text(0.08, 1.00, r"$\dot{y}$", ha="left", va="bottom")


# ---------------------------------------------------------------------
# Angle annotation
# ---------------------------------------------------------------------
angle_arc(
    ax,
    contact,
    radius=0.55,
    theta1=0.0,
    theta2=theta,
    label=r"$\theta$",
    label_radius=0.78,
)


# ---------------------------------------------------------------------
# No-side-slip constraint label
# ---------------------------------------------------------------------
ax.text(
    0.75,
    -0.72,
    r"$-\dot{x}\sin\theta+\dot{y}\cos\theta=0$",
    ha="center",
    va="center",
)


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-2.6, 3.3)
ax.set_ylim(-1.55, 3.05)
ax.axis("off")


# ---------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------
fig_dir = Path(__file__).resolve().parent / "saved_figures"
fig_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(fig_dir / "rolling_disk_no_lateral_slip.pdf", bbox_inches="tight")
plt.savefig(fig_dir / "rolling_disk_no_lateral_slip.png", dpi=300, bbox_inches="tight")

plt.show()