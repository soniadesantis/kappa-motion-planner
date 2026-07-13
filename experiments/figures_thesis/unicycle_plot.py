import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Polygon
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


def curved_arrow(ax, center, radius, theta1, theta2, **kwargs):
    """Draw a curved arrow from theta1 to theta2, with angles in radians."""
    t = np.linspace(theta1, theta2, 60)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)

    ax.plot(x, y, linewidth=1.2, **kwargs)

    # Arrowhead near the end
    p_end = np.array([x[-1], y[-1]])
    p_prev = np.array([x[-4], y[-4]])

    arr = FancyArrowPatch(
        p_prev,
        p_end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        **kwargs,
    )
    ax.add_patch(arr)


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


def rotate(points, theta):
    """Rotate an array of 2D points by theta."""
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])
    return points @ R.T


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
theta = np.deg2rad(35)

# Robot reference point / wheel midpoint
p = np.array([0.0, 0.0])

# Body-frame unit vectors
e_heading = np.array([np.cos(theta), np.sin(theta)])
e_lateral = np.array([-np.sin(theta), np.cos(theta)])

# Wheel/body dimensions for top view
length = 1.35
width = 0.42

# Rectangle in body coordinates, centered at p
rect_body = np.array([
    [-length / 2, -width / 2],
    [ length / 2, -width / 2],
    [ length / 2,  width / 2],
    [-length / 2,  width / 2],
])

rect_world = rotate(rect_body, theta) + p


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))


# ---------------------------------------------------------------------
# Global x-y axes
# ---------------------------------------------------------------------
axis_origin = np.array([-2.2, -1.35])

arrow(ax, axis_origin, axis_origin + np.array([5.0, 0.0]))
arrow(ax, axis_origin, axis_origin + np.array([0.0, 3.6]))

ax.text(axis_origin[0] + 5.15, axis_origin[1] - 0.03, r"$x$", ha="left", va="top")
ax.text(axis_origin[0] - 0.05, axis_origin[1] + 3.75, r"$y$", ha="right", va="bottom")


# ---------------------------------------------------------------------
# Unicycle body / wheel as rectangle
# ---------------------------------------------------------------------
wheel = Polygon(
    rect_world,
    closed=True,
    fill=False,
    linewidth=1.7,
)
ax.add_patch(wheel)

# Midline along the wheel direction
front = p + (length / 2) * e_heading
back = p - (length / 2) * e_heading
ax.plot(
    [back[0], front[0]],
    [back[1], front[1]],
    linewidth=0.9,
    alpha=0.8,
)

# Reference point
ax.plot(p[0], p[1], "o", markersize=3)
ax.text(p[0] - 0.08, p[1] - 0.12, r"$(x,y)$", ha="right", va="top")


# ---------------------------------------------------------------------
# Heading and velocity
# ---------------------------------------------------------------------
# Heading direction
arrow(ax, p, p + 1.45 * e_heading)
ax.text(
    *(p + 1.65 * e_heading),
    r"$v$",
    ha="center",
    va="center",
)

# Optional body-frame heading label
ax.text(
    *(p + 0.85 * e_heading + 0.18 * e_lateral),
    r"$\theta$ direction",
    ha="center",
    va="center",
)


# ---------------------------------------------------------------------
# Angular velocity omega
# ---------------------------------------------------------------------
omega_center = p + 0.10 * e_lateral
curved_arrow(
    ax,
    omega_center,
    radius=0.72,
    theta1=theta + 0.15,
    theta2=theta + 1.55,
)

ax.text(
    *(omega_center + 0.92 * np.array([np.cos(theta + 1.0), np.sin(theta + 1.0)])),
    r"$\omega$",
    ha="center",
    va="center",
)


# ---------------------------------------------------------------------
# Angle theta from world x-axis to heading
# ---------------------------------------------------------------------
angle_arc(
    ax,
    p,
    radius=0.55,
    theta1=0.0,
    theta2=theta,
    label=r"$\theta$",
    label_radius=0.78,
)


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-2.5, 3.0)
ax.set_ylim(-1.6, 2.8)
ax.axis("off")


# ---------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------
fig_dir = Path(__file__).resolve().parent / "saved_figures"
fig_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(fig_dir / "unicycle_model.pdf", bbox_inches="tight")
plt.savefig(fig_dir / "unicycle_model.png", dpi=300, bbox_inches="tight")

plt.show()