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


def double_arrow(ax, start, end, **kwargs):
    """Draw a double-headed arrow."""
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="<->",
        mutation_scale=12,
        linewidth=1.1,
        shrinkA=0,
        shrinkB=0,
        **kwargs,
    )
    ax.add_patch(arr)
    return arr


def curved_arrow(ax, center, radius, theta1, theta2, **kwargs):
    """Draw a curved arrow from theta1 to theta2, with angles in radians."""
    t = np.linspace(theta1, theta2, 80)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)

    ax.plot(x, y, linewidth=1.2, **kwargs)

    p_end = np.array([x[-1], y[-1]])
    p_prev = np.array([x[-5], y[-5]])

    arr = FancyArrowPatch(
        p_prev,
        p_end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
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


def rotate(points, theta):
    """Rotate an array of 2D points by theta."""
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])
    return points @ R.T


def rectangle(center, length, width, theta):
    """Create a rotated rectangle centered at center.

    length is along the heading direction.
    width is along the lateral direction.
    """
    rect_body = np.array([
        [-length / 2, -width / 2],
        [ length / 2, -width / 2],
        [ length / 2,  width / 2],
        [-length / 2,  width / 2],
    ])
    return rotate(rect_body, theta) + center


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
theta = np.deg2rad(35)

# Robot reference point: midpoint of the axle
p = np.array([0.0, 0.0])

# Body-frame unit vectors
e_heading = np.array([np.cos(theta), np.sin(theta)])
e_lateral = np.array([-np.sin(theta), np.cos(theta)])

# Differential-drive geometry
b = 1.55          # distance between wheels
r = 0.42          # wheel radius, visualized as half the wheel length

wheel_length = 2 * r
wheel_width = 0.22

left_wheel_center = p + (b / 2.0) * e_lateral
right_wheel_center = p - (b / 2.0) * e_lateral

left_wheel = rectangle(left_wheel_center, wheel_length, wheel_width, theta)
right_wheel = rectangle(right_wheel_center, wheel_length, wheel_width, theta)


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.4))


# ---------------------------------------------------------------------
# Global x-y axes
# ---------------------------------------------------------------------
axis_origin = np.array([-2.35, -1.45])

arrow(ax, axis_origin, axis_origin + np.array([5.25, 0.0]))
arrow(ax, axis_origin, axis_origin + np.array([0.0, 3.85]))

ax.text(
    axis_origin[0] + 5.40,
    axis_origin[1] - 0.03,
    r"$x$",
    ha="left",
    va="top",
)

ax.text(
    axis_origin[0] - 0.05,
    axis_origin[1] + 4.00,
    r"$y$",
    ha="right",
    va="bottom",
)


# ---------------------------------------------------------------------
# Wheels
# ---------------------------------------------------------------------
left_patch = Polygon(
    left_wheel,
    closed=True,
    fill=False,
    linewidth=1.7,
)
right_patch = Polygon(
    right_wheel,
    closed=True,
    fill=False,
    linewidth=1.7,
)

ax.add_patch(left_patch)
ax.add_patch(right_patch)

# Axle line
ax.plot(
    [left_wheel_center[0], right_wheel_center[0]],
    [left_wheel_center[1], right_wheel_center[1]],
    linewidth=1.0,
    alpha=0.8,
)

# Reference point
ax.plot(p[0], p[1], "o", markersize=3)
ax.text(p[0] - 0.08, p[1] - 0.12, r"$(x,y)$", ha="right", va="top")


# ---------------------------------------------------------------------
# Wheel radius r annotation
# ---------------------------------------------------------------------
# Draw r on the right wheel from its center to its front edge
r_start = right_wheel_center
r_end = right_wheel_center + r * e_heading

arrow(ax, r_start, r_end)
ax.text(
    *(right_wheel_center + 0.58 * r * e_heading - 0.20 * e_lateral),
    r"$r$",
    ha="center",
    va="center",
)


# ---------------------------------------------------------------------
# Distance b annotation between wheels
# ---------------------------------------------------------------------
b_offset = -0.55 * e_heading
b_start = left_wheel_center + b_offset
b_end = right_wheel_center + b_offset

double_arrow(ax, b_start, b_end)

ax.text(
    *((b_start + b_end) / 2.0 - 0.18 * e_heading),
    r"$b$",
    ha="center",
    va="center",
)


# Small guide lines for the b dimension
ax.plot(
    [left_wheel_center[0], b_start[0]],
    [left_wheel_center[1], b_start[1]],
    linewidth=0.8,
    alpha=0.7,
)
ax.plot(
    [right_wheel_center[0], b_end[0]],
    [right_wheel_center[1], b_end[1]],
    linewidth=0.8,
    alpha=0.7,
)


# ---------------------------------------------------------------------
# Forward velocity v
# ---------------------------------------------------------------------
arrow(ax, p, p + 1.45 * e_heading)

ax.text(
    *(p + 1.65 * e_heading),
    r"$v$",
    ha="center",
    va="center",
)


# ---------------------------------------------------------------------
# Angular velocity omega
# ---------------------------------------------------------------------
omega_center = p + 0.12 * e_lateral

curved_arrow(
    ax,
    omega_center,
    radius=0.92,
    theta1=theta + 0.20,
    theta2=theta + 1.55,
)

ax.text(
    *(omega_center + 1.12 * np.array([np.cos(theta + 1.02), np.sin(theta + 1.02)])),
    r"$\omega$",
    ha="center",
    va="center",
)


# ---------------------------------------------------------------------
# Heading angle theta
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
# Optional wheel labels
# ---------------------------------------------------------------------
ax.text(
    *(left_wheel_center + 0.40 * e_lateral),
    r"$L$",
    ha="center",
    va="center",
)

ax.text(
    *(right_wheel_center - 0.40 * e_lateral),
    r"$R$",
    ha="center",
    va="center",
)


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-2.6, 3.1)
ax.set_ylim(-1.7, 2.95)
ax.axis("off")


# ---------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------
fig_dir = Path(__file__).resolve().parent / "saved_figures"
fig_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(fig_dir / "differential_drive_model.pdf", bbox_inches="tight")
plt.savefig(fig_dir / "differential_drive_model.png", dpi=300, bbox_inches="tight")

plt.show()