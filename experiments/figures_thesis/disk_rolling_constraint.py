import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
from pathlib import Path


# ---------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 16,
    "axes.linewidth": 0.8,
})


# ---------------------------------------------------------------------
# Colours and common dimensions
# ---------------------------------------------------------------------
geometry_color = "0.45"
wheel_fill_color = "0.94"

X_LIMITS = (-1.72, 2.62)
Y_LIMITS = (-1.35, 2.65)

AXIS_ORIGIN = np.array([-1.55, -1.05])
X_AXIS_END = AXIS_ORIGIN + np.array([3.85, 0.0])
Y_AXIS_END = AXIS_ORIGIN + np.array([0.0, 3.30])


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
    mutation_scale=11,
    color="black",
    linestyle="-",
    zorder=6,
):
    """Draw a straight arrow."""
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=arrowstyle,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )

    ax.add_patch(patch)
    return patch


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
    """Draw an angular arc and, optionally, its label."""
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
        + label_radius
        * np.array([
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


def draw_coordinate_system(ax, p):
    """Draw the world axes and the coordinate projections of p."""
    arrow(
        ax,
        AXIS_ORIGIN,
        X_AXIS_END,
        arrowstyle="-|>",
        linewidth=1.0,
        mutation_scale=10,
        zorder=9,
    )

    arrow(
        ax,
        AXIS_ORIGIN,
        Y_AXIS_END,
        arrowstyle="-|>",
        linewidth=1.0,
        mutation_scale=10,
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

    # Coordinate projections of p
    ax.plot(
        [p[0], p[0]],
        [AXIS_ORIGIN[1], p[1]],
        color="black",
        linewidth=0.75,
        linestyle="--",
        zorder=2,
    )

    ax.plot(
        [AXIS_ORIGIN[0], p[0]],
        [p[1], p[1]],
        color="black",
        linewidth=0.75,
        linestyle="--",
        zorder=2,
    )

    ax.text(
        p[0],
        AXIS_ORIGIN[1] - 0.09,
        r"$x$",
        ha="center",
        va="top",
        zorder=10,
    )

    ax.text(
        AXIS_ORIGIN[0] - 0.09,
        p[1],
        r"$y$",
        ha="right",
        va="center",
        zorder=10,
    )


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
theta = np.deg2rad(35)

# Reference point
p = np.array([0.0, 0.0])

# Longitudinal and perpendicular wheel directions
e_parallel = np.array([
    np.cos(theta),
    np.sin(theta),
])

e_perp = np.array([
    -np.sin(theta),
    np.cos(theta),
])


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.3, 4.8))

draw_coordinate_system(ax, p)


# ---------------------------------------------------------------------
# Horizontal reference and heading angle
# ---------------------------------------------------------------------
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
    radius=0.48,
    theta1=0.0,
    theta2=theta,
    label=r"$\theta$",
    label_radius=0.70,
    label_offset=(0.02, -0.06),
)


# ---------------------------------------------------------------------
# Allowed longitudinal velocity
# ---------------------------------------------------------------------
velocity_end = p + 1.48 * e_parallel

arrow(
    ax,
    p,
    velocity_end,
    arrowstyle="-|>",
    linewidth=1.2,
    mutation_scale=10,
    zorder=8,
)

ax.text(
    *(
        velocity_end
        + 0.10 * e_parallel
        + 0.10 * e_perp
    ),
    r"$\dot{\mathbf{p}}"
    r"=v\,\mathbf{e}_{\parallel}(\theta)$",
    ha="left",
    va="center",
    zorder=10,
)


# ---------------------------------------------------------------------
# Forbidden perpendicular direction
# ---------------------------------------------------------------------
lateral_end = p + 1.05 * e_perp

arrow(
    ax,
    p,
    lateral_end,
    arrowstyle="->",
    linewidth=1.0,
    mutation_scale=10,
    color=geometry_color,
    linestyle="--",
    zorder=7,
)


# ---------------------------------------------------------------------
# Cross indicating that lateral velocity is forbidden
# ---------------------------------------------------------------------
cross_center = p + 0.68 * e_perp
cross_size = 0.075

d1 = (
    cross_size
    * (e_parallel + e_perp)
    / np.sqrt(2.0)
)

d2 = (
    cross_size
    * (e_parallel - e_perp)
    / np.sqrt(2.0)
)

ax.plot(
    [
        cross_center[0] - d1[0],
        cross_center[0] + d1[0],
    ],
    [
        cross_center[1] - d1[1],
        cross_center[1] + d1[1],
    ],
    color="black",
    linewidth=1.2,
    zorder=9,
)

ax.plot(
    [
        cross_center[0] - d2[0],
        cross_center[0] + d2[0],
    ],
    [
        cross_center[1] - d2[1],
        cross_center[1] + d2[1],
    ],
    color="black",
    linewidth=1.2,
    zorder=9,
)


# ---------------------------------------------------------------------
# Perpendicular-direction label
# ---------------------------------------------------------------------
e_perp_label_position = (
    lateral_end
    + 0.07 * e_perp
    + 0.10 * e_parallel
)

ax.text(
    e_perp_label_position[0],
    e_perp_label_position[1],
    r"$\mathbf{e}_{\perp}(\theta)$",
    ha="center",
    va="bottom",
    color=geometry_color,
    zorder=10,
)


# ---------------------------------------------------------------------
# No-lateral-slip constraint
# ---------------------------------------------------------------------
constraint_position = (
    lateral_end
    + 0.38 * e_perp
    + 0.12 * e_parallel
)

ax.text(
    constraint_position[0],
    constraint_position[1],
    r"$\mathbf{e}_{\perp}^{\top}(\theta)"
    r"\,\dot{\mathbf{p}}=0$",
    ha="center",
    va="bottom",
    zorder=10,
)


# ---------------------------------------------------------------------
# Reference point
# ---------------------------------------------------------------------
ax.plot(
    p[0],
    p[1],
    marker="o",
    markersize=3.8,
    color="black",
    zorder=11,
)

ax.text(
    p[0] - 0.09,
    p[1] - 0.12,
    r"$\mathbf{p}=(x,y)$",
    ha="right",
    va="top",
    zorder=10,
)


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(*X_LIMITS)
ax.set_ylim(*Y_LIMITS)
ax.axis("off")

fig.subplots_adjust(
    left=0.02,
    right=0.98,
    bottom=0.02,
    top=0.98,
)


# ---------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------
fig_dir = (
    Path(__file__).resolve().parent
    / "saved_figures"
)

fig_dir.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    fig_dir / "rolling_wheel_no_lateral_slip.pdf",
    bbox_inches="tight",
)

fig.savefig(
    fig_dir / "rolling_wheel_no_lateral_slip.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
