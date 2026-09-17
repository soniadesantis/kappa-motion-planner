from math import cos, pi, sin

import matplotlib.pyplot as plt

from kappa_planner import (
    MotionPlanner,
    CorridorWorld,
    Unicycle,
    get_corner_point,
    get_corridor_from_vector,
    plot_corridors,
)
from kappa_planner.helpers.geometry_operations import compute_angular_difference


# ============================================================
# Vehicle
# ============================================================

vehicle = Unicycle(model="Rosbot circular")

vehicle.update(
    width=0.4,
    v_max=0.8,
    omega_max=1.0,
)

R = vehicle.max_radius
r = 0.5 * vehicle.width

print(f"R = {R:.4f} m")
print(f"r = {r:.4f} m")


# ============================================================
# Corridor geometry
#
# Junction angle changes:
# C1 -> C2 : 90 deg
# C2 -> C3 : 30 deg
# C3 -> C4 : 90 deg
# ============================================================

width1 = 0.90
width2 = 0.90
width3 = 0.90
width4 = 0.90

height1 = 3.0
height2 = 2.5
height3 = 2.5
height4 = 3.0

phi1 = 0.0
phi2 = pi / 2
phi3 = pi / 3
phi4 = -pi / 6

add_height = 1.5


# ============================================================
# Corridor 1
# ============================================================

corridor1 = CorridorWorld(
    width=width1,
    height=height1,
    center=[0.0, 0.0],
    tilt=phi1,
)


# ============================================================
# Corridor 2
# ============================================================

tail_vector2 = corridor1.head

head_vector2 = [
    tail_vector2[0] + height2 * cos(phi2),
    tail_vector2[1] + height2 * sin(phi2),
]

corridor2 = get_corridor_from_vector(
    tail_vector2,
    head_vector2,
    width2,
    add_height=add_height,
)


# ============================================================
# Corridor 3
# ============================================================

tail_vector3 = corridor2.head

head_vector3 = [
    tail_vector3[0] + height3 * cos(phi3),
    tail_vector3[1] + height3 * sin(phi3),
]

corridor3 = get_corridor_from_vector(
    tail_vector3,
    head_vector3,
    width3,
    add_height=add_height,
)


# ============================================================
# Corridor 4
# ============================================================

tail_vector4 = corridor3.head

head_vector4 = [
    tail_vector4[0] + height4 * cos(phi4),
    tail_vector4[1] + height4 * sin(phi4),
]

corridor4 = get_corridor_from_vector(
    tail_vector4,
    head_vector4,
    width4,
    add_height=add_height,
)


corridor_list = [
    corridor1,
    corridor2,
    corridor3,
    corridor4,
]


# ============================================================
# Motion planner
# ============================================================

mp = MotionPlanner(
    vehicle,
    corridor_list,
    assumptions="standing",
)


# ============================================================
# Compute beta values
# ============================================================

phis = [c.tilt for c in corridor_list]

betas = [
    0.5
    * abs(
        compute_angular_difference(
            phis[i],
            phis[i + 1],
        )
    )
    for i in range(len(phis) - 1)
]


# ============================================================
# Compute q_j and local minimum widths
# ============================================================

q = [
    (R - r) * cos(beta)
    for beta in betas
]

local_min_widths = [
    R + r - q_i
    for q_i in q
]


print("\nLocal junction minimum widths:")

for i in range(len(local_min_widths)):
    print(
        f"Junction {i + 1}: "
        f"beta = {betas[i] * 180 / pi:.2f} deg, "
        f"local minimum width = {local_min_widths[i]:.4f} m"
    )


print("\nCorridor-level minimum widths from planner:")

for i, min_width in enumerate(mp.min_corridor_widths):
    print(
        f"Corridor {i + 1}: "
        f"minimum width = {min_width:.4f} m"
    )


# ============================================================
# Compute current and proposed s_max values
# ============================================================

s_max_current = []
s_max_local = []

for i in range(len(corridor_list) - 1):

    denom = cos(betas[i])

    if abs(denom) < 1e-12:
        raise ValueError(
            f"Invalid corridor configuration at junction {i}: "
            "cos(beta) is approximately zero."
        )

    # Current formula
    s_current = (
        min(
            corridor_list[i].width,
            corridor_list[i + 1].width,
        )
        -
        min(
            mp.min_corridor_widths[i],
            mp.min_corridor_widths[i + 1],
        )
    ) / denom

    s_max_current.append(s_current)

    # Proposed local-junction formula
    s_local = (
        min(
            corridor_list[i].width,
            corridor_list[i + 1].width,
        )
        -
        local_min_widths[i]
    ) / denom

    s_max_local.append(s_local)


print("\nComparison of s_max values:")

for i in range(len(s_max_current)):
    print(
        f"Junction {i + 1}: "
        f"current = {s_max_current[i]:.4f} m, "
        f"local = {s_max_local[i]:.4f} m, "
        f"difference = "
        f"{s_max_local[i] - s_max_current[i]:.4f} m"
    )


# ============================================================
# Compute geometric data for every junction
# ============================================================

junction_data = []

for j in range(len(corridor_list) - 1):

    corridor_a = corridor_list[j]
    corridor_b = corridor_list[j + 1]

    turn = corridor_a.compute_relative_turn_direction(
        corridor_b
    )

    corner_point = get_corner_point(
        corridor_a,
        corridor_b,
        turn,
    )

    angle_bisector = (
        phis[j + 1]
        + turn
        * 0.5
        * (
            pi
            - abs(
                compute_angular_difference(
                    phis[j + 1],
                    phis[j],
                )
            )
        )
    )

    junction_data.append(
        {
            "corner_point": corner_point,
            "angle_bisector": angle_bisector,
        }
    )


# ============================================================
# Helper: compute circle center
# ============================================================

def compute_shifted_center(
    corner_point,
    angle_bisector,
    s,
):

    xc = (
        corner_point[0]
        + (R - r - s) * cos(angle_bisector)
    )

    yc = (
        corner_point[1]
        + (R - r - s) * sin(angle_bisector)
    )

    return xc, yc


# ============================================================
# Helper: plot all junctions
# ============================================================

def plot_all_junctions(
    s_values,
    title,
):

    figure = plot_corridors(
        corridor_list,
        linestyle="solid",
    )

    ax = figure.gca()

    for j in range(len(corridor_list) - 1):

        corner_point = junction_data[j]["corner_point"]
        angle_bisector = junction_data[j]["angle_bisector"]

        # Nominal center
        xc_nom, yc_nom = compute_shifted_center(
            corner_point,
            angle_bisector,
            0.0,
        )

        # Shifted center
        xc_shift, yc_shift = compute_shifted_center(
            corner_point,
            angle_bisector,
            s_values[j],
        )

        # ----------------------------------------------------
        # Corner
        # ----------------------------------------------------

        ax.plot(
            corner_point[0],
            corner_point[1],
            "ko",
            markersize=4,
        )

        # ----------------------------------------------------
        # Nominal path circle
        # ----------------------------------------------------

        nominal_path_circle = plt.Circle(
            (xc_nom, yc_nom),
            R,
            fill=False,
            linestyle="--",
            linewidth=1.0,
        )

        ax.add_artist(
            nominal_path_circle
        )

        # ----------------------------------------------------
        # Nominal swept envelope
        # ----------------------------------------------------

        nominal_envelope = plt.Circle(
            (xc_nom, yc_nom),
            R + r,
            fill=False,
            linestyle="--",
            linewidth=0.8,
        )

        ax.add_artist(
            nominal_envelope
        )

        # ----------------------------------------------------
        # Shifted path circle
        # ----------------------------------------------------

        shifted_path_circle = plt.Circle(
            (xc_shift, yc_shift),
            R,
            fill=False,
            linewidth=2.0,
        )

        ax.add_artist(
            shifted_path_circle
        )

        # ----------------------------------------------------
        # Shifted swept envelope
        # ----------------------------------------------------

        shifted_envelope = plt.Circle(
            (xc_shift, yc_shift),
            R + r,
            fill=False,
            linewidth=2.0,
        )

        ax.add_artist(
            shifted_envelope
        )

        # ----------------------------------------------------
        # Centers
        # ----------------------------------------------------

        ax.plot(
            xc_nom,
            yc_nom,
            marker="o",
            markersize=4,
        )

        ax.plot(
            xc_shift,
            yc_shift,
            "ko",
            markersize=4,
        )

        # ----------------------------------------------------
        # Bisector shift
        # ----------------------------------------------------

        ax.plot(
            [xc_nom, xc_shift],
            [yc_nom, yc_shift],
            linestyle=":",
            linewidth=1.0,
        )

        # ----------------------------------------------------
        # Junction label
        # ----------------------------------------------------

        ax.text(
            xc_shift,
            yc_shift,
            f"  J{j + 1}",
            fontsize=9,
        )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_title(
        title
    )

    return figure


# ============================================================
# Plot 1: current formula at all junctions
# ============================================================

figure_current = plot_all_junctions(
    s_max_current,
    "Current $s_{j,max}$ formula - all junctions",
)


# ============================================================
# Plot 2: local-junction formula at all junctions
# ============================================================

figure_local = plot_all_junctions(
    s_max_local,
    "Local-junction $s_{j,max}$ formula - all junctions",
)


# ============================================================
# Display
# ============================================================

plt.show(block=True)