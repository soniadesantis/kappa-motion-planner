from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.corridor_geometry import get_corner_point
from kappa_planner.helpers.plot_helpers import plot_corridors


# ================================================================
# Geometry helpers
# ================================================================

def point_to_array(point):
    """Convert a Point-like or array-like object to a NumPy vector."""
    if hasattr(point, "x") and hasattr(point, "y"):
        return np.array(
            [point.x, point.y],
            dtype=float,
        )

    return np.asarray(
        point,
        dtype=float,
    )


def segment_to_array(segment):
    """Convert a two-point segment to a 2x2 NumPy array."""
    point_1, point_2 = segment

    return np.vstack(
        (
            point_to_array(point_1),
            point_to_array(point_2),
        )
    )


def local_to_world(
    local_points,
    corner,
    local_x,
    local_y,
):
    """
    Convert local transition coordinates to world coordinates.

    A local point (x, y) is mapped according to

        p = corner + x local_x + y local_y.
    """
    local_points = np.asarray(
        local_points,
        dtype=float,
    )

    corner = np.asarray(
        corner,
        dtype=float,
    )

    local_x = np.asarray(
        local_x,
        dtype=float,
    )

    local_y = np.asarray(
        local_y,
        dtype=float,
    )

    return (
        corner[None, :]
        + local_points[:, [0]] * local_x[None, :]
        + local_points[:, [1]] * local_y[None, :]
    )


# ================================================================
# Arrow helpers
# ================================================================

def arrow(
    ax,
    start,
    end,
    *,
    arrowstyle="-|>",
    linewidth=1.2,
    mutation_scale=11,
    color="black",
    zorder=10,
):
    """
    Draw an arrow using the same FancyArrowPatch style adopted
    in the kinematic-model figures.
    """
    start = np.asarray(
        start,
        dtype=float,
    )

    end = np.asarray(
        end,
        dtype=float,
    )

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


def draw_double_arrow(
    ax,
    start,
    end,
    *,
    color="0.25",
    linewidth=1.3,
    mutation_scale=11,
    zorder=15,
):
    """Draw a double-headed dimension arrow without text."""
    return arrow(
        ax=ax,
        start=start,
        end=end,
        arrowstyle="<->",
        linewidth=linewidth,
        mutation_scale=mutation_scale,
        color=color,
        zorder=zorder,
    )


def draw_corridor_direction_arrow(
    ax,
    corridor,
    arrow_fraction=0.20,
    linewidth=1.8,
    color="0.45",
):
    """
    Draw a short arrow along the longitudinal direction of a corridor.
    """
    center = np.asarray(
        corridor.center,
        dtype=float,
    )

    direction = np.asarray(
        corridor.unit_vector,
        dtype=float,
    )

    direction /= np.linalg.norm(
        direction
    )

    arrow_length = (
        arrow_fraction
        * corridor.height
    )

    start = (
        center
        - 0.5 * arrow_length * direction
    )

    end = (
        center
        + 0.5 * arrow_length * direction
    )

    arrow(
        ax=ax,
        start=start,
        end=end,
        arrowstyle="-|>",
        linewidth=linewidth,
        mutation_scale=11,
        color=color,
        zorder=8,
    )


# ================================================================
# Feasible-set construction
# ================================================================

def build_feasible_center_polygon(
    D,
    a,
    b,
    number_of_arc_points=250,
):
    """
    Build the local feasible-center region

        x^2 + y^2 <= D^2,
        x >= a,
        y >= b.
    """
    if D <= 0.0:
        raise ValueError(
            "D = R-r must be strictly positive."
        )

    if a < 0.0 or b < 0.0:
        raise ValueError(
            "The lower bounds a and b must be nonnegative."
        )

    if a**2 + b**2 > D**2 + 1e-12:
        raise ValueError(
            "The feasible-center set is empty because "
            "a^2+b^2 > D^2."
        )

    x_at_y_b = np.sqrt(
        max(D**2 - b**2, 0.0)
    )

    y_at_x_a = np.sqrt(
        max(D**2 - a**2, 0.0)
    )

    theta_start = np.arctan2(
        b,
        x_at_y_b,
    )

    theta_end = np.arctan2(
        y_at_x_a,
        a,
    )

    theta_values = np.linspace(
        theta_start,
        theta_end,
        number_of_arc_points,
    )

    circular_arc = np.column_stack(
        (
            D * np.cos(theta_values),
            D * np.sin(theta_values),
        )
    )

    return np.vstack(
        (
            np.array([[a, b]]),
            np.array([[x_at_y_b, b]]),
            circular_arc,
            np.array([[a, y_at_x_a]]),
        )
    )


# ================================================================
# Main plotting function
# ================================================================

def plot_local_transition_geometry(
    corridor1,
    corridor2,
    R,
    r,
    output_path="local_transition_geometry_feasible_set.pdf",
):
    """
    Plot the feasible intermediate-circle center set.

    The figure contains:
        - original corridors in solid grey;
        - corridors shrunk by r in dashed grey;
        - short corridor-orientation arrows in grey;
        - opposite corridor walls in blue and orange;
        - corresponding clearance constraints in matching colors;
        - thick positive local coordinate axes;
        - the footprint circle of radius r;
        - a double-arrow radius of length R-r;
        - the colored feasible-center set.
    """
    if R <= r:
        raise ValueError(
            f"Expected R > r, but received "
            f"R={R} and r={r}."
        )

    # ------------------------------------------------------------
    # Transition direction and corner
    # ------------------------------------------------------------

    turn = (
        corridor1.compute_relative_turn_direction(
            corridor2
        )
    )

    if turn == 0:
        raise ValueError(
            "The corridors are aligned. "
            "This figure expects an orthogonal representation."
        )

    corner = get_corner_point(
        corridor1,
        corridor2,
        turn,
    )

    if corner is None:
        raise ValueError(
            "No valid corner point was found."
        )

    corner = point_to_array(
        corner
    )

    # ------------------------------------------------------------
    # Side-edge selection
    # ------------------------------------------------------------

    if turn > 0:
        corner_edge_1 = corridor1.LFT
        corner_edge_2 = corridor2.LFT

        opposite_edge_1 = corridor1.RGT
        opposite_edge_2 = corridor2.RGT

    else:
        corner_edge_1 = corridor1.RGT
        corner_edge_2 = corridor2.RGT

        opposite_edge_1 = corridor1.LFT
        opposite_edge_2 = corridor2.LFT

    opposite_wall_1 = segment_to_array(
        corridor1.get_edge_segment(
            opposite_edge_1
        )
    )

    opposite_wall_2 = segment_to_array(
        corridor2.get_edge_segment(
            opposite_edge_2
        )
    )

    # Consistency checks.
    corridor1.get_edge_segment(
        corner_edge_1
    )

    corridor2.get_edge_segment(
        corner_edge_2
    )

    # ------------------------------------------------------------
    # Local frame
    # ------------------------------------------------------------

    local_x = -np.asarray(
        corridor1.outward_normals[
            opposite_edge_1
        ],
        dtype=float,
    )

    local_y = -np.asarray(
        corridor2.outward_normals[
            opposite_edge_2
        ],
        dtype=float,
    )

    local_x /= np.linalg.norm(
        local_x
    )

    local_y /= np.linalg.norm(
        local_y
    )

    if not np.isclose(
        np.dot(local_x, local_y),
        0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "The selected local axes are not orthogonal."
        )

    forward_1 = np.asarray(
        corridor1.unit_vector,
        dtype=float,
    )

    forward_2 = np.asarray(
        corridor2.unit_vector,
        dtype=float,
    )

    # ------------------------------------------------------------
    # Feasible-center constraints
    # ------------------------------------------------------------

    D = R - r
    S = R + r

    a = max(
        0.0,
        S - corridor1.width,
    )

    b = max(
        0.0,
        S - corridor2.width,
    )

    if a**2 + b**2 > D**2 + 1e-12:
        raise ValueError(
            "The local feasible-center set is empty:\n"
            f"a^2+b^2 = {a**2+b**2:.6f}, "
            f"D^2 = {D**2:.6f}."
        )

    print("turn:", turn)
    print("corner:", corner)
    print("local_x:", local_x)
    print("local_y:", local_y)
    print("D = R-r:", D)
    print("S = R+r:", S)
    print("a_j:", a)
    print("b_j:", b)

    # ------------------------------------------------------------
    # Drawing scales
    # ------------------------------------------------------------

    characteristic_width = min(
        corridor1.width,
        corridor2.width,
    )

    axis_length = max(
        1.15 * D,
        0.60 * characteristic_width,
    )

    width_arrow_offset_1 = (
        0.34 * corridor1.height
    )

    width_arrow_offset_2 = (
        0.30 * corridor2.height
    )

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    figure, ax = plt.subplots(
        figsize=(8.8, 7.2)
    )

    # ------------------------------------------------------------
    # Original corridors: solid grey
    # ------------------------------------------------------------

    plot_corridors(
        corridor_list=[
            corridor1,
            corridor2,
        ],
        figure=figure,
        plot_vectors=False,
        plot_corridor_index=False,
        linestyle="-",
        color="0.58",
        linewidth=1.6,
    )

    # ------------------------------------------------------------
    # Shrunken corridors: dashed grey
    # ------------------------------------------------------------

    shrunken_corridor1 = corridor1.shrink(r)
    shrunken_corridor2 = corridor2.shrink(r)

    plot_corridors(
        corridor_list=[
            shrunken_corridor1,
            shrunken_corridor2,
        ],
        figure=figure,
        plot_vectors=False,
        plot_corridor_index=False,
        linestyle="--",
        color="0.72",
        linewidth=1.3,
    )

    # ------------------------------------------------------------
    # Short corridor direction arrows
    # ------------------------------------------------------------

    draw_corridor_direction_arrow(
        ax=ax,
        corridor=corridor1,
        arrow_fraction=0.20,
        color="0.42",
    )

    draw_corridor_direction_arrow(
        ax=ax,
        corridor=corridor2,
        arrow_fraction=0.20,
        color="0.42",
    )

    # ------------------------------------------------------------
    # Opposite walls
    # ------------------------------------------------------------

    wall_1_color = "tab:blue"
    wall_2_color = "tab:orange"

    ax.plot(
        opposite_wall_1[:, 0],
        opposite_wall_1[:, 1],
        color=wall_1_color,
        linewidth=3.2,
        solid_capstyle="round",
        zorder=9,
    )

    ax.plot(
        opposite_wall_2[:, 0],
        opposite_wall_2[:, 1],
        color=wall_2_color,
        linewidth=3.2,
        solid_capstyle="round",
        zorder=9,
    )

    # ------------------------------------------------------------
    # Feasible-center set
    # ------------------------------------------------------------

    feasible_polygon_local = (
        build_feasible_center_polygon(
            D=D,
            a=a,
            b=b,
        )
    )

    feasible_polygon_world = local_to_world(
        local_points=feasible_polygon_local,
        corner=corner,
        local_x=local_x,
        local_y=local_y,
    )

    feasible_patch = Polygon(
        feasible_polygon_world,
        closed=True,
        facecolor="tab:green",
        edgecolor="none",
        alpha=0.25,
        zorder=10,
    )

    ax.add_patch(
        feasible_patch
    )

    # ------------------------------------------------------------
    # Quarter-circle boundary
    # ------------------------------------------------------------

    theta_quarter = np.linspace(
        0.0,
        0.5 * np.pi,
        250,
    )

    quarter_circle_local = np.column_stack(
        (
            D * np.cos(theta_quarter),
            D * np.sin(theta_quarter),
        )
    )

    quarter_circle_world = local_to_world(
        local_points=quarter_circle_local,
        corner=corner,
        local_x=local_x,
        local_y=local_y,
    )

    ax.plot(
        quarter_circle_world[:, 0],
        quarter_circle_world[:, 1],
        color="black",
        linestyle="--",
        linewidth=1.6,
        zorder=12,
    )

    # ------------------------------------------------------------
    # Constraint x = a_j
    # ------------------------------------------------------------

    y_at_x_a = np.sqrt(
        max(D**2 - a**2, 0.0)
    )

    vertical_constraint_local = np.array(
        [
            [a, 0.0],
            [a, y_at_x_a],
        ]
    )

    vertical_constraint_world = local_to_world(
        local_points=vertical_constraint_local,
        corner=corner,
        local_x=local_x,
        local_y=local_y,
    )

    ax.plot(
        vertical_constraint_world[:, 0],
        vertical_constraint_world[:, 1],
        color=wall_1_color,
        linewidth=2.4,
        zorder=14,
    )

    # ------------------------------------------------------------
    # Constraint y = b_j
    # ------------------------------------------------------------

    x_at_y_b = np.sqrt(
        max(D**2 - b**2, 0.0)
    )

    horizontal_constraint_local = np.array(
        [
            [0.0, b],
            [x_at_y_b, b],
        ]
    )

    horizontal_constraint_world = local_to_world(
        local_points=horizontal_constraint_local,
        corner=corner,
        local_x=local_x,
        local_y=local_y,
    )

    ax.plot(
        horizontal_constraint_world[:, 0],
        horizontal_constraint_world[:, 1],
        color=wall_2_color,
        linewidth=2.4,
        zorder=14,
    )

    # ------------------------------------------------------------
    # Footprint circle of radius r
    # ------------------------------------------------------------

    footprint_circle = Circle(
        xy=corner,
        radius=r,
        fill=False,
        edgecolor="black",
        linewidth=1.5,
        zorder=17,
    )

    ax.add_patch(
        footprint_circle
    )

    # The small radius is intentionally not aligned with R-r.
    angle_r = np.deg2rad(210.0)

    direction_r_world = (
        np.cos(angle_r) * local_x
        + np.sin(angle_r) * local_y
    )

    radius_r_end = (
        corner
        + r * direction_r_world
    )

    ax.plot(
        [
            corner[0],
            radius_r_end[0],
        ],
        [
            corner[1],
            radius_r_end[1],
        ],
        color="black",
        linewidth=1.15,
        zorder=18,
    )

    # ------------------------------------------------------------
    # Radius R-r as a double-headed dimension arrow
    # ------------------------------------------------------------

    angle_D = np.pi / 4.0

    direction_D_world = (
        np.cos(angle_D) * local_x
        + np.sin(angle_D) * local_y
    )

    # The annulus arrow starts on the boundary of the radius-r circle
    # and ends on the outer boundary of radius D = R-r.
    annulus_arrow_start = (
        corner
        + r * direction_D_world
    )

    annulus_arrow_end = (
        corner
        + D * direction_D_world
    )

    draw_double_arrow(
        ax=ax,
        start=annulus_arrow_start,
        end=annulus_arrow_end,
        color="black",
        linewidth=1.15,
        mutation_scale=10,
        zorder=18,
    )

    # ------------------------------------------------------------
    # Corner point
    # ------------------------------------------------------------

    ax.plot(
        corner[0],
        corner[1],
        marker="o",
        markersize=7,
        color="black",
        zorder=25,
    )

    # ------------------------------------------------------------
    # Positive local frame
    #
    # Same FancyArrowPatch style as in the other kinematic figure,
    # but thicker.
    # ------------------------------------------------------------

    local_axis_linewidth = 2.7
    local_axis_mutation_scale = 11

    arrow(
        ax=ax,
        start=corner,
        end=corner + axis_length * local_x,
        arrowstyle="-|>",
        linewidth=local_axis_linewidth,
        mutation_scale=local_axis_mutation_scale,
        color="black",
        zorder=30,
    )

    arrow(
        ax=ax,
        start=corner,
        end=corner + axis_length * local_y,
        arrowstyle="-|>",
        linewidth=local_axis_linewidth,
        mutation_scale=local_axis_mutation_scale,
        color="black",
        zorder=30,
    )

    # ------------------------------------------------------------
    # Width arrow for corridor 1
    # ------------------------------------------------------------

    width_j_start = corner

    width_j_end = (
        corner
        - corridor1.width * local_x
    )

    offset_j = (
        -width_arrow_offset_1
        * forward_1
    )

    draw_double_arrow(
        ax=ax,
        start=width_j_start + offset_j,
        end=width_j_end + offset_j,
        color="0.25",
        linewidth=1.3,
        mutation_scale=11,
        zorder=15,
    )

    # ------------------------------------------------------------
    # Width arrow for corridor 2
    # ------------------------------------------------------------

    width_j1_start = corner

    width_j1_end = (
        corner
        - corridor2.width * local_y
    )

    offset_j1 = (
        width_arrow_offset_2
        * forward_2
    )

    draw_double_arrow(
        ax=ax,
        start=width_j1_start + offset_j1,
        end=width_j1_end + offset_j1,
        color="0.25",
        linewidth=1.3,
        mutation_scale=11,
        zorder=15,
    )

    # ------------------------------------------------------------
    # Plot limits
    # ------------------------------------------------------------

    all_corners = np.vstack(
        (
            np.asarray(
                corridor1.corners,
                dtype=float,
            ),
            np.asarray(
                corridor2.corners,
                dtype=float,
            ),
        )
    )

    total_span_x = np.ptp(
        all_corners[:, 0]
    )

    total_span_y = np.ptp(
        all_corners[:, 1]
    )

    margin_x = (
        0.06 * total_span_x
    )

    margin_y = (
        0.06 * total_span_y
    )

    ax.set_xlim(
        np.min(
            all_corners[:, 0]
        ) - margin_x,
        np.max(
            all_corners[:, 0]
        ) + margin_x,
    )

    ax.set_ylim(
        np.min(
            all_corners[:, 1]
        ) - margin_y,
        np.max(
            all_corners[:, 1]
        ) + margin_y,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.axis("off")

    figure.tight_layout()

    figure.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.show()

    return figure, ax


# ================================================================
# Example
# ================================================================

if __name__ == "__main__":
    corridor1 = CorridorWorld(
        width=4.0,
        height=8.0,
        center=[0.0, 0.0],
        tilt=-np.pi / 2.0,
    )

    corridor2 = CorridorWorld(
        width=3.0,
        height=10.0,
        center=[3.0, -2.5],
        tilt=0.0,
    )

    output_directory = (
        Path(__file__).resolve().parent
        / "saved_figures"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_local_transition_geometry(
        corridor1=corridor1,
        corridor2=corridor2,
        R=3.8,
        r=0.5,
        output_path=(
            output_directory
            / "local_transition_geometry_feasible_set.pdf"
        ),
    )