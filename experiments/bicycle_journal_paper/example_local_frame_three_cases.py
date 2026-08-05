from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.corridor_geometry import get_corner_point
from kappa_planner.helpers.plot_helpers import plot_corridors


# ================================================================
# Matplotlib style
# ================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.linewidth": 0.8,
    }
)


# ================================================================
# Visual settings
# ================================================================

CORRIDOR_FILL_COLOR = "0.96"
CORRIDOR_EDGE_COLOR = "0.58"
SHRUNKEN_EDGE_COLOR = "0.72"
DIRECTION_ARROW_COLOR = "0.42"

WALL_1_COLOR = "tab:blue"
WALL_2_COLOR = "tab:orange"

FEASIBLE_SET_COLOR = "tab:green"
INFEASIBLE_POINT_COLOR = "tab:red"


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

    A local point (x, y) is mapped as

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
    """Draw a FancyArrowPatch arrow."""
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
    arrow_fraction=0.17,
    linewidth=1.8,
    color=DIRECTION_ARROW_COLOR,
):
    """Draw a short arrow along a corridor longitudinal direction."""
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
# Corridor drawing helpers
# ================================================================

def fill_corridor_free_space(
    ax,
    corridor_list,
    color=CORRIDOR_FILL_COLOR,
):
    """
    Fill the rectangular free-space regions of the corridors.

    The fill is opaque and identical for both rectangles, so their overlap
    has the same shade as the remaining corridor free space.
    """
    for corridor in corridor_list:
        corners = np.asarray(
            corridor.corners,
            dtype=float,
        )

        patch = Polygon(
            corners,
            closed=True,
            facecolor=color,
            edgecolor="none",
            alpha=1.0,
            zorder=1,
        )

        ax.add_patch(patch)


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
    Build the local candidate-center set

        x^2 + y^2 <= D^2,
        x >= a,
        y >= b.

    Returns None when the set is empty.
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
        return None

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
# Corridor-case construction
# ================================================================

def make_corridor_pair(
    width_1,
    width_2,
    *,
    height_1=8.0,
    height_2=10.0,
):
    """
    Construct an orthogonal left-turn corridor pair.

    The selected left-left corner is fixed at the world origin.
    """
    corridor1 = CorridorWorld(
        width=width_1,
        height=height_1,
        center=[
            -0.5 * width_1,
            0.0,
        ],
        tilt=-np.pi / 2.0,
    )

    corridor2 = CorridorWorld(
        width=width_2,
        height=height_2,
        center=[
            2.5,
            -0.5 * width_2,
        ],
        tilt=0.0,
    )

    return corridor1, corridor2


# ================================================================
# Local transition geometry
# ================================================================

def extract_transition_geometry(
    corridor1,
    corridor2,
    R,
    r,
):
    """Compute the geometric quantities used by one panel."""
    if R <= r:
        raise ValueError(
            f"Expected R > r, but received R={R}, r={r}."
        )

    turn = (
        corridor1.compute_relative_turn_direction(
            corridor2
        )
    )

    if turn == 0:
        raise ValueError(
            "The corridor pair must be represented orthogonally."
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

    if turn > 0:
        opposite_edge_1 = corridor1.RGT
        opposite_edge_2 = corridor2.RGT
    else:
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

    feasible = (
        a**2 + b**2
        <= D**2 + 1e-12
    )

    return {
        "turn": turn,
        "corner": corner,
        "local_x": local_x,
        "local_y": local_y,
        "forward_1": np.asarray(
            corridor1.unit_vector,
            dtype=float,
        ),
        "forward_2": np.asarray(
            corridor2.unit_vector,
            dtype=float,
        ),
        "opposite_wall_1": opposite_wall_1,
        "opposite_wall_2": opposite_wall_2,
        "D": D,
        "S": S,
        "a": a,
        "b": b,
        "feasible": feasible,
    }


# ================================================================
# One figure panel
# ================================================================

def draw_transition_panel(
    ax,
    corridor1,
    corridor2,
    R,
    r,
    *,
    show_infeasible_point=True,
):
    """Draw one local transition-feasibility panel."""
    geometry = extract_transition_geometry(
        corridor1,
        corridor2,
        R,
        r,
    )

    corner = geometry["corner"]
    local_x = geometry["local_x"]
    local_y = geometry["local_y"]

    forward_1 = geometry["forward_1"]
    forward_2 = geometry["forward_2"]

    opposite_wall_1 = geometry[
        "opposite_wall_1"
    ]

    opposite_wall_2 = geometry[
        "opposite_wall_2"
    ]

    D = geometry["D"]
    a = geometry["a"]
    b = geometry["b"]
    feasible = geometry["feasible"]

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
    # Filled free space
    # ------------------------------------------------------------

    fill_corridor_free_space(
        ax,
        [
            corridor1,
            corridor2,
        ],
    )

    # ------------------------------------------------------------
    # Original corridor boundaries
    # ------------------------------------------------------------

    figure = ax.figure

    plot_corridors(
        corridor_list=[
            corridor1,
            corridor2,
        ],
        figure=figure,
        plot_vectors=False,
        plot_corridor_index=False,
        linestyle="-",
        color=CORRIDOR_EDGE_COLOR,
        linewidth=1.6,
    )

    # ------------------------------------------------------------
    # Shrunken corridors
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
        color=SHRUNKEN_EDGE_COLOR,
        linewidth=1.3,
    )

    # ------------------------------------------------------------
    # Corridor directions
    # ------------------------------------------------------------

    draw_corridor_direction_arrow(
        ax=ax,
        corridor=corridor1,
        arrow_fraction=0.17,
    )

    draw_corridor_direction_arrow(
        ax=ax,
        corridor=corridor2,
        arrow_fraction=0.17,
    )

    # ------------------------------------------------------------
    # Opposite walls
    # ------------------------------------------------------------

    ax.plot(
        opposite_wall_1[:, 0],
        opposite_wall_1[:, 1],
        color=WALL_1_COLOR,
        linewidth=3.0,
        solid_capstyle="round",
        zorder=9,
    )

    ax.plot(
        opposite_wall_2[:, 0],
        opposite_wall_2[:, 1],
        color=WALL_2_COLOR,
        linewidth=3.0,
        solid_capstyle="round",
        zorder=9,
    )

    # ------------------------------------------------------------
    # Candidate-center set
    # ------------------------------------------------------------

    feasible_polygon_local = (
        build_feasible_center_polygon(
            D=D,
            a=a,
            b=b,
        )
    )

    if feasible_polygon_local is not None:
        feasible_polygon_world = local_to_world(
            local_points=feasible_polygon_local,
            corner=corner,
            local_x=local_x,
            local_y=local_y,
        )

        feasible_patch = Polygon(
            feasible_polygon_world,
            closed=True,
            facecolor=FEASIBLE_SET_COLOR,
            edgecolor="none",
            alpha=0.25,
            zorder=10,
        )

        ax.add_patch(
            feasible_patch
        )

    # ------------------------------------------------------------
    # Quarter-disk boundary
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
        linewidth=1.5,
        zorder=12,
    )

    # ------------------------------------------------------------
    # Opposite-wall constraints
    # ------------------------------------------------------------

    constraint_extent = 1.08 * D

    if a > 1e-10:
        if a <= D:
            vertical_end = np.sqrt(
                max(D**2 - a**2, 0.0)
            )
        else:
            vertical_end = constraint_extent

        vertical_constraint_local = np.array(
            [
                [a, 0.0],
                [a, max(vertical_end, b)],
            ]
        )

        vertical_constraint_world = local_to_world(
            vertical_constraint_local,
            corner,
            local_x,
            local_y,
        )

        ax.plot(
            vertical_constraint_world[:, 0],
            vertical_constraint_world[:, 1],
            color=WALL_1_COLOR,
            linewidth=2.3,
            zorder=14,
        )

    if b > 1e-10:
        if b <= D:
            horizontal_end = np.sqrt(
                max(D**2 - b**2, 0.0)
            )
        else:
            horizontal_end = constraint_extent

        horizontal_constraint_local = np.array(
            [
                [0.0, b],
                [max(horizontal_end, a), b],
            ]
        )

        horizontal_constraint_world = local_to_world(
            horizontal_constraint_local,
            corner,
            local_x,
            local_y,
        )

        ax.plot(
            horizontal_constraint_world[:, 0],
            horizontal_constraint_world[:, 1],
            color=WALL_2_COLOR,
            linewidth=2.3,
            zorder=14,
        )

    # ------------------------------------------------------------
    # Infeasible closest point
    # ------------------------------------------------------------

    if (
        not feasible
        and show_infeasible_point
    ):
        infeasible_point_world = local_to_world(
            np.array([[a, b]]),
            corner,
            local_x,
            local_y,
        )[0]

        ax.plot(
            infeasible_point_world[0],
            infeasible_point_world[1],
            marker="x",
            markersize=10,
            markeredgewidth=2.2,
            color=INFEASIBLE_POINT_COLOR,
            zorder=20,
        )

    # ------------------------------------------------------------
    # Radius-r corner disk
    # ------------------------------------------------------------

    footprint_circle = Circle(
        xy=corner,
        radius=r,
        fill=False,
        edgecolor="black",
        linewidth=1.4,
        zorder=17,
    )

    ax.add_patch(
        footprint_circle
    )

    angle_r = np.deg2rad(
        210.0
    )

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
        linewidth=1.1,
        zorder=18,
    )

    # ------------------------------------------------------------
    # Radial annulus indicator
    # ------------------------------------------------------------

    angle_D = np.pi / 4.0

    direction_D_world = (
        np.cos(angle_D) * local_x
        + np.sin(angle_D) * local_y
    )

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
        linewidth=1.1,
        mutation_scale=9,
        zorder=18,
    )

    # ------------------------------------------------------------
    # Corner
    # ------------------------------------------------------------

    ax.plot(
        corner[0],
        corner[1],
        marker="o",
        markersize=6,
        color="black",
        zorder=25,
    )

    # ------------------------------------------------------------
    # Positive local frame
    # ------------------------------------------------------------

    arrow(
        ax=ax,
        start=corner,
        end=corner + axis_length * local_x,
        arrowstyle="-|>",
        linewidth=2.6,
        mutation_scale=11,
        color="black",
        zorder=30,
    )

    arrow(
        ax=ax,
        start=corner,
        end=corner + axis_length * local_y,
        arrowstyle="-|>",
        linewidth=2.6,
        mutation_scale=11,
        color="black",
        zorder=30,
    )

    # ------------------------------------------------------------
    # Width arrows
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
        linewidth=1.25,
        mutation_scale=10,
        zorder=15,
    )

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
        linewidth=1.25,
        mutation_scale=10,
        zorder=15,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.axis("off")

    return geometry


# ================================================================
# Shared panel limits
# ================================================================

def compute_common_limits(
    cases,
    R,
    r,
    margin_fraction=0.05,
):
    """Compute common world-coordinate limits for all panels."""
    all_points = []

    D = R - r

    for corridor1, corridor2 in cases:
        geometry = extract_transition_geometry(
            corridor1,
            corridor2,
            R,
            r,
        )

        all_points.extend(
            np.asarray(
                corridor1.corners,
                dtype=float,
            )
        )

        all_points.extend(
            np.asarray(
                corridor2.corners,
                dtype=float,
            )
        )

        corner = geometry["corner"]
        local_x = geometry["local_x"]
        local_y = geometry["local_y"]

        local_boundary = np.array(
            [
                [0.0, 0.0],
                [1.2 * D, 0.0],
                [0.0, 1.2 * D],
                [1.2 * D, 1.2 * D],
            ]
        )

        all_points.extend(
            local_to_world(
                local_boundary,
                corner,
                local_x,
                local_y,
            )
        )

    all_points = np.asarray(
        all_points,
        dtype=float,
    )

    span_x = np.ptp(
        all_points[:, 0]
    )

    span_y = np.ptp(
        all_points[:, 1]
    )

    margin_x = (
        margin_fraction * span_x
    )

    margin_y = (
        margin_fraction * span_y
    )

    x_limits = (
        np.min(all_points[:, 0]) - margin_x,
        np.max(all_points[:, 0]) + margin_x,
    )

    y_limits = (
        np.min(all_points[:, 1]) - margin_y,
        np.max(all_points[:, 1]) + margin_y,
    )

    return x_limits, y_limits


# ================================================================
# Three-panel figure
# ================================================================

def plot_three_transition_cases(
    R,
    r,
    output_path="local_transition_geometry_three_cases.pdf",
):
    """
    Plot three local transition cases:

        (a) no active opposite-wall constraints;
        (b) active constraints with nonempty candidate set;
        (c) active constraints with empty candidate set.
    """
    S = R + r

    # ------------------------------------------------------------
    # Panel (a): wide corridors
    #
    # Both widths exceed R+r, hence a=b=0.
    # ------------------------------------------------------------

    case_1 = make_corridor_pair(
        width_1=S + 0.9,
        width_2=S + 0.7,
    )

    # ------------------------------------------------------------
    # Panel (b): active but feasible constraints
    #
    # This reproduces the central case used previously.
    # ------------------------------------------------------------

    case_2 = make_corridor_pair(
        width_1=4.0,
        width_2=3.0,
    )

    # ------------------------------------------------------------
    # Panel (c): empty candidate set
    # ------------------------------------------------------------

    case_3 = make_corridor_pair(
        width_1=1.6,
        width_2=1.6,
    )

    cases = [
        case_1,
        case_2,
        case_3,
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 5.2),
    )

    for ax, corridor_pair in zip(
        axes,
        cases,
    ):
        corridor1, corridor2 = corridor_pair

        geometry = draw_transition_panel(
            ax=ax,
            corridor1=corridor1,
            corridor2=corridor2,
            R=R,
            r=r,
        )

        print(
            {
                "width_1": corridor1.width,
                "width_2": corridor2.width,
                "a": geometry["a"],
                "b": geometry["b"],
                "D": geometry["D"],
                "feasible": geometry["feasible"],
            }
        )

    # Same geometric scale in all three panels.
    x_limits, y_limits = compute_common_limits(
        cases=cases,
        R=R,
        r=r,
    )

    for ax in axes:
        ax.set_xlim(
            *x_limits
        )

        ax.set_ylim(
            *y_limits
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.axis("off")

    figure.subplots_adjust(
        left=0.015,
        right=0.995,
        bottom=0.035,
        top=0.985,
        wspace=0.02,
    )

    figure.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    plt.show()

    return figure, axes


# ================================================================
# Example
# ================================================================

if __name__ == "__main__":
    output_directory = (
        Path(__file__).resolve().parent
        / "saved_figures"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_three_transition_cases(
        R=3.8,
        r=0.5,
        output_path=(
            output_directory
            / "local_transition_geometry_three_cases.pdf"
        ),
    )