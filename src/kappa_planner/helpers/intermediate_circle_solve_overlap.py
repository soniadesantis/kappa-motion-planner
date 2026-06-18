from ..geometry import Point, IntermediateCircle, IntermediateCirclesSequence, IntermediateCircleChoice, IntermediateCircleChoicesSequence, Circle
from ..corridor import CorridorWorld
from ..vehicle import Unicycle, Bicycle
from .intersections import compute_intersection_points_circle_segment, compute_line_corridor_intersections
from .corridor_geometry import get_corner_point_and_intersecting_edges
from .intermediate_circles_geometry import compute_center_coordinates_second_circle_according_to_edges, compute_center_coordinates_second_circle_given_two_points
from .plot_helpers import plot_corridors
from .geometry_operations import (
    select_tangency_point_from_point_circle,
    compute_turn_direction,
    compute_turn_direction_from_three_points,
    efficient_sign,
    compute_distance_two_points,
    project_point_onto_segment,
    check_point_inside_segment,
    minimum_distance_between_segments,
)
from math import sqrt, pi, cos, sin, asin, tau
from matplotlib import pyplot as plt
import numpy as np


def classify_overlap_cluster(cluster):
    cluster_size = len(cluster["indices"])

    pair_turns = [pair_info["turns"] for pair_info in cluster["pairs"]]

    if cluster_size == 2:
        t1, t2 = pair_turns[0]

        if t1 == t2:
            return "pair_same_turn"
        else:
            return "pair_opposite_turn"

    # For larger clusters, collect the turn of each circle index.
    circle_turns = {}

    for pair_info in cluster["pairs"]:
        i, j = pair_info["pair"]
        ti, tj = pair_info["turns"]

        circle_turns[i] = ti
        circle_turns[j] = tj

    unique_turns = set(circle_turns.values())

    if len(unique_turns) == 1:
        return "multi_same_turn"
    else:
        return "multi_mixed_turn"
    

def extract_overlap_clusters(overlapping_pairs):
    """
    Convert a list of overlapping consecutive pairs into overlap clusters.

    Example:
        [(0, 1), (1, 2), (3, 4)]
        -> clusters [0, 1, 2] and [3, 4]
    """

    if not overlapping_pairs:
        return []

    clusters = []

    current_start = overlapping_pairs[0]["pair"][0]
    current_end = overlapping_pairs[0]["pair"][1]
    current_pairs = [overlapping_pairs[0]]

    for pair_info in overlapping_pairs[1:]:
        i, j = pair_info["pair"]

        if i <= current_end:
            # This pair touches the current cluster.
            current_end = max(current_end, j)
            current_pairs.append(pair_info)
        else:
            clusters.append({
                "start": current_start,
                "end": current_end,
                "indices": list(range(current_start, current_end + 1)),
                "pairs": current_pairs,
            })

            current_start = i
            current_end = j
            current_pairs = [pair_info]

    clusters.append({
        "start": current_start,
        "end": current_end,
        "indices": list(range(current_start, current_end + 1)),
        "pairs": current_pairs,
    })

    return clusters


def compute_consecutive_overlap_pairs(circle_choices_sequence):
    overlapping_pairs = []

    for index in range(len(circle_choices_sequence) - 1):
        choice1 = circle_choices_sequence[index]
        choice2 = circle_choices_sequence[index + 1]

        circle1 = select_preferred_circle(choice1)
        circle2 = select_preferred_circle(choice2)

        status = overlap_status_for_intermediate_circles(circle1, circle2)

        if status["nominal_overlap"]:
            overlapping_pairs.append({
                "pair": (index, index + 1),
                "status": status,
                "turns": (circle1.turn_direction, circle2.turn_direction),
            })

    return overlapping_pairs


def select_preferred_circle(choice: IntermediateCircleChoice) -> IntermediateCircle:
    """
    Select the active circle from one IntermediateCircleChoice.

    If the choice is ambiguous, select the candidate whose turn_direction
    matches choice.preferred_turn_direction.
    Otherwise, return the only candidate.
    """

    if len(choice) == 0:
        raise ValueError("IntermediateCircleChoice has no candidates")

    if not choice.is_ambiguous:
        return choice.first

    if choice.preferred_turn_direction is None:
        raise ValueError(
            f"Ambiguous choice at index {choice.index} has no preferred_turn_direction"
        )

    matching_candidates = [
        circle for circle in choice
        if circle.turn_direction == choice.preferred_turn_direction
    ]

    if len(matching_candidates) != 1:
        raise ValueError(
            f"Expected exactly one candidate with turn_direction "
            f"{choice.preferred_turn_direction}, got {len(matching_candidates)}"
        )

    return matching_candidates[0]


def intermediate_circle_center_at_s(circle, s):
    """
    Return the center Point of an IntermediateCircle at parameter s,
    without mutating the circle.
    """

    if s < 0 or s > circle.s_max:
        raise ValueError(f"s must be in [0, {circle.s_max}]")

    dx = cos(circle.bisector_direction)
    dy = sin(circle.bisector_direction)

    return Point(
        circle.canonical_center.x + s * dx,
        circle.canonical_center.y + s * dy,
    )


def circles_overlap_from_centers(center1, radius1, center2, radius2, tol=1e-9):
    """
    Return True if two circles overlap or touch.
    """

    distance = compute_distance_two_points(center1, center2)

    return distance <= radius1 + radius2 + tol


def intermediate_circles_overlap_at_s(circle1, s1, circle2, s2, tol=1e-9):
    """
    Check overlap between two IntermediateCircle objects
    at arbitrary shift parameters s1 and s2.
    """

    center1 = intermediate_circle_center_at_s(circle1, s1)
    center2 = intermediate_circle_center_at_s(circle2, s2)

    return circles_overlap_from_centers(
        center1,
        circle1.radius,
        center2,
        circle2.radius,
        tol=tol,
    )


def overlap_status_for_intermediate_circles(circle1, circle2, tol=1e-9):
    """
    Check overlap between two intermediate circles in:
    - nominal configuration (s = 0)
    - maximally shifted configuration (s = s_max)
    """

    nominal_overlap = intermediate_circles_overlap_at_s(
        circle1, 0,
        circle2, 0,
        tol=tol,
    )

    max_shift_overlap = intermediate_circles_overlap_at_s(
        circle1, circle1.s_max,
        circle2, circle2.s_max,
        tol=tol,
    )

    min_distance = minimum_distance_between_intermediate_circle_center_segments(
        circle1,
        circle2,
    )

    required_distance = circle1.radius + circle2.radius 

    return {
        "nominal_overlap": nominal_overlap,
        "max_shift_overlap": max_shift_overlap,
        "can_overlap": min_distance < required_distance - tol,
        "min_distance": min_distance,
    }



def compute_circle_center_through_two_points_with_radius(
    point1,
    point2,
    radius,
    turn_direction,
    tol=1e-9,
):
    """
    Compute the center of a circle with given radius passing through two points.

    turn_direction:
        +1 selects the center on the left side of point1 -> point2
        -1 selects the center on the right side of point1 -> point2
    """

    if turn_direction not in (-1, 1):
        raise ValueError("turn_direction must be -1 or +1")

    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y

    dx = x2 - x1
    dy = y2 - y1

    d = compute_distance_two_points(point1, point2)

    if d < tol:
        raise ValueError("The two points are coincident; infinitely many circles exist.")

    if d > 2 * radius + tol:
        raise ValueError(
            "No circle with the given radius can pass through both points."
        )

    # Midpoint of the chord
    mx = 0.5 * (x1 + x2)
    my = 0.5 * (y1 + y2)

    # Distance from midpoint to circle center
    half_chord = 0.5 * d
    h_sq = radius**2 - half_chord**2

    # Numerical safety
    if h_sq < 0 and abs(h_sq) < tol:
        h_sq = 0.0

    h = sqrt(h_sq)

    # Unit perpendicular to point1 -> point2
    ux = -dy / d
    uy = dx / d

    if turn_direction == 1:
        cx = mx + h * ux
        cy = my + h * uy
    else:
        cx = mx - h * ux
        cy = my - h * uy

    return Point(cx, cy)


def compute_circle_through_two_points_with_radius(
    point1,
    point2,
    radius,
    turn_direction,
):
    center = compute_circle_center_through_two_points_with_radius(
        point1,
        point2,
        radius,
        turn_direction,
    )

    return Circle(center=center, radius=radius)


def plot_shift_direction(ax, origin, shift_direction, length=1.0, color="magenta", label="shift direction"):
    """
    Plot a shift direction as an arrow.

    Parameters
    ----------
    ax : matplotlib axis
    origin : Point
        Starting point of the arrow.
    shift_direction : tuple[float, float]
        Direction vector, e.g. (sx, sy).
    length : float
        Arrow length used only for visualization.
    """

    sx, sy = shift_direction

    norm = sqrt(sx**2 + sy**2)
    if norm == 0:
        raise ValueError("Cannot plot zero shift direction.")

    sx /= norm
    sy /= norm

    ax.arrow(
        origin.x,
        origin.y,
        length * sx,
        length * sy,
        head_width=0.15 * length,
        head_length=0.2 * length,
        length_includes_head=True,
        color=color,
        linewidth=2.0,
        label=label,
    )

    ax.plot(origin.x, origin.y, "o", color=color)


def get_half_dimension_perpendicular_to_edge(corridor, edge_index):
    """
    Return the distance from an edge to the corridor centerline parallel to it.
    """

    if edge_index in (corridor.FWD, corridor.BCK):
        return corridor.height / 2

    elif edge_index in (corridor.RGT, corridor.LFT):
        return corridor.width / 2

    else:
        raise ValueError(f"Invalid edge index: {edge_index}")


def shift_segment_inside_corridor(corridor, edge_index, P, Q):
    """
    Shift an edge segment inward, opposite to the outward normal.

    P, Q are segment endpoints as arrays/lists [x, y].
    """

    normal = corridor.outward_normals[edge_index]
    distance = get_half_dimension_perpendicular_to_edge(corridor, edge_index)

    P_shifted = np.array(P) + distance * normal
    Q_shifted = np.array(Q) + distance * normal

    return P_shifted, Q_shifted

    

def build_merged_intermediate_circle(
    circle1,
    circle2,
    corridor1,
    corridor2,
    corridor3,
    merged_circle,
    vehicle,
    index=None,
    s_max=0.0,
):
    R = vehicle.max_radius
    r = vehicle.width / 2
    R_check = R + r

    small_circle1 = Circle(circle1.corner_point, r)
    small_circle2 = Circle(circle2.corner_point, r)

    corner_point1 = circle1.corner_point
    corner_point2 = circle2.corner_point

    # ------------------------------------------------------------
    # Edge of the middle corridor associated with the merged corner
    # ------------------------------------------------------------
    if circle1.edge_pair[1] == 0:
        A = corridor2.get_corners()[3]
        B = corridor2.get_corners()[0]
    elif circle1.edge_pair[1] == 1:
        A = corridor2.get_corners()[0]
        B = corridor2.get_corners()[1]
    elif circle1.edge_pair[1] == 2:
        A = corridor2.get_corners()[1]
        B = corridor2.get_corners()[2]
    elif circle1.edge_pair[1] == 3:
        A = corridor2.get_corners()[2]
        B = corridor2.get_corners()[3]
    else:
        raise ValueError(f"Invalid edge index: {circle1.edge_pair[1]}")


    # This is still useful for plotting/debugging, but not used by the
    # edge-tangent repair below.
    shift_direction = unit_vector_from_points(A, B)

    # ------------------------------------------------------------
    # Critical outer edge of corridor1
    # ------------------------------------------------------------
    edge_corridor1 = circle1.edge_pair[0]

    if edge_corridor1 == 0:
        C = corridor1.get_corners()[1]
        D = corridor1.get_corners()[2]
    elif edge_corridor1 == 1:
        C = corridor1.get_corners()[2]
        D = corridor1.get_corners()[3]
    elif edge_corridor1 == 2:
        C = corridor1.get_corners()[3]
        D = corridor1.get_corners()[0]
    elif edge_corridor1 == 3:
        C = corridor1.get_corners()[0]
        D = corridor1.get_corners()[1]
    else:
        raise ValueError(f"Invalid edge index: {edge_corridor1}")
    
    C, D = shift_segment_inside_corridor(
        corridor=corridor1,
        edge_index=edge_corridor1,
        P=C,
        Q=D,
    )
    # ------------------------------------------------------------
    # Critical outer edge of corridor3
    # ------------------------------------------------------------
    edge_corridor3 = circle2.edge_pair[1]

    if edge_corridor3 == 0:
        E = corridor3.get_corners()[1]
        F = corridor3.get_corners()[2]
    elif edge_corridor3 == 1:
        E = corridor3.get_corners()[2]
        F = corridor3.get_corners()[3]
    elif edge_corridor3 == 2:
        E = corridor3.get_corners()[3]
        F = corridor3.get_corners()[0]
    elif edge_corridor3 == 3:
        E = corridor3.get_corners()[0]
        F = corridor3.get_corners()[1]
    else:
        raise ValueError(f"Invalid edge index: {edge_corridor3}")


    E, F = shift_segment_inside_corridor(
        corridor=corridor3,
        edge_index=edge_corridor3,
        P=E,
        Q=F,
    )
    # ------------------------------------------------------------
    # Check whether the clearance circle intersects critical edges
    # ------------------------------------------------------------
    center = merged_circle.center
    # figure = plot_corridors(
    #     [corridor1, corridor2, corridor3],
    # )
    # plt.plot([C[0], D[0]], [C[1], D[1]], "r-")
    # plt.plot([E[0], F[0]], [E[1], F[1]], "b-")
    # plt.gca().add_patch(plt.Circle((center.x, center.y), R_check, color='green', fill=False, linestyle='--'))
    # plt.show(block=True)


    int_point_corridor1 = compute_intersection_points_circle_segment(
        C[0], C[1],
        D[0], D[1],
        center.x,
        center.y,
        R_check,
        tol=1e-5,
    )

    int_point_corridor3 = compute_intersection_points_circle_segment(
        E[0], E[1],
        F[0], F[1],
        center.x,
        center.y,
        R_check,
        tol=1e-5,
    )

    # ------------------------------------------------------------
    # Repair if exactly one outer edge is violated
    # ------------------------------------------------------------
    repaired_circle = None

    if int_point_corridor1 and int_point_corridor3:
        raise ValueError(
            "Merged circle does not fit: both critical outer edges are intersected."
        )

    elif int_point_corridor1:
        # corridor1 is problematic, so use small_circle1 as the active constraint
        repaired_circle = compute_big_circle_tangent_to_small_circle_and_edge(
            active_small_circle=small_circle1,
            other_small_circle=small_circle2,
            violated_edge_start=C,
            violated_edge_end=D,
            radius=R,
            vehicle=vehicle,
            reference_center=merged_circle.center,
        )

        if repaired_circle is None:
            raise ValueError(
                "Merged circle cannot be repaired after intersecting corridor1 edge."
            )

        # Recheck against the other outer edge
        other_edges = [(E, F)]

    elif int_point_corridor3:
        # corridor3 is problematic, so use small_circle2 as the active constraint
        repaired_circle = compute_big_circle_tangent_to_small_circle_and_edge(
            active_small_circle=small_circle2,
            other_small_circle=small_circle1,
            violated_edge_start=E,
            violated_edge_end=F,
            radius=R,
            vehicle=vehicle,
            reference_center=merged_circle.center,
        )

        if repaired_circle is None:
            raise ValueError(
                "Merged circle cannot be repaired after intersecting corridor3 edge."
            )

        # Recheck against the other outer edge
        other_edges = [(C, D)]

    # If repair was performed, validate and update merged_circle
    if repaired_circle is not None:
        clearance_circle = Circle(repaired_circle.center, R_check)

        if circle_intersects_any_segment(clearance_circle, other_edges):
            raise ValueError(
                "Repaired merged circle still intersects another critical edge."
            )

        merged_circle = repaired_circle

    # ------------------------------------------------------------
    # Recompute center and projected merged corner after possible repair
    # ------------------------------------------------------------
    center_merged_circle = merged_circle.center

    corner_point_merged = project_point_onto_segment(
        point=center_merged_circle,
        segment_start=corner_point1,
        segment_end=corner_point2,
    )

    # ------------------------------------------------------------
    # Recompute s_max after possible repair
    # ------------------------------------------------------------
    is_corner_point_inside = check_point_inside_segment(
        A,
        B,
        corner_point_merged,
    )

    if is_corner_point_inside:
        dist1 = compute_distance_two_points(
            corner_point_merged,
            center_merged_circle,
        )

        s_max = dist1 + corridor2.width - R - r

        if s_max < 0:
            raise ValueError(
                "Merged circle cannot fit in the middle corridor."
            )

    return IntermediateCircle(
        center=center_merged_circle,
        radius=R,
        corner_point=corner_point_merged,
        turn_direction=circle1.turn_direction,
        index=index if index is not None else circle1.index,
        s_max=s_max,
        edge_pair=None,
        door_point=None,
        door_type="merged",
    )
  

def choose_shift_direction_away_from_edge(
    center,
    shift_direction,
    edge_start,
    edge_end,
    tol=1e-9,
):
    sx, sy = shift_direction

    ax, ay = edge_start[0], edge_start[1]
    bx, by = edge_end[0], edge_end[1]

    ex = bx - ax
    ey = by - ay
    edge_norm = sqrt(ex**2 + ey**2)

    if edge_norm < tol:
        raise ValueError("Degenerate edge.")

    # One unit normal to the violated edge
    nx = -ey / edge_norm
    ny = ex / edge_norm

    # Signed distance of the current center to the edge line
    d0 = nx * (center.x - ax) + ny * (center.y - ay)

    # Derivative of signed distance if we shift along +s
    dd = nx * sx + ny * sy

    # We want the absolute distance from the edge to increase.
    # If d0 and dd have the same sign, +s moves away.
    # Otherwise, -s moves away.
    if d0 * dd >= 0:
        return sx, sy
    else:
        return -sx, -sy
    

def unit_vector_from_points(A, B, tol=1e-9):
    dx = B[0] - A[0]
    dy = B[1] - A[1]

    norm = sqrt(dx**2 + dy**2)

    if norm < tol:
        raise ValueError("Cannot define unit vector from coincident points.")

    return dx / norm, dy / norm


def compute_big_circle_tangent_to_small_circle_and_edge(
    active_small_circle,
    other_small_circle,
    violated_edge_start,
    violated_edge_end,
    radius,
    vehicle,
    reference_center=None,
    tol=1e-6,
):
    """
    Compute candidate big circles of fixed radius R that are:
    - internally tangent to active_small_circle;
    - clearance-tangent to the violated edge;
    - containing other_small_circle.

    Returns the best candidate Circle, or None.
    """

    R = radius
    r = vehicle.width / 2
    R_inner = R - r      # distance from big center to active small center
    R_check = R + r      # clearance radius used for edge tangency

    p = active_small_circle.center

    ax, ay = violated_edge_start[0], violated_edge_start[1]
    bx, by = violated_edge_end[0], violated_edge_end[1]

    ex = bx - ax
    ey = by - ay
    edge_norm = sqrt(ex**2 + ey**2)

    if edge_norm < tol:
        return None

    # Unit tangent and normal of the violated edge
    tx = ex / edge_norm
    ty = ey / edge_norm

    nx = -ty
    ny = tx

    # Signed distance from active small center to violated edge line
    d_p = nx * (p.x - ax) + ny * (p.y - ay)

    candidates = []

    # Two parallel lines at signed distance +/- R_check from the violated edge
    for sigma in (+1, -1):
        target_signed_distance = sigma * R_check

        # We need center o = q0 + alpha * t,
        # where q0 is one point on the parallel line.
        # Choose q0 by shifting edge_start along normal.
        q0x = ax + target_signed_distance * nx
        q0y = ay + target_signed_distance * ny

        # Intersect this line with circle centered at p of radius R_inner.
        # o(alpha) = q0 + alpha * t
        # ||o(alpha) - p||^2 = R_inner^2
        wx = q0x - p.x
        wy = q0y - p.y

        # Since t is unit:
        # alpha^2 + 2 alpha (w dot t) + ||w||^2 - R_inner^2 = 0
        b = 2.0 * (wx * tx + wy * ty)
        c = wx**2 + wy**2 - R_inner**2

        disc = b**2 - 4.0 * c

        if disc < -tol:
            continue

        if disc < 0:
            disc = 0.0

        sqrt_disc = sqrt(disc)

        for alpha in (
            (-b + sqrt_disc) / 2.0,
            (-b - sqrt_disc) / 2.0,
        ):
            ox = q0x + alpha * tx
            oy = q0y + alpha * ty

            candidate = Circle(
                center=Point(ox, oy),
                radius=R,
            )

            if not circle_contains_small_circle(
                candidate,
                other_small_circle,
                tol=tol,
            ):
                continue

            candidates.append(candidate)

    if not candidates:
        return None

    if reference_center is None:
        return candidates[0]

    # Choose the candidate closest to the original merged center
    candidates.sort(
        key=lambda c: compute_distance_two_points(c.center, reference_center)
    )

    return candidates[0]


def circle_contains_small_circle(big_circle, small_circle, tol=1e-6):
    d = compute_distance_two_points(big_circle.center, small_circle.center)
    return d + small_circle.radius <= big_circle.radius + tol


def circle_intersects_any_segment(circle, segments, tol=1e-6):
    for A, B in segments:
        points = compute_intersection_points_circle_segment(
            A[0], A[1],
            B[0], B[1],
            circle.center.x,
            circle.center.y,
            circle.radius,
            tol=tol,
        )

        if points:
            return True

    return False


def minimum_distance_between_intermediate_circle_center_segments(circle1, circle2):
    c1_start = intermediate_circle_center_at_s(circle1, 0.0)
    c1_end = intermediate_circle_center_at_s(circle1, circle1.s_max)

    c2_start = intermediate_circle_center_at_s(circle2, 0.0)
    c2_end = intermediate_circle_center_at_s(circle2, circle2.s_max)

    return minimum_distance_between_segments(
        c1_start,
        c1_end,
        c2_start,
        c2_end,
    )


def build_circle_overlap_opposite_turn_direction(
    original_circle,
    middle_corridor,
    vehicle,
):
    
    corner_point = original_circle.corner_point

    if original_circle.edge_pair is None:
        raise ValueError("Original circle must have edge_pair defined.")

    edge_middle_circle = original_circle.edge_pair[1]
    new_orientation = middle_corridor.outward_normals[edge_middle_circle]
            
    R = vehicle.max_radius
    r = vehicle.width * 0.5
    edge_pair=original_circle.edge_pair
    circle_index=original_circle.index
    door_type=original_circle.door_type
    s_max = middle_corridor.width - 2 * r
    xc2 = corner_point.x + (R - r) * new_orientation[0]
    yc2 = corner_point.y + (R - r) * new_orientation[1]

    return IntermediateCircle(
            Point(xc2, yc2),
            R,
            Point(corner_point[0], corner_point[1]),
            original_circle.turn_direction,
            index=circle_index,
            s_max=s_max,
            edge_pair=edge_pair,
            door_type=door_type,
        )       
