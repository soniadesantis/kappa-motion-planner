from ..geometry import Point, IntermediateCircle, IntermediateCirclesSequence, IntermediateCircleChoice, IntermediateCircleChoicesSequence
from ..corridor import CorridorWorld
from ..vehicle import Unicycle, Bicycle
from .intersections import compute_intersection_two_segments, compute_line_corridor_intersections
from .corridor_geometry import get_corner_point_and_intersecting_edges
from .intermediate_circles_geometry import compute_center_coordinates_second_circle_according_to_edges, compute_center_coordinates_second_circle_given_two_points
from .intermediate_circle_solve_overlap import (
    select_preferred_circle,
    overlap_status_for_intermediate_circles,
    compute_circle_through_two_points_with_radius,
)
from .inputs_check import compute_min_width_s_max_corridor_pair
from .plot_helpers import plot_corridors
from .geometry_operations import (
    select_tangency_point_from_point_circle,
    compute_turn_direction,
    compute_turn_direction_from_three_points,
    efficient_sign,
    compute_distance_two_points
)
from math import sqrt, atan2, cos, sin, asin, tau
from matplotlib import pyplot as plt
import numpy as np


import numpy as np

import numpy as np
import matplotlib.pyplot as plt


def solve_circles_overlap(
    circle_choices_sequence,
    corridor_sequence,
    vehicle,
    start_pose,
    end_pose,
):
    """
    Resolve overlaps between consecutive selected intermediate circles.

    For ambiguous choices, the preferred candidate is selected.
    """

    index = 0

    while index < len(circle_choices_sequence) - 1:

        choice1 = circle_choices_sequence[index]
        choice2 = circle_choices_sequence[index + 1]

        circle1 = select_preferred_circle(choice1)
        circle2 = select_preferred_circle(choice2)

        status = overlap_status_for_intermediate_circles(
            circle1,
            circle2,
        )

        nominal_overlap = status["nominal_overlap"]
        max_shift_overlap = status["max_shift_overlap"]

        if not nominal_overlap and not max_shift_overlap:
            index += 1
            continue

        print(
            f"Detected overlap between circles at indices "
            f"{index} and {index + 1}"
        )

        # try replacing the two intermediate circles with one direct circle
        corridor1 = corridor_sequence[index]
        corridor3 = corridor_sequence[index + 2]

        replacement_choice = try_build_replacement_choice_between_corridors(
            corridor1,
            corridor3,
            index,
            vehicle,
        )

        if replacement_choice is not None:
            circle_choices_sequence.replace_two_with_one(
                index,
                replacement_choice,
            )

            reindex_intermediate_circle_choices_sequence(
                circle_choices_sequence
            )

            circle_choices_sequence = assign_preferred_candidates(
                circle_choices_sequence,
                start_pose,
                end_pose,
            )

            # Re-check from previous local neighborhood
            index = max(index - 1, 0)
            continue

        # If replacement is not possible, later try shifting logic here
        else:
            if circle1.turn_direction == circle2.turn_direction:
                R = vehicle.max_radius

                p_critical1 = Point(
                    circle1.center.x + R * cos(circle1.bisector_direction),
                    circle1.center.y + R * sin(circle1.bisector_direction),
                )

                p_critical2 = Point(
                    circle2.center.x + R * cos(circle2.bisector_direction),
                    circle2.center.y + R * sin(circle2.bisector_direction),
                )

                if compute_distance_two_points(p_critical1, circle2.center) < R:
                    merged_circle = compute_circle_through_two_points_with_radius(
                        p_critical1,
                        p_critical2,
                        R,
                        circle1.turn_direction,
                    )
            else: 
                pass
        index += 1

    return circle_choices_sequence


def reindex_intermediate_circle_choices_sequence(circle_choices_sequence):
    """
    Reassign consistent indices to choices and their candidate circles.
    """

    for i, choice in enumerate(circle_choices_sequence):
        choice.index = i

        for circle in choice:
            circle.index = i


def try_build_replacement_choice_between_corridors(
    corridor1,
    corridor3,
    index,
    vehicle,
):
    """
    Try to build one IntermediateCircleChoice directly between corridor1 and corridor3.

    Returns None if no valid direct transition exists.
    """

    try:
        tau = corridor1.compute_relative_turn_direction(corridor3)

        if tau != 0:
            candidates = [
                build_intermediate_circle_candidate(
                    corridor1,
                    corridor3,
                    tau,
                    index,
                    vehicle,
                )
            ]

        else:
            candidates = [
                build_intermediate_circle_candidate(
                    corridor1,
                    corridor3,
                    tau=1,
                    index=index,
                    vehicle=vehicle,
                ),
                build_intermediate_circle_candidate(
                    corridor1,
                    corridor3,
                    tau=-1,
                    index=index,
                    vehicle=vehicle,
                ),
            ]

        return IntermediateCircleChoice(
            candidates=candidates,
            index=index,
        )

    except ValueError:
        return None
    
    
def plot_line_from_w(ax, w, xlim, ylim, label=None, linestyle="-"):
    """
    Plot implicit line w[0]*x + w[1]*y + w[2] = 0.
    """
    wa, wb, wc = w

    if abs(wb) > 1e-9:
        xs = np.linspace(xlim[0], xlim[1], 200)
        ys = -(wa * xs + wc) / wb
        ax.plot(xs, ys, linestyle=linestyle, label=label)

    elif abs(wa) > 1e-9:
        x = -wc / wa
        ax.plot([x, x], ylim, linestyle=linestyle, label=label)


def plot_circle(ax, center, radius, label=None, linestyle="-"):
    theta = np.linspace(0, 2 * np.pi, 200)
    center = np.asarray(center, dtype=float)

    xs = center[0] + radius * np.cos(theta)
    ys = center[1] + radius * np.sin(theta)

    ax.plot(xs, ys, linestyle=linestyle, label=label)


def plot_shift_debug(
    corridor1,
    corridor2,
    corner_point,
    int_point,
    xc2,
    yc2,
    u,
    s_max_endpoint,
    s_max,
    rho,
    relevant_side_lines,
    plot_corridors,
):
    """
    Debug plot for the intermediate-circle shift limit.
    """

    figure = plot_corridors([corridor1, corridor2])
    ax = figure.gca()

    p0 = np.array([xc2, yc2], dtype=float)

    p_endpoint = p0 + s_max_endpoint * u
    p_final = p0 + s_max * u

    # Current limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Plot original geometric segment corner -> int_point
    corner_point = np.asarray(corner_point, dtype=float)
    int_point = np.asarray(int_point, dtype=float)

    ax.plot(
        [corner_point[0], int_point[0]],
        [corner_point[1], int_point[1]],
        "--",
        label="corner_point to int_point",
    )

    # Plot center path up to endpoint limit
    ax.plot(
        [p0[0], p_endpoint[0]],
        [p0[1], p_endpoint[1]],
        "-",
        linewidth=2,
        label="circle-center path to s_max_endpoint",
    )

    # Plot accepted center path up to final s_max
    ax.plot(
        [p0[0], p_final[0]],
        [p0[1], p_final[1]],
        "-",
        linewidth=4,
        label="accepted center path to final s_max",
    )

    # Key points
    ax.scatter(p0[0], p0[1], marker="o", label="p0 = initial circle center")
    ax.scatter(p_endpoint[0], p_endpoint[1], marker="x", label="endpoint limit center")
    ax.scatter(p_final[0], p_final[1], marker="*", label="final s_max center")
    ax.scatter(corner_point[0], corner_point[1], marker="s", label="corner_point")
    ax.scatter(int_point[0], int_point[1], marker="s", label="int_point")

    # Plot circle of radius rho at start, endpoint, and final selected point
    plot_circle(ax, p0, rho, label="rho-circle at s=0", linestyle=":")
    plot_circle(ax, p_endpoint, rho, label="rho-circle at s_max_endpoint", linestyle=":")
    plot_circle(ax, p_final, rho, label="rho-circle at final s_max", linestyle="--")

    # Plot relevant side lines and their rho-offset lines
    for i, side_line in enumerate(relevant_side_lines):
        w = np.asarray(side_line, dtype=float)
        wa, wb, wc = w
        normal_norm = np.hypot(wa, wb)

        plot_line_from_w(
            ax,
            w,
            xlim,
            ylim,
            label=f"checked side line {i}",
            linestyle="-",
        )

        # Offset lines at distance rho:
        # wa*x + wb*y + wc = +/- rho * ||normal||
        offset = rho * normal_norm

        w_plus = np.array([wa, wb, wc - offset])
        w_minus = np.array([wa, wb, wc + offset])

        plot_line_from_w(
            ax,
            w_plus,
            xlim,
            ylim,
            label=f"rho-offset + side {i}",
            linestyle=":",
        )

        plot_line_from_w(
            ax,
            w_minus,
            xlim,
            ylim,
            label=f"rho-offset - side {i}",
            linestyle=":",
        )

        safe, points_at_rho, min_dist = segment_line_rho_intersections(
            w,
            p0=p0,
            u=u,
            s_max=s_max_endpoint,
            rho=rho,
        )

        for s_hit, p_hit in points_at_rho:
            ax.scatter(
                p_hit[0],
                p_hit[1],
                marker="D",
                label=f"side {i}: distance rho at s={s_hit:.3f}",
            )

        print(
            f"side {i}: safe={safe}, min_dist={min_dist:.4f}, "
            f"points_at_rho={[float(s) for s, _ in points_at_rho]}"
        )

    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(True)

    return figure


def segment_line_rho_intersections(
    w,
    p0,
    u,
    s_max,
    rho,
    eps=1e-9,
):
    """
    Check whether the segment

        p(s) = p0 + s*u,    s in [0, s_max]

    stays at least distance rho away from the line

        w[0]*x + w[1]*y + w[2] = 0

    :param w:
        Line parameters [wa, wb, wc].
    :type w: array-like, shape (3,)

    :param p0:
        Initial point of the segment.
    :type p0: array-like, shape (2,)

    :param u:
        Segment direction vector.
        Preferably unit norm so that s corresponds to distance.
    :type u: array-like, shape (2,)

    :param s_max:
        Maximum parameter value along the segment.
    :type s_max: float

    :param rho:
        Safety distance.
    :type rho: float

    :param eps:
        Numerical tolerance.
    :type eps: float

    :return:
        Tuple containing:

        - safe:
            True if the whole segment stays at least rho away.

        - intersections:
            List containing tuples (s, point) where the segment
            reaches exactly distance rho from the line.

        - min_distance:
            Minimum distance between the segment and the line.
    :rtype:
        tuple(bool, list[tuple[float, numpy.ndarray]], float)
    """

    # Convert inputs to numpy arrays
    w = np.asarray(w, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    u = np.asarray(u, dtype=float)

    wa, wb, wc = w

    # Norm of the line normal vector
    normal_norm = np.hypot(wa, wb)

    if normal_norm < eps:
        raise ValueError(
            "Invalid line: normal vector has near-zero length."
        )

    # Signed line function along the segment:
    #
    # g(s) = wa*x(s) + wb*y(s) + wc
    #
    # with:
    #
    # p(s) = p0 + s*u
    #
    # Therefore:
    #
    # g(s) = g0 + s*g1
    #
    g0 = wa * p0[0] + wb * p0[1] + wc
    g1 = wa * u[0] + wb * u[1]

    # Distance threshold in implicit-line coordinates
    threshold = rho * normal_norm

    # Candidate points where minimum distance may occur
    candidates = [0.0, s_max]

    # If not parallel, check whether the segment crosses the line
    if abs(g1) > eps:
        s_cross = -g0 / g1

        if 0.0 <= s_cross <= s_max:
            candidates.append(s_cross)

    # Compute minimum distance
    min_abs_g = min(abs(g0 + s * g1) for s in candidates)
    min_distance = min_abs_g / normal_norm

    # Safety check
    safe = min_distance >= rho - eps

    intersections = []

    # Parallel case
    if abs(g1) < eps:

        # Entire segment lies exactly at distance rho
        if abs(abs(g0) - threshold) <= eps:

            intersections = [
                (0.0, p0.copy()),
                (s_max, p0 + s_max * u),
            ]

    # General case
    else:

        # Solve:
        #
        # g(s) = +threshold
        # g(s) = -threshold
        #
        for target in [threshold, -threshold]:

            s = (target - g0) / g1

            if -eps <= s <= s_max + eps:

                s_clamped = min(max(s, 0.0), s_max)

                point = p0 + s_clamped * u

                intersections.append((s_clamped, point))

    return safe, intersections, min_distance


def get_relevant_side_edges_side_side(corridor1, corridor2, tau):
    """
    If tau == 1, use right edges.
    Otherwise, use left edges.
    """
    if tau == 1:
        edge_idx = CorridorWorld.RGT
    else:
        edge_idx = CorridorWorld.LFT

    return corridor1.W[:, edge_idx], corridor2.W[:, edge_idx]


def get_relevant_side_edges_front_side(corridor1, corridor2, tau):
    """
    If tau == 1, use right edges.
    Otherwise, use left edges.
    """
    edge_idx1 = CorridorWorld.BCK
    if tau == 1:
        edge_idx2 = CorridorWorld.RGT
    else:
        edge_idx2 = CorridorWorld.LFT

    return corridor1.W[:, edge_idx1], corridor2.W[:, edge_idx2]


def get_relevant_side_edges_side_back(corridor1, corridor2, tau):
    """
    If tau == 1, use right edges.
    Otherwise, use left edges.
    """
    edge_idx2 = CorridorWorld.FWD
    if tau == 1:
        edge_idx1 = CorridorWorld.RGT
    else:
        edge_idx1 = CorridorWorld.LFT

    return corridor1.W[:, edge_idx1], corridor2.W[:, edge_idx2]


def not_ambiguous_circle_choices(intermediate_circles):
    """
    Convert an IntermediateCirclesSequence into an
    IntermediateCircleChoicesSequence with no ambiguity.

    Each circle is wrapped into a single-candidate choice.

    :param intermediate_circles: sequence of IntermediateCircle
    :type intermediate_circles: IntermediateCirclesSequence or iterable

    :return: sequence of choices (each with exactly one candidate)
    :rtype: IntermediateCircleChoicesSequence
    """
    choices = []

    for i, circle in enumerate(intermediate_circles):
        choice = IntermediateCircleChoice(
            candidates=[circle],
            index=i,
        )
        choices.append(choice)

    return IntermediateCircleChoicesSequence(choices)


def side_side_circle(
    corridor1,
    corridor2,
    tau,
    edge_pair,
    corner_point,
    vehicle,
    circle_index=None,
):
    """
    Compute the intermediate circle for a side-side corridor intersection.

    This function handles the case where two consecutive corridors intersect
    through their side edges. If the opposite corridor edges also intersect,
    the circle center is computed from the corner point and that intersection
    point. Otherwise, the circle center is computed using the edge-based rule.

    :param corridor1: first corridor
    :param corridor2: second corridor
    :param tau: turn direction (+1 for left, -1 for right)
    :param edge_pair: pair of intersecting edge indices
    :param corner_point: selected corridor corner point [x, y]
    :param vehicle: vehicle model
    :param circle_index: optional index of the intermediate circle

    :return: intermediate circle
    :rtype: IntermediateCircle
    """
    R = vehicle.max_radius
    vehicle_width = vehicle.width

    corners1 = corridor1.get_corners()
    corners2 = corridor2.get_corners()

    A1 = corners1[3]  # top left
    A2 = corners1[0]  # top right
    B1 = corners2[2]  # bottom left
    B2 = corners2[1]  # bottom right

    int_point, intersects = compute_intersection_two_segments(
        A1,
        A2,
        B1,
        B2,
        tol=1e-12,
    )

    if not intersects:
        xc2, yc2 = compute_center_coordinates_second_circle_according_to_edges(
            corner_point,
            tau,
            R,
            vehicle_width,
            0,
            edge_pair,
            corridor1,
            corridor2,
        )

        min_width, s_max = compute_min_width_s_max_corridor_pair(
            corridor1,
            corridor2,
            vehicle,
        )

        if corridor1.width < min_width or corridor2.width < min_width:
            raise ValueError(
                "Corridor widths are too small for the vehicle to make the turn. "
                f"Minimum required width: {min_width}"
            )

        return IntermediateCircle(
            Point(xc2, yc2),
            R,
            Point(corner_point[0], corner_point[1]),
            tau,
            index=circle_index,
            s_max=s_max,
        )

    xc2, yc2 = compute_center_coordinates_second_circle_given_two_points(
        corner_point,
        int_point,
        tau,
        R,
        vehicle_width,
    )

    length_section = compute_distance_two_points(corner_point, int_point)
    r = vehicle_width / 2
    rho = R + r

    if length_section < vehicle_width:
        raise ValueError(
            "The distance between the corner point and the intersection point "
            "is too small for the vehicle to make the turn. "
            f"Minimum required distance: {vehicle_width}"
        )

    s_max_endpoint = length_section - rho + (R-r)

    s_max_candidates = [s_max_endpoint]

    # Direction from corner_point to int_point
    u = np.array(int_point, dtype=float) - np.array(corner_point, dtype=float)
    u = u / np.linalg.norm(u)

    w1, w2 = get_relevant_side_edges_side_side(corridor1, corridor2, tau)
    relevant_side_lines = [w1, w2]
    p0 = np.array([xc2, yc2], dtype=float)

    for side_line in relevant_side_lines:
        safe, points_at_rho, min_dist = segment_line_rho_intersections(
            side_line,
            p0=p0,
            u=u,
            s_max=s_max_endpoint,
            rho=rho,
        )

        if not safe:
            valid_s = [s for s, p in points_at_rho if 0 <= s <= s_max_endpoint]

            if not valid_s:
                raise ValueError(
                    "The circle-center path violates a side constraint, "
                    "but no valid rho-boundary point was found."
                )

            s_limit = min(valid_s)
            s_max_candidates.append(s_limit)

    s_max = min(s_max_candidates)

    # fig = plot_shift_debug(
    #     corridor1=corridor1,
    #     corridor2=corridor2,
    #     corner_point=corner_point,
    #     int_point=int_point,
    #     xc2=xc2,
    #     yc2=yc2,
    #     u=u,
    #     s_max_endpoint=s_max_endpoint,
    #     s_max=s_max,
    #     rho=rho,
    #     relevant_side_lines=relevant_side_lines,
    #     plot_corridors=plot_corridors,
    # )

    # plt.show()

    return IntermediateCircle(
        Point(xc2, yc2),
        R,
        Point(corner_point[0], corner_point[1]),
        tau,
        index=circle_index,
        s_max=s_max,
    )


def front_side_circle(
    corridor1,
    corridor2,
    tau,
    edge_pair,
    corner_point,
    vehicle,
    circle_index=None,
):
    corners1 = corridor1.get_corners()
    corners2 = corridor2.get_corners()

    def build_circle_from_intersection(int_point, add_s_check = False):
        xc2, yc2 = compute_center_coordinates_second_circle_given_two_points(
            corner_point,
            int_point,
            tau,
            vehicle.max_radius,
            vehicle.width,
        )

        length_section = compute_distance_two_points(corner_point, int_point)

        r = vehicle.width / 2
        R = vehicle.max_radius
        rho = R + r

        if length_section < vehicle.width:
            raise ValueError(
                "The distance between the corner point and the intersection point "
                "is too small for the vehicle to make the turn. "
                f"Minimum required distance: {vehicle.width}"
            )

        s_max_endpoint = length_section - vehicle.width

        s_max_candidates = [s_max_endpoint]

        if add_s_check:
            # Direction from corner_point to int_point
            u = np.array(int_point, dtype=float) - np.array(corner_point, dtype=float)
            u = u / np.linalg.norm(u)

            w1, w2 = get_relevant_side_edges_front_side(corridor1, corridor2, tau)

            relevant_side_lines = [w1, w2]
            p0 = np.array([xc2, yc2], dtype=float)

            for side_line in relevant_side_lines:
                safe, points_at_rho, min_dist = segment_line_rho_intersections(
                    side_line,
                    p0=p0,
                    u=u,
                    s_max=s_max_endpoint,
                    rho=rho,
                )

                if not safe:
                    valid_s = [s for s, p in points_at_rho if 0 <= s <= s_max_endpoint]

                    if not valid_s:
                        raise ValueError(
                            "The circle-center path violates a side constraint, "
                            "but no valid rho-boundary point was found."
                        )

                    s_limit = min(valid_s)
                    s_max_candidates.append(s_limit)

        s_max = min(s_max_candidates)

        return IntermediateCircle(
            Point(xc2, yc2),
            vehicle.max_radius,
            Point(corner_point[0], corner_point[1]),
            tau,
            index=circle_index,
            s_max=s_max,
        )

    # First attempt: front edge of corridor1 with side edge of corridor2
    A1 = corners1[3]
    A2 = corners1[0]

    if edge_pair[1] == 1:
        B1 = corners2[2]
        B2 = corners2[3]
    else:
        B1 = corners2[1]
        B2 = corners2[0]

    int_point, intersects = compute_intersection_two_segments(
        A1,
        A2,
        B1,
        B2,
        tol=1e-12,
    )

    if intersects:
        return build_circle_from_intersection(int_point)

    # Second attempt: side edge of corridor1 with back edge of corridor2
    # Here we need to check for s_max
    if edge_pair[1] == 1:
        A1 = corners1[2]
        A2 = corners1[3]
    else:
        A1 = corners1[1]
        A2 = corners1[0]

    B1 = corners2[2]
    B2 = corners2[1]

    int_point, intersects = compute_intersection_two_segments(
        A1,
        A2,
        B1,
        B2,
        tol=1e-12,
    )

    if intersects:
        return build_circle_from_intersection(int_point, add_s_check=True)

    # Fallback: compute circle from edge-based rule
    xc2, yc2 = compute_center_coordinates_second_circle_according_to_edges(
        corner_point,
        tau,
        vehicle.max_radius,
        vehicle.width,
        0,
        edge_pair,
        corridor1,
        corridor2,
    )

    angle = atan2(
        yc2 - corner_point[1],
        xc2 - corner_point[0],
    )

    intersections = compute_line_corridor_intersections(
        corner_point,
        angle,
        corridor2,
        tol=1e-9,
    )

    if len(intersections) != 2:
        raise ValueError(
            "Unexpected number of intersections between the line from the "
            f"corner point to the circle center and the corridor: {len(intersections)}"
        )

    length = compute_distance_two_points(
        intersections[0],
        intersections[1],
    )

    if length < vehicle.width:
        raise ValueError(
            "The distance between the two intersections of the line from the "
            "corner point to the circle center with the corridor is too small "
            "for the vehicle to make the turn. "
            f"Minimum required distance: {vehicle.width}"
        )

    return IntermediateCircle(
        Point(xc2, yc2),
        vehicle.max_radius,
        Point(corner_point[0], corner_point[1]),
        tau,
        index=circle_index,
        s_max=length - 2 * vehicle.max_radius,
    )


def side_back_circle(
    corridor1,
    corridor2,
    tau,
    edge_pair,
    corner_point,
    vehicle,
    circle_index=None,
):
    corners1 = corridor1.get_corners()
    corners2 = corridor2.get_corners()

    def build_circle_from_intersection(int_point, add_s_check = False):
        xc2, yc2 = compute_center_coordinates_second_circle_given_two_points(
            corner_point,
            int_point,
            tau,
            vehicle.max_radius,
            vehicle.width,
        )

        length_section = compute_distance_two_points(corner_point, int_point)

        r = vehicle.width / 2
        R = vehicle.max_radius
        rho = R + r

        if length_section < vehicle.width:
            raise ValueError(
                "The distance between the corner point and the intersection point "
                "is too small for the vehicle to make the turn. "
                f"Minimum required distance: {vehicle.width}"
            )

        s_max_end_point = length_section - vehicle.width
        s_max_candidates = [s_max_end_point]

        if add_s_check:
            # Direction from corner_point to int_point
            u = np.array(int_point, dtype=float) - np.array(corner_point, dtype=float)
            u = u / np.linalg.norm(u)

            w1, w2 = get_relevant_side_edges_side_back(corridor1, corridor2, tau)

            relevant_side_lines = [w1, w2]
            p0 = np.array([xc2, yc2], dtype=float)

            for side_line in relevant_side_lines:
                safe, points_at_rho, min_dist = segment_line_rho_intersections(
                    side_line,
                    p0=p0,
                    u=u,
                    s_max=s_max_end_point,
                    rho=rho,
                )

                if not safe:
                    valid_s = [s for s, p in points_at_rho if 0 <= s <= s_max_end_point]

                    if not valid_s:
                        raise ValueError(
                            "The circle-center path violates a side constraint, "
                            "but no valid rho-boundary point was found."
                        )

                    s_limit = min(valid_s)
                    s_max_candidates.append(s_limit)

        s_max = min(s_max_candidates)
        
        return IntermediateCircle(
            Point(xc2, yc2),
            vehicle.max_radius,
            Point(corner_point[0], corner_point[1]),
            tau,
            index=circle_index,
            s_max=s_max,
        )

    # First attempt: side edge of corridor1 with back edge of corridor2
    if edge_pair[0] == 1:
        A1 = corners1[2]  # bottom left
        A2 = corners1[3]  # top left
    else:
        A1 = corners1[1]  # bottom right
        A2 = corners1[0]  # top right

    B1 = corners2[2]  # bottom left
    B2 = corners2[1]  # bottom right

    int_point, intersects = compute_intersection_two_segments(
        A1,
        A2,
        B1,
        B2,
        tol=1e-12,
    )

    if intersects:
        return build_circle_from_intersection(int_point)

    # Second attempt: front edge of corridor1 with side edge of corridor2
    A1 = corners1[3]  # top left
    A2 = corners1[0]  # top right

    if edge_pair[0] == 1:
        B1 = corners2[2]  # bottom left
        B2 = corners2[3]  # top left
    else:
        B1 = corners2[1]  # bottom right
        B2 = corners2[0]  # top right

    int_point, intersects = compute_intersection_two_segments(
        A1,
        A2,
        B1,
        B2,
        tol=1e-12,
    )

    if not intersects:
        raise ValueError(
            "Could not find a valid intersection point for the side-back case. "
            "This may indicate invalid corridor geometry or an unexpected edge pair."
        )

    return build_circle_from_intersection(int_point)
        

def assign_preferred_turn_directions(circle_choices_sequence):
    """
    Assign preferred turn directions to ambiguous circle choices.

    For each ambiguous block, look at the nearest nonzero turn direction
    before and after the block. If they are equal, use that direction as
    the preferred candidate direction for all choices in the block.
    Otherwise, leave the preference as None.
    """
    turn_directions = []

    for choice in circle_choices_sequence:
        if choice.is_ambiguous:
            turn_directions.append(0)
        else:
            turn_directions.append(choice.first.turn_direction)

    n = len(turn_directions)
    i = 0

    while i < n:
        if turn_directions[i] != 0:
            i += 1
            continue

        # Found start of ambiguous block
        start = i
        while i < n and turn_directions[i] == 0:
            i += 1
        end = i - 1

        previous_turn = turn_directions[start - 1] if start > 0 else None
        next_turn = turn_directions[i] if i < n else None

        if previous_turn is not None and previous_turn == next_turn:
            preferred_turn = previous_turn
        else:
            preferred_turn = None

        for j in range(start, end + 1):
            circle_choices_sequence[j].preferred_turn_direction = preferred_turn

    return circle_choices_sequence


# def create_intermediate_circle_choice_sequence(corridor_list, vehicle, start_pose, end_pose):
#     choices = []
#     left_turn = 1
#     right_turn = -1

#     def build_candidate(corridor1, corridor2, tau, index):
#         corner_point, edge_pair = get_corner_point_and_intersecting_edges(
#             corridor1,
#             corridor2,
#             tau,
#         )

#         if corner_point is None or edge_pair == []:
#             raise ValueError(
#                 f"No valid corner point found for turn direction {tau}."
#             )

#         if edge_pair in ((1, 1), (3, 3)):
#             return side_side_circle(
#                 corridor1,
#                 corridor2,
#                 tau,
#                 edge_pair,
#                 corner_point,
#                 vehicle,
#                 circle_index=index,
#             )

#         # front (0) side (2,3) case
#         if edge_pair in ((0, 1), (0, 3)):
#             return front_side_circle(
#                 corridor1,
#                 corridor2,
#                 tau,
#                 edge_pair,
#                 corner_point,
#                 vehicle,
#                 circle_index=index,
#             )

#         # side (1,3) back (2) case
#         if edge_pair in ((3, 2), (1, 2)):
#             return side_back_circle(
#                 corridor1,
#                 corridor2,
#                 tau,
#                 edge_pair,
#                 corner_point,
#                 vehicle,
#                 circle_index=index,
#             )
#         plot_corridors([corridor1, corridor2])
#         plt.title(f"Unexpected edge pair: {edge_pair}, tau={tau}")
#         plt.show()
#         raise ValueError(f"Invalid edge pair: {edge_pair}")

#     for i in range(len(corridor_list) - 1):
#         corridor1 = corridor_list[i]
#         corridor2 = corridor_list[i + 1]

#         tau = corridor1.compute_relative_turn_direction(corridor2)

#         if tau != 0:
#             candidates = [
#                 build_candidate(corridor1, corridor2, tau, i)
#             ]
#         else:
#             candidates = [
#                 build_candidate(corridor1, corridor2, left_turn, i),
#                 build_candidate(corridor1, corridor2, right_turn, i),
#             ]

#         choices.append(
#             IntermediateCircleChoice(
#                 candidates=candidates,
#                 index=i,
#             )
#         )

#     choices_sequence = IntermediateCircleChoicesSequence(choices)
#     choices_sequence = assign_preferred_candidates(choices_sequence, start_pose, end_pose)
#     choices_sequence = solve_circles_overlap(choices_sequence)
#     return choices_sequence

def build_intermediate_circle_candidate(
    corridor1,
    corridor2,
    tau,
    index,
    vehicle,
):
    corner_point, edge_pair = get_corner_point_and_intersecting_edges(
        corridor1,
        corridor2,
        tau,
    )

    if corner_point is None or edge_pair == []:
        raise ValueError(
            f"No valid corner point found for turn direction {tau}."
        )

    if edge_pair in ((1, 1), (3, 3)):
        return side_side_circle(
            corridor1,
            corridor2,
            tau,
            edge_pair,
            corner_point,
            vehicle,
            circle_index=index,
        )

    if edge_pair in ((0, 1), (0, 3)):
        return front_side_circle(
            corridor1,
            corridor2,
            tau,
            edge_pair,
            corner_point,
            vehicle,
            circle_index=index,
        )

    if edge_pair in ((3, 2), (1, 2)):
        return side_back_circle(
            corridor1,
            corridor2,
            tau,
            edge_pair,
            corner_point,
            vehicle,
            circle_index=index,
        )

    plot_corridors([corridor1, corridor2])
    plt.title(f"Unexpected edge pair: {edge_pair}, tau={tau}")
    plt.show()

    raise ValueError(f"Invalid edge pair: {edge_pair}")


def create_intermediate_circle_choice_sequence(
    corridor_list,
    vehicle,
    start_pose,
    end_pose,
):
    choices = []
    left_turn = 1
    right_turn = -1

    for i in range(len(corridor_list) - 1):
        corridor1 = corridor_list[i]
        corridor2 = corridor_list[i + 1]

        tau = corridor1.compute_relative_turn_direction(corridor2)

        if tau != 0:
            candidates = [
                build_intermediate_circle_candidate(
                    corridor1,
                    corridor2,
                    tau,
                    i,
                    vehicle,
                )
            ]
        else:
            candidates = [
                build_intermediate_circle_candidate(
                    corridor1,
                    corridor2,
                    left_turn,
                    i,
                    vehicle,
                ),
                build_intermediate_circle_candidate(
                    corridor1,
                    corridor2,
                    right_turn,
                    i,
                    vehicle,
                ),
            ]

        choices.append(
            IntermediateCircleChoice(
                candidates=candidates,
                index=i,
            )
        )

    choices_sequence = IntermediateCircleChoicesSequence(choices)

    choices_sequence = assign_preferred_candidates(
        choices_sequence,
        start_pose,
        end_pose,
    )

    choices_sequence = solve_circles_overlap(
        choices_sequence,
        corridor_list,
        vehicle,
        start_pose,
        end_pose,
        )

    return choices_sequence


def detect_ambiguous_blocks(circle_choices_sequence):
    ambiguous_blocks = []
    block = []

    for choice in circle_choices_sequence:
        if choice.is_ambiguous:
            block.append(choice.index)
        # Close current block when a non-ambiguous choice is found
        else:
            if block:
                ambiguous_blocks.append(block)
                block = []

    # Add final block if the sequence ends with ambiguous choices
    if block:
        ambiguous_blocks.append(block)

    return ambiguous_blocks


def assign_preferred_candidates(circle_choices_sequence, start_pose, end_pose):
    """
    Assign a preferred turn direction (±1) to all ambiguous circle choices.

    The decision is based on local geometry:
    previous point → door midpoint → next point.

    :param circle_choices_sequence: sequence of IntermediateCircleChoice
    :param start_pose: [x, y, theta]
    :param end_pose: [x, y, theta]
    :return: updated circle_choices_sequence
    """

    n_circles = len(circle_choices_sequence)
    ambiguous_blocks = detect_ambiguous_blocks(circle_choices_sequence)

    start_point = Point(start_pose[0], start_pose[1])
    end_point = Point(end_pose[0], end_pose[1])

    for block in ambiguous_blocks:

        # -----------------------------------------
        # Step 1 — compute mid-door points
        # -----------------------------------------
        mid_door_sequence = []

        for index in block:
            choice = circle_choices_sequence[index]

            door_A = choice[0].corner_point
            door_B = choice[1].corner_point

            mid_door = Point(
                (door_A.x + door_B.x) / 2,
                (door_A.y + door_B.y) / 2,
            )

            mid_door_sequence.append(mid_door)

        # -----------------------------------------
        # Step 2 — assign preferred turn
        # -----------------------------------------
        for i, choice_index in enumerate(block):

            p_curr = mid_door_sequence[i]

            # ---- previous reference ----
            if i > 0:
                p_prev = mid_door_sequence[i - 1]
            else:
                if choice_index > 0:
                    previous_circle = circle_choices_sequence[
                        choice_index - 1
                    ].first

                    p_prev = select_tangency_point_from_point_circle(
                        p_curr,
                        previous_circle,
                        turn_direction=-previous_circle.turn_direction,
                    )
                else:
                    p_prev = start_point

            # ---- next reference ----
            if i < len(block) - 1:
                p_next = mid_door_sequence[i + 1]
            else:
                if choice_index < n_circles - 1:
                    next_circle = circle_choices_sequence[choice_index + 1].first

                    p_next = select_tangency_point_from_point_circle(
                        p_curr,
                        next_circle,
                    )
                else:
                    p_next = end_point

            # ---- compute turn ----
            preferred_turn = compute_turn_direction_from_three_points(
                p_prev,
                p_curr,
                p_next,
            )

            choice = circle_choices_sequence[choice_index]

            # fallback if perfectly aligned
            if preferred_turn == 0:
                preferred_turn = choice[0].turn_direction

            choice.preferred_turn_direction = preferred_turn

        # -----------------------------------------
        # Step 3 — update circles (apply heuristic)
        # -----------------------------------------
        for choice_index in block:
            choice = circle_choices_sequence[choice_index]

            if choice.preferred_turn_direction is None:
                continue

            for circle in choice:
                if circle.turn_direction == choice.preferred_turn_direction:
                    s_new = circle.s_max / 2
                    circle.update_s(s=s_new)

    return circle_choices_sequence
 

def plot_intermediate_circle_choices(ax, circle_choices_sequence, r = 0):
    """
    Plot all intermediate circle candidates at their nominal position s=0.

    The red segment shows the maximum shift from s=0 to s=s_max.
    """

    for i, choice in enumerate(circle_choices_sequence):
        for j, circle in enumerate(choice):
            # Plot nominal circle center, not current shifted center
            if hasattr(circle, "canonical_center"):
                xc = circle.canonical_center.x
                yc = circle.canonical_center.y
            else:
                xc = circle.center.x
                yc = circle.center.y

            R = circle.radius
            rho = R + r

            is_preferred = (
                choice.is_ambiguous
                and choice.preferred_turn_direction is not None
                and circle.turn_direction == choice.preferred_turn_direction
            )

            if choice.is_ambiguous:
                if is_preferred:
                    color = "green"
                    linestyle = "-"
                    linewidth = 2.5
                    alpha = 1.0
                else:
                    color = "orange"
                    linestyle = "--"
                    linewidth = 1.0
                    alpha = 0.6
            else:
                color = "blue"
                linestyle = "-"
                linewidth = 1.5
                alpha = 0.8

            circ = plt.Circle(
                (xc, yc),
                rho,
                fill=False,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )

            circ_path = plt.Circle(
                (xc, yc),
                R,
                fill=False,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )

            ax.add_patch(circ)
            ax.add_patch(circ_path)

            ax.plot(xc, yc, "o", color=color, alpha=alpha)

            ax.text(
                xc,
                yc,
                f"{i}:{j}",
                fontsize=8,
                color=color,
            )

            if hasattr(circle, "s_max") and hasattr(circle, "bisector_direction"):
                x_shift_end = xc + circle.s_max * cos(circle.bisector_direction)
                y_shift_end = yc + circle.s_max * sin(circle.bisector_direction)

                ax.plot(
                    [xc, x_shift_end],
                    [yc, y_shift_end],
                    "r-",
                    linewidth=1.5,
                    alpha=0.8,
                )

                circ_extreme = plt.Circle(
                    (x_shift_end, y_shift_end),
                    rho,
                    fill=False,
                    color="red",
                    linestyle=":",
                    alpha=0.7,
                )

                circ_extreme_path = plt.Circle(
                    (x_shift_end, y_shift_end),
                    R,
                    fill=False,
                    color="red",
                    linestyle=":",
                    alpha=0.7,
                )

                ax.add_patch(circ_extreme)
                ax.add_patch(circ_extreme_path)

                ax.plot(
                    x_shift_end,
                    y_shift_end,
                    "x",
                    color="red",
                    alpha=0.7,
                )

        if choice.is_ambiguous and len(choice) == 2:
            c1 = choice[0]
            c2 = choice[1]

            c1_center = c1.canonical_center if hasattr(c1, "canonical_center") else c1.center
            c2_center = c2.canonical_center if hasattr(c2, "canonical_center") else c2.center

            ax.plot(
                [c1_center.x, c2_center.x],
                [c1_center.y, c2_center.y],
                "k--",
                linewidth=1,
                alpha=0.6,
            )


def selected_sequence_from_preferences(circle_choices_sequence):
    selected_circles = []

    for choice in circle_choices_sequence:
        if not choice.is_ambiguous:
            selected_circles.append(choice.first)
            continue

        selected_circle = None

        if choice.preferred_turn_direction is not None:
            for circle in choice:
                if circle.turn_direction == choice.preferred_turn_direction:
                    selected_circle = circle
                    break

        if selected_circle is None:
            selected_circle = choice.first

        selected_circles.append(selected_circle)

    return IntermediateCirclesSequence(selected_circles)


