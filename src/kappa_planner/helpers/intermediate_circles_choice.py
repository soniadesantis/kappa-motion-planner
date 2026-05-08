from ..geometry import Point, IntermediateCircle, IntermediateCirclesSequence, IntermediateCircleChoice, IntermediateCircleChoicesSequence
from ..corridor import CorridorWorld
from ..vehicle import Unicycle, Bicycle
from .intersections import compute_intersection_two_segments, compute_line_corridor_intersections
from .corridor_geometry import get_corner_point_and_intersecting_edges
from .intermediate_circles_geometry import compute_center_coordinates_second_circle_according_to_edges, compute_center_coordinates_second_circle_given_two_points
from .inputs_check import compute_min_width_s_max_corridor_pair
from .plot_helpers import plot_corridors
from .geometry_operations import (
    select_tangency_point_from_point_circle,
    compute_turn_direction,
    compute_turn_direction_from_three_points,
    efficient_sign,
    compute_distance_two_points
)
from math import sqrt, atan2, cos, sin, asin
from matplotlib import pyplot as plt


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

    if length_section < vehicle_width:
        raise ValueError(
            "The distance between the corner point and the intersection point "
            "is too small for the vehicle to make the turn. "
            f"Minimum required distance: {vehicle_width}"
        )

    return IntermediateCircle(
        Point(xc2, yc2),
        R,
        Point(corner_point[0], corner_point[1]),
        tau,
        index=circle_index,
        s_max=length_section - vehicle_width,
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

    def build_circle_from_intersection(int_point):
        xc2, yc2 = compute_center_coordinates_second_circle_given_two_points(
            corner_point,
            int_point,
            tau,
            vehicle.max_radius,
            vehicle.width,
        )

        length_section = compute_distance_two_points(corner_point, int_point)

        if length_section < vehicle.width:
            raise ValueError(
                "The distance between the corner point and the intersection point "
                "is too small for the vehicle to make the turn. "
                f"Minimum required distance: {vehicle.width}"
            )

        return IntermediateCircle(
            Point(xc2, yc2),
            vehicle.max_radius,
            Point(corner_point[0], corner_point[1]),
            tau,
            index=circle_index,
            s_max=length_section - vehicle.width,
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
        return build_circle_from_intersection(int_point)

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
        s_max=length - vehicle.width,
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

    def build_circle_from_intersection(int_point):
        xc2, yc2 = compute_center_coordinates_second_circle_given_two_points(
            corner_point,
            int_point,
            tau,
            vehicle.max_radius,
            vehicle.width,
        )

        length_section = compute_distance_two_points(corner_point, int_point)

        if length_section < vehicle.width:
            raise ValueError(
                "The distance between the corner point and the intersection point "
                "is too small for the vehicle to make the turn. "
                f"Minimum required distance: {vehicle.width}"
            )

        return IntermediateCircle(
            Point(xc2, yc2),
            vehicle.max_radius,
            Point(corner_point[0], corner_point[1]),
            tau,
            index=circle_index,
            s_max=length_section - vehicle.width,
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


def create_intermediate_circle_choice_sequence(corridor_list, vehicle, start_pose, end_pose):
    choices = []
    left_turn = 1
    right_turn = -1

    def build_candidate(corridor1, corridor2, tau, index):
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

        # front (0) side (2,3) case
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

        # side (1,3) back (2) case
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

    for i in range(len(corridor_list) - 1):
        corridor1 = corridor_list[i]
        corridor2 = corridor_list[i + 1]

        tau = corridor1.compute_relative_turn_direction(corridor2)

        if tau != 0:
            candidates = [
                build_candidate(corridor1, corridor2, tau, i)
            ]
        else:
            candidates = [
                build_candidate(corridor1, corridor2, left_turn, i),
                build_candidate(corridor1, corridor2, right_turn, i),
            ]

        choices.append(
            IntermediateCircleChoice(
                candidates=candidates,
                index=i,
            )
        )

    choices_sequence = IntermediateCircleChoicesSequence(choices)
    choices_sequence = assign_preferred_candidates(choices_sequence, start_pose, end_pose)
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
 

def plot_intermediate_circle_choices(ax, circle_choices_sequence):
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

            r = circle.radius

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
                r,
                fill=False,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )
            ax.add_patch(circ)

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
                    r,
                    fill=False,
                    color="red",
                    linestyle=":",
                    alpha=0.7,
                )
                ax.add_patch(circ_extreme)

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


