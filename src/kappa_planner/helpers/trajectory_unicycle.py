"""
Analytical trajectory construction for corridor-based motion planning.

This module contains the core logic to build unicycle trajectories
across one or multiple corridors using motion primitives.

It includes:
- construction of intermediate segments between circles,
- generation of trajectories for two or multiple corridors,
- resolution of geometric inconsistencies (e.g., segment intersections)
  by shifting intermediate circles,
"""

from ..geometry import IntermediateCircle, Point

from .corridor_geometry import (
    compute_corner_point_vector,
    compute_turn_direction_vector,
    get_corner_point,
)
from .intermediate_circles_geometry import (
    create_intermediate_circles_sequence,
    second_circle,
    compute_center_coordinates_vector,
)
from .intersections import check_intersection_case
from .invert_inputs import invert_inputs
from .pose_to_circle_unicycle import compute_three_maneuvers_compact
from .primitives import (
    compute_arc_from_two_tangents,
    compute_arc_from_two_tangents_objects,
    compute_segment_between_two_circles,
    invert_maneuvers,
    correct_angles,
)


def compute_P_mid(intermediate_circles, unicycle):
    n_segments = len(intermediate_circles) - 1
    if n_segments <= 0:
        return []

    segments = [0] * n_segments

    for index in range(n_segments):
        segments[index] = compute_segment_between_two_circles(
            intermediate_circles[index].xc,
            intermediate_circles[index].yc,
            intermediate_circles[index + 1].xc,
            intermediate_circles[index + 1].yc,
            intermediate_circles[index].turn_direction,
            intermediate_circles[index + 1].turn_direction,
            unicycle,
        )
    return segments


def compute_P_mid_unicycle(
    center_circumference_vector,
    turn_direction_vector,
    unicycle,
):
    """
    Compute intermediate segments between consecutive circles.

    :param center_circumference_vector: list of circle centers [(x, y), ...]
    :param turn_direction_vector: list of turn directions
    :param unicycle: vehicle model
    :return: list of segments
    """
    n_segments = len(center_circumference_vector) - 1
    if n_segments <= 0:
        return []

    segments = [
        compute_segment_between_two_circles(
            center_circumference_vector[i][0],
            center_circumference_vector[i][1],
            center_circumference_vector[i + 1][0],
            center_circumference_vector[i + 1][1],
            turn_direction_vector[i],
            turn_direction_vector[i + 1],
            unicycle,
        )
        for i in range(n_segments)
    ]

    return segments


def shift_circles(
    intermediate_circles,
    segments,
    unicycle,
    start_pose,
    corridor_list,
    end_pose,
    start_maneuvers,
    end_maneuvers,
):
    """
    Adjust intermediate circle positions to resolve intersections
    between consecutive segments.

    For N corridors:
      - there are (N - 1) intermediate circles
      - there are N connecting segments

    If an intersection is resolved at circle i, the checking process
    restarts from circle max(i-1, 0), since the update may affect the
    neighboring intersection on the left.
    """
    step = 0.1
    i = 0

    while i < len(intermediate_circles):
        if not check_intersection_case(segments[i], segments[i + 1]):
            i += 1
            continue

        # Resolve the intersection at circle i completely
        while check_intersection_case(segments[i], segments[i + 1]):

            # First intermediate circle
            if i == 0:
                circle = intermediate_circles.first
                if circle.s >= circle.s_max and check_intersection_case(segments[i], segments[i+1]):
                    raise ValueError(
                        f"Intersection unresolved at circle {i} even at maximum shift."
                    )
                if circle.s == 0:
                    s_new = 0.5 * circle.s_max
                else:
                    s_new = min(circle.s + step * circle.s_max, circle.s_max)

                circle.update_s(s=s_new)
                first_int_circ = circle
                circ2 = intermediate_circles[1]

                T1, C1, S1 = compute_three_maneuvers_compact(
                    corridor_list[0],
                    corridor_list[1],
                    start_pose,
                    unicycle,
                    first_int_circ.xc,
                    first_int_circ.yc,
                    first_int_circ.turn_direction,
                    t0=0,
                    turn1=0,
                )
                start_maneuvers = [T1, C1, S1]

                new_segment = compute_segment_between_two_circles(
                    first_int_circ.xc,
                    first_int_circ.yc,
                    circ2.xc,
                    circ2.yc,
                    first_int_circ.turn_direction,
                    circ2.turn_direction,
                    unicycle,
                    overlap=False,
                )

                segments[0] = S1
                segments[1] = new_segment

            # Interior intermediate circle
            elif 0 < i < len(intermediate_circles) - 1:
                circle = intermediate_circles[i]
                if circle.s >= circle.s_max and check_intersection_case(segments[i], segments[i+1]):
                    raise ValueError(
                        f"Intersection unresolved at circle {i} even at maximum shift."
                    )
                if circle.s == 0:
                    s_new = 0.5 * circle.s_max
                else:
                    s_new = min(circle.s + step * circle.s_max, circle.s_max)

                circle.update_s(s=s_new)

                circ1 = intermediate_circles[i - 1]
                circ2 = intermediate_circles[i]
                circ3 = intermediate_circles[i + 1]

                new_segment1 = compute_segment_between_two_circles(
                    circ1.xc,
                    circ1.yc,
                    circ2.xc,
                    circ2.yc,
                    circ1.turn_direction,
                    circ2.turn_direction,
                    unicycle,
                    overlap=False,
                )

                new_segment2 = compute_segment_between_two_circles(
                    circ2.xc,
                    circ2.yc,
                    circ3.xc,
                    circ3.yc,
                    circ2.turn_direction,
                    circ3.turn_direction,
                    unicycle,
                    overlap=False,
                )

                segments[i] = new_segment1
                segments[i + 1] = new_segment2

            # Last intermediate circle
            else:
                circle = intermediate_circles.last
                if circle.s >= circle.s_max and check_intersection_case(segments[i], segments[i+1]):
                    raise ValueError(
                        f"Intersection unresolved at circle {i} even at maximum shift."
                    )
                if circle.s == 0:
                    s_new = 0.5 * circle.s_max
                else:
                    s_new = min(circle.s + step * circle.s_max, circle.s_max)

                circle.update_s(s=s_new)
                last_int_circ = circle
                circ2 = intermediate_circles[-2]

                T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(
                    *invert_inputs(
                        corridor_list[-1],
                        corridor_list[-2],
                        end_pose,
                        unicycle,
                        last_int_circ.xc,
                        last_int_circ.yc,
                        last_int_circ.turn_direction,
                        t0=0,
                        turn1=0,
                    )
                )
                Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0=0)
                end_maneuvers = [Sn, Cn_plus_1, T2]

                new_segment = compute_segment_between_two_circles(
                    circ2.xc,
                    circ2.yc,
                    last_int_circ.xc,
                    last_int_circ.yc,
                    circ2.turn_direction,
                    last_int_circ.turn_direction,
                    unicycle,
                    overlap=False,
                )

                segments[-2] = new_segment
                segments[-1] = Sn

            # Optional safety stop if no more shift is possible
            # and the intersection still persists
            if circle.s >= circle.s_max and check_intersection_case(segments[i], segments[i + 1]):
                break

        # After changing circle i, restart from the previous circle
        i = max(i - 1, 0)

    return intermediate_circles, segments, start_maneuvers, end_maneuvers


def shift_circles_two_corridors(
    intermediate_circle,
    start_maneuvers,
    end_maneuvers,
    start_pose, 
    end_pose,
    unicycle,
    corridor_list,
):
    """
    Shift intermediate circle to resolve possible intersection.
    S: segment
    C: arc
    T: turn on-the-spot
    """
    processed_finished = False
    step = 0.1 

    while not processed_finished:
        S3 = start_maneuvers[-1]
        S5 = end_maneuvers[0]
        changed = False

        if check_intersection_case(S3, S5): 
            if intermediate_circle.s == 0:
                s_new = 0.5 * intermediate_circle.s_max
            else:
                s_new = min(
                    intermediate_circle.s + step * intermediate_circle.s_max,
                    intermediate_circle.s_max
                    )

            intermediate_circle.update_s(s=s_new)
            # Compute first three maneuvers
            T1, C2, S3 = compute_three_maneuvers_compact(
                corridor_list[0],
                corridor_list[1],
                start_pose,
                unicycle,
                intermediate_circle.xc,
                intermediate_circle.yc,
                intermediate_circle.turn_direction,
                t0=0,
                turn1=0,
            )
            start_maneuvers = [T1, C2, S3]
        
            # Compute last three maneuvers 
            # Compute last three maneuvers
            T7_inv, C6_inv, S5_inv = compute_three_maneuvers_compact(
                *invert_inputs(
                    corridor_list[-1],
                    corridor_list[-2],
                    end_pose,
                    unicycle,
                    intermediate_circle.xc,
                    intermediate_circle.yc,
                    intermediate_circle.turn_direction,
                    t0=0,
                    turn1=0,
                )
            )
            S5, C6, T7 = invert_maneuvers([T7_inv, C6_inv, S5_inv], t0 = 0)
            end_maneuvers = [S5, C6, T7]
            changed = True

        processed_finished = not changed
            
    return intermediate_circle, start_maneuvers, end_maneuvers


def compute_trajectory_unicycle_two_corridors(
    corridor1,
    corridor2,
    start_pose,
    end_pose,
    unicycle,
    intermediate_circles,
):
    intermediate_circle = intermediate_circles.first
    
    xc2, yc2 = intermediate_circle.xc, intermediate_circle.yc
    turn_direction = intermediate_circle.turn_direction

    # intermediate_circle = IntermediateCircle(
    #     center=Point(xc2, yc2),
    #     radius=unicycle.max_radius,
    #     corner_point=Point(corner_point[0], corner_point[1]),
    #     turn_direction=turn_direction,
    #     index=0,
    #     s_max=s_max,
    # )

    T1, C2, S3 = compute_three_maneuvers_compact(
        corridor1,
        corridor2,
        start_pose,
        unicycle,
        xc2,
        yc2,
        turn_direction,
        t0=0,
        turn1=0,
    )

    raw_maneuvers = compute_three_maneuvers_compact(
        *invert_inputs(
            corridor2,
            corridor1,
            end_pose,
            unicycle,
            xc2,
            yc2,
            turn_direction,
            t0=0,
            turn1=0,
        )
    )

    S5, C6, T7 = invert_maneuvers(raw_maneuvers, t0=0)

    intermediate_circle, start_maneuvers, end_maneuvers = shift_circles_two_corridors(
        intermediate_circle,
        [T1, C2, S3],
        [S5, C6, T7],
        start_pose,
        end_pose,
        unicycle,
        [corridor1, corridor2],
    )

    T1, C2, S3 = start_maneuvers
    S5, C6, T7 = end_maneuvers

    C4 = compute_arc_from_two_tangents(
        S3,
        S5,
        turn_direction,
        intermediate_circle.xc,
        intermediate_circle.yc,
        unicycle,
    )

    trajectory = [T1, C2, S3, C4, S5, C6, T7]

    for i in range(len(trajectory) - 1):
        if trajectory[i + 1].time_grid[0] != trajectory[i].time_grid[-1]:
            trajectory[i + 1].add_time_offset(
                abs(trajectory[i + 1].time_grid[0] - trajectory[i].time_grid[-1])
            )
    correct_angles(trajectory)

    return trajectory, intermediate_circle


def compute_trajectory_unicycle_multiple_corridors_optimal(
    corridor_list,
    shrunken_corridor_list,
    start_pose,
    end_pose,
    unicycle,
    intermediate_circles,
):
    """
    Compute the sequence of primitives that build the time-optimal trajectory
    for a unicycle vehicle within multiple corridors.

    :param corridor_list: list of corridors
    :type corridor_list: list of CorridorWorld
    :param start_pose: initial pose within the first corridor
    :type start_pose: list of floats
    :param end_pose: final pose within the last corridor
    :type end_pose: list of floats
    :param unicycle: unicycle vehicle
    :type Unicycle: Unicycle
    :param intermediate_circles: list of circles to be reached at the intersection between two subsequent corridors
    :type intermediate_circles: IntermediateCircleSequence object

    :return: sequence of primitives
    :rtype: list of primitives
    :return: boolean indicating whether an intersection has been detected
    :rtype: Boolean
    """

    # # Compute the turn direction vector
    # turn_direction_vector = compute_turn_direction_vector(
    #     corridor_list
    #     )

    # # Compute the corner_point vector
    # corner_point_vector = compute_corner_point_vector(
    #     corridor_list,
    #     turn_direction_vector
    #     )
    
    # # Compute the centers of the intermediate circles 
    # center_circumference_vector = compute_center_coordinates_vector(
    #     corridor_list,
    #     turn_direction_vector,
    #     corner_point_vector,
    #     unicycle,
    #     margin = 0
    #     )

    # # Create intermediate circles sequence object
    # intermediate_circles = create_intermediate_circles_sequence(
    #     center_circumference_vector,
    #     turn_direction_vector,
    #     corner_point_vector,
    #     unicycle.max_radius,
    #     s_max_circles,
    # )

    # Compute P^init
    T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0],
                                                 corridor_list[1],
                                                 start_pose,
                                                 unicycle,
                                                 intermediate_circles.first.xc,
                                                 intermediate_circles.first.yc,
                                                 intermediate_circles.first.turn_direction,
                                                 t0 = 0,
                                                 turn1 = 0
                                                 )
    start_maneuvers = [T1, C1, S1]

    # Compute P^final
    T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1],
                                                                       corridor_list[-2],
                                                                       end_pose,
                                                                       unicycle,
                                                                       intermediate_circles.last.xc,
                                                                       intermediate_circles.last.yc,
                                                                       intermediate_circles.last.turn_direction,
                                                                       t0 = 0,
                                                                       turn1 = 0))
    Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
    end_maneuvers = [Sn, Cn_plus_1, T2]

    # Compute the segments and prune the intermediate circles if necessary
    segments = compute_P_mid(
        intermediate_circles,
        unicycle
        )
    segments.insert(0,start_maneuvers[-1])
    segments.append(end_maneuvers[0])

    # Prune segments and intermediate circles if necessary
    (
    intermediate_circles,
    segments,
    start_maneuvers,
    end_maneuvers,
    ) = shift_circles(
    intermediate_circles,
    segments,
    unicycle,
    start_pose,
    shrunken_corridor_list,
    end_pose,
    start_maneuvers,
    end_maneuvers,
    )

    if intermediate_circles is None:
        return "Invalid inputs"
    # figure = plot_corridors(corridor_list)
    # plot_analytical_trajectory(segments + start_maneuvers + end_maneuvers, figure)
    # # plt.plot(intermediate_circles[i].xc + intermediate_circles[i].radius * np.cos(angle_array), intermediate_circles[i].yc + intermediate_circles[i].radius * np.sin(angle_array), 'r--')
    # plt.show(block = True)
    # Compute the arcs
    arcs = []

    for i in range(len(segments)-1): 
        # print('Segment ', i)
        arcs.append(compute_arc_from_two_tangents_objects(segments[i], segments[i+1], intermediate_circles[i], unicycle))

    ## Attach everything together
    middle_sequence = [0] * (len(segments)-2 + len(arcs))
    middle_segments = segments[1:-1]
    ind_seg, ind_arc = 0, 0
    for index in range(0, len(middle_sequence), 2):
        middle_sequence[index] = arcs[ind_arc]
        ind_arc += 1
    
    for index in range(1, len(middle_sequence), 2):
        middle_sequence[index] = middle_segments[ind_seg]
        ind_seg += 1
    trajectory = start_maneuvers + middle_sequence + end_maneuvers

    # Adjust the time grid and angles
    for i in range(len(trajectory)-1):
        if trajectory[i+1].time_grid[0] != trajectory[i].time_grid[-1]:
            trajectory[i+1].add_time_offset(abs(trajectory[i+1].time_grid[0] - trajectory[i].time_grid[-1]))
    correct_angles(trajectory)
    
    return trajectory, intermediate_circles


