from .plot_helpers import plot_corridors, plot_analytical_trajectory
from .intermediate_circles_choice import selected_sequence_from_preferences
from .invert_inputs import invert_inputs_all, invert_inputs
from .helper_functions import (
    compute_center_coordinates_first_circle,

)
from .geometry_operations import (
    compute_angular_difference,
    wrapPositiveAngle,
    compute_angular_difference_with_turn_direction,
)
from .poses import (
    relative_to_absolute_pose,
    absolute_to_relative_pose,
)
from .intersections import (
    circle_intersection,
    check_intersection_case,
)
from .collision_avoidance import collision_avoidance_check_bicycle
from .primitives import (
    correct_angles,
    invert_maneuvers,
    compute_segment_between_two_circles_objects,
    compute_arc_from_two_tangents_objects,
)

from ..geometry import Point, Pose, Circle
from ..trajectory import BackwardArc, CurvilinearArcUnicycle, LinearSegmentUnicycle
from .pose_to_circle_unicycle import compute_three_maneuvers_compact

import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, pi, sqrt, atan2, asin


def compute_P_mid(intermediate_circles, bicycle):
    n_segments = len(intermediate_circles) - 1
    if n_segments <= 0:
        return []

    segments = [0] * n_segments

    for index in range(n_segments):
        segments[index] = compute_segment_between_two_circles_objects(
                intermediate_circles[index],
                intermediate_circles[index + 1],
                bicycle,
        )

    return segments


def compute_traj_to_circle_unicycle(corridor1, corridor2, start_pose, unicycle, circ1, t0 =0,tau0 = 0):
    """
    Build the initial part of the trajectory from the start pose to the first intermediate circle.
    
    :param corridor1: first corridor in the sequence
    :type corridor1: CorridorWorld
    :param corridor2: second corridor in the sequence
    :type corridor2: CorridorWorld
    :param start_pose: initial pose of the vehicle
    :type start_pose: list of floats
    :param unicycle: unicycle vehicle
    :type unicycle: Unicycle
    :param circ1: first intermediate circle
    :type circ1: IntermediateCircle object
    """
    T1, C1, S1 = compute_three_maneuvers_compact(corridor1,
                                                 corridor2,
                                                 start_pose,
                                                 unicycle,
                                                 circ1.xc,
                                                 circ1.yc,
                                                 circ1.turn_direction,
                                                 t0 = t0,
                                                 turn1 = tau0
                                                 )

    return [T1, C1, S1]


def shift_circles_unicycle(
    intermediate_circles,
    intermediate_circles_choices,
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
    initial_step = 0.1
    step = 0.05
    i = 0
    tried_other_side = [False] * len(intermediate_circles)

    figure = plot_corridors(corridor_list)
    plot_analytical_trajectory(segments, figure=figure)
    for circle in intermediate_circles:
        plt.plot(circle.center.x, circle.center.y, 'ro')    
        plt.plot(circle.xc + circle.radius * np.cos(np.linspace(0, 2*pi, 100)), circle.yc + circle.radius * np.sin(np.linspace(0, 2*pi, 100)), 'r--')
    plt.show(block = True)
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
                    s_new = initial_step * circle.s_max
                else:
                    s_new = min(circle.s + step * circle.s_max, circle.s_max)

                circle.update_s(s=s_new)
                first_int_circ = circle
                circ2 = intermediate_circles[1]

                start_maneuvers = compute_traj_to_circle_unicycle(
                    corridor_list[0],
                    corridor_list[1],
                    start_pose,
                    unicycle,
                    first_int_circ,
                )

                new_segment =  compute_segment_between_two_circles_objects(
                    first_int_circ,
                    intermediate_circles[1],
                    unicycle,
                )

                segments[0] = start_maneuvers[-1]
                segments[1] = new_segment

            # Interior intermediate circle
            elif 0 < i < len(intermediate_circles) - 1:
                circle = intermediate_circles[i]
                if circle.s >= circle.s_max and check_intersection_case(segments[i], segments[i+1]):
                    figure = plot_corridors(corridor_list)
                    plot_analytical_trajectory(segments, figure=figure)
                    plt.plot(circle.center.x, circle.center.y, 'ro')    
                    plt.plot(circle.xc + circle.radius * np.cos(np.linspace(0, 2*pi, 100)), circle.yc + circle.radius * np.sin(np.linspace(0, 2*pi, 100)), 'r--')
                    plt.show(block = True)

                    raise ValueError(
                        f"Intersection unresolved at circle {i} even at maximum shift."
                    )
                if circle.s == 0:
                    s_new = initial_step * circle.s_max
                else:
                    s_new = min(circle.s + step * circle.s_max, circle.s_max)

                circle.update_s(s=s_new)
                
                circ1 = intermediate_circles[i - 1]
                circ2 = intermediate_circles[i]
                circ3 = intermediate_circles[i + 1]

                new_segment1 =  compute_segment_between_two_circles_objects(
                    circ1,
                    circ2,
                    unicycle,
                )

                new_segment2 =  compute_segment_between_two_circles_objects(
                    circ2,
                    circ3,
                    unicycle,
                )

                segments[i] = new_segment1
                segments[i + 1] = new_segment2

                figure = plot_corridors(corridor_list)
                plot_analytical_trajectory(segments, figure=figure)
                plt.plot(circle.center.x, circle.center.y, 'ro')    
                plt.plot(circle.xc + circle.radius * np.cos(np.linspace(0, 2*pi, 100)), circle.yc + circle.radius * np.sin(np.linspace(0, 2*pi, 100)), 'r--')
                plt.plot(circ3.center.x, circ3.center.y, 'ro')
                plt.plot(circ3.xc + circ3.radius * np.cos(np.linspace(0, 2*pi, 100)), circ3.yc + circ3.radius * np.sin(np.linspace(0, 2*pi, 100)), 'r--')
                plt.show(block = True)

            # Last intermediate circle
            else:
                circle = intermediate_circles.last
                if circle.s >= circle.s_max and check_intersection_case(segments[i], segments[i+1]):
                    raise ValueError(
                        f"Intersection unresolved at circle {i} even at maximum shift."
                    )
                if circle.s == 0:
                    s_new = initial_step * circle.s_max
                else:
                    s_new = min(circle.s + step * circle.s_max, circle.s_max)

                circle.update_s(s=s_new)
                last_int_circ = circle
                circ2 = intermediate_circles[-2]

                last_corridor = corridor_list[-1]
                penultimate_corridor = corridor_list[-2]
                (
                    inv_last_corridor,
                    inv_penultimate_corridor,
                    inv_last_int_circ,
                    inv_end_pose) = invert_inputs_all(
                        last_corridor,
                        penultimate_corridor,
                        last_int_circ,
                        end_pose
                        )

                inv_end_maneuvers = compute_traj_to_circle_unicycle(
                    inv_last_corridor,
                    inv_penultimate_corridor,
                    inv_end_pose,
                    unicycle,
                    inv_last_int_circ,
                )
                
                end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)

                new_segment =  compute_segment_between_two_circles_objects(
                    circ2,
                    last_int_circ,
                    unicycle,
                )

                segments[-2] = new_segment
                segments[-1] = end_maneuvers[0]

            # Optional safety stop if no more shift is possible
            # and the intersection still persists
            if circle.s >= circle.s_max and check_intersection_case(segments[i], segments[i + 1]):
                break

        # After changing circle i, restart from the previous circle
        i = max(i - 1, 0)

    return intermediate_circles, segments, start_maneuvers, end_maneuvers


def compute_trajectory_unicycle_multiple_corridors_core(
    corridor_list,
    start_pose,
    end_pose,
    unicycle,
    intermediate_circles_choices,
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
    :param bicycle: unicycle vehicle
    :type Bicycle: Unicycle
    :param intermediate_circles_choices: list of choices for intermediate circles
    :type intermediate_circles_choices: list of IntermediateCircleChoice objects

    :return: sequence of primitives
    :rtype: list of primitives
    """

    # Extract the sequence of intermediate circles from the choices
    intermediate_circles = selected_sequence_from_preferences(
        intermediate_circles_choices
)
    # figure = plot_corridors(corridor_list)
    # plot_analytical_trajectory(segments, figure)
    # plt.show(block = True)
    # 1 — Compute P^init
    first_int_circ = intermediate_circles.first

    start_maneuvers = compute_traj_to_circle_unicycle(
        corridor_list[0],
        corridor_list[1],
        start_pose,
        unicycle,
        first_int_circ,
        t0=0,
        tau0=0)

    # 2 — Compute P^final
    last_int_circ = intermediate_circles.last
    penultimate_corridor = corridor_list[-2]
    last_corridor = corridor_list[-1]


    (inv_last_corridor,
     inv_penultimate_corridor,
     inv_last_int_circ,
     inv_end_pose) = invert_inputs_all(
            last_corridor,
            penultimate_corridor,
            last_int_circ,
            end_pose
            )

    inv_end_maneuvers = compute_traj_to_circle_unicycle(
        inv_last_corridor,
        inv_penultimate_corridor,
        inv_end_pose,
        unicycle,
        inv_last_int_circ,
        t0=0,
        tau0=0)
    

    end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)

    # Compute the segments and prune the intermediate circles if necessary
    segments = compute_P_mid(intermediate_circles, unicycle)
    segments.insert(0,start_maneuvers[-1])
    segments.append(end_maneuvers[0])

    # angle_array = np.linspace(0, 2*pi, 100)
    # figure = plot_corridors(corridor_list)
    # plot_analytical_trajectory(segments + start_maneuvers + end_maneuvers, figure)
    # # plt.plot(intermediate_circles[i].xc + intermediate_circles[i].radius * np.cos(angle_array), intermediate_circles[i].yc + intermediate_circles[i].radius * np.sin(angle_array), 'r--')
    # plt.show(block = True)

    ## Switch circles in case needed
    # (
    # intermediate_circles,
    # segments,
    # start_maneuvers,
    # end_maneuvers,
    # ) = adjust_trajectory_switch_circles(
    # intermediate_circles,
    # segments,
    # bicycle,
    # start_pose,
    # corridor_list,
    # end_pose,
    # start_maneuvers,
    # end_maneuvers,
    # )

    (
    intermediate_circles,
    segments,
    start_maneuvers,
    end_maneuvers,
    ) = shift_circles_unicycle(
    intermediate_circles,
    intermediate_circles_choices,
    segments,
    unicycle,
    start_pose,
    corridor_list,
    end_pose,
    start_maneuvers,
    end_maneuvers,
    )

    # figure = plot_corridors(corridor_list)
    # plot_analytical_trajectory([segments[0], segments[2]], figure)
    # for circle in intermediate_circles:
    #     plt.plot(circle.xc, circle.yc, 'ro')
    #     plt.plot(circle.xc + circle.radius * np.cos(angle_array), circle.yc + circle.radius * np.sin(angle_array), 'r--')
    # # plt.plot(intermediate_circles[i].xc + intermediate_circles[i].radius * np.cos(angle_array), intermediate_circles[i].yc + intermediate_circles[i].radius * np.sin(angle_array), 'r--')
    # plt.show(block = True)
    # Compute the arcs
    arcs = []
    angle_array = np.linspace(0, 2*pi, 100)
    # figure = plot_corridors(corridor_list)
    # plot_analytical_trajectory(segments, figure)
    # # plt.plot(circle.xc, circle.yc, 'ro')
    # # plt.plot(circle.xc + circle.radius * np.cos(angle_array), circle.yc + circle.radius * np.sin(angle_array), 'r--')
    # # plt.plot(intermediate_circles[i].xc + intermediate_circles[i].radius * np.cos(angle_array), intermediate_circles[i].yc + intermediate_circles[i].radius * np.sin(angle_array), 'r--')
    # # plt.plot(intermediate_circles[i+1].xc + intermediate_circles[i+1].radius * np.cos(angle_array), intermediate_circles[i+1].yc + intermediate_circles[i+1].radius * np.sin(angle_array), 'r--')

    # plt.show(block = True)
    for i in range(len(segments)-1): 
        print('Segment ', i)
        circle = intermediate_circles[i]
        angle_array = np.linspace(0, 2*pi, 100)
        # figure = plot_corridors(corridor_list)
        # plot_analytical_trajectory([segments[i], segments[i+1]], figure)
        # plt.plot(circle.xc, circle.yc, 'ro')
        # plt.plot(circle.xc + circle.radius * np.cos(angle_array), circle.yc + circle.radius * np.sin(angle_array), 'r--')
        # plt.plot(intermediate_circles[i].xc + intermediate_circles[i].radius * np.cos(angle_array), intermediate_circles[i].yc + intermediate_circles[i].radius * np.sin(angle_array), 'r--')
        # plt.plot(intermediate_circles[i+1].xc + intermediate_circles[i+1].radius * np.cos(angle_array), intermediate_circles[i+1].yc + intermediate_circles[i+1].radius * np.sin(angle_array), 'r--')
        
        # plt.show(block = True)
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
    
    return trajectory


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


def compute_trajectory_unicycle_two_corridors_core(
    corridor1,
    corridor2,
    start_pose,
    end_pose,
    unicycle,
    intermediate_circles_choices,
):
    # Extract the sequence of intermediate circles from the choices
    intermediate_circles = selected_sequence_from_preferences(
        intermediate_circles_choices
    )
    intermediate_circle = intermediate_circles.first
    

    start_maneuvers = compute_traj_to_circle_unicycle(
        corridor1,
        corridor2,
        start_pose,
        unicycle,
        intermediate_circle,
        t0=0,
        tau0=0)

    # 2 — Compute P^final
    (inv_corridor2,
     inv_corridor1,
     inv_intermediate_circle,
     inv_end_pose) = invert_inputs_all(
            corridor2,
            corridor1,
            intermediate_circle,
            end_pose
            )

    inv_end_maneuvers = compute_traj_to_circle_unicycle(
        inv_corridor2,
        inv_corridor1,
        inv_end_pose,
        unicycle,
        inv_intermediate_circle,
        t0=0,
        tau0=0)
    

    end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)

    intermediate_circle, start_maneuvers, end_maneuvers = shift_circles_two_corridors(
        intermediate_circle,
        start_maneuvers,
        end_maneuvers,
        start_pose,
        end_pose,
        unicycle,
        [corridor1, corridor2],
    )

    C4 = compute_arc_from_two_tangents_objects(
        start_maneuvers[-1],
        end_maneuvers[0],
        intermediate_circle,
        unicycle)


    trajectory = start_maneuvers + [C4] + end_maneuvers

    for i in range(len(trajectory) - 1):
        if trajectory[i + 1].time_grid[0] != trajectory[i].time_grid[-1]:
            trajectory[i + 1].add_time_offset(
                abs(trajectory[i + 1].time_grid[0] - trajectory[i].time_grid[-1])
            )
    correct_angles(trajectory)

    return trajectory