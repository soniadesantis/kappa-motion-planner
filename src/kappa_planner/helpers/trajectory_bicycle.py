from .plot_helpers import plot_corridors, plot_analytical_trajectory
from .intermediate_circles_choice import selected_sequence_from_preferences
from .invert_inputs import invert_inputs_all

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

from .primitives import (
    correct_angles,
    invert_maneuvers,
    compute_segment_between_two_circles_objects,
    compute_arc_from_two_tangents_objects,
)
from .axis_aligned_int_circle_sequence import (
    detect_tangent_intersections_blocks,
    update_intermediate_circle_centers_from_extended_sequence,
    solve_tangent_intersections_blocks_centers,
    print_failed_tangent_shift_summary,
    update_turn_direction_circles,
)

from .pose_to_circle_bicycle import compute_traj_to_circle_bicycle, compute_traj_to_circle_bicycle_with_fixed_forward_circle, compute_full_traj_bicycle_with_two_fixed_circles


from ..geometry import IntermediateCircle, Point, Pose, Circle
from ..trajectory import BackwardArc, CurvilinearArcUnicycle, LinearSegmentUnicycle
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, pi, sqrt, atan2, asin


def get_active_intermediate_circles(intermediate_circles):
    """
    Return intermediate circles that are active in final trajectory construction.

    :param intermediate_circles: Intermediate circles.
    :type intermediate_circles: IntermediateCirclesSequence | list

    :return: Active intermediate circles.
    :rtype: list
    """
    active_circles = []

    for circle in intermediate_circles:
        if getattr(circle, "skip", False):
            continue

        active_circles.append(circle)

    return active_circles


def compute_P_mid(intermediate_circles, bicycle, return_active_circles=False):
    """
    Compute tangent segments between consecutive active intermediate circles.
    """
    active_circles = get_active_intermediate_circles(intermediate_circles)

    n_segments = len(active_circles) - 1

    if n_segments <= 0:
        if return_active_circles:
            return [], active_circles
        return []

    segments = []

    for index in range(n_segments):
        segment = compute_segment_between_two_circles_objects(
            active_circles[index],
            active_circles[index + 1],
            bicycle,
        )

        segments.append(segment)

    if return_active_circles:
        return segments, active_circles

    return segments, active_circles



 


    
















def shift_circles_bicycle(
    intermediate_circles,
    # intermediate_circles_choices,
    segments,
    bicycle,
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

    # figure = plot_corridors(corridor_list)
    # plot_analytical_trajectory(segments, figure=figure)
    # for circle in intermediate_circles:
    #     plt.plot(circle.center.x, circle.center.y, 'ro')    
    #     plt.plot(circle.xc + circle.radius * np.cos(np.linspace(0, 2*pi, 100)), circle.yc + circle.radius * np.sin(np.linspace(0, 2*pi, 100)), 'r--')
    # plt.show(block = True)
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

                start_maneuvers = compute_traj_to_circle_bicycle(
                    corridor_list[0],
                    start_pose,
                    bicycle,
                    first_int_circ,
                )

                new_segment =  compute_segment_between_two_circles_objects(
                    first_int_circ,
                    intermediate_circles[1],
                    bicycle,
                )

                segments[0] = start_maneuvers[-1]
                segments[1] = new_segment

            # Interior intermediate circle
            elif 0 < i < len(intermediate_circles) - 1:
                circle = intermediate_circles[i]
                if circle.s >= circle.s_max and check_intersection_case(segments[i], segments[i+1]):
                    # figure = plot_corridors(corridor_list)
                    # plot_analytical_trajectory(segments, figure=figure)
                    # plt.plot(circle.center.x, circle.center.y, 'ro')    
                    # plt.plot(circle.xc + circle.radius * np.cos(np.linspace(0, 2*pi, 100)), circle.yc + circle.radius * np.sin(np.linspace(0, 2*pi, 100)), 'r--')
                    # plt.show(block = True)

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
                    bicycle,
                )

                new_segment2 =  compute_segment_between_two_circles_objects(
                    circ2,
                    circ3,
                    bicycle,
                )

                segments[i] = new_segment1
                segments[i + 1] = new_segment2

                # figure = plot_corridors(corridor_list)
                # plot_analytical_trajectory(segments, figure=figure)
                # plt.plot(circle.center.x, circle.center.y, 'ro')    
                # plt.plot(circle.xc + circle.radius * np.cos(np.linspace(0, 2*pi, 100)), circle.yc + circle.radius * np.sin(np.linspace(0, 2*pi, 100)), 'r--')
                # plt.plot(circ3.center.x, circ3.center.y, 'ro')
                # plt.plot(circ3.xc + circ3.radius * np.cos(np.linspace(0, 2*pi, 100)), circ3.yc + circ3.radius * np.sin(np.linspace(0, 2*pi, 100)), 'r--')
                # plt.show(block = True)

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
                inv_last_corridor, inv_last_int_circ, inv_end_pose = invert_inputs_all(last_corridor, last_int_circ, end_pose)

                inv_end_maneuvers = compute_traj_to_circle_bicycle(
                    inv_last_corridor,
                    inv_end_pose,
                    bicycle,
                    inv_last_int_circ,
                )
                
                end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)

                new_segment =  compute_segment_between_two_circles_objects(
                    circ2,
                    last_int_circ,
                    bicycle,
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


def extract_intermediate_circle_from_maneuver_list(maneuver_list):
    """
    Extract the intermediate circle from a list of maneuvers.

    :param maneuver_list: list of maneuvers
    :type maneuver_list: list of primitives

    :return: intermediate circle
    :rtype: IntermediateCircle object
    """
    for maneuver in maneuver_list:
        if isinstance(maneuver, CurvilinearArcUnicycle):
            intermediate_circle = IntermediateCircle(
                center=Point(x=maneuver.xc, y=maneuver.yc),
                radius=maneuver.radius,
                corner_point=Point(x=maneuver.xc, y=maneuver.yc),
                turn_direction=maneuver.turn_direction,
                index=None,
            )
            return intermediate_circle
    return None


def all_intermediate_circles_are_skipped(intermediate_circles):
    """
    Check whether all intermediate circles are skipped.

    :param intermediate_circles: Intermediate circles.
    :type intermediate_circles: IntermediateCirclesSequence | list

    :return: True if all circles are skipped.
    :rtype: bool
    """
    return all(
        getattr(circle, "skip", False)
        for circle in intermediate_circles
    )


def compute_trajectory_bicycle_multiple_corridors_optimal(
    corridor_list,
    start_pose,
    end_pose,
    bicycle,
    intermediate_circles,
):
    """
    Compute the sequence of primitives that build the time-optimal trajectory
    for a bicycle vehicle within multiple corridors.
    Backward maneuver both for collision avoidance and time-optimality.

    :param corridor_list: list of corridors
    :type corridor_list: list of CorridorWorld
    :param start_pose: initial pose within the first corridor
    :type start_pose: list of floats
    :param end_pose: final pose within the last corridor
    :type end_pose: list of floats
    :param bicycle: bicycle vehicle
    :type Bicycle: Bicycle
    :param intermediate_circles_choices: list of choices for intermediate circles
    :type intermediate_circles_choices: list of IntermediateCircleChoice objects

    :return: sequence of primitives
    :rtype: list of primitives
    :return: boolean indicating whether an intersection has been detected
    :rtype: Boolean
    """

    # Extract the sequence of intermediate circles from the choices
    # intermediate_circles = selected_sequence_from_preferences(
    #     intermediate_circles_choices
    # )
    # figure = plot_corridors(corridor_list)
    # plot_analytical_trajectory(segments, figure)
    # plt.show(block = True)
    # 1 — Compute P^init
    first_int_circ = intermediate_circles.first
    corridor1 = corridor_list[0]

    start_maneuvers = compute_traj_to_circle_bicycle(
        corridor1,
        start_pose,
        bicycle,
        first_int_circ,
    )

    # 2 — Compute P^final
    last_int_circ = intermediate_circles.last
    last_corridor = corridor_list[-1]
    inv_last_corridor, inv_last_int_circ, inv_end_pose = invert_inputs_all(last_corridor, last_int_circ, end_pose)

    inv_end_maneuvers = compute_traj_to_circle_bicycle(
        inv_last_corridor,
        inv_end_pose,
        bicycle,
        inv_last_int_circ,
    )

    end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)

    # Update Intermediate Circles depending on intersections
    intermediate_circle0 = extract_intermediate_circle_from_maneuver_list(start_maneuvers)
    intermediate_circleN = extract_intermediate_circle_from_maneuver_list(end_maneuvers)

    extended_circle_sequence = [intermediate_circle0] + list(intermediate_circles) + [intermediate_circleN]

    flags, blocks = detect_tangent_intersections_blocks(
        circle_sequence=extended_circle_sequence,
        bicycle=bicycle,
    )

    intermediate_circles_before_update = [circle.center for circle in intermediate_circles]



    max_iterations = 100
    iteration = 0
    all_skipped = False

    while any(flags) and iteration < max_iterations:
        figure = plot_corridors(corridor_list)
        angle_array = np.linspace(0, 2 * np.pi, 100)
        segments, _ = compute_P_mid(intermediate_circles, bicycle)
        radius = bicycle.max_radius
        for center in intermediate_circles_before_update:
            plt.plot(center.x, center.y, 'ro', markersize=5)
            plt.plot(center.x + radius * np.cos(angle_array), center.y + radius * np.sin(angle_array), 'g-', markersize=5)
        for circle in extended_circle_sequence:
            plt.plot(
                circle.xc + circle.radius * np.cos(angle_array),
                circle.yc + circle.radius * np.sin(angle_array),
                "r--",
            ) 
        plot_analytical_trajectory(start_maneuvers+segments+end_maneuvers, figure)
        plt.show(block = True)
        

        iteration += 1

        (
            new_extended_centers,
            corrected_centers,
            failed_shifts,
            boundary_updates,
        ) = solve_tangent_intersections_blocks_centers(
            blocks,
            extended_circle_sequence,
            start_pose=start_pose,
            end_pose=end_pose,
            corridor_list=corridor_list,
            bicycle=bicycle,
            tol=1e-9,
        )
        print(f"corrected_centers at iteration {iteration}: {corrected_centers}")
        # ------------------------------------------------------------
        # 1. Apply all successful center shifts, even if some other
        #    circles failed in other blocks.
        # ------------------------------------------------------------
        intermediate_circles_before_update = [circle.center for circle in intermediate_circles]
        if corrected_centers:
            update_intermediate_circle_centers_from_extended_sequence(
                intermediate_circles,
                new_extended_centers,
                corrected_centers=corrected_centers,
                mark_corrected_as_skip=True,
            )

        for circle in intermediate_circles:
            print(f"Center circle {circle.index}: {circle.center.x}, {circle.center.y}, s: {circle.s}")

        # ------------------------------------------------------------
        # 2. If there are failed shifts, structurally repair them.
        #    Important: this happens AFTER applying successful shifts.
        # ------------------------------------------------------------
        if failed_shifts:
            old_number_of_intermediate_circles = len(intermediate_circles)

            touches_start = (
                1 in failed_shifts
                or 1 in corrected_centers
            )

            touches_end = (
                old_number_of_intermediate_circles in failed_shifts
                or old_number_of_intermediate_circles in corrected_centers
            )

            updated_any = update_turn_direction_circles(
                failed_shifts=failed_shifts,
                intermediate_circles=intermediate_circles,
                corridor_list=corridor_list,
                vehicle=bicycle,
                tol=1e-9,
            )

            if not updated_any:
                print("No failed circle could be updated.")
                break

            # Recompute the start maneuver only if the first actual
            # intermediate circle was shifted or structurally repaired.
            if touches_start:
                skip = True
                ind = 0
                while skip:
                    if intermediate_circles[ind].skip == True: 
                        ind += 1
                    else:
                        first_int_circ = intermediate_circles[ind]
                        skip = False
                corridor1 = corridor_list[0]

                start_maneuvers = compute_traj_to_circle_bicycle_with_fixed_forward_circle(start_maneuvers, bicycle, first_int_circ)
                first_int_circ = intermediate_circles.first


                intermediate_circle0 = extract_intermediate_circle_from_maneuver_list(
                    start_maneuvers
                )

            # Recompute the end maneuver only if the last actual
            # intermediate circle was shifted or structurally repaired.
            if touches_end:
                skip = True
                ind = -1
                while skip:
                    if intermediate_circles[ind].skip == True: 
                        ind -= 1
                    else:
                        last_int_circ = intermediate_circles[ind]
                        skip = False
                
                

                inv_last_int_circ, inv_end_pose = invert_inputs_all(
                    last_int_circ,
                    end_pose,
                )
                inv_end_maneuvers = invert_maneuvers(end_maneuvers, t0=0)
                inv_end_maneuvers = compute_traj_to_circle_bicycle_with_fixed_forward_circle(inv_end_maneuvers, bicycle, inv_last_int_circ)
                end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0=0)

                intermediate_circleN = extract_intermediate_circle_from_maneuver_list(
                    end_maneuvers
                )

            # Always rebuild the extended sequence and rerun tangent detection
            # globally, because internal shifts/structural repairs can affect
            # neighboring tangent triples.
            extended_circle_sequence = (
                [intermediate_circle0]
                + list(intermediate_circles)
                + [intermediate_circleN]
            )

            flags, blocks = detect_tangent_intersections_blocks(
                circle_sequence=extended_circle_sequence,
                bicycle=bicycle,
            )

            continue

        # ------------------------------------------------------------
        # 3. If there are no failed shifts but also no successful shifts,
        #    then the solver did not make progress.
        # ------------------------------------------------------------
        if not corrected_centers:
            print("Tangent intersections remain, but no centers were corrected.")
            break

        # ------------------------------------------------------------
        # 4. If successful shifts touched the first or last actual
        #    intermediate circle, recompute boundary maneuvers.
        # ------------------------------------------------------------
        # if "start_maneuvers" in boundary_updates:
        #     start_maneuvers = boundary_updates["start_maneuvers"]
        #     intermediate_circle0 = extract_intermediate_circle_from_maneuver_list(
        #         start_maneuvers
        #     )
        #     intermediate_circle0 = extract_intermediate_circle_from_maneuver_list(
        #         start_maneuvers
        #     )

        # if "end_maneuvers" in boundary_updates:
        #     end_maneuvers = boundary_updates["end_maneuvers"]
        #     intermediate_circleN = extract_intermediate_circle_from_maneuver_list(
        #         end_maneuvers
        #     )

        #     intermediate_circleN = extract_intermediate_circle_from_maneuver_list(
        #         end_maneuvers
        #     )

        if all_intermediate_circles_are_skipped(intermediate_circles):
            start_maneuvers, end_maneuvers = compute_full_traj_bicycle_with_two_fixed_circles(start_maneuvers, end_maneuvers, bicycle)
            all_skipped = True
            
        else: 
            if 1 in corrected_centers:
                skip = True
                ind = 0
                while skip:
                    if intermediate_circles[ind].skip == True: 
                        ind += 1
                    else:
                        first_int_circ = intermediate_circles[ind]
                        skip = False
                # first_int_circ = intermediate_circles.first
                corridor1 = corridor_list[0]

                # start_maneuvers = compute_traj_to_circle_bicycle(
                #     corridor1,
                #     start_pose,
                #     bicycle,
                #     first_int_circ,
                # )

                start_maneuvers = compute_traj_to_circle_bicycle_with_fixed_forward_circle(start_maneuvers, bicycle, first_int_circ)

                intermediate_circle0 = extract_intermediate_circle_from_maneuver_list(
                    start_maneuvers
                )

            if len(intermediate_circles) in corrected_centers:
                skip = True
                ind = -1
                while skip:
                    if intermediate_circles[ind].skip == True: 
                        ind -= 1
                    else:
                        last_int_circ = intermediate_circles[ind]
                        skip = False
                # last_int_circ = intermediate_circles.last
                last_corridor = corridor_list[-1]

                inv_last_corridor, inv_last_int_circ, inv_end_pose = invert_inputs_all(
                    last_corridor,
                    last_int_circ,
                    end_pose,
                )

                # inv_end_maneuvers = compute_traj_to_circle_bicycle(
                #     inv_last_corridor,
                #     inv_end_pose,
                #     bicycle,
                #     inv_last_int_circ,
                # )

                inv_end_maneuvers = invert_maneuvers(end_maneuvers, t0=0)
                inv_end_maneuvers = compute_traj_to_circle_bicycle_with_fixed_forward_circle(inv_end_maneuvers, bicycle, inv_last_int_circ)

                end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0=0)

                intermediate_circleN = extract_intermediate_circle_from_maneuver_list(
                    end_maneuvers
                )

        extended_circle_sequence = (
            [intermediate_circle0]
            + list(intermediate_circles)
            + [intermediate_circleN]
        )

        flags, blocks = detect_tangent_intersections_blocks(
            circle_sequence=extended_circle_sequence,
            bicycle=bicycle,
        )
        print(f"Iteration {iteration}: flags = {flags}, blocks = {blocks}")

    if any(flags):
        print("Warning: tangent intersections remain after correction loop.")


    # Compute the segments and prune the intermediate circles if necessary
    if all_skipped:
        trajectory = start_maneuvers + end_maneuvers
            # Adjust the time grid and angles
        for i in range(len(trajectory)-1):
            if trajectory[i+1].time_grid[0] != trajectory[i].time_grid[-1]:
                trajectory[i+1].add_time_offset(abs(trajectory[i+1].time_grid[0] - trajectory[i].time_grid[-1]))
        correct_angles(trajectory)
        return trajectory

    segments, active_intermediate_circles = compute_P_mid(intermediate_circles, bicycle, return_active_circles=True)
    segments.insert(0,start_maneuvers[-1])
    segments.append(end_maneuvers[0])

    # Compute the arcs.
    arcs = []

    for i in range(len(segments) - 1):
        arcs.append(
            compute_arc_from_two_tangents_objects(
                segments[i],
                segments[i + 1],
                active_intermediate_circles[i],
                bicycle,
            )
        )

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


def compute_trajectory_bicycle_two_corridors_optimal(corridor1, corridor2, start_pose, end_pose, bicycle, intermediate_circles):
    '''
    Compute the sequence of primitives that build the time-optimal trajectory for a bicycle vehicle within two corridors.
    Backward maneuver both for collision avoidance and time-optimality.

    :param corridor1: first corridor
    :type corridor1: CorridorWorld
    :param corridor2: second corridor
    :type corridor2: CorridorWorld
    :param start_pose: initial pose within the first corridor
    :type start_pose: list of floats
    :param end_pose: final pose within the second corridor
    :type end_pose: list of floats
    :param bicycle: bicycle vehicle
    :type Bicycle: Bicycle

    :return: sequence of primitives
    :rtype: list of primitives
    :return: boolean indicating whether an intersection has been detected
    :rtype: Boolean
    '''
    # Extract the sequence of intermediate circles from the choices
    # intermediate_circles = selected_sequence_from_preferences(
    #     intermediate_circles_choices
    # )
    # 1 — Compute P^init
    intermediate_circle = intermediate_circles.first

    start_maneuvers = compute_traj_to_circle_bicycle(
        corridor1,
        start_pose,
        bicycle,
        intermediate_circle,
    )

    # 2 — Compute P^final
    inv_last_corridor, inv_last_int_circ, inv_end_pose = invert_inputs_all(corridor2, intermediate_circle, end_pose)

    inv_end_maneuvers = compute_traj_to_circle_bicycle(
        inv_last_corridor,
        inv_end_pose,
        bicycle,
        inv_last_int_circ,
    )

    end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)


    intermediate_circle, start_maneuvers, end_maneuvers = shift_circles_two_corridors_bicycle(
    intermediate_circle,
    start_maneuvers,
    end_maneuvers,
    start_pose,
    end_pose,
    bicycle,
    corridor1,
    corridor2,
    )
    
    intermediate_arc = compute_arc_from_two_tangents_objects(start_maneuvers[-1], end_maneuvers[0], intermediate_circle, bicycle)
    trajectory = start_maneuvers + [intermediate_arc] + end_maneuvers

    for i in range(len(trajectory)-1):
        if trajectory[i+1].time_grid[0] != trajectory[i].time_grid[-1]:
            trajectory[i+1].add_time_offset(abs(trajectory[i+1].time_grid[0] - trajectory[i].time_grid[-1]))
    correct_angles(trajectory)

    return trajectory


def shift_circles_two_corridors_bicycle(
    intermediate_circle,
    start_maneuvers,
    end_maneuvers,
    start_pose, 
    end_pose,
    bicycle,
    corridor1,
    corridor2
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
            start_maneuvers = compute_traj_to_circle_bicycle(
                corridor1,
                start_pose,
                bicycle,
                intermediate_circle,
            )

            # 2 — Compute P^final
            inv_last_corridor, inv_last_int_circ, inv_end_pose = invert_inputs_all(corridor2, intermediate_circle, end_pose)

            inv_end_maneuvers = compute_traj_to_circle_bicycle(
                inv_last_corridor,
                inv_end_pose,
                bicycle,
                inv_last_int_circ,
            )

            end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)
            changed = True

        processed_finished = not changed
            
    return intermediate_circle, start_maneuvers, end_maneuvers