from pyexpat.errors import messages
import warnings
from .corridor_geometry import (
    check_point_inside_corridor,
    compute_corner_point_vector_and_intersecting_edges,
    compute_turn_direction_vector,
    remove_zeros_from_turn_direction_vector,
    compute_corner_point_vector,
)
from .intermediate_circle_solve_overlap import select_preferred_circle
from .intermediate_circles_geometry import compute_center_coordinates_vector, compute_center_coordinates_vector_according_to_edges, create_intermediate_circles_sequence
import matplotlib.pyplot as plt
from .plot_helpers import plot_corridors, plot_analytical_trajectory
from .geometry_operations import compute_distance_two_points, check_point_inside_segment, compute_angular_difference, wrapPositiveAngle
from math import sqrt, atan2, cos, sin, pi


# def check_inputs_analytical_planner_bicycle(planner):
#     '''
#     Check whether the inputs are valid for the analytical planner.
#     In case the inputs are not valid, write a warning message
        
#     :param planner: analytical planner
#     :type planner: MotionPlanner object
#     '''
#     check_passed = True
#     messages = []
#     center_circumference_vector = None

#     corridor_list = planner.corridor_list
#     shrunken_corridor_list = planner.shrunken_corridor_list
#     R = planner.vehicle.max_radius

#     # Start pose inside first shrunken corridor
#     if not check_point_inside_corridor(shrunken_corridor_list[0], planner.start_pose[:2]):
#         msg = "Start pose is not inside the first shrunken corridor."
#         messages.append(msg)
#         warnings.warn(msg, UserWarning)
#         check_passed = False

#     # End pose inside last shrunken corridor
#     if not check_point_inside_corridor(shrunken_corridor_list[-1], planner.end_pose[:2]):
#         msg = "End pose is not inside the last shrunken corridor."
#         messages.append(msg)
#         warnings.warn(msg, UserWarning)
#         check_passed = False

#     # The corridors properly intersect and the intermediate circles can be defined
#     # try: 
#     # Compute turn directions for each pair of adjacent corridors with different tilt
#     turn_direction_choices = compute_turn_direction_choices(
#         corridor_list
#     )

#     corner_point_choices, intersecting_edges_choices = (
#         compute_corner_point_choices_and_intersecting_edges(
#             corridor_list,
#             turn_direction_choices,
#         )
#     )

#     center_circumference_choices = compute_center_coordinate_choices_according_to_edges(
#         corridor_list,
#         turn_direction_choices,
#         corner_point_choices,
#         intersecting_edges_choices,
#         planner.vehicle,
#         margin=0,
#     )
    
#     s_max_circles = [None] * len(center_circumference_choices)

#     intermediate_circle_choices_sequence = create_intermediate_circle_choices_sequence(
#     center_circumference_choices,
#     turn_direction_choices,
#     corner_point_choices,
#     planner.vehicle.max_radius,
#     s_max_circles,
# )

#     fig = plot_corridors(corridor_list)
#     ax = plt.gca()

#     plot_intermediate_circle_choices(ax, intermediate_circle_choices_sequence)
#     plt.axis("equal")
#     plt.show()

    
#     # The start pose is outside the first intermediate circle
#     # dist_start = compute_distance_two_points(
#     #     planner.start_pose[:2],
#     #     center_circumference_vector[0])
#     # if dist_start < planner.vehicle.max_radius - 1e-3:
#     #     msg = 'The start pose is inside first intermediate circle.'
#     #     messages.append(msg)
#     #     warnings.warn(msg, UserWarning)
#     #     check_passed = False
    
#     # # The end pose is outside the last intermediate circle
#     # dist_end = compute_distance_two_points(
#     #     planner.end_pose[:2],
#     #     center_circumference_vector[-1])
#     # if dist_end < planner.vehicle.max_radius - 1e-3:
#     #     msg = 'The end pose is inside last intermediate circle.'
#     #     messages.append(msg)
#     #     warnings.warn(msg, UserWarning)
#     #     check_passed = False
    
#     # # The intermediate circles don't overlap 
#     # for i in range(len(center_circumference_vector)-1):
#     #     dist_centers = compute_distance_two_points(
#     #         center_circumference_vector[i],
#     #         center_circumference_vector[i+1])
#     #     if dist_centers - 2 * R < 1e-3: # and turn_direction_vector[i] != turn_direction_vector[i+1]: # if the distance between the centers is less than 2R and the turn directions are not the same, then the circles overlap
#     #         msg = f'Intermediate circles {i} and {i+1} overlap and they have different turn directions.'
#     #         messages.append(msg)
#     #         warnings.warn(msg, UserWarning)
#     #         check_passed = False

#     # except Exception as e:
#     #     warnings.warn(f'Error computing corner point vector: {e}', UserWarning)
#     #     check_passed = False
    
#     # Minimum corridors widths
#     # min_corridor_widths = planner.min_corridor_widths
#     # corridor_widths = [corridor.width for corridor in corridor_list]

#     # if not all(x >= y for x, y in zip(corridor_widths, min_corridor_widths)):
#     #     msg = "At least one corridor is not wide enough to guarantee collision-free maneuvers."
#     #     messages.append(msg)
#     #     warnings.warn(msg, UserWarning)
#     #     check_passed = False

#     return [], [10]*len(center_circumference_vector), check_passed, messages, center_circumference_vector, turn_direction_vector, corner_point_vector


# def compute_minimum_widths_bicycle(planner, corner_point_vector, turn_direction_vector, center_circumference_vector):
#     min_corridor_widths, s_max_circles = compute_minimum_widths_bicycle(planner) 
#     corridor_list = planner.corridor_list 

#     for i in range(len(corridor_list)-1): 
#         corner_point = corner_point_vector[i]
#         corners1 = corridor_list[i].get_corners()
#         corners2 = corridor_list[i+1].get_corners()

#         if turn_direction_vector[i] == 1: 
#             corridor1_side = check_point_inside_segment(corners1[0], corners1[1], corner_point)
#             corridor1_front = check_point_inside_segment(corners1[3], corners1[0], corner_point)
#             corridor2_side = check_point_inside_segment(corners2[0], corners2[1], corner_point)
#             corridor2_back = check_point_inside_segment(corners2[1], corners2[2], corner_point)
#             if corridor1_side and corridor2_side: 
#                 continue 
#             elif corridor1_front and corridor2_side: 
#                 min_corridor_widhts[i] = update
#             elif corridor1_side and corridor2_back:
#                 min_corridor_widths[i+1] = update
#             elif corridor1_front and corridor2_back:
#                 min_corridor_widths[i] = update
#                 min_corridor_widths[i+1] = update


#     return min_corridor_widths, s_max_circles

def check_start_and_end_poses(shrunken_corridor_list, start_pose, end_pose, messages): 
    """
    Check the core validity assumptions for the start and end poses.

    The function verifies that:
    - the start pose lies inside the first shrunken corridor,
    - the start pose does not already lie inside the second shrunken corridor,
    - the end pose lies inside the last shrunken corridor,
    - the end pose does not already lie inside the penultimate shrunken corridor.

    :param shrunken_corridor_list: sequence of shrunken corridors
    :type shrunken_corridor_list: list of CorridorWorld
    :param start_pose: initial pose [x, y, theta]
    :type start_pose: list or numpy.ndarray
    :param end_pose: final pose [x, y, theta]
    :type end_pose: list or numpy.ndarray
    :param messages: list used to collect warning/error messages
    :type messages: list of str

    :return: updated messages and boolean indicating whether all checks passed
    :rtype: tuple[list[str], bool]
    """
    check_passed = True
    # Start pose inside first shrunken corridor
    if not check_point_inside_corridor(shrunken_corridor_list[0], start_pose[:2]):
        msg = "Start pose is not inside the first shrunken corridor."
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check_passed = False

    if check_point_inside_corridor(shrunken_corridor_list[1], start_pose[:2]):
        msg = "Start pose is inside the second shrunken corridor."
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check_passed = False

    # End pose inside last shrunken corridor
    if not check_point_inside_corridor(shrunken_corridor_list[-1], end_pose[:2]):
        msg = "End pose is not inside the last shrunken corridor."
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check_passed = False

    if check_point_inside_corridor(shrunken_corridor_list[-2], end_pose[:2]):
        msg = "End pose is inside the penultimate shrunken corridor."
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check_passed = False
        
    return messages, check_passed


def check_corridor_widths_feasibility(shrunken_corridor_list, messages):
    """
    Check that all shrunken corridors have a positive width.

    A non-positive width indicates that the original corridor is too narrow
    for the vehicle (i.e., the vehicle does not fit within the corridor).

    :param shrunken_corridor_list: list of shrunken corridors
    :type shrunken_corridor_list: list of CorridorWorld
    :param messages: list used to collect warning/error messages
    :type messages: list of str

    :return: updated messages and boolean indicating feasibility
    :rtype: tuple[list[str], bool]
    """
    check_passed = True

    for i, corridor in enumerate(shrunken_corridor_list):
        if corridor.width <= 0:
            msg = (
                f"Corridor {i} has non-positive width after shrinking "
                "(vehicle does not fit in the corridor)."
            )
            messages.append(msg)
            warnings.warn(msg, UserWarning)
            check_passed = False

    return messages, check_passed


def check_core_assumptions(planner):
    """
    Check the core validity assumptions for the analytical planner inputs.

    This function currently checks the validity of the start and end poses with respect to the shrunken corridors. 
    Additional checks can be added in the future as needed.

    :param planner: analytical planner
    :type planner: MotionPlanner object

    :return: boolean indicating whether all checks passed and list of warning/error messages
    :rtype: tuple[bool, list[str]]
    """
    messages = []
    check_passed = True
    check1 = True

    # Check robot footprint 
    if planner.vehicle.max_radius < planner.vehicle.width * 0.5:
        msg = "The robot footprint is not valid: the maximum radius must be at least half of the width."
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check1 = False

    # Check start and end poses
    messages, check2 = check_start_and_end_poses(
        planner.shrunken_corridor_list,
        planner.start_pose,
        planner.end_pose,
        messages
    )

    # Check shrunken corridor widths
    messages, check3 = check_corridor_widths_feasibility(
        planner.shrunken_corridor_list,
        messages
    ) 

    check_passed = check1 and check2 and check3
    return check_passed, messages


def check_standing_assumptions(planner):
    """
    Check the standing assumptions for the analytical planner inputs.

    This function checks the validity of the corridor intersections and the intermediate circles. 
    Additional checks can be added in the future as needed.

    :param planner: analytical planner
    :type planner: MotionPlanner object

    :return: boolean indicating whether all checks passed, list of warning/error messages, and intermediate circles if the checks passed
    :rtype: tuple[bool, list[str], IntermediateCirclesSequence or None]
    """
    messages = []
    check_passed = True
    intermediate_circles = None
    corridor_list = planner.corridor_list

    # Consecutive corridors have different tilt
    turn_direction_vector = compute_turn_direction_vector(
    corridor_list
    )

    check1 = True
    if 0 in turn_direction_vector:
        msg = "At least one pair of adjacent corridors has the same tilt."
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check1 = False

    # Consecutive corridors properly intersect
    (
        corner_point_vector,
        intersecting_edges
    ) = compute_corner_point_vector_and_intersecting_edges(
        corridor_list,
        turn_direction_vector
    )

    check2 = True
    if any(point is None for point in corner_point_vector):
        msg = "At least one pair of adjacent corridors does not properly intersect."
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check2 = False

    check3 = True
    # Consecutive corridors intersect on the side edges
    for i, edge_pair in enumerate(intersecting_edges):
        if edge_pair is None or edge_pair == []:
            continue
        edge1, edge2 = edge_pair
        if edge1 not in (1, 3) or edge2 not in (1, 3):
            msg = (
                f"Corridor pair {i}-{i + 1} does not intersect on side edges. "
                f"Intersecting edges are ({edge1}, {edge2})."
            )
            messages.append(msg)
            warnings.warn(msg, UserWarning)
            check3 = False
    
    # Intermediate circles are 2R apart
    center_circumference_vector = compute_center_coordinates_vector_according_to_edges(
            corridor_list,
            turn_direction_vector,
            corner_point_vector,
            intersecting_edges,
            planner.vehicle,
            margin = 0)  
    
    check4 = True
    for i in range(len(center_circumference_vector)-1):
        dist_centers = compute_distance_two_points(
            center_circumference_vector[i],
            center_circumference_vector[i+1])
        if dist_centers - 2 * planner.vehicle.max_radius < 1e-3: # and turn_direction_vector[i] != turn_direction_vector[i+1]: # if the distance between the centers is less than 2R and the turn directions are not the same, then the circles overlap
            msg = f'Intermediate circles {i} and {i+1} overlap.'
            messages.append(msg)
            warnings.warn(msg, UserWarning)
            check4 = False

    # The start pose is outside the first intermediate circle
    dist_start = compute_distance_two_points(
        planner.start_pose[:2],
        center_circumference_vector[0])
    check5 = True
    if dist_start < planner.vehicle.max_radius - 1e-3:
        msg = 'The start pose is inside first intermediate circle.'
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check5 = False
    
    # The end pose is outside the last intermediate circle
    dist_end = compute_distance_two_points(
        planner.end_pose[:2],
        center_circumference_vector[-1])
    check6 = True
    if dist_end < planner.vehicle.max_radius - 1e-3:
        msg = 'The end pose is inside last intermediate circle.'
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check6 = False

    # Minimum corridors widths
    min_corridor_widths = planner.min_corridor_widths
    corridor_widths = [corridor.width for corridor in corridor_list]

    check7 = True
    if not all(x >= y for x, y in zip(corridor_widths, min_corridor_widths)):
        msg = "At least one corridor is not wide enough to guarantee collision-free maneuvers."
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check7 = False

    check_passed = check1 and check2 and check3 and check4 and check5 and check6 and check7

    if check_passed:
        intermediate_circles = create_intermediate_circles_sequence(
                center_circumference_vector,
                turn_direction_vector,
                corner_point_vector,
                planner.vehicle.max_radius,
                planner.s_max_circles,
            )

    return check_passed, messages, intermediate_circles



def check_position_out_of_circles_assumption(planner):
    check_passed = True
    first_circle = select_preferred_circle(
        planner.intermediate_circles_choice_sequence.first
    )

    last_circle = select_preferred_circle(
        planner.intermediate_circles_choice_sequence.last
    )

    messages = []
    inside_first_circle = False
    inside_last_circle = False
    
    if compute_distance_two_points(planner.start_pose[:2], (first_circle.center.x, first_circle.center.y)) < planner.vehicle.max_radius - 1e-3:
        msg = 'The start pose is inside first intermediate circle.'
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check_passed = False
        inside_first_circle = True
    elif compute_distance_two_points(planner.end_pose[:2], (last_circle.center.x, last_circle.center.y)) < planner.vehicle.max_radius - 1e-3:
        msg = 'The end pose is inside last intermediate circle.'
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check_passed = False
        inside_last_circle = True

    return check_passed, messages, inside_first_circle, inside_last_circle
    

def check_inputs_analytical_planner(planner):
    '''
    Check whether the inputs are valid for the analytical planner.
    In case the inputs are not valid, write a warning message
        
    :param planner: analytical planner
    :type planner: MotionPlanner object
    '''
    check_passed = True
    messages = []
    center_circumference_vector = None

    corridor_list = planner.corridor_list
    shrunken_corridor_list = planner.shrunken_corridor_list
    R = planner.vehicle.max_radius

    # Start pose inside first shrunken corridor
    if not check_point_inside_corridor(shrunken_corridor_list[0], planner.start_pose[:2]):
        msg = "Start pose is not inside the first shrunken corridor."
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check_passed = False

    # End pose inside last shrunken corridor
    if not check_point_inside_corridor(shrunken_corridor_list[-1], planner.end_pose[:2]):
        msg = "End pose is not inside the last shrunken corridor."
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check_passed = False

    # The corridors properly intersect and the intermediate circles can be defined
    try: 
        turn_direction_vector = compute_turn_direction_vector(
            corridor_list
        )
        if 0 in turn_direction_vector:
            turn_direction_vector = remove_zeros_from_turn_direction_vector(corridor_list, planner.end_pose, turn_direction_vector)

        corner_point_vector = compute_corner_point_vector(
            corridor_list,
            turn_direction_vector
        )

        for i, corner_point in enumerate(corner_point_vector):
            corners1 = corridor_list[i].get_corners()
            corners2 = corridor_list[i+1].get_corners()

            if turn_direction_vector[i] == -1:
                # figure = plot_corridors([corridor_list[i], corridor_list[i+1]])
                # plt.plot(corner_point[0], corner_point[1], 'ro', label = f'Corner point {i}')
                # plt.plot([corners1[0][0], corners1[1][0]], [corners1[0][1], corners1[1][1]], 'g-', label = f'Corridor {i} segment')
                # plt.plot([corners2[0][0], corners2[1][0]], [corners2[0][1], corners2[1][1]], 'b-', label = f'Corridor {i+1} segment')
                # plt.legend()
                # plt.show(block=True)
                if not check_point_inside_segment(corners1[0], corners1[1], corner_point) or not check_point_inside_segment(corners2[0], corners2[1], corner_point):
                    msg = f'Corner point {i} is not inside the segment defined by the two corners of the corridors.'
                    messages.append(msg)
                    warnings.warn(msg, UserWarning)
                    check_passed = False
            elif turn_direction_vector[i] == 1:
                # figure = plot_corridors([corridor_list[i], corridor_list[i+1]])
                # plt.plot(corner_point[0], corner_point[1], 'ro', label = f'Corner point {i}')
                # plt.plot([corners1[2][0], corners1[3][0]], [corners1[2][1], corners1[3][1]], 'g-', label = f'Corridor {i} segment')
                # plt.plot([corners2[2][0], corners2[3][0]], [corners2[2][1], corners2[3][1]], 'b-', label = f'Corridor {i+1} segment')
                # plt.legend()
                # plt.show(block=True)
                if not check_point_inside_segment(corners1[2], corners1[3], corner_point) or not check_point_inside_segment(corners2[2], corners2[3], corner_point):
                    msg = f'Corner point {i} is not inside the segment defined by the two corners of the corridors.'
                    messages.append(msg)
                    warnings.warn(msg, UserWarning)
                    check_passed = False
   
        center_circumference_vector = compute_center_coordinates_vector(
            corridor_list,
            turn_direction_vector,
            corner_point_vector, planner.vehicle, margin = 0
        )
        
        # The start pose is outside the first intermediate circle
        dist_start = compute_distance_two_points(
            planner.start_pose[:2],
            center_circumference_vector[0])
        if dist_start < planner.vehicle.max_radius - 1e-3:
            msg = 'The start pose is inside first intermediate circle.'
            messages.append(msg)
            warnings.warn(msg, UserWarning)
            check_passed = False
        
        # The end pose is outside the last intermediate circle
        dist_end = compute_distance_two_points(
            planner.end_pose[:2],
            center_circumference_vector[-1])
        if dist_end < planner.vehicle.max_radius - 1e-3:
            msg = 'The end pose is inside last intermediate circle.'
            messages.append(msg)
            warnings.warn(msg, UserWarning)
            check_passed = False
        
        # The intermediate circles don't overlap 
        for i in range(len(center_circumference_vector)-1):
            dist_centers = compute_distance_two_points(
                center_circumference_vector[i],
                center_circumference_vector[i+1])
            if dist_centers - 2 * R < 1e-3: # and turn_direction_vector[i] != turn_direction_vector[i+1]: # if the distance between the centers is less than 2R and the turn directions are not the same, then the circles overlap
                msg = f'Intermediate circles {i} and {i+1} overlap and they have different turn directions.'
                messages.append(msg)
                warnings.warn(msg, UserWarning)
                check_passed = False

    except Exception as e:
        warnings.warn(f'Error computing corner point vector: {e}', UserWarning)
        check_passed = False
    
    # Minimum corridors widths
    min_corridor_widths = planner.min_corridor_widths
    corridor_widths = [corridor.width for corridor in corridor_list]

    if not all(x >= y for x, y in zip(corridor_widths, min_corridor_widths)):
        msg = "At least one corridor is not wide enough to guarantee collision-free maneuvers."
        messages.append(msg)
        warnings.warn(msg, UserWarning)
        check_passed = False

    return check_passed, messages, center_circumference_vector, turn_direction_vector, corner_point_vector

def compute_minimum_widths(planner):
    """
    Compute the minimum corridor widths required to guarantee
    collision-free intermediate-circle maneuvers, together with
    the maximum admissible shift of each intermediate circle
    along the corresponding junction bisector.

    Parameters
    ----------
    planner : MotionPlanner
        Analytical motion planner.

    Returns
    -------
    min_widths : list of float
        Minimum admissible width for each corridor.

    s_max_list : list of float
        Maximum admissible shift for each intermediate circle.
        For junction i, s_max is the displacement along the
        junction bisector at which the swept robot footprint
        becomes tangent to the opposite wall of one of the two
        adjacent corridors.
    """

    corridor_list = planner.corridor_list

    R = planner.vehicle.max_radius
    r = planner.vehicle.width * 0.5

    n_corridors = len(corridor_list)

    if n_corridors < 2:
        raise ValueError(
            "At least two corridors are required."
        )

    # --------------------------------------------------------
    # Corridor orientations
    # --------------------------------------------------------

    phis = [
        corridor.tilt
        for corridor in corridor_list
    ]

    # --------------------------------------------------------
    # Half of the angular difference at each junction
    #
    # beta_i corresponds to the junction between
    # corridor i and corridor i+1.
    # --------------------------------------------------------

    betas = [
        0.5 * abs(
            compute_angular_difference(
                phis[i],
                phis[i + 1],
            )
        )
        for i in range(n_corridors - 1)
    ]

    # --------------------------------------------------------
    # Junction geometry
    #
    # q_i = (R-r) cos(beta_i)
    #
    # The local minimum corridor width required by junction i
    # is
    #
    #     w_req_i = R + r - q_i
    #
    # --------------------------------------------------------

    q = [
        (R - r) * cos(beta)
        for beta in betas
    ]

    local_min_widths = [
        R + r - q_i
        for q_i in q
    ]

    # --------------------------------------------------------
    # Minimum admissible width of each corridor
    #
    # Boundary corridors participate in one junction.
    # Interior corridors participate in two junctions and
    # must satisfy the more restrictive one.
    # --------------------------------------------------------

    min_widths = []

    # First corridor
    min_widths.append(
        local_min_widths[0]
    )

    # Interior corridors
    for i in range(1, n_corridors - 1):

        min_width = max(
            local_min_widths[i - 1],
            local_min_widths[i],
        )

        min_widths.append(
            min_width
        )

    # Last corridor
    min_widths.append(
        local_min_widths[-1]
    )

    # --------------------------------------------------------
    # Maximum admissible shift of each intermediate circle
    #
    # For junction i, the relevant nominal width requirement
    # is the LOCAL junction requirement:
    #
    #     w_req_i = R + r - q_i
    #
    # If the circle is shifted by s_i along the bisector,
    # the transverse displacement toward the opposite wall is
    #
    #     s_i cos(beta_i).
    #
    # Hence:
    #
    #     w_req_i + s_i cos(beta_i)
    #         <= min(w_i, w_{i+1})
    #
    # which yields
    #
    #     s_max_i =
    #       [min(w_i,w_{i+1}) - w_req_i] / cos(beta_i)
    #
    # --------------------------------------------------------

    s_max_list = []

    for i in range(n_corridors - 1):

        denom = cos(
            betas[i]
        )

        if abs(denom) < 1e-12:
            raise ValueError(
                f"Invalid corridor configuration at junction {i}: "
                "the angular difference is pi (180 degrees), "
                "so cos(beta) is zero."
            )

        limiting_width = min(
            corridor_list[i].width,
            corridor_list[i + 1].width,
        )

        local_min_width = (
            local_min_widths[i]
        )

        s_max = (
            limiting_width
            - local_min_width
        ) / denom

        # Under the standing assumptions this should be
        # non-negative. Keep this check because a negative
        # value indicates that at least one adjacent corridor
        # violates the local minimum-width requirement.
        if s_max < -1e-12:
            raise ValueError(
                f"Invalid corridor configuration at junction {i}: "
                f"available width is smaller than the local "
                f"minimum-width requirement. "
                f"s_max = {s_max:.6e}."
            )

        # Remove tiny negative values due only to floating-
        # point roundoff.
        s_max = max(
            0.0,
            s_max,
        )

        s_max_list.append(
            s_max
        )

    return min_widths, s_max_list


def compute_min_width_s_max_corridor_pair(corridor1, corridor2, vehicle):
    R = vehicle.max_radius
    r = vehicle.width * 0.5

    # Extract tilt angles
    phi1 = corridor1.tilt
    phi2 = corridor2.tilt

    # Compute betas
    beta = 0.5 * abs(compute_angular_difference(phi1, phi2))
    q = (R - r) * cos(beta)
    min_width = r + R - q

    denom = cos(beta)

    if abs(denom) < 1e-12:
        raise ValueError(
            f"Invalid corridor configuration: "
            "angle difference is pi (180 degrees), making cos(beta)=0."
        )
    s_max = (min(corridor1.width, corridor2.width) - min(min_width, min_width))/denom

    return min_width, s_max


def check_merge_corridors(corridor1, corridor2):
    '''
    Check whether two corridors can be merged into one corridor.

    :param corridor1: first corridor
    :type corridor1: CorridorWorld

    :param corridor2: second corridor
    :type corridor2: CorridorWorld

    :return: True if the corridors can be merged, False otherwise
    :rtype: bool
    '''

    # Conditions to merge two corridors: 
    # 1- The corridors have the same tilt
    # 2- The corridors are aligned 
    # 3- The corridors have the same width

    xc1, xc2 = corridor1.center[0], corridor2.center[0]
    yc1, yc2 = corridor1.center[1], corridor2.center[1]
    tol_cond1, tol_cond2, tol_cond3 = 1e-5, 1e-3, 1e-2

    # Check the three conditions
    if (abs(corridor1.tilt - corridor2.tilt) < tol_cond1) and \
        ((wrapPositiveAngle(atan2(yc2 - yc1, xc2 - xc1)) - corridor1.tilt)) < tol_cond2 and \
        (corridor1.width - corridor2.width < tol_cond3):
        return True
    
    return False