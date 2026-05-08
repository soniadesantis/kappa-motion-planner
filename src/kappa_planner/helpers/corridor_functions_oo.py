import numpy as np
from math import sin, cos, pi, sqrt, atan2, asin, acos, tan, inf
import math as m
import matplotlib.pyplot as plt
from copy import copy
from ..geometry import Point, Circle, Pose, IntermediateCircle, IntermediateCirclesSequence
from ..corridor import CorridorWorld
from ..trajectory import BackwardArc, LinearSegmentUnicycle, CurvilinearArcUnicycle, UnicycleTrajectory, UnicycleTrajectoryOptimal, TurnOnTheSpot
from .helper_functions import compute_initial_turn_direction
from .poses import absolute_to_relative_pose, relative_to_absolute_pose
from .plot_helpers import plot_corridors, plot_analytical_trajectory
from .intersections import circle_intersection, check_intersection_case, get_intersection
from .primitives import compute_arc_from_two_tangents, compute_arc_from_two_tangents_objects
from .geometry_operations import (
    compute_distance_two_points,
    compute_angular_difference,
    wrapPositiveAngle,
    compute_angular_difference_with_turn_direction,
    efficient_sign,
    check_point_inside_segment,
)

def compute_trajectory_bicycle_two_corridors(corridor1, corridor2, start_pose, end_pose, bicycle):
    '''
    Compute the sequence of primitives that build a trajectory for a bicycle vehicle within two corridors. 
    This is not the time-optimal version.
    Backward maneuver only for collision avoidance.

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
    ## Merge two corridors if needed #TO BE DONE
    # if check_merge_corridors(corridor1, corridor2): # If the two corridors can be merged
    #     new_corridor = get_corridor_from_vector(corridor1.tail, corridor2.head, corridor1.width, add_height = 0)
    #     # Compute the trajectory within the new corridor and return false as the check_intersection Boolean
    #     return compute_trajectory_unicycle_one_corridor(new_corridor, start_pose, unicycle, end_pose), False

    ## Compute main turn direction tau2
    tau2 = corridor1.compute_relative_turn_direction(corridor2)
    
    ## Compute the corner point
    corner_point = get_corner_point(corridor1, corridor2, tau2)

    ## Compute the center of the second circumference
    circ2 = second_circle_bicycle(corridor1, corridor2, tau2, corner_point, bicycle, start_pose)
    
    ## Compute the first two maneuvers from start pose to the second circumference
    collision_check = True
    start_pose_fw_drive = start_pose.copy()
    start_pose_object = Pose(position=Point(start_pose[0], start_pose[1]), theta = start_pose[2])
    start_maneuvers = []
    while collision_check:
        arc1, segment2 = compute_two_maneuvers_bicycle(start_pose_fw_drive, bicycle, circ2, tau2, t0 = 0, tau1 = 0)
        collision_check, wall = collision_avoidance_check(arc1, corridor1, bicycle, margin = 0)
        if collision_check:
            bw_arc = compute_backward_arc(corridor1, start_pose_object, bicycle, arc1.turn_direction, bicycle.max_radius, wall, corner_point)
            start_pose_fw_drive = [bw_arc.xf, bw_arc.yf, bw_arc.thetaf]
            start_maneuvers.append(bw_arc)
        else:
            start_maneuvers.append(arc1)
            start_maneuvers.append(segment2)
            
    ## Compute the last two maneuvers from the second circumference to end pose
    collision_check = True
    start_pose_fw_drive = end_pose.copy()
    start_pose_fw_drive[2] = start_pose_fw_drive[2] + pi
    start_pose_object = Pose(position=Point(end_pose[0], end_pose[1]), theta = end_pose[2]+pi)
    corridor2_inverted = corridor2.rotate_corridor(pi)
    end_maneuvers_inverted = []
    while collision_check:
        arc5, segment4 = compute_two_maneuvers_bicycle(start_pose_fw_drive, bicycle, circ2, -tau2, t0 = 0, tau1 = 0)
        collision_check, wall = collision_avoidance_check(arc5, corridor2_inverted, bicycle, margin = 0)
        if collision_check:
            bw_arc = compute_backward_arc(corridor2_inverted, start_pose_object, bicycle, arc5.turn_direction, bicycle.max_radius, wall, corner_point)
            start_pose_fw_drive = [bw_arc.xf, bw_arc.yf, bw_arc.thetaf]
            end_maneuvers_inverted.append(bw_arc)
        else:
            end_maneuvers_inverted.append(arc5)
            end_maneuvers_inverted.append(segment4)
    
    ## Compute the last three maneuvers from end pose to the second circumference and invert them
    # arc5, segment4 = compute_two_maneuvers([end_pose[0], end_pose[1], end_pose[2]+pi], bicycle, circ2.center.x, circ2.center.y, -tau2, t0 = 0, turn1 = 0)
    # raw_maneuvers = compute_three_maneuvers_compact(*invert_inputs(corridor2, corridor1, end_pose, bicycle, xc2, yc2, tau2, t0 = 0, turn1 = 0))
    end_maneuvers = invert_maneuvers(end_maneuvers_inverted, t0 = 0)
    segment4 = end_maneuvers[0]
    arc5 = end_maneuvers[1]
    if len(end_maneuvers) > 2:
        arc6 = end_maneuvers[2]
    # end_maneuvers = []
    # Check whether the intersection case occurs
    # check_intersection = check_intersection_case(segment2, segment4)
    check_intersection = False

    if check_intersection:
        pass
        # maneuvers = compute_trajectory_intersection_case_without_optimization(corridor1, corridor2, start_pose, end_pose, unicycle)
    else:
        # Build the arc along the second circle
        arc3 = compute_arc_from_two_tangents(segment2, segment4, tau2, circ2.center.x, circ2.center.y, bicycle)
        segment4.add_time_offset(arc3.tf)
        arc5.add_time_offset(arc3.tf)
        
        if len(end_maneuvers) > 2:
            arc6.add_time_offset(arc3.tf)
            end_maneuvers = [segment4, arc5, arc6]
        else:
            end_maneuvers = [segment4, arc5]
        trajectory = start_maneuvers + [arc3] + end_maneuvers

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
    n = 2
    # 1 — Compute P^init
    first_int_circ = intermediate_circles.first

    start_maneuvers = compute_traj_to_circle_bicycle(
        corridor1,
        start_pose,
        bicycle,
        first_int_circ,
    )

    # 2 — Compute P^final
    last_int_circ = intermediate_circles.last
    inv_last_corridor, inv_last_int_circ, inv_end_pose = invert_inputs(corridor2, last_int_circ, end_pose)

    inv_end_maneuvers = compute_traj_to_circle_bicycle(
        inv_last_corridor,
        inv_end_pose,
        bicycle,
        inv_last_int_circ,
    )

    end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)
    
    intermediate_arc = compute_arc_from_two_tangents_objects(start_maneuvers[-1], end_maneuvers[0], first_int_circ, bicycle)
    trajectory = start_maneuvers + [intermediate_arc] + end_maneuvers

    for i in range(len(trajectory)-1):
        if trajectory[i+1].time_grid[0] != trajectory[i].time_grid[-1]:
            trajectory[i+1].add_time_offset(abs(trajectory[i+1].time_grid[0] - trajectory[i].time_grid[-1]))
    correct_angles(trajectory)

    return trajectory

def get_corner_point(corridor1, corridor2, turn):
    '''
    Given two corridors, compute the corner point at the intersection of their edges. 
    The selected edges depend on the turn direction between the two corridors.

    :param corridor1: starting corridor
    :type corridor1: CorridorWorld
    :param corridor2: arrival corridor
    :type corridor2: CorridorWorld
    :param turn: turn direction between the two corridors
    :type turn: float [-1, 1]

    :return: x and y coordinates of the corner point
    :rtype: Point object
    '''
    # Exclude the case where the two corridors are aligned and have the same dimensions
    # IMPORTANT: this check should not be needed anymore, since it is checked before in the code
    # if check_merge_corridors(corridor1, corridor2):
    #     print("These two corridors are aligned and have the same dimensions: can merge into one corridor.")
    #     return None
    
    # Exclude the case where the two corridors have the same tilt 
    # and the right faces of corridor1 and corridor2 (if turn right) 
    # the left faces of corridor1 and corridor2 (if turn left) coincide
    if (corridor1.tilt == corridor2.tilt) and (turn > 0) and (get_intersection(corridor1.W[:, 3], corridor2.W[:,3]) == m.inf): #turn left
        print("The two left faces of the corridors coincide, it is not possible to compute a corner point to the left.")
        return None
    elif (corridor1.tilt == corridor2.tilt) and (turn < 0) and (get_intersection(corridor1.W[:, 1], corridor2.W[:,1]) == m.inf): # turn right
        print("The two right faces of the corridors coincide, it is not possible to compute a corner point to the right.")
        return None
    # Continue with the case where the two corridors don't have the same tilt, 
    # or have the same tilt but the selected faces needed to compute the corner 
    # point don't coincide
    else:
        corner_point_list = [] # initialize a list containing the candidates corner points
        if turn > 0: # turn left
            select_edges1 = np.array([0, 3])  # check F and L faces of corridor1 
            select_edges2 = np.array([2, 3])  # check B and L faces of corridor2
        else:  # turn right
            select_edges1 = np.array([0, 1])  # check F and R faces of corridor1
            select_edges2 = np.array([2, 1])  # check B and R faces of corridor2
        # Check the intersection points between the selected faces of the corridors
        for i in select_edges2:
            for k in select_edges1:
                w1 = corridor2.W[:, i]
                w2 = corridor1.W[:, k]
                # check whether the two faces don't coincide (can happen especially for 90 degrees turns)
                intersection_point = get_intersection_oo(w1, w2)
                # intersection_point = Point(x = int_point[0], y = int_point[1])
                if (intersection_point != m.inf) and (intersection_point != []):
                    if (check_point_inside_corridor(corridor1, intersection_point) and check_point_inside_corridor(corridor2, intersection_point)):
                        corner_point_list.append(intersection_point)
        
        # If more than one intersection point exists, initialize the first vector with the first point
        vector1 = [corner_point_list[0].x, corner_point_list[0].y]
        for i in range(1, len(corner_point_list)):
            vector2 = [corner_point_list[i].x, corner_point_list[i].y]
            turn_direction_new_corner_point = compute_turn_direction([vector1[0] - corridor1.tail[0], vector1[1] - corridor1.tail[1]],
                                                                     [vector2[0] - corridor1.tail[0], vector2[1] - corridor1.tail[1]])
            if turn > 0 and turn_direction_new_corner_point > 0:
                vector1 = vector2
            elif turn < 0 and turn_direction_new_corner_point < 0:
                vector1 = vector2
        try:
            return Point(x = vector1[0], y = vector1[1])
        except:
            print("No corner point was found between these two corridors")
            return None
        
def get_intersection_oo(w1, w2):
    '''
    Use Cramer's rule to solve the system of equations
    ``w1.T @ ph = 0 ; w2.T @ ph = 0;`` for p, where ph = [p; 1]

    :param np.array w1: line parameter vector of shape [w0, w1, w2]
    :param np.array w2: line parameter vector of shape [w0, w1, w2]

    :return: intersection point between w1 and w2
    :rtype: Point object
    '''
    tol = 1e-3
    denominator = (w1[0]*w2[1] - w1[1]*w2[0])
    if (abs(denominator) < tol):
        if (abs(w2[0]) < tol and abs(w2[2]) > tol) or (abs(w2[0]) > tol and abs(w2[2]) < tol):
            return inf
        elif (abs(w2[0]) > tol and abs(w2[2]) > tol):
            if (abs(w1[0]/w2[0] - w1[2]/w2[2]) < tol):
                return inf
            else: 
                return []
    else:
        return Point(x = (w1[1]*w2[2] - w1[2]*w2[1])/denominator, 
                     y = (w1[2]*w2[0] - w1[0]*w2[2])/denominator)

def check_point_inside_corridor(corridor, point):
    '''Check if a given point is inside a given corridor.

    :param point: point to be checked
    :type point: Point object
    :param corridor: corridor to be checked whether it contains the point
    :type corridor: CorridorWorld

    :return: bool with True if point is inside the corridor
    :rtype: bool
    '''
    tol = 1e-6
    rot_angle = pi * 0.5 - corridor.tilt
    cos_rot_angle, sin_rot_angle = cos(rot_angle), sin(rot_angle)
    relative_point = Point(x = (point.x - corridor.center[0]) * cos_rot_angle - (point.y - corridor.center[1]) * sin_rot_angle,
                           y = (point.x - corridor.center[0]) * sin_rot_angle + (point.y - corridor.center[1]) * cos_rot_angle)
    
    x_ub, y_ub = 0.5 * corridor.width + tol, 0.5 * corridor.height + tol
    return (-x_ub <= relative_point.x <= x_ub) and (-y_ub <= relative_point.y <= y_ub)

def second_circle_bicycle(corridor1, corridor2, turn_direction, corner_point, bicycle, start_pose):
    '''
    Compute the center of the second circle. The position of the second circle can vary if
    1. The second corridor is narrow
    2. The start pose is inside the second circle

    :param corridor1: first corridor
    :type corridor1: CorridorWorld

    :param corridor2: second corridor
    :type corridor2: CorridorWorld

    :param turn_direction: turn direction
    :type turn_direction: either [-1, 1]

    :param corner_point: corner point
    :type corner_point: list of floats

    :param bicycle: bicycle model
    :type bicycle: Bicycle

    :param start_pose: start pose
    :type start_pose: list of floats

    :return: circle to be reached
    :rtype: Circle

    :return: y coordinate of the center of the second circle
    :rtype: float
    '''
    circ2 = compute_second_circle(corner_point, turn_direction, bicycle.max_radius, bicycle.width, 0, corridor1.tilt, corridor2.tilt)

    # if corridor2.width > bicycle.width:
    #     xc2, yc2 = compute_center_coordinates_second_circle_oo(corner_point, turn_direction, bicycle.max_radius, bicycle.width, 0, corridor1.tilt, corridor2.tilt)
    # else:
    #     # If the second corridor is narrow the position of the second circle is adjusted
    #     xc2, yc2 = circ_center_narrow_corridors(corridor2.tilt, corner_point, bicycle.max_radius, bicycle.width, turn_direction)
    # # If the start pose is inside the second circumference, adjust the position of the second circle
    # if ((xc2 - start_pose[0])**2 + (yc2 - start_pose[1])**2) < bicycle.max_radius**2:
    #     xc2, yc2 = start_inside_second_circle_case(corridor1, corridor2, turn_direction, corner_point, bicycle)
    return circ2

def compute_second_circle(corner_point, turn, R, vehicle_width, margin, tilt1, tilt2):
    '''
    Compute the coordinates of the center of a circumference placed at the intersection between two subsequent corridors (intermediate circumference).
    The center is placed at a distance equal to (R - half of vehicle_width - margin) along the bisector between two edges of the two corridors.
    The selected edges depend on the main turn direction between the two corridors. 

    :param corner_point: selected corner point at the intersection between two corridors
    :type corner_point: Point object
    :param turn: turn direction along the circumference to be reached
    :type turn: float [-1, 1]
    :param R: radius of the circumference to be reached == radius of arc maneuver of the considered vehicle
    :type R: float
    :param vehicle_width: width of the vehicle
    :type vehicle_width: float
    :param margin: additional margin to avoid collision with the walls of the corridors
    :type margin: float
    :param tilt1: tilt of first corridor
    :type tilt1: float
    :param tilt2: tilt of second corridor
    :type tilt2: float

    :return: center point of the circumference to be reached
    :rtype: Point object
    '''
    angle_circle_center_direction = tilt2 + turn * 0.5 * (pi - abs(compute_angular_difference(tilt2, tilt1)))
    margin = 0.5 * vehicle_width - margin
    # margin = 0
    return Circle(center =  Point(x = corner_point.x + (R - margin) * cos(angle_circle_center_direction),
                                  y = corner_point.y + (R - margin) * sin(angle_circle_center_direction)),
                    radius = R)







    





    











def create_intermediate_circles_sequence(
    intermediate_circle_centers,
    turn_direction_vector,
    corner_point_vector,
    radius,
    s_max_circles,
):
    """
    Create the sequence of intermediate circles used in the trajectory algorithm.
    Each intermediate circle is associated with a corner point and a turn direction.

    :param intermediate_circle_centers: centers of the intermediate circles
    :type intermediate_circle_centers: list of list of floats
    :param turn_direction_vector: turn directions of the intermediate circles
    :type turn_direction_vector: list of int
    :param corner_point_vector: corner points associated with the intermediate circles
    :type corner_point_vector: list of list of floats
    :param radius: radius of the intermediate circles
    :type radius: float

    :return: ordered sequence of intermediate circles
    :rtype: IntermediateCirclesSequence
    """

    n = len(intermediate_circle_centers)

    if not (len(turn_direction_vector) == len(corner_point_vector) == n):
        raise ValueError("Impossible to define IntermediateCirclesSequence:" \
        "all input vectors must have the same length.")

    items = []

    for i in range(n):
        cx, cy = intermediate_circle_centers[i]
        px, py = corner_point_vector[i]
        td = turn_direction_vector[i]

        circle = IntermediateCircle(
            center=Point(cx, cy),
            radius=radius,
            corner_point=Point(px, py),
            turn_direction=td,
            index = i,
            s_max=s_max_circles[i],
        )

        items.append(circle)

    return IntermediateCirclesSequence(items)



def compute_arc_from_two_tangents_objects(segment1, segment2, circ, vehicle):
    """
    Given two linear segments, compute the circular arc connecting them.
    The arc belongs to the circle `circ` and its turning direction is given by
    `circ.turn_direction`.

    :param segment1: first segment
    :type segment1: LinearSegmentUnicycle
    :param segment2: second segment
    :type segment2: LinearSegmentUnicycle
    :param circ: circle to which the arc belongs (includes turn direction)
    :type circ: IntermediateCircle
    :param vehicle: considered vehicle
    :type vehicle: Unicycle

    :return: arc in between the two segments
    :rtype: CurvilinearArcUnicycle
    """
    x2, y2 = segment1.end_position
    theta2 = segment1.thetaf

    x3, y3 = segment2.start_position
    theta3 = segment2.theta0

    t0 = segment1.tf
    radius = vehicle.max_radius

    omega = vehicle.omega_max if circ.turn_direction > 0 else vehicle.omega_min

    arc = CurvilinearArcUnicycle(
        xc=circ.xc,
        yc=circ.yc,
        x0=x2,
        y0=y2,
        theta0=theta2,
        xf=x3,
        yf=y3,
        thetaf=theta3,
        radius=radius,
        turn_direction=circ.turn_direction,
        v=vehicle.v_max,
        omega=omega,
        unicycle=vehicle,
        t0=t0,
        samples_number=10,
    )

    return arc





def compute_traj_to_circle_free_space_bicycle(start_pose, bicycle, circ1, tau0 = 0):
    """
    Build the initial part of the trajectory from the start pose to the first intermediate circle.
    
    :param corridor1: first corridor in the sequence
    :type corridor1: CorridorWorld
    :param start_pose: initial pose of the vehicle
    :type start_pose: list of floats
    :param bicycle: bicycle vehicle
    :type Bicycle: Bicycle
    :param circ1: first intermediate circle
    :type circ1: IntermediateCircle object
    """
    tau1 = circ1.turn_direction
    tau0 = compute_initial_turn_direction(
        circ1.xc,
        circ1.yc,
        circ1.radius,
        start_pose[0],
        start_pose[1],
        start_pose[2],
        tau1) if tau0 == 0 else tau0
    
    ## Compute the first two maneuvers from start pose to the second circumference
    start_pose_fw_drive = start_pose.copy()
    start_pose_object = Pose(position=Point(start_pose[0], start_pose[1]), theta = start_pose[2])
    start_maneuvers = []

    # First check whether a backward maneuver is required for time-optimality
    not_optimal, _, _, _ = rule_initial_backward_maneuver(start_pose, circ1, tau0)
    if not_optimal: # Case tau1 = tau2 and iota > 90 degrees
        bw_arc = compute_backward_arc_optimal(start_pose_object, tau0, tau1, circ1, bicycle)
        start_pose_fw_drive = [bw_arc.xf, bw_arc.yf, bw_arc.thetaf]
        start_pose_object = Pose(position=Point(bw_arc.xf, bw_arc.yf), theta = bw_arc.thetaf)
        start_maneuvers.append(bw_arc)  
    # Compute the free space solution
    arc1, segment2 = compute_two_maneuvers_bicycle(start_pose_fw_drive, bicycle, circ1, tau1, t0 = 0, tau1 = tau0)
    start_maneuvers.append(arc1)
    start_maneuvers.append(segment2)
    
    # Adjust the time grid and angles
    for i in range(len(start_maneuvers)-1):
        if start_maneuvers[i+1].time_grid[0] != start_maneuvers[i].time_grid[-1]:
            start_maneuvers[i+1].add_time_offset(abs(start_maneuvers[i+1].time_grid[0] - start_maneuvers[i].time_grid[-1]))
    correct_angles(start_maneuvers)
    return start_maneuvers


    
# def compute_P_mid(intermediate_circles, bicycle):
#     n_segments = len(intermediate_circles) - 1
#     if n_segments <= 0:
#         return [], intermediate_circles

#     circles_to_prune = []
#     segments = []

#     segment1 = compute_segment_between_two_circles_objects(
#         intermediate_circles[0],
#         intermediate_circles[1],
#         bicycle,
#     )

#     for i in range(1, n_segments):
#         segment2 = compute_segment_between_two_circles_objects(
#             intermediate_circles[i],
#             intermediate_circles[i + 1],
#             bicycle,
#         )

#         if check_intersection_case(segment1, segment2):
#             print("Pruning circle at index", i)
#             # circle i is unnecessary
#             circles_to_prune.append(i)

#             # bridge from (i-1) directly to (i+1)
#             segment1 = compute_segment_between_two_circles_objects(
#                 intermediate_circles[i - 1],
#                 intermediate_circles[i + 1],
#                 bicycle,
#             )
#         else:
#             segments.append(segment1)
#             segment1 = segment2

#     # append the last active segment
#     segments.append(segment1)

#     # prune circles after the scan (descending indices)
#     for idx in sorted(set(circles_to_prune), reverse=True):
#         if 0 <= idx < len(intermediate_circles):
#             intermediate_circles.remove_at(idx)

#     return segments, intermediate_circles
            
def prune_circles_adjust_trajectory(
    intermediate_circles,
    segments,
    bicycle,
    start_pose,
    corridor_list,
    end_pose,
    start_maneuvers,
    end_maneuvers,
):
    """
    Prune unnecessary intermediate circles and rebuild/adjust the trajectory accordingly.

    For N corridors:
      - there are (N - 1) intermediate circles
      - there are N connecting segments
    """
    processed_finished = False
    while not(processed_finished):
        changed = False
        for i in range(len(intermediate_circles)):
            # If an intersection between two segments is found
            if check_intersection_case(segments[i], segments[i+1]): 
                print("Pruning circle at index", i)
                removed_circle = intermediate_circles[i]
                intermediate_circles.remove_at(i)
                if len(intermediate_circles) == 0:
                    raise ValueError("No intermediate circles are necessary.")

                del segments[i : i + 2]
                
                # If the first int circle is pruned, recompute P^init 
                if i == 0: 
                    first_int_circ = intermediate_circles.first
                    corridor1 = corridor_list[0]

                    start_maneuvers = compute_traj_to_circle_bicycle(
                        corridor1,
                        start_pose,
                        bicycle,
                        first_int_circ,
                    )
                    if not(check_segment_inside_two_corridors(corridor1, corridor_list[1], start_maneuvers[-1])):
                        new_corner_point =  get_corner_point(
                                            corridor1,
                                            corridor_list[1],
                                            -removed_circle.turn_direction,
                                            )
                        xc, yc = compute_center_coordinates_second_circle(
                            new_corner_point,
                            removed_circle.turn_direction,
                            bicycle.max_radius,
                            bicycle.width,
                            0,
                            corridor1.tilt,
                            corridor_list[1].tilt,
                            corridor1,
                            corridor_list[1])
     
                        new_int_circ = IntermediateCircle(
                            center=Point(xc, yc),
                            radius=removed_circle.radius,
                            corner_point=new_corner_point,
                            turn_direction=-removed_circle.turn_direction,
                            index = removed_circle.index,
                        )
                        start_maneuvers = compute_traj_to_circle_bicycle(
                        corridor1,
                        start_pose,
                        bicycle,
                        new_int_circ,
                    )
                        
                    segments.insert(0, start_maneuvers[-1])
                    # Restart the cycle 
                    

                if i < len(intermediate_circles) - 1: 
                    new_segment =  compute_segment_between_two_circles_objects(
                        intermediate_circles[i-1],
                        intermediate_circles[i],
                        bicycle,
                    )
                    check_segment_inside_two_corridors(corridor_list[removed_circle.index-1], corridor_list[removed_circle.index], new_segment)
                    segments.insert(i,new_segment)
                    # Restart the cycle 
                    

                else:
                    last_int_circ = intermediate_circles.last
                    last_corridor = corridor_list[-1]
                    inv_last_corridor, inv_last_int_circ, inv_end_pose = invert_inputs(last_corridor, last_int_circ, end_pose)

                    inv_end_maneuvers = compute_traj_to_circle_bicycle(
                        inv_last_corridor,
                        inv_end_pose,
                        bicycle,
                        inv_last_int_circ,
                    )
                    
                    end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)
                    check_segment_inside_two_corridors(corridor_list[-1], corridor_list[-2], end_maneuvers[0])
                    # figure = plot_corridors(corridor_list)
                    # plot_analytical_trajectory(segments, figure=figure)
                    # plt.show(block = True)
                    segments.append(end_maneuvers[0])
                    # figure = plot_corridors(corridor_list)
                    # plot_analytical_trajectory(segments[0:3], figure=figure)
                    # plt.show(block = True)

                    # Restart the cycle

                changed = True
                break 

        processed_finished = not(changed)
            
    return intermediate_circles, segments, start_maneuvers, end_maneuvers

def check_segment_inside_two_corridors(corridor1, corridor2, segment): 
    A1, A2 = segment.start_position, segment.end_position
    for index in range(0, 4): 
        B1 = corridor1.corners[index]
        B2 = corridor1.corners[index+1] if index < 3 else corridor1.corners[0]
        # figure = plot_corridors([corridor1])
        # plot_analytical_trajectory([segment], figure=figure)
        # plt.plot([B1[0], B2[0]], [B1[1], B2[1]], 'r-')
        # plt.show(block = True)
        int_point = check_intersection_two_segments(A1, A2, B1, B2)
        if int_point is not None:
            break
    if int_point is None:
        print("No intersection between segment and corridor when removing circle")
        return True
    else: 
        check_point_inside_corridor1 = check_point_inside_corridor(corridor1, int_point)
        check_point_inside_corridor2 = check_point_inside_corridor(corridor2, int_point)
        if check_point_inside_corridor1 and check_point_inside_corridor2:   
            return True
        else:
            return False


def check_intersection_two_segments(A1, A2, B1, B2):
    '''
    Given the extreme points of two segments, check whether the two segments are intersecting.

    :param A1: extreme point of first segment
    :type A1: list of floats or np.ndarray
    :param A2: extreme point of first segment
    :type A2: list of floats or np.ndarray
    :param B1: extreme point of second segment
    :type B1: list of floats or np.ndarray
    :param B2: extreme point of second segment
    :type B2: list of floats or np.ndarray

    :return: a boolean variable equal to True if the segments intersect, False otherwise
    :rtype: Boolean
    '''
    point_inside_two_segments = False
    x1 = A1[0]
    y1 = A1[1]
    x2 = A2[0]
    y2 = A2[1]
    x3 = B1[0]
    y3 = B1[1]
    x4 = B2[0]
    y4 = B2[1]
    # two vertical segments: no intersection possible
    if x2 == x1 and x3 == x4:
        return point_inside_two_segments
    # one vertical segment
    elif x2 == x1:
        m2 = (y4 - y3)/ (x4 - x3)
        q2 = y3 - ((y4 - y3)/ (x4 - x3)) * x3
        x_int = x2
        y_int = m2 * x_int + q2
    # one vertical segment
    elif x3 == x4:
        m1 = (y2 - y1)/ (x2 - x1)
        q1 = y1 - ((y2 - y1)/ (x2 - x1)) * x1   
        x_int = x3
        y_int = m1 * x_int + q1 
    # no vertical segment
    else:
        m1 = (y2 - y1)/ (x2 - x1)
        q1 = y1 - ((y2 - y1)/ (x2 - x1)) * x1
        m2 = (y4 - y3)/ (x4 - x3)
        q2 = y3 - ((y4 - y3)/ (x4 - x3)) * x3
        #Parallel segments: no intersection possible
        if m1 == m2:
            return point_inside_two_segments
        else:
            x_int = (q1 - q2) / (m2 - m1)
            y_int = m1 * ((q1 - q2)/ (m2 - m1)) + q1    
            point_inside_seg1 = check_point_inside_segment([x1,y1], [x2,y2], [x_int,y_int])
            point_inside_seg2 = check_point_inside_segment([x3,y3], [x4,y4], [x_int,y_int])
            if point_inside_seg2 and point_inside_seg1:
                point_inside_two_segments = True
    return Point(x_int, y_int) if point_inside_two_segments else None


def adjust_trajectory_switch_circles(
    intermediate_circles,
    segments,
    bicycle,
    start_pose,
    corridor_list,
    end_pose,
    start_maneuvers,
    end_maneuvers,
):
    n_circles = len(intermediate_circles)

    for index in range(n_circles - 1):
        # Compute distance between consecutive circles
        point1 = intermediate_circles[index].center
        point2 = intermediate_circles[index + 1].center
        dist = compute_distance_two_points(point1, point2)

        # If the circles intersect and have same turn direction
        # there might be a problem 
        if (
            dist < 2 * intermediate_circles[index].radius
            and intermediate_circles[index].turn_direction
            == intermediate_circles[index + 1].turn_direction
        ):
            # Case 1: invert first and second circle
            if index == 0:
                point_to_check = start_maneuvers[-1].end_position

                if (
                    compute_distance_two_points(point_to_check, point2)
                    <= intermediate_circles[index].radius
                ):
                    switch = True

                    circ1 = intermediate_circles[index+1]
                    circ2 = intermediate_circles[index]
                    intermediate_circles[index], intermediate_circles[index + 1] = circ1, circ2

                    start_maneuvers = compute_traj_to_circle_bicycle(
                        corridor_list[0],
                        start_pose,
                        bicycle,
                        circ1,
                    )
                    segments[0] = start_maneuvers[-1]

                    segments[index+1] = compute_segment_between_two_circles_objects(
                        intermediate_circles[index],
                        intermediate_circles[index + 1],
                        bicycle,
                    )

                    segments[index + 2] = compute_segment_between_two_circles_objects(
                        intermediate_circles[index + 1],
                        intermediate_circles[index + 2],
                        bicycle,
                    )
                    
            # Case 2: invert circles between 2 and (n-2)
            # Substitute segments i, i+1, i+2
            elif index < n_circles - 2:
                point_to_check = segments[index].end_position

                if (
                    compute_distance_two_points(point_to_check, point2)
                    <= intermediate_circles[index].radius
                ):
                    circ1 = intermediate_circles[index+1]
                    circ2 = intermediate_circles[index]
                    intermediate_circles[index], intermediate_circles[index + 1] = circ1, circ2

                    segments[index+1] = compute_segment_between_two_circles_objects(
                        intermediate_circles[index],
                        intermediate_circles[index + 1],
                        bicycle,
                    )

                    segments[index] = compute_segment_between_two_circles_objects(
                        intermediate_circles[index - 1],
                        intermediate_circles[index],
                        bicycle,
                    )

                    segments[index + 2] = compute_segment_between_two_circles_objects(
                        intermediate_circles[index + 1],
                        intermediate_circles[index + 2],
                        bicycle,
                    )
            
            # Case 3: invert (n-2) and (n-1)
            # Substitute segments n, n-1, n-2
            else:
                point_to_check = end_maneuvers[0].start_position
                
                if (
                    compute_distance_two_points(point_to_check, point1)
                    <= intermediate_circles[index].radius
                ):
                    circ1 = intermediate_circles[index+1]
                    circ2 = intermediate_circles[index]
                    intermediate_circles[index], intermediate_circles[index + 1] = circ1, circ2
                    segments[-2] = compute_segment_between_two_circles_objects(
                        intermediate_circles[index],
                        intermediate_circles[index + 1],
                        bicycle,
                    )
                    inv_last_corridor, inv_last_int_circ, inv_end_pose = invert_inputs(corridor_list[-1], intermediate_circles[index + 1], end_pose)

                    # figure = plot_corridors(corridor_list)
                    # plot_analytical_trajectory(segments, figure=figure)
                    # plt.plot(point_to_check[0], point_to_check[1], 'ro')
                    # plt.plot(intermediate_circles[index].xc, intermediate_circles[index].yc, 'go')
                    # angle_array = np.linspace(0, 2*pi, 100)
                    # # plt.plot(intermediate_circles[index].xc + intermediate_circles[index].radius * np.cos(angle_array), intermediate_circles[index].yc + intermediate_circles[index].radius * np.sin(angle_array), 'r--')
                    # plt.plot(intermediate_circles[index+1].xc + intermediate_circles[index+1].radius * np.cos(angle_array), intermediate_circles[index+1].yc + intermediate_circles[index+1].radius * np.sin(angle_array), 'g--')
                    # plt.show(block = True)
                    inv_end_maneuvers = compute_traj_to_circle_bicycle(
                        inv_last_corridor,
                        inv_end_pose,
                        bicycle,
                        inv_last_int_circ,
                    )

                    end_maneuvers = invert_maneuvers(inv_end_maneuvers, t0 = 0)
                    segments[-1] = end_maneuvers[0]
                    segments[-3] = compute_segment_between_two_circles_objects(
                        intermediate_circles[index-1],
                        intermediate_circles[index],
                        bicycle,
                    )

    return (
        intermediate_circles,
        segments,
        start_maneuvers,
        end_maneuvers,
    )


