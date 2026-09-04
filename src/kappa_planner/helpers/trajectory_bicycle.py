from math import asin, atan2, cos, pi, sin, sqrt

import numpy as np

from ..geometry import Circle, Point, Pose
from ..trajectory import BackwardArc, CurvilinearArcUnicycle, LinearSegmentUnicycle
from .collision_avoidance import collision_avoidance_check_bicycle
from .geometry_operations import (
    compute_angular_difference,
    compute_angular_difference_with_turn_direction,
    wrapPositiveAngle,
)
from .helper_functions import (
    compute_center_coordinates_first_circle,
)
from .intermediate_circles_choice import selected_sequence_from_preferences
from .intersections import (
    check_intersection_case,
    circle_intersection,
)
from .invert_inputs import invert_inputs_all
from .poses import (
    absolute_to_relative_pose,
    relative_to_absolute_pose,
)
from .primitives import (
    compute_arc_from_two_tangents_objects,
    compute_segment_between_two_circles_objects,
    correct_angles,
    invert_maneuvers,
)


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


def backward_maneuver_optimality(pose, tau1, tau2, circ2):
    '''
    Compute the circle for the backward maneuver and the final pose to be reached after the backward arc.
    
    :param pose: Start pose
    :type pose: Pose object

    :param tau1: initial turn direction
    :type tau1: float

    :param tau2: turn direction circle to reach
    :type tau2: float

    :param circ2: Circle for the backward maneuver
    :type circ2: Circle object
    '''
    # The bw circle is centered at the opposite side of the initial circle
    bw_circle = Circle(
        center=Point(
            x=pose.x + circ2.radius * cos(pose.theta - tau1 * pi * 0.5),
            y=pose.y + circ2.radius * sin(pose.theta - tau1 * pi * 0.5),
        ),
        radius=circ2.radius,
    )

    # Compute the final pose of the backward maneuver
    if tau1 == tau2:
    # when tau0 = tau1, the final pose is the point on the bw circle that is aligned with the center of circ2
        alpha = atan2(circ2.yc - bw_circle.yc, circ2.xc - bw_circle.xc)
        final_bw_position = Point(x = bw_circle.xc + circ2.radius * cos(alpha), y = bw_circle.yc + circ2.radius * sin(alpha))
        final_bw_pose = Pose(position = final_bw_position, theta = alpha - tau1 * 0.5 * pi)
    
    else:
    # when tau0 != tau1, the final pose is the point on the bw circle that is aligned with the center of circ2 reflected
        a = sqrt((circ2.xc - bw_circle.xc)**2 + (circ2.yc - bw_circle.yc)**2)
        beta_2r = asin(2 * circ2.radius / a)
        alpha0 = atan2(circ2.yc - bw_circle.yc, circ2.xc - bw_circle.xc)
        alpha = alpha0 + tau1 * beta_2r
        final_bw_position = Point(x = bw_circle.xc + circ2.radius * cos(alpha), y = bw_circle.yc + circ2.radius * sin(alpha))
        final_bw_pose = Pose(position = final_bw_position, theta = alpha - tau1 * 0.5 * pi)
    
    return final_bw_pose, bw_circle


def compute_backward_arc_optimal(pose, tau1, tau2, circ2, bicycle):

    final_bw_pose, bw_circle = backward_maneuver_optimality(pose, tau1, tau2, circ2)
    # final_bw_theta = pose.theta + compute_angular_difference_with_turn_direction(pose.theta, final_bw_pose.theta, -tau)
    
    # plt.plot(bw_circle.xc, bw_circle.yc, 'ro')
    # angle_array = np.linspace(0, 2*pi, 100)
    # x_circle = circ2.radius * np.cos(angle_array) + bw_circle.xc
    # y_circle = circ2.radius * np.sin(angle_array) + bw_circle.yc
    # plt.plot(x_circle, y_circle)
    # plt.plot(pose.x, pose.y, 'bo')
    # plt.plot(final_bw_pose.x, final_bw_pose.y, 'go')
    # plt.show(block = True)
    backward_arc =BackwardArc(xc=bw_circle.xc, yc=bw_circle.yc, x0 = pose.x, y0 = pose.y,
                                        theta0 = pose.theta, xf = final_bw_pose.x, yf = final_bw_pose.y,
                                        thetaf = final_bw_pose.theta, radius = circ2.radius,
                                        turn_direction = -tau1, v = -bicycle.v_max,
                                        omega = -tau1 * bicycle.omega_max, bicycle = bicycle,
                                        t0 = 0, samples_number = 100)
    return backward_arc


def rule_initial_backward_maneuver(start_pose, int_circ, turn1):
    '''
    Compute the initial angular displacement required to achieve time optimal motion. 
    If no rotation is needed, the angular displacement will be zero.

    :param start_pose: initial pose
    :type start_pose: list or np.ndarray
    :param xc2: x coordinate of center of intermediate circle
    :type xc2: float
    :param yc2: y coordinate of center of intermediate circle
    :type yc2: float
    :param turn1: turn direction along the first circumference
    :type turn1: float [-1,1]
    :param turn2: turn direction along the second circumference
    :type turn2: float [-1,1]
    :param radius: v_max/omega_max
    :type radius: float
    :param omega_max: maximum angular velocity
    :type omega_max: float
    :param omega_min: minimum angular velocity
    :type omega_min: float

    :return: angular displacement
    :rtype: float
    :return: angular velocity of the initial turn on-the-spot
    :rtype: float
    :return: x coordinate of the initial circle
    :rtype: float
    :return: y coordinate of the initial circle
    :rtype: float
    '''
    tol = 1e-6
    include_backward_maneuver = False
    # Extract initial pose
    x0, y0, theta0 = start_pose
    # Extract intermediate circle parameters
    xc2, yc2 = int_circ.xc, int_circ.yc
    radius = int_circ.radius
    turn2 = int_circ.turn_direction
    # Compute alpha0, the orientation of the segment connecting x0,y0 to xc2,yc2
    alpha0 = wrapPositiveAngle(atan2((yc2 - y0), (xc2 - x0)))
    # Initialize the delta_angle
    delta_angle = 0
    # Compute the angle beta
    a = sqrt((xc2 - x0)**2 + (yc2 - y0)**2)
    beta = asin(radius / a)
    # Compute beta_2R
    if 2*radius / a <= 1: 
        beta_2R = asin(2*radius / a)
    else:
        beta_2R = pi/6 #asin(2*radius / a)

    ## Rule for Left-left and Right-right case
    if turn1 == turn2:
        # Compute the angular difference between theta0 and alpha0 - tau1*beta when rotating according to turn1
        ang_disp = compute_angular_difference_with_turn_direction(theta0, alpha0 - turn2 * beta, turn1)

        # If the amplitude of the obtained angle is > pi/2 - beta, theta0 lies in the region where
        # a turn on-the-spot is needed until angle_to_be_reached
        if abs(ang_disp) > (pi * 0.5 - beta) + tol:
            angle_to_be_reached = wrapPositiveAngle(alpha0 - turn2 * pi * 0.5)
            delta_angle += compute_angular_difference_with_turn_direction(theta0, angle_to_be_reached, turn1)
            include_backward_maneuver = True

    ## Rule for Left-right and Right-left case
    elif turn1 != turn2:
        # Compute the angular difference between theta0 and alpha0 - turn2 * beta when rotating according to turn1
        ang_disp = compute_angular_difference_with_turn_direction(theta0, alpha0 - turn2 * beta, turn1)

        # If the amplitude of the obtained angle is > pi * 0.5 - (beta_2R - beta), theta0 lies in the region where
        # a turn on-the-spot is needed until angle_to_be_reached
        if abs(ang_disp) > (pi * 0.5 - (beta_2R - beta)) + tol:
            angle_to_be_reached = wrapPositiveAngle(alpha0 - turn1 * pi* 0.5 + turn1 * beta_2R)
            delta_angle += compute_angular_difference_with_turn_direction(theta0, angle_to_be_reached, turn1)
            include_backward_maneuver = True

    xc1, yc1 = compute_center_coordinates_first_circle(x0, y0, theta0 + delta_angle, turn1, radius)

    return include_backward_maneuver, delta_angle, xc1, yc1


def compute_initial_turn_direction(xc2, yc2, R, x0, y0, theta0, turn):
    '''Compute the turn direction to reach a circumference with center xc2, yc2, with a tangential direction that is clockwise (right) if turn == -1,
    or counterclockwise (left) if turn == 1, starting from pose (x0, y0, theta0).

    :param xc2: x coordinate of the center of the circumference to be reached
    :type xc2: float
    :param yc2: y coordinate of the center of the circumference to be reached
    :type yc2: float
    :param R: radius of the circumference to be reached == radius of arc maneuver of the considered vehicle
    :type R: float
    :param x0: x coordinate of the start pose of the vehicle
    :type x0: float
    :param y0: y coordinate of the start pose of the vehicle
    :type y0: float
    :param theta0: initial orientation of the vehicle
    :type theta0: float
    :param turn: turn direction along the circumference to be reached
    :type turn: float [-1, 1]

    :return: initial turn direction. -1 if turn right, else 1
    :rtype: float [-1, 1]
    '''
    a = sqrt((xc2 - x0)**2 + (yc2 - y0)**2)
    if abs(R/a) > 1: 
        raise ValueError("Given point is inside the circle, it is not possible to compute the initial turn direction.")
    beta = asin(R/a)
    alpha0 = wrapPositiveAngle(atan2((yc2 - y0), (xc2 - x0)))
    # if (theta0 > alpha0 - turn * beta) and (theta0 < alpha0 + pi):
    if compute_angular_difference(theta0, alpha0 - turn * beta) < 0 and abs(compute_angular_difference(theta0, alpha0)) < pi:
        return -1 # turn right
    else:
        return 1 # turn left
    

def compute_extreme_poses_arc_line_two_radii_oo(circ1, circ2, tau1, tau2, overlap = False):
    '''
    Compute the points and orientation (x1, y1, theta1), (x2, y2, theta2) belonging to a line tangent 
    to two circumferences and to the two circumferences themselves. 

    :param circ1: first circumference
    :type circ1: Circle object
    :param circ2: second circumference
    :type circ2: Circle object
    :param tau1: turn direction along the first circumference
    :type tau1: float 
    :param tau2: turn direction along the second circumference
    :type tau2: float
    :param overlap: boolean indicating whether the two circumferences are overlapping
    :type overlap: bool

    :return: pose1 is the initial pose of the segment that is tangent to the two circumferences
    :rtype: Pose object 
    :return: pose2 is the final pose of the segment that is tangent to the two circumferences
    :rtype: Pose object
    '''
    try:
        # If the provided circles are touching (distance between the centers = 2R) and turn1 is different than turn2
        # if ((yc2 - yc1)**2 + (xc2 - xc1)**2 == (2 * R)**2) and (turn1 != turn2):
        if overlap and (tau1 != tau2):
            # Compute the point where the two circles are touching
            x1, y1, _, _ = circle_intersection(circ1.center.x, circ1.center.y, circ1.radius, circ2.center.x, circ2.center.y, circ2.radius)
            theta1 = wrapPositiveAngle(atan2(circ2.center.y - circ1.center.y, circ2.center.x - circ1.center.x) + tau1 * 0.5 * pi) 
            x2, y2 = x1, y1
        # If the distance between the two circles is > 2R or turn1 = turn2
        else: 
            xc1, yc1 = circ1.center.x, circ1.center.y
            xc2, yc2 = circ2.center.x, circ2.center.y
            R1, R2 = circ1.radius, circ2.radius

            zeta = - (tau1 + tau2) * pi * 0.25
            eta = (tau1 - tau2) * pi * 0.25
            a1 = sqrt((yc1 - yc2)**2 + (xc1 - xc2)**2)
            c1 = sqrt((a1)**2 - (R1 - tau1 * tau2 * R2)**2)
            alfa1 = wrapPositiveAngle(atan2((yc2 - yc1), (xc2 - xc1)))
            gamma1 = asin((R1 - R2)/a1) if tau1 * tau2 > 0 else asin((c1/a1))
            beta1 = alfa1 - tau1 * gamma1
            x1 = xc1 + R1 * cos(beta1 + zeta)
            y1 = yc1 + R1 * sin(beta1 + zeta)
            x2 = x1 + c1 * cos(beta1 + eta)
            y2 = y1 + c1 * sin(beta1 + eta)
            theta1 = wrapPositiveAngle(atan2((y2 - y1),(x2 - x1)))
            pose1 = Pose(position = Point(x = x1, y = y1), theta = theta1)
            pose2 = Pose(position = Point(x = x2, y = y2), theta = theta1)
        return pose1, pose2
    except ValueError: 
        print('Overlapping circles ERROR')
        return None, None, None, None, None, None
    

def compute_first_circle(x0, y0, theta0, tau, R):
    '''
    Compute the coordinates of the center of the first circumference. 
    Starting from the initial robot position (x0, y0), the center of the first circumference is placed along a direction 
    parallel to theta0, depending on the initial turn direction and at a distance R.

    :param x0: x coordinate start pose
    :type x0: float
    :param y0: y coordinate start pose
    :type y0: float
    :param theta0: initial orientation
    :type theta0: float
    :param turn: initial turn direction
    :type turn: float

    :return: first circumference
    :rtype: Circle object
    '''
    return Circle(center = Point(x = x0 + R * cos(theta0 + tau * 0.5 * pi),
                                 y = y0 + R * sin(theta0 + tau * 0.5 * pi)),
                  radius = R)


def compute_two_maneuvers_bicycle(start_pose, bicycle, circ2, tau2, t0 = 0, tau1 = 0):
    '''
    Compute the two maneuvers required to reach a circumference centered in (xc2, yc2): arc and segment.

    :param corridor: considered corridor
    :type corridor: Corridor
    :param start_pose: initial pose
    :type start_pose: list or np.ndarray
    :param bicycle: x coordinate of the center of the second circumference
    :type unicycle: Bicycle
    :param circ2: circumference to be reached
    :type xc2: Circle object
    :param tau2: turn direction along the second circumference
    :type tau2: float 
    :param t0: initial time
    :type t0: float
    :param tau1: turn direction along the first circumference
    :type tau1: float

    :return: primitive1, arc
    :rtype: CurvilinearArcUnicycle
    :return: primitive2, segment
    :rtype: LinearSegmentUnicycle 
    '''
    #Extract variables
    x0, y0, theta0  = start_pose
    r_nominal = bicycle.max_radius
    v_max = bicycle.v_max
    omega_max = bicycle.omega_max
    omega_min = bicycle.omega_min

    # If turn1 is not provided, compute it
    tau1 = compute_initial_turn_direction(circ2.center.x, circ2.center.y, r_nominal, x0, y0, theta0, tau2) if tau1 == 0 else tau1
    # Compute the coordinates of the first circumference
    circ1 = compute_first_circle(x0, y0, theta0, tau1, r_nominal)
    # Check for collision avoidance 
    # lw_intersected, rw_intersected, points_lw, points_rw = compute_intersection_points_between_line_circle_new(x0 = x0, y0 = y0, xc = xc1, yc = yc1, turn = tau1, radius = r_nominal, corridor = corridor, margin = bicycle.width*0.5) 
    # if lw_intersected:
    #     R1 = get_max_radius(corridor.shrink(bicycle.width*0.5), start_pose, tau1, wall = 'left')
    #     xc1, yc1 = compute_center_coordinates_first_circle(x0, y0, theta0, tau1, R1)
    # elif rw_intersected:
    #     R1 = get_max_radius(corridor.shrink(bicycle.width*0.5), start_pose, tau1, wall = 'right')
    #     xc1, yc1 = compute_center_coordinates_first_circle(x0, y0, theta0, tau1, R1)
    # else:
    #     R1 = r_nominal

    # R2 = r_nominal
    # # Compute the extreme poses for each maneuver
    # x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line_two_radii(xc1, yc1, xc2, yc2, tau1, tau2, R1, R2)
     
    # # Compute the extreme poses for each maneuver
    pose1, pose2 = compute_extreme_poses_arc_line_two_radii_oo(circ1, circ2, tau1, tau2)
    #Compute orientations for each primitive
    theta0_p1 = theta0
    thetaf_p1 = theta0_p1 + compute_angular_difference_with_turn_direction(theta0, pose1.theta, tau1)
    theta_p2 = thetaf_p1

    omega1 = omega_max if tau1 > 0 else omega_min

    # Primitive 1: arc
    primitive1 = CurvilinearArcUnicycle(xc=circ1.xc, yc=circ1.yc, x0 = x0, y0 = y0,
                                        theta0 = theta0_p1, xf = pose1.x, yf = pose1.y,
                                        thetaf = thetaf_p1, radius = circ1.radius,
                                        turn_direction = tau1, v = v_max,
                                        omega = omega1, unicycle = bicycle,
                                        t0 = t0, samples_number = 100)
    # Primitive 2: segment
    primitive2 = LinearSegmentUnicycle(x0=pose1.x, y0=pose1.y, xf=pose2.x, yf=pose2.y, theta=theta_p2, v=v_max, t0 = primitive1.tf, unicycle = bicycle, samples_number=10)

    return primitive1, primitive2


def backward_maneuver(corridor, pose, tau, R, wall, corner_point):
    '''
    Compute the backward maneuver in case of collision with one of the corridor's walls

    :param corridor: corridor
    :type corridor: CorridorWorld

    :param pose: initial pose
    :type pose: list of floats

    :param tau: turn direction
    :type tau: float

    :param R: radius of the arc
    :type R: float

    :param wall: wall causing the collision
    :type wall: int

    :return: final pose
    :rtype: Pose object
    '''
    # The corner point is used to select one of two possible circumferences
    x_corner, y_corner = corner_point.x, corner_point.y
    # Compute the rotation angle to obtain a vertical corridor (with tilt pi/2)
    # rot_angle = pi/2 - corridor.tilt
    # Compute the x,y and theta coordinates in the rotated corridor
    x0, y0, theta0 = absolute_to_relative_pose(corridor, [pose.x, pose.y, pose.theta])

    # Compute coordinates of the fixed circumference O1
    circ1 = compute_first_circle(x0, y0, theta0, -tau, R)
    xc1, yc1 = circ1.xc, circ1.yc

    # Compute x,y coordinate of the transformed corridor center: reminder that we consider a vertical corridor with center in (0, 0)
    # cx = cy = 0 

    # Compute the x coordinate of O2, depending on the wall that causes a collision
    xc2 = - wall * corridor.width * 0.5 + wall * R

    # Compute the y coordinate of O2
    a = 1
    b = -2*yc1
    c = xc2**2 + xc1**2 -2*xc2*xc1 + yc1**2 - 4*R**2
    discr = np.sqrt(b**2 - 4*a*c)
    yc2_sol1 = (-b + discr)/(2*a)
    yc2_sol2 = (-b - discr)/(2*a)

    # Pick the closest circumference to the robot
    # yc2 = yc2_sol1 if abs(yc2_sol1 - y0) < abs(yc2_sol2 - y0) else yc2_sol2
    yc2 = yc2_sol1 if abs(yc2_sol1 - y_corner) < abs(yc2_sol2 - y_corner) else yc2_sol2

    # Compute the relative end pose of the backward maneuver
    xb_rel, yb_rel = (xc2 + xc1) * 0.5, (yc2 + yc1) * 0.5
    thetab_rel = wrapPositiveAngle(atan2(yb_rel - yc1, xb_rel - xc1) - tau * 0.5 * pi)

    # Compute the absolute end pose of the backward maneuver
    absolute_pose = relative_to_absolute_pose(corridor, [xb_rel, yb_rel, thetab_rel])
    absolute_circle = relative_to_absolute_pose(corridor, [xc1, yc1, 0])
    final_bw_pose = Pose(position = Point(x = absolute_pose[0], y = absolute_pose[1]), theta = absolute_pose[2])
    bw_circle = Circle(center = Point(x = absolute_circle[0], y = absolute_circle[1]), radius = R)

    return final_bw_pose, bw_circle


def compute_backward_arc(corridor, pose, bicycle, tau, R, wall, corner_point):

    final_bw_pose, bw_circle = backward_maneuver(corridor.shrink(bicycle.width*0.5), pose, tau, R, wall, corner_point)
    final_bw_theta = pose.theta + compute_angular_difference_with_turn_direction(pose.theta, final_bw_pose.theta, -tau)
    backward_arc =BackwardArc(xc=bw_circle.xc, yc=bw_circle.yc, x0 = pose.x, y0 = pose.y,
                                        theta0 = pose.theta, xf = final_bw_pose.x, yf = final_bw_pose.y,
                                        thetaf = final_bw_theta, radius = R,
                                        turn_direction = -tau, v = -bicycle.v_max,
                                        omega = -tau * bicycle.omega_max, bicycle = bicycle,
                                        t0 = 0, samples_number = 100)
    return backward_arc


def compute_traj_to_circle_bicycle(corridor1, start_pose, bicycle, circ1, tau0 = 0):
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
    corner_point1 = circ1.corner_point

    # plot_corridors([corridor1])
    # plt.plot(start_pose[0], start_pose[1], 'ro')
    # plt.plot(circ1.xc, circ1.yc, 'bo')
    # plt.plot(circ1.xc + circ1.radius * np.cos(np.linspace(0, 2*pi, 100)), circ1.yc + circ1.radius * np.sin(np.linspace(0, 2*pi, 100)), 'b--')
    # plt.show(block = True)

    tau0 = compute_initial_turn_direction(
        circ1.xc,
        circ1.yc,
        circ1.radius,
        start_pose[0],
        start_pose[1],
        start_pose[2],
        tau1) if tau0 == 0 else tau0
    
    ## Compute the first two maneuvers from start pose to the second circumference
    collision_check = True
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
    
    # Check for collidions avoidance, if necessary add a backward maneuver
    while collision_check:
        collision_check, wall = collision_avoidance_check_bicycle(arc1, corridor1, bicycle, margin = 0)
        if collision_check:
            bw_arc = compute_backward_arc(corridor1, start_pose_object, bicycle, arc1.turn_direction, bicycle.max_radius, wall, corner_point1)
            start_pose_fw_drive = [bw_arc.xf, bw_arc.yf, bw_arc.thetaf]
            start_maneuvers.append(bw_arc)
            arc1, segment2 = compute_two_maneuvers_bicycle(start_pose_fw_drive, bicycle, circ1, tau1, t0 = 0, tau1 = tau0)
        else:
            start_maneuvers.append(arc1)
            start_maneuvers.append(segment2)
    
    # Adjust the time grid and angles
    for i in range(len(start_maneuvers)-1):
        if start_maneuvers[i+1].time_grid[0] != start_maneuvers[i].time_grid[-1]:
            start_maneuvers[i+1].add_time_offset(abs(start_maneuvers[i+1].time_grid[0] - start_maneuvers[i].time_grid[-1]))
    correct_angles(start_maneuvers)
    return start_maneuvers


def shift_circles_bicycle(
    intermediate_circles,
    intermediate_circles_choices,
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



def compute_trajectory_bicycle_multiple_corridors_optimal(
    corridor_list,
    start_pose,
    end_pose,
    bicycle,
    intermediate_circles_choices,
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
    intermediate_circles = selected_sequence_from_preferences(
        intermediate_circles_choices
    )
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

    # Compute the segments and prune the intermediate circles if necessary
    segments = compute_P_mid(intermediate_circles, bicycle)
    segments.insert(0,start_maneuvers[-1])
    segments.append(end_maneuvers[0])

    angle_array = np.linspace(0, 2*pi, 100)
    # figure = plot_corridors(corridor_list)
    # plot_analytical_trajectory(segments + start_maneuvers + end_maneuvers, figure)
    # # plt.plot(intermediate_circles[i].xc + intermediate_circles[i].radius * np.cos(angle_array), intermediate_circles[i].yc + intermediate_circles[i].radius * np.sin(angle_array), 'r--')
    # plt.show(block = True)
    ok = 1

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
    ) = shift_circles_bicycle(
    intermediate_circles,
    intermediate_circles_choices,
    segments,
    bicycle,
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
        # print('Segment ', i)
        # circle = intermediate_circles[i]
        # angle_array = np.linspace(0, 2*pi, 100)
        # figure = plot_corridors(corridor_list)
        # plot_analytical_trajectory([segments[i], segments[i+1]], figure)
        # plt.plot(circle.xc, circle.yc, 'ro')
        # plt.plot(circle.xc + circle.radius * np.cos(angle_array), circle.yc + circle.radius * np.sin(angle_array), 'r--')
        # plt.plot(intermediate_circles[i].xc + intermediate_circles[i].radius * np.cos(angle_array), intermediate_circles[i].yc + intermediate_circles[i].radius * np.sin(angle_array), 'r--')
        # plt.plot(intermediate_circles[i+1].xc + intermediate_circles[i+1].radius * np.cos(angle_array), intermediate_circles[i+1].yc + intermediate_circles[i+1].radius * np.sin(angle_array), 'r--')
        
        # plt.show(block = True)
        arcs.append(compute_arc_from_two_tangents_objects(segments[i], segments[i+1], intermediate_circles[i], bicycle))

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


def compute_trajectory_bicycle_two_corridors_optimal(corridor1, corridor2, start_pose, end_pose, bicycle, intermediate_circles_choices):
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
    intermediate_circles = selected_sequence_from_preferences(
        intermediate_circles_choices
    )
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