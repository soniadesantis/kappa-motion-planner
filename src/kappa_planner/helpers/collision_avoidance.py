from math import asin, atan2, cos, pi, sin

import sympy as sp

from .helper_functions import (
    compute_center_coordinates_first_circle,
)
from .geometry_operations import (
        compute_angular_difference_with_turn_direction,
        wrapPositiveAngle
)
from .intersections import (
    circles_overlap,
    compute_intersection_points_between_line_circle,
    compute_intersection_points_circle_segment,
    select_closest_intersection,
)
from .primitives import (
    compute_extreme_poses_arc_line,
    compute_central_angle
)
from .corridor_geometry import check_point_inside_corridor
from .poses import absolute_to_relative_pose


def get_max_radius(corridor, pose, tau, wall = 'left'):
    # Compute the rotation angle to obtain a vertical corridor (with tilt pi/2)
    # rot_angle = pi/2 - corridor.tilt
    # Compute the x,y and theta coordinates in the rotated corridor
    x0, y0, theta0 = absolute_to_relative_pose(corridor, pose)

    # Compute x,y coordinate of the transformed corridor center: reminder that we consider a vertical corridor with center in (0, 0)
    # cx = cy = 0 
    if wall == 'left':
        d = - corridor.width * 0.5
    else:
        d = corridor.width * 0.5

    r = sp.symbols('r')

    xc1 = x0 + r * sp.cos(theta0 + tau * pi * 0.5)
    yc1 = y0 + r * sp.sin(theta0 + tau * pi * 0.5)
    a = 1
    b = -2*yc1
    c = d**2 - 2*d*xc1 + xc1**2 + yc1**2 - r**2
    discriminant = b**2 - 4*a*c

    return float(abs(sp.solve(discriminant, r)[0]))


def get_theta_collision_avoidance(pose, xc2, yc2, corridor, R, turn, margin = 0, left_wall = False, right_wall = False):
    '''
    This helper function returns the vehicle heading such that the robot does not crash into the wall of the
    corridor. The corridor is first rotated, such that it is vertical (tilt of pi/2).

    :param pose: considered pose
    :type pose: list or np.ndarray
    :param xc2: x coordinate of the center of intermediate circle
    :type xc2: float
    :param yc2: y coordinate of the center of intermediate circle
    :type yc2: float
    :param corridor: considered corridor
    :type corridor: corridor
    :param R: radius of the circumference at which the robot moves at higher speed (R = vmax/omegamax)
    :type R: float
    :param turn: initial turn direction
    :type turn: float [-1, 1]
    :param margin: additional margin to avoid collision with the corridor's walls
    :type margin: float 

    :return: x coordinate of the center of the first circumference
    :rtype: float 
    :return: y coordinate of the center of the first circumference
    :rtype: float S
    :return: initial orientation such that the robot does not collide with the corridor's walls
    :rtype: float 
    '''
    width = corridor.width * 0.5 - margin
    xc1 = pose[0] + R * cos(pose[2] + turn * pi * 0.5)
    yc1 = pose[1] + R * sin(pose[2] + turn * pi * 0.5)
    # Compute whether there is an intersection between the current osculating circle and one of the two walls of the corridor

    # left_wall, right_wall, _, _ = compute_intersection_points_between_line_circle(xc = xc1, yc = yc1, x0 = pose[0], y0 = pose[1], turn = turn, radius = R, corridor = corridor, margin = margin)
    if not(left_wall) and not(right_wall):
        return xc1, yc1, pose[2]
    elif left_wall:
        # Compute the rotation angle to obtain a vertical corridor (with tilt pi/2)
        rot_angle = pi*0.5 - corridor.tilt
        # Compute the x,y and theta coordinates in the rotated corridor
        x_tilted, y_tilted, theta_tilted = absolute_to_relative_pose(corridor, pose)
        # Compute x,y coordinate of the transformed corridor center: reminder that we consider a vertical corridor with center in (0, 0)
        cx = 0 # = cy

        # Initialize the two possible theta 
        theta_left_wall1, theta_left_wall2 = theta_tilted, theta_tilted

        ######################################################
        ### Case 1: Left side of the corridor, turn left
        ######################################################
        if x_tilted < cx and turn == 1:
            ### Left wall  
            asin_argument_left_wall = (R - width + (cx-x_tilted))/R
            if abs(asin_argument_left_wall) < 1:
                alpha = asin(asin_argument_left_wall)
                theta_left_wall1 = pi + alpha
                theta_left_wall2 = wrapPositiveAngle(-alpha)

        ######################################################
        ### Case 2: Left side of the corridor, turn right
        ######################################################
        elif x_tilted < cx and turn == -1:
            ### Left wall  
            asin_argument_left_wall = (R - width + (cx-x_tilted))/R
            if abs(asin_argument_left_wall) < 1:
                alpha = asin(asin_argument_left_wall)
                theta_left_wall1 = pi - alpha
                theta_left_wall2 = alpha

        ######################################################
        ### Case 3: Right side of the corridor, turn left
        ######################################################
        elif x_tilted >= cx and turn == 1:
            ### Left wall  
            asin_argument_left_wall = (R - width + (cx-x_tilted))/R
            if abs(asin_argument_left_wall) < 1:
                alpha = asin(asin_argument_left_wall)
                theta_left_wall1 =  wrapPositiveAngle(-alpha)
                theta_left_wall2 = pi + alpha
        
        ######################################################
        ### Case 4: Right side of the corridor, turn right
        ######################################################
        elif x_tilted >= cx and turn == -1:
            ### Left wall  
            asin_argument_left_wall = (R - width + (cx-x_tilted))/R
            if abs(asin_argument_left_wall) < 1:
                alpha = asin(asin_argument_left_wall)
                theta_left_wall1 = alpha 
                theta_left_wall2 = pi - alpha

        ### Select the best theta
        # Get back to actual corridor coordinates
        theta_transformed1 = theta_left_wall1 - rot_angle
        theta_transformed2 = theta_left_wall2 - rot_angle

        # Compute the center of the osculating circle
        xc1_theta_left_wall1 = pose[0] + R * cos(theta_transformed1 + turn * pi * 0.5)
        yc1_theta_left_wall1 = pose[1] + R * sin(theta_transformed1 + turn * pi * 0.5)
        # Compute the center of the osculating circle
        xc1_theta_left_wall2 = pose[0] + R * cos(theta_transformed2 + turn * pi * 0.5)
        yc1_theta_left_wall2 = pose[1] + R * sin(theta_transformed2 + turn * pi * 0.5)

        distance1 = (xc1_theta_left_wall1 - xc2)**2 + (yc1_theta_left_wall1 - yc2)**2
        distance2 = (xc1_theta_left_wall2 - xc2)**2 + (yc1_theta_left_wall2 - yc2)**2

        if distance1 < distance2:
            theta_transformed = theta_transformed1
            xc1 = xc1_theta_left_wall1
            yc1 = yc1_theta_left_wall1
        else:
            theta_transformed = theta_transformed2
            xc1 = xc1_theta_left_wall2
            yc1 = yc1_theta_left_wall2

    else:
        # Compute the rotation angle to obtain a vertical corridor (with tilt pi/2)
        rot_angle = pi*0.5 - corridor.tilt
        # Compute the x,y and theta coordinates in the rotated corridor
        x_tilted, y_tilted, theta_tilted = absolute_to_relative_pose(corridor, pose)
        # Compute x,y coordinate of the transformed corridor center: reminder that we consider a vertical corridor with center in (0, 0)
        cx = cy = 0 

        # Initialize the two possible theta 

        theta_right_wall1, theta_right_wall2 = theta_tilted, theta_tilted

        ######################################################
        ### Case 1: Left side of the corridor, turn left
        ######################################################
        if x_tilted < cx and turn == 1:
            ### Right wall  
            asin_argument_right_wall = (R - width - (cx-x_tilted))/R
            if abs(asin_argument_right_wall) < 1:
                alpha = asin(asin_argument_right_wall)
                theta_right_wall1 = pi - alpha
                theta_right_wall2 = alpha

        ######################################################
        ### Case 2: Left side of the corridor, turn right
        ######################################################
        elif x_tilted < cx and turn == -1:
            ### Right wall  
            asin_argument_right_wall = (R - width - (cx-x_tilted))/R

            if abs(asin_argument_right_wall) < 1:
                alpha = asin(asin_argument_right_wall)
                theta_right_wall1 = pi + alpha
                theta_right_wall2 = wrapPositiveAngle(-alpha)

        ######################################################
        ### Case 3: Right side of the corridor, turn left
        ######################################################
        elif x_tilted >= cx and turn == 1:
            ### Right wall  
            asin_argument_right_wall = (R - width - (cx-x_tilted))/R
            if abs(asin_argument_right_wall) < 1:
                alpha = asin(asin_argument_right_wall)
                theta_right_wall1 = alpha
                theta_right_wall2 = pi - alpha

        ######################################################
        ### Case 4: Right side of the corridor, turn right
        ######################################################
        elif x_tilted >= cx and turn == -1:
            ### Right wall  
            asin_argument_right_wall = (R - width - (cx-x_tilted))/R
            if abs(asin_argument_right_wall) < 1:
                alpha = asin(asin_argument_right_wall)
                theta_right_wall1 = wrapPositiveAngle(-alpha) 
                theta_right_wall2 = pi + alpha

        ### Select the best theta
        # Get back to actual corridor coordinates
        theta_transformed1 = theta_right_wall1 - rot_angle
        theta_transformed2 = theta_right_wall2 - rot_angle

        # Compute the center of the osculating circle
        xc1_theta_right_wall1 = pose[0] + R * cos(theta_transformed1 + turn * pi * 0.5)
        yc1_theta_right_wall1 = pose[1] + R * sin(theta_transformed1 + turn * pi * 0.5)
        # Compute the center of the osculating circle
        xc1_theta_right_wall2 = pose[0] + R * cos(theta_transformed2 + turn * pi * 0.5)
        yc1_theta_right_wall2 = pose[1] + R * sin(theta_transformed2 + turn * pi * 0.5)

        distance1 = (xc1_theta_right_wall1 - xc2)**2 + (yc1_theta_right_wall1 - yc2)**2
        distance2 = (xc1_theta_right_wall2 - xc2)**2 + (yc1_theta_right_wall2 - yc2)**2

        if distance1 < distance2:
            theta_transformed = theta_transformed1
            xc1 = xc1_theta_right_wall1
            yc1 = yc1_theta_right_wall1
        else:
            theta_transformed = theta_transformed2
            xc1 = xc1_theta_right_wall2
            yc1 = yc1_theta_right_wall2
        
    return xc1, yc1, theta_transformed


def collision_avoidance_check(start_pose, turn1, turn2, xc1, yc1, R, xc2, yc2, corridor, corridor2, unicycle):
    delta_angle = 0
    overlap = False
    overlap_tol = 1e-3

    # Initial overlap check
    if turn1 != turn2 and circles_overlap(xc1, yc1, xc2, yc2, R, tol=overlap_tol):
        overlap = True
    else:
        x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(
            xc1, yc1, xc2, yc2, turn1, turn2, R
            )
        x0, y0, theta0 = start_pose

        ## 2- Compute whether there are intersection points between the first circumference and the right and left walls of the corridor.
        left_wall_int, right_wall_int, int_points_left_wall, int_points_right_wall = (
            compute_intersection_points_between_line_circle(
                xc=xc1,
                yc=yc1,
                radius=R,
                corridor=corridor,
                margin=unicycle.width * 0.5,
            )
        )  

        if left_wall_int and int_points_left_wall:
            int_points_left_wall = select_closest_intersection(x0, y0, int_points_left_wall)
            iota_primitive = compute_central_angle(
                x0=x0, y0=y0, xf=x1, yf=y1, xc1=xc1, yc1=yc1, turn=turn1, radius=R
            )
            if not check_point_inside_corridor(corridor2, int_points_left_wall[0]):
                iota = compute_central_angle(
                    x0=x0, y0=y0,
                    xf=int_points_left_wall[0][0], yf=int_points_left_wall[0][1],
                    xc1=xc1, yc1=yc1, turn=turn1, radius=R
                )
                if iota <= iota_primitive:
                    xc1, yc1, theta0_2 = get_theta_collision_avoidance(
                        [x0, y0, theta0], xc2, yc2, corridor, R, turn1,
                        margin=unicycle.width * 0.5,
                        left_wall=True, right_wall=False
                    )
                    delta_angle = atan2(sin(theta0_2 - theta0), cos(theta0_2 - theta0))

        if right_wall_int and int_points_right_wall:
            int_points_right_wall = select_closest_intersection(x0, y0, int_points_right_wall)
            iota_primitive = compute_central_angle(
                x0=x0, y0=y0, xf=x1, yf=y1, xc1=xc1, yc1=yc1, turn=turn1, radius=R
            )
            if not check_point_inside_corridor(corridor2, int_points_right_wall[0]):
                iota = compute_central_angle(
                    x0=x0, y0=y0,
                    xf=int_points_right_wall[0][0], yf=int_points_right_wall[0][1],
                    xc1=xc1, yc1=yc1, turn=turn1, radius=R
                )
                if iota <= iota_primitive:
                    xc1, yc1, theta0_2 = get_theta_collision_avoidance(
                        [x0, y0, theta0], xc2, yc2, corridor, R, turn1,
                        margin=unicycle.width * 0.5,
                        left_wall=False, right_wall=True
                    )
                    delta_angle = atan2(sin(theta0_2 - theta0), cos(theta0_2 - theta0))

    # 4. Final overlap check after possible update
    if turn1 != turn2 and circles_overlap(xc1, yc1, xc2, yc2, R):
        overlap = True
        
    return delta_angle, xc1, yc1, overlap


def collision_avoidance_check_tb(start_pose, turn1, turn2, xc1, yc1, R, xc2, yc2, corridor, corridor2):
    delta_angle_tb = 0
    corners = corridor.get_corners()
    x1, y1 = corners[1]
    x2, y2 = corners[2] 

    int_points = compute_intersection_points_circle_segment(x1, y1, x2, y2, xc1, yc1, R, tol=1e-5)
    if int_points: 
            x0,y0,theta0 = start_pose
            int_point = select_closest_intersection(x0, y0, int_points)
            x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(
            xc1, yc1, xc2, yc2, turn1, turn2, R
            )
            if not check_point_inside_corridor(corridor2, int_point[0]):
                iota_primitive = compute_central_angle(
                    x0=x0, y0=y0, xf=x1, yf=y1, xc1=xc1, yc1=yc1, turn=turn1, radius=R
                    )
                iota = compute_central_angle(
                    x0=x0, y0=y0,
                    xf=int_point[0][0], yf=int_point[0][1],
                    xc1=xc1, yc1=yc1, turn=turn1, radius=R
                )
                if iota <= iota_primitive:
                    delta_angle1 = compute_angular_difference_with_turn_direction(theta0, corridor.tilt + pi/2, turn1)
                    delta_angle2 = compute_angular_difference_with_turn_direction(theta0, corridor.tilt - pi/2, turn1)
                    if abs(delta_angle1) < abs(delta_angle2):
                        delta_angle_tb = delta_angle1
                    else:                         
                        delta_angle_tb = delta_angle2

                    xc1, yc1 = compute_center_coordinates_first_circle(x0, y0, theta0 + delta_angle_tb, turn1, R)

    return delta_angle_tb, xc1, yc1


def collision_avoidance_check_after_overlap(start_pose, turn1, turn2, xc1, yc1, R, xc2, yc2, corridor, corridor2, unicycle):
    delta_angle, omega_turn_on_the_spot = 0, 0
    
    # 1- Check whether the two circles already overlap. This is a problem only if the turn directions are not the same
    # if turn1 != turn2 and (abs((xc2 - xc1)**2 + (yc2 - yc1)**2 - 4 * R**2) < 1e-3):
    #     overlap = True
    # else:
    x1, y1 = (xc1 + xc2) * 0.5, (yc1 + yc2) * 0.5
    x0, y0, theta0 = start_pose
    overlap = False
    ## 2- Compute whether there are intersection points between the first circumference and the right and left walls of the corridor.
    left_wall_int, right_wall_int, int_points_left_wall, int_points_right_wall = compute_intersection_points_between_line_circle(xc = xc1, yc = yc1, radius = R, corridor = corridor, margin = unicycle.width/2) 
    
    ## 3.a- If the collision with the left wall of the corridor is detected
    if left_wall_int:
        # Compute the angle of the arc that is going to be performed
        iota_primitive = compute_central_angle(x0 = x0, y0 = y0, xf = x1, yf = y1, xc1 = xc1, yc1 = yc1, turn = turn1, radius = R)
        # Compute the angle of the arc between the start point of the robot and the intersection points with the left wall
        for i in range(len(int_points_left_wall)):
            if not(check_point_inside_corridor(corridor2, int_points_left_wall[i])):
                iota = compute_central_angle(x0 = x0, y0 = y0, xf = int_points_left_wall[i][0], yf = int_points_left_wall[i][1], xc1 = xc1, yc1 = yc1, turn = turn1, radius = R)
                # If the angle of the arc between the start point of the robot and the intersection point is less than the primitive angle, a collision actually occurs
                if iota <= iota_primitive:
                    xc1, yc1, theta0_2 = get_theta_collision_avoidance([x0, y0, theta0], xc2, yc2, corridor, R, turn1, margin = unicycle.width * 0.5)
                    delta_angle += atan2(sin(theta0_2 - theta0), cos(theta0_2 - theta0)) # add rotation 
                    omega_turn_on_the_spot = unicycle.omega_max if delta_angle > 0 else unicycle.omega_min
        
    # 3.b- If the collision with the right wall of the corridor is detected
    if right_wall_int:
        # Compute the angle of the arc that is going to be performed
        iota_primitive = compute_central_angle(x0 = x0, y0 = y0, xf = x1, yf = y1, xc1 = xc1, yc1 = yc1, turn = turn1, radius = R)
        for i in range(len(int_points_right_wall)):
            if not(check_point_inside_corridor(corridor2, int_points_right_wall[i])):
                # Compute the angle of the arc between the start point of the robot and the intersection points with the left wall      
                iota = compute_central_angle(x0 = x0, y0 = y0, xf = int_points_right_wall[i][0], yf = int_points_right_wall[i][1], xc1 = xc1, yc1 = yc1, turn = turn1, radius = R)
                # If the angle of the arc between the start point of the robot and the intersection point is less than the primitive angle, a collision actually occurs
                if iota <= iota_primitive:
                    xc1, yc1, theta0_2 = get_theta_collision_avoidance([x0, y0, theta0], xc2, yc2, corridor, R, turn1, margin = unicycle.width * 0.5)
                    delta_angle += atan2(sin(theta0_2 - theta0), cos(theta0_2 - theta0)) # add rotation 
                    omega_turn_on_the_spot = unicycle.omega_max if delta_angle > 0 else unicycle.omega_min
        
    # 4- Check whether in the end, the two circles overlap. This is a problem only if the turn directions are not the same
    if turn1 != turn2 and ((xc2 - xc1)**2 + (yc2 - yc1)**2 - 4 * R**2 <= 1e-3):
        overlap = True
        
    return delta_angle, omega_turn_on_the_spot, xc1, yc1, overlap


def collision_avoidance_check_bicycle(arc, corridor, bicycle, margin = 0):
    
    x0, y0, xc, yc, tau, radius, margin = arc.x0, arc.y0, arc.xc, arc.yc, arc.turn_direction, arc.radius, bicycle.width*0.5 + margin
    ## 2- Compute whether there are intersection points between the first circumference and the right and left walls of the corridor.
    left_int, right_int, int_points_left, int_points_right = compute_intersection_points_between_line_circle(xc = xc,
                                                                                                                yc = yc,
                                                                                                                 radius = radius,
                                                                                                                 corridor = corridor,
                                                                                                                 margin = margin) 
    iota_primitive = arc.iota # Extract the central angle of the arc primitive
    one_point_left = len(int_points_left) == 1 # Check if there is only one intersection point
    ## 3.a- If a collision with the left wall of the corridor is detected (left_int = True and there's more than one intersection point)
    if left_int and not(one_point_left):
        # Compute the angle of the arc between the start point of the robot and the intersection points with the left wall
        for i in range(len(int_points_left)):
            iota_wall = compute_central_angle(x0 = x0, y0 = y0, xf = int_points_left[i][0], yf = int_points_left[i][1], xc1 = xc, yc1 = yc, turn = tau, radius = radius)
            # If the angle of the arc between the start point of the robot and the intersection point is less than the primitive angle, a collision actually occurs
            if iota_wall <= iota_primitive:
                return True, 1

    one_point_right = len(int_points_right) == 1 # Check if there is only one intersection point
    # 3.b- If the collision with the right wall of the corridor is detected (right_int = True and there's more than one intersection point)
    if right_int and not(one_point_right):
        for i in range(len(int_points_right)):
            # Compute the angle of the arc between the start point of the robot and the intersection points with the left wall      
            iota_wall = compute_central_angle(x0 = x0, y0 = y0, xf = int_points_right[i][0], yf = int_points_right[i][1], xc1 = xc, yc1 = yc, turn = tau, radius = radius)
            # If the angle of the arc between the start point of the robot and the intersection point is less than the primitive angle, a collision actually occurs
            if iota_wall <=  iota_primitive:
                return True, -1

    return False, 0