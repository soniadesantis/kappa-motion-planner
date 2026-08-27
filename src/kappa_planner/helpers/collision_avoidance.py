from math import asin, atan2, cos, pi, sin, sqrt, ceil

import sympy as sp
import numpy as np

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
    intersect_circle_line,
)
from .primitives import (
    compute_extreme_poses_arc_line,
    compute_central_angle
)
from .corridor_geometry import check_point_inside_corridor
from .poses import absolute_to_relative_pose

from ..geometry import Circle, Point


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
    one_point_left = len(int_points_left) == 1 # Check if there is only one intersection point (the circle is tangent and that is ok)
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


def allowed_maneuvers(
    start_pose,
    corridor,
    vehicle,
    tol=1e-9,
    max_amplitude=2.0 * pi,
):
    """
    Compute:

    1. Whether the left and right turning circles intersect the corridor's
       left, back, and right walls.
    2. The corresponding intersection points.
    3. The maximum allowed amplitudes for:
       - forward left
       - backward left
       - forward right
       - backward right

    The front wall is ignored.

    The robot is modeled as a circle of radius vehicle.width / 2, so wall
    checks are performed against the corridor shrunken by that radius.

    :return:
        {
            "left_circle": {
                "intersects": bool,
                "points": [[x, y], ...],
            },
            "right_circle": {
                "intersects": bool,
                "points": [[x, y], ...],
            },
            "amplitudes": {
                "forward_left": float,
                "backward_left": float,
                "forward_right": float,
                "backward_right": float,
            },
        }
    """
    x0, y0, theta0 = start_pose

    R = float(vehicle.max_radius)
    robot_radius = 0.5 * float(vehicle.width)

    if R <= 0.0:
        raise ValueError("vehicle.max_radius must be positive")

    if corridor.width <= 2.0 * robot_radius:
        raise ValueError("The corridor is too narrow for the robot")

    if corridor.height <= 2.0 * robot_radius:
        raise ValueError("The corridor is too short for the robot")

    effective_corridor = corridor.shrink(robot_radius)

    left_circle = Circle(
        Point(
            x0 + R * cos(theta0 + pi / 2.0),
            y0 + R * sin(theta0 + pi / 2.0),
        ),
        R,
    )

    right_circle = Circle(
        Point(
            x0 + R * cos(theta0 - pi / 2.0),
            y0 + R * sin(theta0 - pi / 2.0),
        ),
        R,
    )

    wall_indices = (
        effective_corridor.LFT,
        effective_corridor.BCK,
        effective_corridor.RGT,
    )

    def circle_wall_intersections(circle):
        points = []

        for wall_index in wall_indices:
            wall_start, wall_end = effective_corridor.get_edge_segment(
                wall_index
            )

            intersects, wall_points = intersect_circle_line(
                circle=circle,
                line_start=wall_start,
                line_end=wall_end,
                segment=True,
                tol=tol,
            )

            if intersects:
                points.extend(wall_points)

        # Remove duplicate points, which can occur at corridor corners.
        unique_points = []

        for point in points:
            if not any(
                (point[0] - existing[0]) ** 2
                + (point[1] - existing[1]) ** 2
                <= tol**2
                for existing in unique_points
            ):
                unique_points.append(point)

        return bool(unique_points), unique_points

    left_intersects, left_points = circle_wall_intersections(left_circle)
    right_intersects, right_points = circle_wall_intersections(right_circle)


    def maximum_amplitude(circle, points, angular_direction):
        """
        Find the first wall-intersection point reached while moving around
        the circle.

        angular_direction:
            +1 = counterclockwise
            -1 = clockwise
        """
        if not points:
            return max_amplitude

        start_angle = atan2(
            y0 - circle.yc,
            x0 - circle.xc,
        )

        minimum_amplitude = max_amplitude

        for px, py in points:
            point_angle = atan2(
                py - circle.yc,
                px - circle.xc,
            )

            if angular_direction == 1:
                amplitude = (point_angle - start_angle) % (2.0 * pi)
            else:
                amplitude = (start_angle - point_angle) % (2.0 * pi)

            # Ignore an intersection at the initial position.
            if amplitude <= tol:
                continue

            if amplitude < minimum_amplitude:
                minimum_amplitude = amplitude

        return minimum_amplitude

    amplitudes = {
        # Left circle:
        # forward left moves counterclockwise,
        # backward left moves clockwise.
        "forward_left": maximum_amplitude(
            left_circle,
            left_points,
            angular_direction=1,
        ),
        "backward_left": maximum_amplitude(
            left_circle,
            left_points,
            angular_direction=-1,
        ),

        # Right circle:
        # forward right moves clockwise,
        # backward right moves counterclockwise.
        "forward_right": maximum_amplitude(
            right_circle,
            right_points,
            angular_direction=-1,
        ),
        "backward_right": maximum_amplitude(
            right_circle,
            right_points,
            angular_direction=1,
        ),
    }

    return {
        "left_circle": {
            "intersects": left_intersects,
            "points": left_points,
        },
        "right_circle": {
            "intersects": right_intersects,
            "points": right_points,
        },
        "amplitudes": amplitudes,
    }


def directed_angular_distance(start_angle, end_angle, direction):
    """
    Return the positive angular distance from start_angle to end_angle.

    :param direction:
        +1 for counterclockwise motion
        -1 for clockwise motion
    """
    if direction == 1:
        return wrapPositiveAngle(end_angle - start_angle)

    if direction == -1:
        return wrapPositiveAngle(start_angle - end_angle)

    raise ValueError("direction must be either +1 or -1")


def check_arc_collision(arc, corridor, tol=1e-9):
    """
    Check whether a CurvilinearArcUnicycle or BackwardArc crosses a wall
    of the corridor shrunken by the robot footprint radius.

    Tangential contact with a wall is considered collision-free.

    :param arc: CurvilinearArcUnicycle or BackwardArc
    :param corridor: CorridorWorld
    :param tol: numerical tolerance

    :return:
        collision: True if the arc crosses a wall
        wall: index of the first crossed wall, or None
    :rtype: tuple[bool, int | None]
    """

    # Retrieve the vehicle stored in the trajectory primitive.
    vehicle = getattr(arc, "bicycle", None)
    if vehicle is None:
        vehicle = getattr(arc, "unicycle", None)

    if vehicle is None:
        raise ValueError(
            "The arc must contain either a 'bicycle' or 'unicycle' attribute"
        )

    # Circular robot footprint.
    robot_radius = 0.5 * vehicle.width
    effective_corridor = corridor.shrink(robot_radius)

    arc_circle = Circle(
        center=Point(arc.xc, arc.yc),
        radius=arc.radius,
    )

    # Direction followed around the geometric circle.
    #
    # Forward:
    #   left  -> counterclockwise
    #   right -> clockwise
    #
    # Backward motion reverses the angular direction.
    is_backward = getattr(arc, "label", "") == "backward arc"

    angular_direction = (
        -arc.turn_direction
        if is_backward
        else arc.turn_direction
    )

    start_angle = atan2(
        arc.y0 - arc.yc,
        arc.x0 - arc.xc,
    )

    first_collision_amplitude = float("inf")
    first_collision_wall = None

    walls = (
        effective_corridor.LFT,
        effective_corridor.RGT,
        effective_corridor.BCK,
        effective_corridor.FWD,
    )

    for wall in walls:
        wall_start, wall_end = effective_corridor.get_edge_segment(wall)

        intersects, points = intersect_circle_line(
            circle=arc_circle,
            line_start=wall_start,
            line_end=wall_end,
            segment=True,
            tol=tol,
        )

        if not intersects:
            continue

        # One intersection means that the arc circle is tangent to the wall.
        # Tangency to a shrunken wall is collision-free.
        if len(points) == 1:
            continue

        for px, py in points:
            point_angle = atan2(
                py - arc.yc,
                px - arc.xc,
            )

            if angular_direction > 0:
                amplitude = (point_angle - start_angle) % (2.0 * pi)
            else:
                amplitude = (start_angle - point_angle) % (2.0 * pi)

            # Ignore the initial point if it lies on a wall.
            if amplitude <= tol:
                continue

            if amplitude < first_collision_amplitude:
                first_collision_amplitude = amplitude
                first_collision_wall = wall

    # Reaching the wall exactly at the arc endpoint is permitted.
    collision = (
        first_collision_wall is not None
        and arc.iota > first_collision_amplitude + tol
    )

    if collision:
        return True, first_collision_wall

    return False, None


def compute_wall_tangent_circle_centers(
    corridor,
    wall,
    fixed_circle,
    radius,
    tol=1e-9,
):
    """
    Compute centers of radius-R circles that are:

    - externally tangent to ``fixed_circle``;
    - tangent to the selected corridor wall from the corridor interior.

    The corridor is assumed to already be shrunken by the robot radius.

    :return: list of candidate centers as Point objects
    """
    wall_start, wall_end = corridor.get_edge_segment(wall)

    wall_start = np.asarray(wall_start, dtype=float)
    wall_end = np.asarray(wall_end, dtype=float)

    wall_vector = wall_end - wall_start
    wall_length = np.linalg.norm(wall_vector)

    if wall_length <= tol:
        raise ValueError("The selected wall has zero length")

    wall_tangent = wall_vector / wall_length

    # Corridor stores outward normals; negate to obtain the inward normal.
    inward_normal = -np.asarray(
        corridor.outward_normals[wall],
        dtype=float,
    )

    fixed_center = np.array(
        [fixed_circle.xc, fixed_circle.yc],
        dtype=float,
    )

    # A point on the wall shifted inward by R.
    offset_line_point = wall_start + radius * inward_normal

    # Candidate center:
    # C(s) = offset_line_point + s * wall_tangent
    delta = offset_line_point - fixed_center

    # Solve ||delta + s*t||² = (2R)².
    b = 2.0 * np.dot(delta, wall_tangent)
    c = np.dot(delta, delta) - (2.0 * radius) ** 2

    discriminant = b * b - 4.0 * c

    if discriminant < -tol:
        return []

    if abs(discriminant) <= tol:
        s_values = [-0.5 * b]
    else:
        sqrt_discriminant = sqrt(max(0.0, discriminant))
        s_values = [
            0.5 * (-b + sqrt_discriminant),
            0.5 * (-b - sqrt_discriminant),
        ]

    candidate_centers = []

    for s in s_values:
        center = offset_line_point + s * wall_tangent

        # Point where the new circle touches the wall.
        wall_contact = center - radius * inward_normal

        # Require contact with the finite wall, not only its extension.
        wall_coordinate = np.dot(
            wall_contact - wall_start,
            wall_tangent,
        )

        if -tol <= wall_coordinate <= wall_length + tol:
            candidate_centers.append(
                Point(float(center[0]), float(center[1]))
            )

    return candidate_centers


def check_robot_footprint_inside_corridor_union(
    corridors,
    center,
    robot_radius,
    angular_samples=72,
    radial_layers=4,
):
    """
    Check whether a circular robot footprint is contained in the union
    of a collection of corridors.

    A footprint point is admissible when it belongs to at least one
    corridor. This permits the robot footprint to straddle two
    consecutive corridors near their intersection.

    The footprint disk is checked numerically using concentric radial
    layers.

    :param corridors: corridors defining the admissible region
    :type corridors: list[CorridorWorld]
    :param center: coordinates of the robot center
    :type center: array-like, shape (2,)
    :param robot_radius: radius of the circular robot footprint
    :type robot_radius: float
    :param angular_samples: samples on each radial layer
    :type angular_samples: int
    :param radial_layers: number of radial layers inside the footprint
    :type radial_layers: int

    :return: True if the footprint is contained in the corridor union
    :rtype: bool
    """
    center = np.asarray(center, dtype=float)

    def point_inside_union(point):
        return any(
            check_point_inside_corridor(corridor, point)
            for corridor in corridors
        )

    # The robot center must itself lie in the admissible union.
    if not point_inside_union(center):
        return False

    if robot_radius <= 0.0:
        return True

    angles = np.linspace(
        0.0,
        2.0 * pi,
        angular_samples,
        endpoint=False,
    )

    # Include interior radial layers, not only the footprint boundary.
    radii = np.linspace(
        robot_radius / radial_layers,
        robot_radius,
        radial_layers,
    )

    for radius in radii:
        for angle in angles:
            footprint_point = center + radius * np.array(
                [
                    cos(angle),
                    sin(angle),
                ]
            )

            if not point_inside_union(footprint_point):
                return False

    return True


def _get_arc_angular_direction(arc):
    """
    Return the direction followed around the geometric arc circle.

    The result is:
        +1 for counterclockwise motion,
        -1 for clockwise motion.
    """
    is_backward = getattr(arc, "label", "") == "backward arc"

    if is_backward:
        return -arc.turn_direction

    return arc.turn_direction


def _compute_point_on_arc(
    arc,
    start_angle,
    angular_direction,
    amplitude,
):
    """
    Compute the robot-center position after travelling the given
    angular amplitude along an arc.
    """
    angle = start_angle + angular_direction * amplitude

    return np.array(
        [
            arc.xc + arc.radius * cos(angle),
            arc.yc + arc.radius * sin(angle),
        ],
        dtype=float,
    )


def check_arc_collision_corridor_union(
    arc,
    corridors,
    max_spatial_step=None,
    angular_samples=72,
    radial_layers=4,
):
    """
    Check whether an arc leaves the union of multiple corridors.

    The robot is represented by a circular footprint. The footprint may
    occupy portions of different corridors simultaneously, which permits
    admissible motion through the wedge region created at the intersection
    of consecutive corridors.

    Tangential contact is accepted according to the tolerance used by
    ``check_point_inside_corridor``.

    :param arc: circular trajectory primitive
    :type arc: CurvilinearArcUnicycle or BackwardArc
    :param corridors: corridors defining the admissible region
    :type corridors: list[CorridorWorld]
    :param max_spatial_step: maximum distance between consecutive arc samples
    :type max_spatial_step: float or None
    :param angular_samples: angular samples for the robot footprint
    :type angular_samples: int
    :param radial_layers: radial samples for the robot footprint
    :type radial_layers: int

    :return:
        collision:
            True if a nonzero portion of the arc leaves the corridor union

        first_invalid_amplitude:
            first sampled angular amplitude outside the admissible region,
            or None when no collision occurs
    :rtype: tuple[bool, float | None]
    """
    vehicle = getattr(arc, "bicycle", None)

    if vehicle is None:
        vehicle = getattr(arc, "unicycle", None)

    if vehicle is None:
        raise ValueError(
            "The arc must contain either a 'bicycle' or "
            "'unicycle' attribute."
        )

    if not corridors:
        raise ValueError(
            "At least one corridor must be provided."
        )

    robot_radius = 0.5 * vehicle.width

    start_angle = atan2(
        arc.y0 - arc.yc,
        arc.x0 - arc.xc,
    )

    angular_direction = _get_arc_angular_direction(arc)

    arc_length = abs(arc.radius * arc.iota)

    if max_spatial_step is None:
        # Keep the trajectory sampling reasonably fine relative to both
        # the maneuver radius and the robot footprint.
        radius_based_step = 0.02 * arc.radius
        footprint_based_step = 0.25 * max(robot_radius, 1e-3)

        max_spatial_step = max(
            1e-3,
            min(
                radius_based_step,
                footprint_based_step,
            ),
        )

    number_of_intervals = max(
        1,
        int(ceil(arc_length / max_spatial_step)),
    )

    amplitudes = np.linspace(
        0.0,
        arc.iota,
        number_of_intervals + 1,
    )

    for amplitude in amplitudes:
        center = _compute_point_on_arc(
            arc,
            start_angle,
            angular_direction,
            amplitude,
        )

        footprint_inside = (
            check_robot_footprint_inside_corridor_union(
                corridors=corridors,
                center=center,
                robot_radius=robot_radius,
                angular_samples=angular_samples,
                radial_layers=radial_layers,
            )
        )

        if not footprint_inside:
            return True, float(amplitude)

    return False, None


def check_segment_collision_corridor_union(
    segment,
    corridors,
    max_spatial_step=None,
    angular_samples=72,
    radial_layers=4,
):
    """
    Check whether a straight trajectory segment leaves the union of
    multiple corridors.

    :param segment: straight trajectory primitive
    :type segment: LinearSegmentUnicycle
    :param corridors: corridors defining the admissible region
    :type corridors: list[CorridorWorld]
    :param max_spatial_step: maximum distance between samples
    :type max_spatial_step: float or None
    :param angular_samples: angular samples for the robot footprint
    :type angular_samples: int
    :param radial_layers: radial samples for the robot footprint
    :type radial_layers: int

    :return:
        collision:
            True if a portion of the segment leaves the corridor union

        first_invalid_fraction:
            normalized location along the segment at which the first
            invalid sample is detected
    :rtype: tuple[bool, float | None]
    """
    vehicle = getattr(segment, "bicycle", None)

    if vehicle is None:
        vehicle = getattr(segment, "unicycle", None)

    if vehicle is None:
        raise ValueError(
            "The segment must contain either a 'bicycle' or "
            "'unicycle' attribute."
        )

    if not corridors:
        raise ValueError(
            "At least one corridor must be provided."
        )

    robot_radius = 0.5 * vehicle.width

    start = np.array(
        [
            segment.x0,
            segment.y0,
        ],
        dtype=float,
    )

    end = np.array(
        [
            segment.xf,
            segment.yf,
        ],
        dtype=float,
    )

    displacement = end - start
    segment_length = np.linalg.norm(displacement)

    if max_spatial_step is None:
        max_spatial_step = max(
            1e-3,
            0.25 * max(robot_radius, 1e-3),
        )

    number_of_intervals = max(
        1,
        int(ceil(segment_length / max_spatial_step)),
    )

    fractions = np.linspace(
        0.0,
        1.0,
        number_of_intervals + 1,
    )

    for fraction in fractions:
        center = start + fraction * displacement

        footprint_inside = (
            check_robot_footprint_inside_corridor_union(
                corridors=corridors,
                center=center,
                robot_radius=robot_radius,
                angular_samples=angular_samples,
                radial_layers=radial_layers,
            )
        )

        if not footprint_inside:
            return True, float(fraction)

    return False, None
    