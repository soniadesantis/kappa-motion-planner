import numpy as np
from math import atan2, floor, ceil, cos, sin, pi, sqrt, asin, acos, inf
from time import perf_counter
import contextlib, time
import sympy as sp

class Timer: 
    def __enter__(self):
        self.start = perf_counter()
        self.end = 0.0
        return lambda: self.end - self.start 
    def __exit__(self, *args):
        self.end = perf_counter()

@contextlib.contextmanager
def measure_elapsed_time(identifier, total_time = 0):
    start_time = time.time_ns()
    yield
    elapsed = (time.time_ns() - start_time)/1000
    print(f"{identifier}: {elapsed} us")
    total_time = elapsed


def get_vehicle_vertices_no_casadi(x, y, theta, w_left, w_right, l_front, l_back):
    
    cos_theta, sin_theta = cos(theta), sin(theta)
    
    # vertices = np.array( 
    #     [[-l_back, l_front, l_front, -l_back],
    #     [-w_right, -w_right, w_left, w_left],
    #     [0, 0, 0, 0],
    #     [1, 1, 1, 1]])
    # # Homogeneous transformation matrix
    
    # homog_transf_matrix = np.array([[cos_theta, -sin_theta, 0, x], 
    #                                 [sin_theta,  cos_theta, 0, y],
    #                                 [0, 0, 1, 0],
    #                                 [0, 0, 0, 1]])

    # # Transform vertices and extract 2D points
    # return (homog_transf_matrix @ vertices)[0:2, :]

    return [[-l_back*cos_theta + w_right*sin_theta + x, l_front*cos_theta + w_right*sin_theta + x, l_front*cos_theta - w_left*sin_theta + x, -l_back*cos_theta - w_left*sin_theta + x], 
                    [-l_back*sin_theta - w_right*cos_theta + y, l_front*sin_theta - w_right*cos_theta + y, l_front*sin_theta + w_left*cos_theta + y, -l_back*sin_theta + w_left*cos_theta + y]]

def get_vehicle_vertices(x, y, theta, w_left, w_right, l_front, l_back):
    '''Get all vertices of a vehicle, using following convention::


        (4)       y     (3)   ^
        .|--------|------|    |  w_left
        .|        o---x  |    x
        .|               |    |
        .|---------------|    |  w_right
        (1)             (2)   v
        . <-------><---->
        .  l_back   l_front


    :param x, y: center point of vehicle
    :type x, y: cs.MX

    :param theta: angle wrt x-axis
    :type theta: cs.MX

    :param w_left: width of vehicle, to the left of center
    :type w_left: float

    :param w_right: width of vehicle, to the right of center
    :type w_right: float

    :param l_front: length of vehicle, front to center
    :type l_front: float

    :param l_back: length of vehicle, back to center
    :type l_back: float

    :return: points - matrix with all vertices
    :rtype: np.ndarray
    '''

    import casadi as cs
    # Vertices of vehicle in 3D homogeneous coordinates
    vertices = cs.vertcat(
        cs.horzcat(-l_back, l_front, l_front, -l_back),
        cs.horzcat(-w_right, -w_right, w_left, w_left),
        cs.horzcat(0, 0, 0, 0),
        cs.horzcat(1, 1, 1, 1))

    # Homogeneous transformation matrix
    homog_transf_matrix = cs.vertcat(
        cs.horzcat(cs.cos(theta), -cs.sin(theta), 0, x),
        cs.horzcat(cs.sin(theta),  cs.cos(theta), 0, y),
        cs.horzcat(0, 0, 1, 0),
        cs.horzcat(0, 0, 0, 1))

    # import pdb; pdb.set_trace()
    # Transform vertices and extract 2D points
    return (homog_transf_matrix @ vertices)[0:2, :]

def compute_path_coordinates_curvilinear_arc(xc, yc, x0, y0, x1, y1, radius, turn_direction, samples_number):
    t = efficient_sign(turn_direction)
    epsilon = wrapPositiveAngle(atan2((y0-yc),(x0-xc)))
    chord = sqrt((x1 - x0)**2+(y1- y0)**2) 
    iota = 2 * asin((0.5*chord)/radius)
    arc_angles_samples = np.linspace(epsilon, epsilon + t * iota, samples_number)
    arc_x_coordinates = xc + radius * np.cos(arc_angles_samples)
    arc_y_coordinates = yc + radius * np.sin(arc_angles_samples)
    path_coordinates = np.column_stack([arc_x_coordinates, arc_y_coordinates])
    return path_coordinates, chord, iota, epsilon


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
    a = sqrt((xc2 - x0)**2 + (yc2 - y0)**2) # distance between start point and circumference center
    beta = asin(R/a)
    alpha0 = wrapPositiveAngle(atan2((yc2 - y0), (xc2 - x0)))
    # if (theta0 > alpha0 - turn * beta) and (theta0 < alpha0 + pi):
    if compute_angular_difference(theta0, alpha0 - turn * beta) < 0 and abs(compute_angular_difference(theta0, alpha0)) < pi:
        return -1 # turn right
    else:
        return 1 # turn left
    
def compute_center_coordinates_second_circle_R_smaller_than_r(corner_point, turn, R, vehicle_width, margin, tilt1, tilt2):
    '''
    Compute the coordinates of the center of a circumference placed at the intersection between two subsequent corridors (intermediate circumference).
    The center is placed at a distance equal to (R - half of vehicle_width - margin) along the bisector between two edges of the two corridors.
    The selected edges depend on the main turn direction between the two corridors. 

    :param corner_point: selected corner point at the intersection between two corridors
    :type corner_point: list of floats or np.ndarray
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

    :return: x coordinate of the center of the circumference to be reached
    :rtype: float
    :return: y coordinate of the center of the circumference to be reached
    :rtype: float
    '''
    angle_circle_center_direction = tilt2 + turn * 0.5 * (pi - abs(compute_angular_difference(tilt2, tilt1)))
    # xc2 = corner_point[0] - (0.5 * vehicle_width + margin + R) * cos(angle_circle_center_direction)
    # yc2 = corner_point[1] - (0.5 * vehicle_width + margin + R) * sin(angle_circle_center_direction)
    xc2 = corner_point[0] - (0.5 * vehicle_width + R) * cos(angle_circle_center_direction)
    yc2 = corner_point[1] - (0.5 * vehicle_width + R) * sin(angle_circle_center_direction)
    return xc2, yc2
    

def get_bisector_direction(corridor1, corridor2):
    turn = corridor1.compute_relative_turn_direction(corridor2)
    tilt1 = corridor1.tilt
    tilt2 = corridor2.tilt

    if tilt1 == tilt2: 
        # tilt2 = tilt1 + turn * 0.5 * pi
        tilt1 = tilt2 - turn * 0.5 * pi
        angle_circle_center_direction = tilt2 + turn * 0.5 * (pi - abs(compute_angular_difference(tilt2, tilt1)))
    
    else:
        angle_circle_center_direction = tilt2 + turn * 0.5 * (pi - abs(compute_angular_difference(tilt2, tilt1)))
    return angle_circle_center_direction

def compute_center_coordinates_first_circle(x0, y0, theta0, turn, R):
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

    :return: x coordinate of the center of the first circumference
    :rtype: float
    :return: y coordinate of the center of the first circumference
    :rtype: float
    '''
    xc1 = x0 + R * cos(theta0 + turn * 0.5 * pi)
    yc1 = y0 + R * sin(theta0 + turn * 0.5 * pi)
    return xc1, yc1


def fit_new_circle(x0, y0, xc2, yc2, R, turn1, alpha0):
    '''
    In case an overlap between the first circle and the second circle is detected,
    a new circle is computed. 
    :param x0: x coordinate of the start pose
    :type x0: float
    :param y0: y coordinate of the start pose
    :type y0: float
    :param xc2: x coordinate of the center of the second circle
    :type xc2: float
    :param yc2: y coordinate of the center of the second circle
    :type yc2: float
    :param R: radius of the circles
    :type R: float
    :param turn1: initial turn direction
    :type turn1: float
    :param alpha0: direction between (x0, y0) and (xc2, yc2)
    :type alpha0: float

    :return: x coordinate of new circle
    :rtype: float
    :return: y coordinate of the new circle
    :rtype: float
    '''
    
    xint1, yint1, xint2, yint2 = circle_intersection(x0, y0, R, xc2, yc2, 2 * R)
    vector_alpha0 = [cos(alpha0), sin(alpha0)]
    angle1 = wrapPositiveAngle(atan2((yint1 - y0), (xint1 - x0)))
    angle2 = wrapPositiveAngle(atan2((yint2 - y0), (xint2 - x0)))

    vector1 = [cos(angle1), sin(angle1)]
    vector2 = [cos(angle2), sin(angle2)]

    turn_point1 = compute_turn_direction(vector_alpha0, vector1)
    turn_point2 = compute_turn_direction(vector_alpha0, vector2)

    if turn_point1 == turn1: 
        return xint1, yint1
    else: 
        return xint2, yint2

def circ_center_narrow_corridors(corridor2_tilt, corner_point, R, r, turn):
    xc2 = corner_point[0] + turn * cos(corridor2_tilt + turn * pi * 0.5) * (R-r)
    yc2 = corner_point[1] + turn * sin(corridor2_tilt + turn * pi * 0.5) * (R-r)
    return xc2, yc2
    

def compute_motion_time_with_orientation(x0, y0, xc1, yc1, xc2, yc2, xt, yt, R, tau1, tau2, omega, v):
    # Compute alpha0 and beta
    a = sqrt((xc2 - x0)**2 + (yc2 - y0)**2)
    beta = asin(R/a)
    alpha0 = wrapPositiveAngle(atan2((yc2 - y0), (xc2 - x0)))

    csi = - (tau1 + tau2)/2 * pi * 0.5
    eta = (tau1 - tau2)/2 * pi * 0.5
    a = sqrt((yc2 - yc1)**2 + (xc2 - xc1)**2)
    alpha = atan2(yc2 - yc1, xc2 - xc1)
    c = sqrt(a**2 - (R - tau1 * R)**2)
    if tau1 * tau2 > 0:
        gamma = asin((R - tau1 * R)/a)  
    else:
        gamma = asin(c/a)
    beta1 = alpha - tau1 * gamma
    x1 = xc1 + R * cos(beta1 + csi)
    y1 = yc1 + R * sin(beta1 + csi)
    x2 = x1 + c * cos(beta1 + eta)
    y2 = y1 + c * sin(beta1 + eta)

    # Compute motion time for angle of separation alpha0 - tau2 * beta
    delta_theta_up = 3 * pi * 0.5 + beta
    Ttots_up = delta_theta_up / omega

    Ta1_up = pi * 0.5 / omega

    Ts_up = c / v

    chord = sqrt((xt - x2)**2 + (yt - y2)**2)
    Ta2_up = 2 * asin(chord * 0.5 / R) / omega

    # Compute motion time for angle of separation alpha0 - tau2 * beta
    delta_theta_low = pi * 0.5 + beta
    Ttots_low = delta_theta_low / omega

    Ta1_low = Ta1_up

    Ts_low = Ts_up

    Ta2_low = Ta2_up

    # Compute the two points
    y1 = Ttots_up + Ta1_up + Ts_up + Ta2_up
    x1 = alpha0 - tau2 * beta

    y2 = Ttots_low + Ta1_low + Ts_low + Ta2_low
    x2 = alpha0 + pi - tau2 * beta
    return x1, y1, x2, y2


def compute_extreme_poses_arc_line_two_radii(xc1, yc1, xc2, yc2, turn1, turn2, R1, R2, overlap = False):
    '''
    Compute the points and orientation (x1, y1, theta1), (x2, y2, theta2) belonging to a line tangent 
    to two circumferences and to the two circumferences themselves. 

    :param xc1: x coordinate of the center of the first circumference
    :type xc1: float
    :param yc1: y coordinate of the center of the first circumference
    :type yc1: float
    :param xc2: x coordinate of the center of the second circumference
    :type xc2: float
    :param yc2: y coordinate of the center of the second circumference
    :type yc2: float
    :param turn1: turn direction along the first circumference
    :type turn1: float 
    :param turn2: turn direction along the second circumference
    :type turn2: float
    :param R: radius of the circumference at which the robot moves at higher speed (R = vmax/omegamax)
    :type R: float
   
    :return: x coordinate of first point
    :rtype: float 
    :return: y coordinate of first point
    :rtype: float
    :return: orientation of the tangent
    :rtype: float 
    :return: x coordinate of second point
    :rtype: float 
    :return: y coordinate of second point
    :rtype: float
    :return: orientation of the tangent
    :rtype: float 
    '''
    try:
        # If the provided circles are touching (distance between the centers = 2R) and turn1 is different than turn2
        # if ((yc2 - yc1)**2 + (xc2 - xc1)**2 == (2 * R)**2) and (turn1 != turn2):
        if overlap and (turn1 != turn2):
            # Compute the point where the two circles are touching
            x1, y1, _, _ = circle_intersection(xc1, yc1, R1, xc2, yc2, R2)
            theta1 = wrapPositiveAngle(atan2(yc2 - yc1, xc2 - xc1) + turn1 * 0.5 * pi) 
            x2, y2 = x1, y1
        # If the distance between the two circles is > 2R or turn1 = turn2
        else: 
            zeta = - (turn1 + turn2) * pi * 0.25
            eta = (turn1 - turn2) * pi * 0.25
            a1 = sqrt((yc1 - yc2)**2 + (xc1 - xc2)**2)
            c1 = sqrt((a1)**2 - (R1 - turn1 * turn2 * R2)**2)
            alfa1 = wrapPositiveAngle(atan2((yc2 - yc1), (xc2 - xc1)))
            gamma1 = asin((R1 - R2)/a1) if turn1 * turn2 > 0 else asin((c1/a1))
            beta1 = alfa1 - turn1 * gamma1
            x1 = xc1 + R1 * cos(beta1 + zeta)
            y1 = yc1 + R1 * sin(beta1 + zeta)
            x2 = x1 + c1 * cos(beta1 + eta)
            y2 = y1 + c1 * sin(beta1 + eta)
            theta1 = wrapPositiveAngle(atan2((y2 - y1),(x2 - x1)))
        return x1, y1, theta1, x2, y2, theta1
    except ValueError: 
        print('Overlapping circles ERROR')
        return None, None, None, None, None, None
    


    







