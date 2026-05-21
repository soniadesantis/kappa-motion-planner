from .helper_functions import compute_center_coordinates_first_circle
from .geometry_operations import wrapPositiveAngle, compute_angular_difference_with_turn_direction
from .pose_to_circle_dubins import compute_two_maneuvers
from math import atan2, asin, sqrt, pi, cos, sin

from matplotlib import pyplot as plt
import numpy as np
from ..trajectory import CurvilinearArcUnicycle


def compute_initial_turn_direction_exact_rule(start_pose, unicycle, xc2, yc2, tau2):
    x0, y0, theta0 = start_pose[0], start_pose[1], start_pose[2]
    R = unicycle.max_radius
    slope = 1/unicycle.omega_max
    alpha0 = wrapPositiveAngle(atan2((yc2 - y0), (xc2 - x0)))
    xt, yt = xc2 + R * cos(alpha0 - tau2 * pi*0.5), yc2 + R * sin(alpha0 - tau2 * pi*0.5)
    thetaf_val = pi*0.5
    a = sqrt((xc2 - x0)**2 + (yc2 - y0)**2)
    if abs(R/a) > 1: 
        return tau2
    if abs(2 * R/a) > 1: # Added this check in case the start point is within the circle with 2R
        beta_2r = pi/6
    else:
        beta_2r = asin(2 * R/a)
    beta = asin(R/a)
    angle_of_separation = alpha0 - tau2 * beta
    xt, yt = xc2 + R* cos(alpha0), yc2 + R * sin(alpha0)
    thetat = alpha0 +tau2 * pi*0.5 # Final orientation of the unicycle
    # Compute the theta0 from which to start turn on-the-spot with planner
    theta0_start_ll = alpha0 - tau2 * pi*0.5
    theta0_start_rl = alpha0 - tau2 * beta_2r + tau2 * pi*0.5

    if (theta0_start_ll < angle_of_separation): 
        theta0_start_ll += 2 * pi
    elif (theta0_start_ll > angle_of_separation + 2 * pi):
        theta0_start_ll -= 2 * pi
    if (theta0_start_rl < angle_of_separation):
        theta0_start_rl += 2 * pi
    elif (theta0_start_rl > angle_of_separation + 2 * pi):
        theta0_start_rl -= 2 * pi

    if (theta0 < angle_of_separation): 
        theta0 += 2 * pi
    elif (theta0 > angle_of_separation + 2 * pi):
        theta0 -= 2 * pi
    if (theta0 < angle_of_separation):
        theta0 += 2 * pi
    elif (theta0 > angle_of_separation + 2 * pi):
        theta0 -= 2 * pi

    C1_ll, S1_ll = compute_two_maneuvers([x0, y0, theta0_start_ll], unicycle, xc2, yc2, tau2, t0 = 0, turn1 = 1)
    C2_ll =  CurvilinearArcUnicycle(xc=xc2, yc=yc2, x0=S1_ll.xf, y0=S1_ll.yf, theta0=S1_ll.theta, xf=xt, yf=yt, thetaf=thetat, radius=R, turn_direction=tau2, v=unicycle.v_max, omega=unicycle.omega_max, unicycle=unicycle, t0=0, samples_number=50)

    motion_time_theta0_start_ll = C1_ll.maneuver_time + S1_ll.maneuver_time + C2_ll.maneuver_time
        
    C1_rl, S1_rl = compute_two_maneuvers([x0, y0, theta0_start_rl], unicycle, xc2, yc2, tau2, t0 = 0, turn1 = -1)
    C2_rl =  CurvilinearArcUnicycle(xc=xc2, yc=yc2, x0=S1_rl.xf, y0=S1_rl.yf, theta0=S1_rl.theta, xf=xt, yf=yt, thetaf=thetat, radius=R, turn_direction=tau2, v=unicycle.v_max, omega=unicycle.omega_max, unicycle=unicycle, t0=0, samples_number=50)
    motion_time_theta0_start_rl = C1_rl.maneuver_time + S1_rl.maneuver_time + C2_rl.maneuver_time

    theta_choice_tau1 = (slope * (theta0_start_ll + theta0_start_rl) + (motion_time_theta0_start_ll - motion_time_theta0_start_rl))/(2*slope)

    if angle_of_separation < theta0 < theta_choice_tau1:
        tau1 = -1 
    else:
        tau1 = 1

    return tau1


def compute_initial_turn_on_the_spot_time_optimality(start_pose, xc2, yc2, turn1, turn2, radius):
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
    # Extract initial pose
    x0, y0, theta0 = start_pose
    # Compute alpha0, the orientation of the segment connecting x0,y0 to xc2,yc2
    alpha0 = wrapPositiveAngle(atan2((yc2 - y0), (xc2 - x0)))
    # Initialize the delta_angle
    delta_angle = 0
    # Compute the angle beta
    a = sqrt((xc2 - x0)**2 + (yc2 - y0)**2)
    if abs(radius / a) > 1:
        plt.plot(x0, y0, 'ro')
        angle_array = np.linspace(0, 2 * pi, 100)
        plt.plot(xc2 + radius * np.cos(angle_array), yc2 + radius * np.sin(angle_array), 'b-')
        plt.show(block=True)
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
        if abs(ang_disp) > pi * 0.5 - beta:
            angle_to_be_reached = wrapPositiveAngle(alpha0 - turn2 * pi * 0.5)
            delta_angle += compute_angular_difference_with_turn_direction(theta0, angle_to_be_reached, turn1)

    ## Rule for Left-right and Right-left case
    elif turn1 != turn2:
        # Compute the angular difference between theta0 and alpha0 - turn2 * beta when rotating according to turn1
        ang_disp = compute_angular_difference_with_turn_direction(theta0, alpha0 - turn2 * beta, turn1)

        # If the amplitude of the obtained angle is > pi * 0.5 - (beta_2R - beta), theta0 lies in the region where
        # a turn on-the-spot is needed until angle_to_be_reached
        if abs(ang_disp) > pi * 0.5 - (beta_2R - beta):
            angle_to_be_reached = wrapPositiveAngle(alpha0 - turn1 * pi* 0.5 + turn1 * beta_2R)
            delta_angle += compute_angular_difference_with_turn_direction(theta0, angle_to_be_reached, turn1)
 
    xc1, yc1 = compute_center_coordinates_first_circle(x0, y0, theta0 + delta_angle, turn1, radius)

    return delta_angle, xc1, yc1