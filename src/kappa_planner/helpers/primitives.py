"""Motion primitive construction utilities.

This module contains functions to construct basic motion primitives
(e.g., arcs, segments) used to build analytical trajectories for
corridor-based motion planning.
"""
from math import asin, atan2, cos, pi, sin, sqrt

import numpy as np

from kappa_planner.trajectory import (
    BackwardArc,
    CurvilinearArcUnicycle,
    LinearSegmentUnicycle,
    TurnOnTheSpot,
)

from .geometry_operations import (
    compute_angular_difference,
    efficient_sign,
    wrapPositiveAngle,
)
from .intersections import circle_intersection


def compute_extreme_poses_arc_line(xc1, yc1, xc2, yc2, turn1, turn2, R, overlap = False):
    """Compute the points and orientation (x1, y1, theta1).

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
    """
    try:
        # If the provided circles are touching (distance between the centers = 2R) and turn1 is different than turn2
        # if ((yc2 - yc1)**2 + (xc2 - xc1)**2 == (2 * R)**2) and (turn1 != turn2):
        if overlap and (turn1 != turn2):
            # Compute the point where the two circles are touching
            x1, y1, _, _ = circle_intersection(xc1, yc1, R, xc2, yc2, R)
            theta1 = wrapPositiveAngle(atan2(yc2 - yc1, xc2 - xc1) + turn1 * 0.5 * pi)
            x2, y2 = x1, y1
            return x1, y1, theta1, x2, y2, theta1

        distance = np.hypot(xc2 - xc1, yc2 - yc1)
        if turn1 != turn2 and abs(distance - 2.0 * R) <= 1e-9:
            # The internal tangent degenerates to the circles' contact point.
            # Its heading cannot be obtained with atan2(y2-y1, x2-x1), since
            # both differences are zero.
            x1 = 0.5 * (xc1 + xc2)
            y1 = 0.5 * (yc1 + yc2)
            theta1 = wrapPositiveAngle(
                atan2(yc2 - yc1, xc2 - xc1) + turn1 * 0.5 * pi
            )
            return x1, y1, theta1, x1, y1, theta1

        # If the distance between the two circles is > 2R or turn1 = turn2
        else:
            zeta = - (turn1 + turn2) * pi * 0.25
            eta = (turn1 - turn2) * pi * 0.25
            a1 = sqrt((yc1 - yc2)**2 + (xc1 - xc2)**2)
            c1 = sqrt((a1)**2 - (R - turn1 * turn2 * R)**2)
            alfa1 = wrapPositiveAngle(atan2((yc2 - yc1), (xc2 - xc1)))
            if (c1/a1) > 1:
                stop = 1
            gamma1 = asin((R - R)/a1) if turn1 * turn2 > 0 else asin(c1/a1)
            beta1 = alfa1 - turn1 * gamma1
            x1 = xc1 + R * cos(beta1 + zeta)
            y1 = yc1 + R * sin(beta1 + zeta)
            x2 = x1 + c1 * cos(beta1 + eta)
            y2 = y1 + c1 * sin(beta1 + eta)
            theta1 = wrapPositiveAngle(atan2((y2 - y1),(x2 - x1)))
        return x1, y1, theta1, x2, y2, theta1
    except ValueError:
        print("Overlapping circles ERROR")
        return None, None, None, None, None, None


def compute_arc_from_two_tangents(segment1, segment2, turn, xc2, yc2, unicycle):
    """Compute the arc connecting two tangent segments.

    :param segment1: first segment
    :type segment1: LinearSegmentUnicycle
    :param segment2: second segment
    :type segment2: LinearSegmentUnicycle
    :param turn: turn direction along the arc
    :type turn: float in [-1, 1]
    :param xc2: x coordinate of the center of the second circumference
    :type xc2: float
    :param yc2: y coordinate of the center of the second circumference
    :type yc2: float
    :param unicycle: considered unicycle vehicle
    :type unicycle: Unicycle

    :return: arc between the two segments
    :rtype: CurvilinearArcUnicycle
    """
    x0, y0 = segment1.end_position
    theta0 = segment1.thetaf

    xf, yf = segment2.start_position
    thetaf = segment2.theta0

    t0 = segment1.tf
    radius = unicycle.max_radius
    omega = unicycle.omega_max if turn > 0 else unicycle.omega_min

    arc = CurvilinearArcUnicycle(
        xc=xc2,
        yc=yc2,
        x0=x0,
        y0=y0,
        theta0=theta0,
        xf=xf,
        yf=yf,
        thetaf=thetaf,
        radius=radius,
        turn_direction=turn,
        v=unicycle.v_max,
        omega=omega,
        unicycle=unicycle,
        t0=t0,
        samples_number=10,
    )

    return arc


def invert_maneuvers(maneuvers_list, t0 = 0):
    """Given a list of maneuvers, invert their order, and the orientation at the beginning and at the end of each maneuver, and the turn directions.

    :param maneuvers_list: list of maneuvers
    :type maneuvers_list: list of trajectory pieces (TurnOnTheSpot, CurvilinearArcUnicycle, LinearSegmentUnicycle)
    :param t0: initial time
    :type t0: float

    :return: inverted list of maneuvers
    :rtype: list of maneuvers
    """
    maneuvers_list_inv = [0] * len(maneuvers_list)
    ind = 0
    for i in np.arange(len(maneuvers_list)-1, -1, -1):
        man = maneuvers_list[i]
        if isinstance(man, TurnOnTheSpot):
            new_maneuver    = TurnOnTheSpot(x=man.x0, y=man.y0, theta0=man.thetaf - pi, thetaf=man.theta0 - pi, omega=-man.omega, unicycle=man.unicycle, t0=t0, samples_number=man.samples_number)
        elif isinstance(man, CurvilinearArcUnicycle):
            new_maneuver    = CurvilinearArcUnicycle(xc=man.xc, yc=man.yc, x0=man.xf, y0=man.yf, theta0=man.thetaf - pi, xf=man.x0, yf=man.y0, thetaf=man.theta0 - pi, radius=man.radius, turn_direction=-man.turn_direction,  v=man.v, omega=-man.omega, unicycle=man.unicycle, t0=t0, samples_number=man.samples_number)
        elif isinstance(man, LinearSegmentUnicycle):
            new_maneuver    = LinearSegmentUnicycle(x0=man.xf, y0=man.yf, xf=man.x0, yf=man.y0, theta=man.theta0 - pi, v=man.v, unicycle = man.unicycle, t0 = t0, samples_number=man.samples_number)
        elif isinstance(man, BackwardArc):
            new_maneuver    =  BackwardArc(xc = man.xc, yc = man.yc, x0 = man.xf, y0 = man.yf, theta0 = man.thetaf - pi, xf = man.x0, yf = man.y0, thetaf = man.theta0 - pi, radius = man.radius, turn_direction=-man.turn_direction, v = man.v, omega = -man.omega, bicycle = man.bicycle, t0 = t0, samples_number = man.samples_number)

        t0 = new_maneuver.tf
        maneuvers_list_inv[ind] = new_maneuver
        ind += 1
    return maneuvers_list_inv


def compute_segment_between_two_circles(xc1, yc1, xc2, yc2, turn1, turn2, unicycle, overlap = False):
    x1, y1, theta1, x2, y2, _ = compute_extreme_poses_arc_line(xc1, yc1, xc2, yc2, turn1, turn2, unicycle.max_radius, overlap = overlap)
    return LinearSegmentUnicycle(x0=x1, y0=y1, xf=x2, yf=y2, theta=theta1, v=unicycle.v_max, t0 = 0, unicycle = unicycle, samples_number=10)


def reverse_maneuvers(maneuvers):
    """ 
    Reverse a list of maneuvers.
    :param maneuvers: list of maneuvers
    :type maneuvers: list of Trajectory

    :return: list of reversed maneuvers
    :rtype: list of Trajectory
    """
    maneuvers_reversed = [0] * (len(maneuvers))
    for i in range(len(maneuvers_reversed)):
        maneuvers_reversed[i] = maneuvers[i].reverse()
    return maneuvers_reversed

def is_it_u_turn(corridor1, corridor3, turn1, turn2):
    tol = 1e-3
    if turn1 == turn2:
        corridor1_corners = corridor1.get_corners()
        corridor3_corners = corridor3.get_corners()
        if turn1 == 1:
            # Take the corners of the left wall of corridor1 and corridor3
            x1, y1 = corridor1_corners[2]
            x2, y2 = corridor1_corners[3]
            x3, y3 = corridor3_corners[2]
            x4, y4 = corridor3_corners[3]
            # Horizontal corridors
            if abs(y1 - y2) < tol and abs(y3 - y4) < tol:
                if abs(y1 - y3) < tol:
                    return True
            # Vertical corridors
            elif abs(x1 - x3) < tol:
                return True
        else:
            # Take the corners of the right wall of corridor1 and corridor3
            x1, y1 = corridor1_corners[0]
            x2, y2 = corridor1_corners[1]
            x3, y3 = corridor3_corners[0]
            x4, y4 = corridor3_corners[1]


            # Horizontal corridors
            if abs(y1 - y2) < tol and abs(y3 - y4) < tol:
                if abs(y1 - y3) < tol:
                    return True
            # Vertical corridors
            elif abs(x1 - x3) < tol:
                return True
    return False


def correct_angles(maneuver_list):
    for i in range(1, len(maneuver_list)):
        if isinstance(maneuver_list[i], LinearSegmentUnicycle):
            maneuver_list[i].change_theta(maneuver_list[i-1].thetaf)
        else:
            maneuver_list[i].change_theta0(maneuver_list[i-1].thetaf)


def compute_arc_from_two_tangents_objects(segment1, segment2, circ, vehicle):
    """Given two linear segments, compute the circular arc connecting them.
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


def compute_central_angle(x0, y0, xf, yf, xc1, yc1, turn, radius):
    """Given a circumference, two points which are the extremes of an arc and a turn direction, computes the central angle.

    :param x0: x coordinate of first extreme point of the arc
    :type x0: float
    :param y0: y coordinat of first extreme point of the arc
    :type y0: float
    :param xf: x coordinate of second extreme point of the arc
    :type xf: float
    :param yf: y coordinate of second extreme point of the arc
    :type yf: float
    :param xc1: x coordinate of the center of circumference to which the arc belongs
    :type xc1: float
    :param yc1: y coordinate of the center of circumference to which the arc belongs
    :type yc1: float
    :param turn: turn direction along the given circumference
    :type turn: float [-1,1]
    :param radius: radius of the circumference v_max/omega_max
    :type radius: float

    :return: central angle in radians
    :rtype: float
    """
    chord  = sqrt((xf - x0)**2 + (yf - y0)**2)
    # Compute the delta angle between initial and final orientation
    # theta0 = wrapPositiveAngle(atan2(y0-yc1, x0-xc1)) and thetaf = wrapPositiveAngle(atan2(yf-yc1, xf-xc1))
    delta_angle = compute_angular_difference(wrapPositiveAngle(atan2(y0-yc1, x0-xc1)), wrapPositiveAngle(atan2(yf-yc1, xf-xc1)))
    # Compute the angle depending on whether the turn direction corresponds to the sign of delta angle
    if efficient_sign(delta_angle) == turn:
        return 2 * asin((chord * 0.5)/radius)
    return 2 * pi - (2 * asin((chord * 0.5)/radius))


def compute_segment_between_two_circles_objects(
    circ1,
    circ2,
    vehicle,
    overlap=False,
):
    x1, y1, theta1, x2, y2, _ = compute_extreme_poses_arc_line(
        circ1.xc,
        circ1.yc,
        circ2.xc,
        circ2.yc,
        circ1.turn_direction,
        circ2.turn_direction,
        vehicle.max_radius,
        overlap=overlap,
    )

    return LinearSegmentUnicycle(
        x0=x1,
        y0=y1,
        xf=x2,
        yf=y2,
        theta=theta1,
        v=vehicle.v_max,
        t0=0,
        unicycle=vehicle,
        samples_number=10,
    )
