from .helper_functions import fit_new_circle
from .geometry_operations import compute_angular_difference_with_turn_direction, wrapPositiveAngle, efficient_sign, compute_angular_difference
from ..trajectory import LinearSegmentUnicycle, CurvilinearArcUnicycle, TurnOnTheSpot
from .collision_avoidance import collision_avoidance_check, collision_avoidance_check_tb, collision_avoidance_check_after_overlap
from .initial_turn_on_the_spot import compute_initial_turn_on_the_spot_time_optimality, compute_initial_turn_direction_exact_rule
from .primitives import compute_extreme_poses_arc_line
from math import atan2, pi


def compute_three_maneuvers_compact(
    corridor,
    corridor2,
    start_pose,
    unicycle,
    xc2,
    yc2,
    turn2,
    t0=0,
    turn1=0,
):
    """
    Compute the three maneuvers needed to reach a target circle.

    The maneuver sequence is:
    1. turn on-the-spot,
    2. circular arc,
    3. linear segment.

    The target circle is centered at (xc2, yc2), and the vehicle reaches it
    with turn direction `turn2`.

    :param corridor: current corridor
    :param corridor2: next corridor
    :param start_pose: initial pose [x, y, theta]
    :param unicycle: unicycle vehicle model
    :param xc2: x-coordinate of target circle center
    :param yc2: y-coordinate of target circle center
    :param turn2: target circle turn direction
    :param t0: initial time
    :param turn1: initial turn direction

    :return: turn-on-the-spot, arc, and segment primitives
    :rtype: list (TurnOnTheSpot, CurvilinearArcUnicycle, LinearSegmentUnicycle)
    """
    # Normalize initial heading
    start_pose[2] = wrapPositiveAngle(start_pose[2])

    # Extract pose
    x0, y0, theta0 = start_pose

    # Extract vehicle limits
    R = unicycle.max_radius
    v_max = unicycle.v_max
    omega_max = unicycle.omega_max
    omega_min = unicycle.omega_min

    # Initialize maneuver state
    overlap = False
    omega_turn_on_the_spot = 0

    # Determine initial turn direction (if not provided)
    if turn1 == 0:
        turn1 = compute_initial_turn_direction_exact_rule(
            start_pose,
            unicycle,
            xc2,
            yc2,
            turn2,
        )

    # Compute additional turn needed for time optimality.
    # If no turn-on-the-spot is needed, delta_angle is zero.
    delta_angle, xc1, yc1 = compute_initial_turn_on_the_spot_time_optimality(
        start_pose,
        xc2,
        yc2,
        turn1,
        turn2,
        R,
    )

    # Compute additional turn needed for collision avoidance.
    delta_angle_tb, xc1, yc1 = collision_avoidance_check_tb(
        [x0, y0, theta0 + delta_angle],
        turn1,
        turn2,
        xc1,
        yc1,
        R,
        xc2,
        yc2,
        corridor,
        corridor2,
    )

    # Compute pose after previous turn adjustments
    pose_after_turns = [
        x0,
        y0,
        theta0 + delta_angle + delta_angle_tb,
    ]

    # Additional turn-on-the-spot for collision avoidance
    delta_angle_collision_avoidance, xc1, yc1, overlap = collision_avoidance_check(
        pose_after_turns,
        turn1,
        turn2,
        xc1,
        yc1,
        R,
        xc2,
        yc2,
        corridor,
        corridor2,
        unicycle,
    )

    # Compute extreme poses defining the primitives
    # If no overlap is detected (or both turns are identical), compute normally
    if (not overlap) or (turn1 == turn2):
        x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(
            xc1,
            yc1,
            xc2,
            yc2,
            turn1,
            turn2,
            R,
        )
    # If overlap is detected, move the first circle to avoid it
    else:
        delta_angle = 0

        alpha0 = wrapPositiveAngle(
            atan2(
                yc2 - y0,
                xc2 - x0,
            )
        )

        xc1, yc1 = fit_new_circle(
            x0,
            y0,
            xc2,
            yc2,
            R,
            turn1,
            alpha0,
        )

        theta0_2 = (
            wrapPositiveAngle(
                atan2(
                    yc1 - y0,
                    xc1 - x0,
                )
            )
            - turn1 * pi * 0.5
        )

        delta_angle += compute_angular_difference_with_turn_direction(
            theta0,
            theta0_2,
            turn1,
        )

        pose_after_overlap_adjustment = [
            x0,
            y0,
            theta0 + delta_angle,
        ]

        (
            delta_angle_collision_avoidance,
            omega_turn_on_the_spot,
            xc1,
            yc1,
            overlap,
        ) = collision_avoidance_check_after_overlap(
            pose_after_overlap_adjustment,
            turn1,
            turn2,
            xc1,
            yc1,
            R,
            xc2,
            yc2,
            corridor,
            corridor2,
            unicycle,
        )

        delta_angle += delta_angle_collision_avoidance

        if delta_angle != 0:
            omega_turn_on_the_spot = omega_max if turn1 > 0 else omega_min

        x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(
            xc1,
            yc1,
            xc2,
            yc2,
            turn1,
            turn2,
            R,
            overlap=overlap,
        )

    if (delta_angle + delta_angle_collision_avoidance) != 0:
        omega_turn_on_the_spot = omega_max if turn1 > 0 else omega_min

    # Compute orientations for each primitive
    theta0_p1 = theta0
    thetaf_p1 = theta0_p1 + delta_angle + delta_angle_collision_avoidance
    theta0_p2 = thetaf_p1
    angular_diff = compute_angular_difference(
    wrapPositiveAngle(theta0_p2),
    wrapPositiveAngle(theta1),
    )
    if turn1 != efficient_sign(angular_diff):
        delta_angle2 = 2 * pi - angular_diff 
    else:
        delta_angle2 = angular_diff
    thetaf_p2 = theta0_p2 + delta_angle2
    theta_p3 = thetaf_p2 

    omega1 = omega_max if turn1 > 0 else omega_min

    # Primitive 1: turn-on-the-spot
    primitive1 = TurnOnTheSpot(
        x=x0,
        y=y0,
        theta0=theta0_p1,
        thetaf=thetaf_p1,
        omega=omega_turn_on_the_spot,
        unicycle=unicycle,
        t0=t0,
        samples_number=5,
    )

    # Primitive 2: arc
    primitive2 = CurvilinearArcUnicycle(
        xc=xc1,
        yc=yc1,
        x0=x0,
        y0=y0,
        theta0=theta0_p2,
        xf=x1,
        yf=y1,
        thetaf=thetaf_p2,
        radius=R,
        turn_direction=turn1,
        v=v_max,
        omega=omega1,
        unicycle=unicycle,
        t0=primitive1.tf,
        samples_number=10,
    )

    # Primitive 3: segment
    primitive3 = LinearSegmentUnicycle(
        x0=x1,
        y0=y1,
        xf=x2,
        yf=y2,
        theta=theta_p3,
        v=v_max,
        t0=primitive2.tf,
        unicycle=unicycle,
        samples_number=10,
    )
    return primitive1, primitive2, primitive3


def compute_three_maneuvers_no_collision_avoidance(start_pose, unicycle, xc2, yc2, turn2, t0 = 0, turn1 = 0):
    '''
    Compute the three maneuvers required to reach a circumference centered in (xc2, yc2): turn on-the-spot, arc and segment.

    :param corridor: considered corridor
    :type corridor: Corridor
    :param start_pose: initial pose
    :type start_pose: list or np.ndarray
    :param unicycle: x coordinate of the center of the second circumference
    :type unicycle: Unicycle
    :param xc2: x coordinate of the center of the second circumference
    :type xc2: float
    :param yc2: y coordinate of the center of the second circumference
    :type yc2: float
    :param turn2: turn direction along the second circumference
    :type turn2: float 
    :param t0: initial time
    :type t0: float
    :param turn1: turn direction along the first circumference
    :type turn1: float

    :return: primitive 1, turn on-the-spot
    :rtype: TurnOnTheSpot 
    :return: primitive2, arc
    :rtype: CurvilinearArcUnicycle
    :return: primitive3, segment
    :rtype: LinearSegmentUnicycle 
    '''
    start_pose[2] = wrapPositiveAngle(start_pose[2])
    # Extract variables
    x0, y0, theta0  = start_pose
    R = unicycle.max_radius
    v_max = unicycle.v_max
    omega_max = unicycle.omega_max
    omega_min = unicycle.omega_min

    # Initialize overlap Boolean variable
    overlap = False
    omega_turn_on_the_spot = 0

    # 1- If turn1 is not provided as input, compute it
    turn1 = compute_initial_turn_direction_exact_rule(start_pose, unicycle, xc2, yc2, turn2) if turn1 == 0 else turn1

    # 2- Turn-on-the-spot for time optimality: returns delta_angle = 0 and omega_turn_on_the_spot = 0 if no turn on-the-spot for time-optimality is required
    delta_angle, xc1, yc1 = compute_initial_turn_on_the_spot_time_optimality(start_pose, xc2, yc2, turn1, turn2, R)
    
    # 3- Compute the extreme poses for each maneuver
    # 4.a- If an overlap is not detected, proceed with the computation of the points on the two circumferences to build the motion primitives
    if not(overlap) or turn1 == turn2: 
        x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(xc1, yc1, xc2, yc2, turn1, turn2, R)
    
    # 4.b- If an overlap is detected, change the position of the first circumference to avoid the overlap
    else:
        delta_angle = 0
        alpha0 = wrapPositiveAngle(atan2((yc2 - y0), (xc2 - x0)))
        xc1, yc1 = fit_new_circle(x0, y0, xc2, yc2, R, turn1, alpha0)
        theta0_2 = wrapPositiveAngle(atan2((yc1 - y0), (xc1 - x0))) - turn1 * pi * 0.5
        delta_angle += compute_angular_difference_with_turn_direction(theta0, theta0_2 , turn1)
        if delta_angle != 0:
            omega_turn_on_the_spot = omega_max if delta_angle > 0 else omega_min

        x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(xc1, yc1, xc2, yc2, turn1, turn2, R, overlap = overlap)

    if delta_angle != 0:
        omega_turn_on_the_spot = omega_max if turn1 > 0 else omega_min

    # Compute orientations for each primitive
    theta0_p1 = theta0
    thetaf_p1 = theta0_p1 + delta_angle
    theta0_p2 = thetaf_p1
    if theta0_p2 == None or theta1 == None:
        problema = True
    angular_diff = compute_angular_difference(wrapPositiveAngle(theta0_p2), wrapPositiveAngle(theta1))
    if turn1 != efficient_sign(angular_diff):
        delta_angle2 = 2*pi - angular_diff 
    else:
        delta_angle2 = angular_diff
    thetaf_p2 = theta0_p2 + delta_angle2
    theta_p3 = thetaf_p2 

    omega1 = omega_max if turn1 > 0 else omega_min

    # Primitive 1: turn-on-the-spot
    primitive1 = TurnOnTheSpot(x=x0, y=y0, theta0=theta0_p1, thetaf=thetaf_p1, omega=omega_turn_on_the_spot, unicycle = unicycle, t0 = t0, samples_number=5)
    # Primitive 2: arc
    primitive2 = CurvilinearArcUnicycle(xc=xc1, yc=yc1, x0 = x0, y0 = y0, theta0 = theta0_p2, xf = x1, yf = y1, thetaf = thetaf_p2, radius = R, turn_direction = turn1, v = v_max, omega = omega1, unicycle = unicycle, t0 = primitive1.tf, samples_number = 10)
    # Primitive 3: segment
    primitive3 = LinearSegmentUnicycle(x0=x1, y0=y1, xf=x2, yf=y2, theta=theta_p3, v=v_max, t0 = primitive2.tf, unicycle = unicycle, samples_number=10)

    return primitive1, primitive2, primitive3