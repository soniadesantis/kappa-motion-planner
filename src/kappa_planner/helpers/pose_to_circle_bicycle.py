from .geometry_operations import compute_angular_difference, wrapPositiveAngle, compute_angular_difference_with_turn_direction
from ..trajectory import CurvilinearArcBicycle, LinearSegmentBicycle, BackwardArcBicycle
from .primitives import correct_angles
from .collision_avoidance import check_arc_collision, compute_wall_tangent_circle_centers, compute_safe_corridor_union, segment_inside_safe_corridor_union
from .intersections import circle_intersection
from ..geometry import Point, Pose, Circle
from .helper_functions import compute_center_coordinates_first_circle
from math import sqrt, asin, atan2, cos, sin, pi
import numpy as np


def compute_traj_to_circle_bicycle(
    corridor1,
    corridor2,
    start_pose,
    bicycle,
    circ1,
    tau0=0,
    figure=None,
    admissible_corridors=None,
):
    """
    Public wrapper for the circular-footprint bicycle pose-to-circle
    construction.

    ``admissible_corridors`` is optional and preserves the original behavior
    when omitted. It is needed when the refinement skips the first or last
    intermediate circle and the boundary tangent consequently traverses more
    than two corridors.
    """
    return compute_traj_to_circle_bicycle_circular(
        corridor1=corridor1,
        corridor2=corridor2,
        start_pose=start_pose,
        bicycle=bicycle,
        circ1=circ1,
        tau0=tau0,
        figure=figure,
        admissible_corridors=admissible_corridors,
    )


def compute_traj_to_circle_bicycle_circular(
    corridor1,
    corridor2,
    start_pose,
    bicycle,
    circ1,
    tau0=0,
    figure=None,
    admissible_corridors=None,
):
    """
    Build a collision-free trajectory from ``start_pose`` to ``circ1``.

    The considered boundary-connection structures are

        CS
        C_back CS

    where the backward arc may either be introduced by the free-space
    candidate-selection rule or as one corrective maneuver to avoid a
    collision with a lateral wall of the first corridor.

    Collision checks:
        - backward arcs must remain inside corridor1;
        - the forward arc must remain inside corridor1;
        - the tangent segment must remain inside
          the admissible corridor union eroded by the robot radius.

    When ``admissible_corridors`` is omitted, the tangent is checked against
    corridor1 union corridor2. A supplied subsequence allows the tangent to
    cross additional corridors when refinement skips boundary circles.

    Only lateral-wall collisions of the nominal circular part are repaired.
    Collisions with unsupported walls, an unsafe tangent segment, or failure
    of the single corrective construction cause the function to return None.

    :return:
        List of trajectory primitives, or None.
    """

    # ---------------------------------------------------------------
    # 0. Target-circle geometry
    # ---------------------------------------------------------------
    tau1 = circ1.turn_direction
    corner_point1 = circ1.corner_point

    side_walls = (
        corridor1.LFT,
        corridor1.RGT,
    )

    # Build the local collision-free reference-point region once.
    #
    # Important:
    #     union first, then erode by the footprint radius.
    if admissible_corridors is None:
        admissible_corridors = [corridor1, corridor2]

    safe_union = compute_safe_corridor_union(
        corridor_list=admissible_corridors,
        r=0.5 * bicycle.width,
    )

    if safe_union is None:
        return None

    # Determine the turn direction of the first forward arc
    # if it has not been prescribed.
    if tau0 == 0:

        tau0 = compute_initial_turn_direction(
            circ1.xc,
            circ1.yc,
            circ1.radius,
            start_pose[0],
            start_pose[1],
            start_pose[2],
            tau1,
        )

    original_pose = Pose(
        position=Point(
            start_pose[0],
            start_pose[1],
        ),
        theta=start_pose[2],
    )

    # ---------------------------------------------------------------
    # Small local helpers
    # ---------------------------------------------------------------
    def build_forward_connection(
        pose,
        t0,
    ):
        """
        Build the forward C-S connection from ``pose`` to circ1.
        """
        return compute_two_maneuvers_bicycle(
            start_pose=pose,
            bicycle=bicycle,
            circ2=circ1,
            tau2=tau1,
            t0=t0,
            tau1=tau0,
            figure=figure,
        )

    def tangent_segment_is_safe(
        segment,
    ):
        """
        Check the complete finite tangent segment in the eroded
        admissible corridor union.
        """
        return segment_inside_safe_corridor_union(
            start_point=np.array(
                [
                    segment.x0,
                    segment.y0,
                ],
                dtype=float,
            ),
            end_point=np.array(
                [
                    segment.xf,
                    segment.yf,
                ],
                dtype=float,
            ),
            safe_union=safe_union,
        )

    def classify_forward_connection(
        arc,
        segment,
    ):
        """
        Classify one forward C-S connection.

        Returns
        -------
        status:
            "safe"
            "repair"
            "unsupported"

        wall:
            Colliding lateral wall for status == "repair",
            otherwise the detected wall or None.
        """

        # Exact geometric check against the first shrunken corridor.
        arc_collision, wall = check_arc_collision(
            arc,
            corridor1,
        )

        if arc_collision:

            if wall in side_walls:
                return "repair", wall

            # No correction rule is introduced for front/back-wall
            # collisions.
            return "unsupported", wall

        # The tangent may legitimately pass through both corridors,
        # therefore check it against the eroded corridor union.
        if not tangent_segment_is_safe(
            segment
        ):
            return "unsupported", None

        return "safe", None

    # ---------------------------------------------------------------
    # 1. Optional backward arc from the free-space rule
    # ---------------------------------------------------------------
    backward_is_better, _, _, _ = (
        rule_initial_backward_maneuver(
            start_pose,
            circ1,
            tau0,
        )
    )

    initial_maneuvers = []

    current_pose = list(
        start_pose
    )

    next_t0 = 0.0

    # If this becomes non-None, discard the nominal candidate and
    # construct one corrective backward arc from the original pose.
    repair_wall = None

    if backward_is_better:

        backward_arc = compute_backward_arc_optimal(
            pose=original_pose,
            tau1=tau0,
            tau2=tau1,
            circ2=circ1,
            bicycle=bicycle,
        )

        if backward_arc is None:
            return None

        backward_collision, wall = (
            check_arc_collision(
                backward_arc,
                corridor1,
            )
        )

        if backward_collision:

            # Only lateral-wall collisions are corrected.
            if wall not in side_walls:
                return None

            repair_wall = wall

        else:

            initial_maneuvers.append(
                backward_arc
            )

            current_pose = [
                backward_arc.xf,
                backward_arc.yf,
                backward_arc.thetaf,
            ]

            next_t0 = backward_arc.tf

    # ---------------------------------------------------------------
    # 2. Try the nominal forward C-S connection
    # ---------------------------------------------------------------
    if repair_wall is None:

        forward_arc, tangent_segment = (
            build_forward_connection(
                pose=current_pose,
                t0=next_t0,
            )
        )

        if (
            forward_arc is None
            or tangent_segment is None
        ):
            return None

        status, wall = (
            classify_forward_connection(
                forward_arc,
                tangent_segment,
            )
        )

        if status == "safe":

            maneuvers = (
                initial_maneuvers
                + [
                    forward_arc,
                    tangent_segment,
                ]
            )

            _finalize_maneuver_sequence(
                maneuvers
            )

            return maneuvers

        if status == "unsupported":
            return None

        repair_wall = wall

    # ---------------------------------------------------------------
    # 3. Construct exactly one corrective backward arc
    # ---------------------------------------------------------------
    corrective_arc = compute_backward_arc(
        corridor=corridor1,
        pose=original_pose,
        bicycle=bicycle,
        tau=tau0,
        radius=bicycle.max_radius,
        wall=repair_wall,
        corner_point=corner_point1,
    )

    if corrective_arc is None:
        return None

    # The corrective backward arc must remain entirely inside C1.
    corrective_collision, _ = (
        check_arc_collision(
            corrective_arc,
            corridor1,
        )
    )

    if corrective_collision:
        return None

    # ---------------------------------------------------------------
    # 4. Recompute the forward C-S connection
    # ---------------------------------------------------------------
    corrected_start_pose = [
        corrective_arc.xf,
        corrective_arc.yf,
        corrective_arc.thetaf,
    ]

    (
        corrected_forward_arc,
        corrected_segment,
    ) = build_forward_connection(
        pose=corrected_start_pose,
        t0=corrective_arc.tf,
    )

    if (
        corrected_forward_arc is None
        or corrected_segment is None
    ):
        return None

    # ---------------------------------------------------------------
    # 5. Validate the corrected forward connection
    #
    # Do not recurse into another correction. One corrective backward
    # maneuver is the complete supported recovery rule.
    # ---------------------------------------------------------------
    corrected_status, _ = (
        classify_forward_connection(
            corrected_forward_arc,
            corrected_segment,
        )
    )

    if corrected_status != "safe":
        return None

    # ---------------------------------------------------------------
    # 6. Successful corrected boundary connection
    # ---------------------------------------------------------------
    corrected_maneuvers = [
        corrective_arc,
        corrected_forward_arc,
        corrected_segment,
    ]

    _finalize_maneuver_sequence(
        corrected_maneuvers
    )

    return corrected_maneuvers


def _finalize_maneuver_sequence(maneuvers, tol=1e-9):
    """Make the timing and heading values continuous across primitives."""
    for index in range(1, len(maneuvers)):
        previous_tf = maneuvers[index - 1].time_grid[-1]
        current_t0 = maneuvers[index].time_grid[0]

        offset = previous_tf - current_t0

        if abs(offset) > tol:
            maneuvers[index].add_time_offset(offset)

    correct_angles(maneuvers)


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
    backward_arc =BackwardArcBicycle(xc=bw_circle.xc, yc=bw_circle.yc, x0 = pose.x, y0 = pose.y,
                                        theta0 = pose.theta, xf = final_bw_pose.x, yf = final_bw_pose.y,
                                        thetaf = final_bw_pose.theta, radius = circ2.radius,
                                        turn_direction = -tau1, v = -bicycle.v_max,
                                        omega = -tau1 * bicycle.omega_max, bicycle = bicycle,
                                        t0 = 0, samples_number = 100)
    return backward_arc


def compute_two_maneuvers_bicycle(start_pose, bicycle, circ2, tau2, t0 = 0, tau1 = 0, figure = None,):
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
    :rtype: CurvilinearArcBicycle
    :return: primitive2, segment
    :rtype: LinearSegmentBicycle
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

    # # # Compute the extreme poses for each maneuver
    # plt.plot(circ1.xc, circ1.yc, 'ro')
    # plt.plot(circ2.xc, circ2.yc, 'bo')
    # plt.arrow(x0, y0, 0.5 * np.cos(theta0), 0.5 * np.sin(theta0), head_width=0.1, head_length=0.1, fc='k', ec='k')
    # plt.plot(circ1.xc + circ1.radius * np.cos(np.linspace(0, 2*pi, 100)), circ1.yc + circ1.radius * np.sin(np.linspace(0, 2*pi, 100)), 'r--')
    # plt.plot(circ2.xc + circ2.radius * np.cos(np.linspace(0, 2*pi, 100)), circ2.yc + circ2.radius * np.sin(np.linspace(0, 2*pi, 100)), 'b--')
    # plt.plot(x0, y0, 'go')
    # plt.title('Initial Pose and Intermediate Circles tau_0 = {}, tau_1 = {}'.format(tau1, tau2))
    # plt.show(block = True)
    pose1, pose2 = compute_extreme_poses_arc_line_two_radii_oo(circ1, circ2, tau1, tau2)
    if pose1 is None or pose2 is None:
        return None, None
    #Compute orientations for each primitive
    theta0_p1 = theta0
    thetaf_p1 = theta0_p1 + compute_angular_difference_with_turn_direction(theta0, pose1.theta, tau1)
    theta_p2 = thetaf_p1

    omega1 = omega_max if tau1 > 0 else omega_min

    # Primitive 1: arc
    primitive1 = CurvilinearArcBicycle(xc=circ1.xc, yc=circ1.yc, x0 = x0, y0 = y0,
                                        theta0 = theta0_p1, xf = pose1.x, yf = pose1.y,
                                        thetaf = thetaf_p1, radius = circ1.radius,
                                        turn_direction = tau1, v = v_max,
                                        omega = omega1, bicycle=bicycle,
                                        t0 = t0, samples_number = 100)
    # Primitive 2: segment
    primitive2 = LinearSegmentBicycle(x0=pose1.x, y0=pose1.y, xf=pose2.x, yf=pose2.y, theta=theta_p2, v=v_max, t0 = primitive1.tf, bicycle=bicycle, samples_number=10, start_circle_index = 0, end_circle_index = circ2.index)

    return primitive1, primitive2


def backward_maneuver(
    corridor,
    pose,
    tau,
    radius,
    wall,
    corner_point,
    tol=1e-9,
):
    """
    Construct a backward maneuver such that the following forward circle
    is tangent to the selected corridor wall.

    The input corridor must already be shrunken by the robot footprint
    radius.

    :param tau: turn direction of the following forward arc
    :return: final backward pose and backward circle, or (None, None)
    """

    # The backward arc uses the opposite steering side.
    backward_turn = -tau

    backward_circle = compute_first_circle(
        pose.x,
        pose.y,
        pose.theta,
        backward_turn,
        radius,
    )

    candidate_centers = compute_wall_tangent_circle_centers(
        corridor=corridor,
        wall=wall,
        fixed_circle=backward_circle,
        radius=radius,
        tol=tol,
    )

    if not candidate_centers:
        return None, None

    corner = np.array(
        [corner_point.x, corner_point.y],
        dtype=float,
    )

    # Preserve your current selection principle.
    # The candidate closest to the transition corner is selected.
    forward_center = min(
        candidate_centers,
        key=lambda point: (
            (point.x - corner[0]) ** 2
            + (point.y - corner[1]) ** 2
        ),
    )

    # Equal-radius externally tangent circles touch at their midpoint.
    xb = 0.5 * (backward_circle.xc + forward_center.x)
    yb = 0.5 * (backward_circle.yc + forward_center.y)

    radial_angle = atan2(
        yb - backward_circle.yc,
        xb - backward_circle.xc,
    )

    # Heading corresponding to the backward circle steering side -tau.
    thetab = wrapPositiveAngle(
        radial_angle - tau * 0.5 * pi
    )

    final_pose = Pose(
        position=Point(xb, yb),
        theta=thetab,
    )

    return final_pose, backward_circle


def compute_backward_arc(
    corridor,
    pose,
    bicycle,
    tau,
    radius,
    wall,
    corner_point,
    tol=1e-9,
):
    """
    Build the corrective backward arc.

    :param tau: turn direction of the forward arc that collided
    :return: BackwardArcBicycle, or None if no construction exists
    """
    effective_corridor = corridor.shrink(0.5 * bicycle.width)

    final_pose, backward_circle = backward_maneuver(
        corridor=effective_corridor,
        pose=pose,
        tau=tau,
        radius=radius,
        wall=wall,
        corner_point=corner_point,
        tol=tol,
    )

    if final_pose is None:
        return None

    backward_turn = -tau

    final_theta = (
        pose.theta
        + compute_angular_difference_with_turn_direction(
            pose.theta,
            final_pose.theta,
            backward_turn,
        )
    )

    backward_arc = BackwardArcBicycle(
        xc=backward_circle.xc,
        yc=backward_circle.yc,
        x0=pose.x,
        y0=pose.y,
        theta0=pose.theta,
        xf=final_pose.x,
        yf=final_pose.y,
        thetaf=final_theta,
        radius=radius,
        turn_direction=backward_turn,
        v=-bicycle.v_max,
        omega=-tau * bicycle.omega_max,
        bicycle=bicycle,
        t0=0,
        samples_number=100,
    )

    return backward_arc


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
        return None, None
