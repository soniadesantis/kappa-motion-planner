from .geometry_operations import compute_angular_difference, wrapPositiveAngle, compute_angular_difference_with_turn_direction
from ..trajectory import CurvilinearArcUnicycle, LinearSegmentUnicycle, BackwardArc

from .primitives import (
    correct_angles,
    invert_maneuvers,
    compute_extreme_poses_arc_line,
)
from .poses import absolute_to_relative_pose, relative_to_absolute_pose
from .collision_avoidance import collision_avoidance_check_bicycle, check_arc_collision, compute_wall_tangent_circle_centers, check_segment_collision_corridor_union, check_arc_collision_corridor_union
from .intersections import circle_intersection
from ..geometry import Point, Pose, Circle
from .helper_functions import (
    compute_center_coordinates_first_circle,

)

from kappa_planner.helpers.plot_helpers import (
    plot_analytical_trajectory,
    plot_corridors,
    plot_circular_footprint,
)
from math import sqrt, asin, atan2, cos, sin, pi
import matplotlib.pyplot as plt
import numpy as np


def compute_traj_to_circle_bicycle(
    corridor1,
    corridor2,
    start_pose,
    bicycle,
    circ1,
    tau0=0,
    figure = None,
):
    # if getattr(bicycle, "rectangular_footprint", False):
    #     return compute_traj_to_circle_bicycle_rectangular(
    #         corridor1=corridor1,
    #         start_pose=start_pose,
    #         bicycle=bicycle,
    #         circ1=circ1,
    #         tau0=tau0,
    #     )

    return compute_traj_to_circle_bicycle_circular(
        corridor1=corridor1,
        corridor2=corridor2,
        start_pose=start_pose,
        bicycle=bicycle,
        circ1=circ1,
        tau0=tau0,
        figure = figure,
    )


# def compute_traj_to_circle_bicycle_circular(corridor1, start_pose, bicycle, circ1, tau0 = 0):
#     """
#     Build the initial part of the trajectory from the start pose to the first intermediate circle.
    
#     :param corridor1: first corridor in the sequence
#     :type corridor1: CorridorWorld
#     :param start_pose: initial pose of the vehicle
#     :type start_pose: list of floats
#     :param bicycle: bicycle vehicle
#     :type Bicycle: Bicycle
#     :param circ1: first intermediate circle
#     :type circ1: IntermediateCircle object
#     """
#     tau1 = circ1.turn_direction
#     corner_point1 = circ1.corner_point

#     # plot_corridors([corridor1])
#     # plt.plot(start_pose[0], start_pose[1], 'ro')
#     # plt.plot(circ1.xc, circ1.yc, 'bo')
#     # plt.plot(circ1.xc + circ1.radius * np.cos(np.linspace(0, 2*pi, 100)), circ1.yc + circ1.radius * np.sin(np.linspace(0, 2*pi, 100)), 'b--')
#     # plt.show(block = True)

#     tau0 = compute_initial_turn_direction(
#         circ1.xc,
#         circ1.yc,
#         circ1.radius,
#         start_pose[0],
#         start_pose[1],
#         start_pose[2],
#         tau1) if tau0 == 0 else tau0
    
#     ## Compute the first two maneuvers from start pose to the second circumference
#     collision_check = True
#     start_pose_fw_drive = start_pose.copy()
#     start_pose_object = Pose(position=Point(start_pose[0], start_pose[1]), theta = start_pose[2])
#     start_maneuvers = []

#     # First check whether a backward maneuver is required for time-optimality
#     not_optimal, _, _, _ = rule_initial_backward_maneuver(start_pose, circ1, tau0)
#     if not_optimal: # Case tau1 = tau2 and iota > 90 degrees
#         bw_arc = compute_backward_arc_optimal(start_pose_object, tau0, tau1, circ1, bicycle)
#         start_pose_fw_drive = [bw_arc.xf, bw_arc.yf, bw_arc.thetaf]
#         start_pose_object = Pose(position=Point(bw_arc.xf, bw_arc.yf), theta = bw_arc.thetaf)
#         start_maneuvers.append(bw_arc)  
#     # Compute the free space solution
#     arc1, segment2 = compute_two_maneuvers_bicycle(start_pose_fw_drive, bicycle, circ1, tau1, t0 = 0, tau1 = tau0)
    
#     # Check for collidions avoidance, if necessary add a backward maneuver
#     while collision_check:
#         collision_check, wall = collision_avoidance_check_bicycle(arc1, corridor1, bicycle, margin = 0)
#         if collision_check:
#             bw_arc = compute_backward_arc(corridor1, start_pose_object, bicycle, arc1.turn_direction, bicycle.max_radius, wall, corner_point1)
#             start_pose_fw_drive = [bw_arc.xf, bw_arc.yf, bw_arc.thetaf]
#             start_maneuvers.append(bw_arc)
#             arc1, segment2 = compute_two_maneuvers_bicycle(start_pose_fw_drive, bicycle, circ1, tau1, t0 = 0, tau1 = tau0)
#         else:
#             start_maneuvers.append(arc1)
#             start_maneuvers.append(segment2)
    
#     # Adjust the time grid and angles
#     for i in range(len(start_maneuvers)-1):
#         if start_maneuvers[i+1].time_grid[0] != start_maneuvers[i].time_grid[-1]:
#             start_maneuvers[i+1].add_time_offset(abs(start_maneuvers[i+1].time_grid[0] - start_maneuvers[i].time_grid[-1]))
#     correct_angles(start_maneuvers)
#     return start_maneuvers


# def compute_traj_to_circle_bicycle_circular(
#     corridor1,
#     corridor2,
#     start_pose,
#     bicycle,
#     circ1,
#     tau0=0,
#     figure=None,
# ):
#     """
#     Build a trajectory from ``start_pose`` to ``circ1``.

#     Candidate families:

#         CS
#         C_back CS

#     The initial backward arc, when present, is required to remain inside
#     the first corridor.

#     The forward arc and tangent segment are checked against the union of
#     the first and second corridors. This permits the trajectory and robot
#     footprint to pass through the admissible intersection region between
#     consecutive corridors.

#     If the free-space solution is infeasible, the function attempts to
#     construct one corrective backward arc.

#     :param corridor1: corridor containing the initial pose
#     :type corridor1: CorridorWorld
#     :param corridor2: subsequent corridor
#     :type corridor2: CorridorWorld
#     :param start_pose: initial pose [x, y, theta]
#     :type start_pose: list or numpy.ndarray
#     :param bicycle: considered bicycle vehicle
#     :type bicycle: Bicycle
#     :param circ1: first intermediate circle
#     :type circ1: IntermediateCircle
#     :param tau0: prescribed initial turn direction, or zero if unknown
#     :type tau0: int
#     :param figure: optional plotting figure
#     :type figure: matplotlib.figure.Figure or None

#     :return: list of trajectory primitives, or None if infeasible
#     :rtype: list or None
#     """
#     tau1 = circ1.turn_direction
#     corner_point1 = circ1.corner_point

#     admissible_corridors = [
#         corridor1,
#         corridor2,
#     ]

#     if tau0 == 0:
#         tau0 = compute_initial_turn_direction(
#             circ1.xc,
#             circ1.yc,
#             circ1.radius,
#             start_pose[0],
#             start_pose[1],
#             start_pose[2],
#             tau1,
#         )

#     original_pose = Pose(
#         position=Point(
#             start_pose[0],
#             start_pose[1],
#         ),
#         theta=start_pose[2],
#     )

#     # ---------------------------------------------------------------
#     # 1. Build the free-space candidate
#     # ---------------------------------------------------------------
#     free_space_maneuvers = []
#     current_pose = list(start_pose)
#     next_t0 = 0.0

#     backward_is_better, _, _, _ = (
#         rule_initial_backward_maneuver(
#             start_pose,
#             circ1,
#             tau0,
#         )
#     )

#     optimal_backward_arc = None

#     if backward_is_better:
#         optimal_backward_arc = compute_backward_arc_optimal(
#             original_pose,
#             tau0,
#             tau1,
#             circ1,
#             bicycle,
#         )

#         free_space_maneuvers.append(
#             optimal_backward_arc
#         )

#         current_pose = [
#             optimal_backward_arc.xf,
#             optimal_backward_arc.yf,
#             optimal_backward_arc.thetaf,
#         ]

#         next_t0 = optimal_backward_arc.tf

#     forward_arc, tangent_segment = (
#         compute_two_maneuvers_bicycle(
#             current_pose,
#             bicycle,
#             circ1,
#             tau1,
#             t0=next_t0,
#             tau1=tau0,
#             figure=figure,
#         )
#     )

#     free_space_maneuvers.extend(
#         [
#             forward_arc,
#             tangent_segment,
#         ]
#     )

#     # if figure is not None:
#     #     plot_analytical_trajectory(
#     #         free_space_maneuvers,
#     #         figure=figure,
#     #     )

#     #     plt.show(block=True)

#     # ---------------------------------------------------------------
#     # 2. Check the optional time-optimal backward arc
#     #
#     # The backward arc must remain in the first corridor.
#     # ---------------------------------------------------------------
#     colliding_wall = None

#     if optimal_backward_arc is not None:
#         collision, wall = check_arc_collision(
#             optimal_backward_arc,
#             corridor1,
#         )

#         if collision:
#             if wall == corridor1.FWD:
#                 return None

#             # Discard the optional backward arc and construct one
#             # corrective backward arc from the original pose.
#             colliding_wall = wall

#     # ---------------------------------------------------------------
#     # 3. Check the forward CS portion
#     #
#     # Both the forward arc and the segment may pass from corridor 1
#     # into corridor 2.
#     # ---------------------------------------------------------------
#     if colliding_wall is None:
#         arc_collision, _ = (
#             check_arc_collision_corridor_union(
#                 arc=forward_arc,
#                 corridors=admissible_corridors,
#             )
#         )

#         segment_collision, _ = (
#             check_segment_collision_corridor_union(
#                 segment=tangent_segment,
#                 corridors=admissible_corridors,
#             )
#         )

#         if (
#             not arc_collision
#             and not segment_collision
#         ):
#             _finalize_maneuver_sequence(
#                 free_space_maneuvers
#             )

#             return free_space_maneuvers

#         # The corrective-arc construction currently requires the wall
#         # of the first corridor that is violated by the forward arc.
#         first_corridor_collision, wall = (
#             check_arc_collision(
#                 forward_arc,
#                 corridor1,
#             )
#         )

#         if not first_corridor_collision:
#             # The trajectory leaves the total corridor union, but not
#             # through a wall for which the current correction method
#             # is defined.
#             return None

#         if wall == corridor1.FWD:
#             return None

#         colliding_wall = wall

#     # ---------------------------------------------------------------
#     # 4. Build exactly one corrective backward arc
#     #
#     # The corrective backward arc remains inside corridor 1.
#     # ---------------------------------------------------------------
#     corrective_arc = compute_backward_arc(
#         corridor=corridor1,
#         pose=original_pose,
#         bicycle=bicycle,
#         tau=forward_arc.turn_direction,
#         radius=bicycle.max_radius,
#         wall=colliding_wall,
#         corner_point=corner_point1,
#     )

#     if corrective_arc is None:
#         return None

#     corrective_collision, _ = check_arc_collision(
#         corrective_arc,
#         corridor1,
#     )

#     if corrective_collision:
#         return None

#     # ---------------------------------------------------------------
#     # 5. Recompute CS after the corrective backward arc
#     # ---------------------------------------------------------------
#     corrected_start_pose = [
#         corrective_arc.xf,
#         corrective_arc.yf,
#         corrective_arc.thetaf,
#     ]

#     corrected_forward_arc, corrected_segment = (
#         compute_two_maneuvers_bicycle(
#             corrected_start_pose,
#             bicycle,
#             circ1,
#             tau1,
#             t0=corrective_arc.tf,
#             tau1=tau0,
#             figure=figure,
#         )
#     )

#     corrected_arc_collision, _ = (
#         check_arc_collision_corridor_union(
#             arc=corrected_forward_arc,
#             corridors=admissible_corridors,
#         )
#     )

#     if corrected_arc_collision:
#         return None

#     corrected_segment_collision, _ = (
#         check_segment_collision_corridor_union(
#             segment=corrected_segment,
#             corridors=admissible_corridors,
#         )
#     )

#     if corrected_segment_collision:
#         return None

#     corrected_maneuvers = [
#         corrective_arc,
#         corrected_forward_arc,
#         corrected_segment,
#     ]

#     _finalize_maneuver_sequence(
#         corrected_maneuvers
#     )

#     return corrected_maneuvers


def compute_traj_to_circle_bicycle_circular(
    corridor1,
    corridor2,
    start_pose,
    bicycle,
    circ1,
    tau0=0,
    figure = None,
):
    """
    Build a trajectory from ``start_pose`` to ``circ1``.

    Candidate families:

        CS
        C_back CS

    If the free-space solution is infeasible, the function attempts to build
    one corrective backward arc. If no feasible trajectory can be constructed,
    the function returns ``None``.

    :return: list of trajectory primitives, or ``None`` if no feasible
             trajectory can be constructed
    """
    tau1 = circ1.turn_direction
    corner_point1 = circ1.corner_point

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
        position=Point(start_pose[0], start_pose[1]),
        theta=start_pose[2],
    )

    # ---------------------------------------------------------------
    # 1. Build the free-space candidate
    # ---------------------------------------------------------------
    free_space_maneuvers = []
    current_pose = list(start_pose)
    next_t0 = 0.0

    backward_is_better, _, _, _ = rule_initial_backward_maneuver(
        start_pose,
        circ1,
        tau0,
    )

    optimal_backward_arc = None

    if backward_is_better:
        optimal_backward_arc = compute_backward_arc_optimal(
            original_pose,
            tau0,
            tau1,
            circ1,
            bicycle,
        )

        free_space_maneuvers.append(optimal_backward_arc)

        current_pose = [
            optimal_backward_arc.xf,
            optimal_backward_arc.yf,
            optimal_backward_arc.thetaf,
        ]

        next_t0 = optimal_backward_arc.tf
    # figure = plot_corridors([corridor1, corridor2])
    forward_arc, tangent_segment = compute_two_maneuvers_bicycle(
        current_pose,
        bicycle,
        circ1,
        tau1,
        t0=next_t0,
        tau1=tau0,
        figure = figure,
    )
    if forward_arc is None or tangent_segment is None:
        raise ValueError("Failed to compute forward arc and tangent segment due to overlapping circles.")

    free_space_maneuvers.extend([
        forward_arc,
        tangent_segment,
    ])

    # plot_analytical_trajectory(
    #     free_space_maneuvers, figure=figure)
    # plt.show(block = True)
    

    # ---------------------------------------------------------------
    # 2. Check the optional time-optimal backward arc
    # ---------------------------------------------------------------
    colliding_wall = None

    if optimal_backward_arc is not None:
        collision, wall = check_arc_collision(
            optimal_backward_arc,
            corridor1,
        )

        if collision:
            if wall == corridor1.FWD:
                return None

            # Discard the optional backward arc and construct one
            # corrective backward arc from the original pose.
            colliding_wall = wall

    # ---------------------------------------------------------------
    # 3. Check the free-space forward arc
    # ---------------------------------------------------------------
    if colliding_wall is None:
        collision, wall = check_arc_collision(
            forward_arc,
            corridor1,
        )

        if not collision:
            _finalize_maneuver_sequence(free_space_maneuvers)
            return free_space_maneuvers

        if wall == corridor1.FWD:
            return None

        colliding_wall = wall

    # ---------------------------------------------------------------
    # 4. Build exactly one corrective backward arc
    # ---------------------------------------------------------------
    corrective_arc = compute_backward_arc(
        corridor=corridor1,
        pose=original_pose,
        bicycle=bicycle,
        tau=forward_arc.turn_direction,
        radius=bicycle.max_radius,
        wall=colliding_wall,
        corner_point=corner_point1,
    )

    if corrective_arc is None:
        return None

    corrective_collision, _ = check_arc_collision(
        corrective_arc,
        corridor1,
    )

    if corrective_collision:
        return None

    # ---------------------------------------------------------------
    # 5. Recompute CS after the corrective backward arc
    # ---------------------------------------------------------------
    corrected_start_pose = [
        corrective_arc.xf,
        corrective_arc.yf,
        corrective_arc.thetaf,
    ]

    corrected_forward_arc, corrected_segment = (
        compute_two_maneuvers_bicycle(
            corrected_start_pose,
            bicycle,
            circ1,
            tau1,
            t0=corrective_arc.tf,
            tau1=tau0,
        )
    )

    if corrected_forward_arc is None or corrected_segment is None:
        raise ValueError("Failed to compute corrected forward arc and tangent segment due to overlapping circles.")

    forward_collision, _ = check_arc_collision(
        corrected_forward_arc,
        corridor1,
    )

    if forward_collision:
        return None

    corrected_maneuvers = [
        corrective_arc,
        corrected_forward_arc,
        corrected_segment,
    ]

    _finalize_maneuver_sequence(corrected_maneuvers)

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

    # ## 3- If collision occurs, compute a backward maneuver to avoid the collision and recompute the two maneuvers to reach the first intermediate circle
    # while collision_check:
    #     collision_check, wall = collision_avoidance_check_bicycle(arc1, corridor1, bicycle, margin = 0)
    #     if collision_check:
    #         bw_arc = compute_backward_arc(corridor1, start_pose_object, bicycle, arc1.turn_direction, bicycle.max_radius, wall, corner_point1)
    #         start_pose_fw_drive = [bw_arc.xf, bw_arc.yf, bw_arc.thetaf]
    #         start_maneuvers.append(bw_arc)
    #         arc1, segment2 = compute_two_maneuvers_bicycle(start_pose_fw_drive, bicycle, circ1, tau1, t0 = 0, tau1 = tau0)
    #     else:
    #         start_maneuvers.append(arc1)
    #         start_maneuvers.append(segment2)
    
    # ## 4- Return trajectory or failure
    # # Adjust the time grid and angles
    # for i in range(len(start_maneuvers)-1):
    #     if start_maneuvers[i+1].time_grid[0] != start_maneuvers[i].time_grid[-1]:
    #         start_maneuvers[i+1].add_time_offset(abs(start_maneuvers[i+1].time_grid[0] - start_maneuvers[i].time_grid[-1]))
    # correct_angles(start_maneuvers)
    # return start_maneuvers


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
    backward_arc =BackwardArc(xc=bw_circle.xc, yc=bw_circle.yc, x0 = pose.x, y0 = pose.y,
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
    primitive1 = CurvilinearArcUnicycle(xc=circ1.xc, yc=circ1.yc, x0 = x0, y0 = y0,
                                        theta0 = theta0_p1, xf = pose1.x, yf = pose1.y,
                                        thetaf = thetaf_p1, radius = circ1.radius,
                                        turn_direction = tau1, v = v_max,
                                        omega = omega1, unicycle = bicycle,
                                        t0 = t0, samples_number = 100)
    # Primitive 2: segment
    primitive2 = LinearSegmentUnicycle(x0=pose1.x, y0=pose1.y, xf=pose2.x, yf=pose2.y, theta=theta_p2, v=v_max, t0 = primitive1.tf, unicycle = bicycle, samples_number=10, start_circle_index = 0, end_circle_index = circ2.index)

    return primitive1, primitive2


# def compute_backward_arc(corridor, pose, bicycle, tau, R, wall, corner_point):

#     final_bw_pose, bw_circle = backward_maneuver(corridor.shrink(bicycle.width*0.5), pose, tau, R, wall, corner_point)
#     final_bw_theta = pose.theta + compute_angular_difference_with_turn_direction(pose.theta, final_bw_pose.theta, -tau)
#     backward_arc =BackwardArc(xc=bw_circle.xc, yc=bw_circle.yc, x0 = pose.x, y0 = pose.y,
#                                         theta0 = pose.theta, xf = final_bw_pose.x, yf = final_bw_pose.y,
#                                         thetaf = final_bw_theta, radius = R,
#                                         turn_direction = -tau, v = -bicycle.v_max,
#                                         omega = -tau * bicycle.omega_max, bicycle = bicycle,
#                                         t0 = 0, samples_number = 100)
#     return backward_arc


# def backward_maneuver(corridor, pose, tau, R, wall, corner_point):
#     '''
#     Compute the backward maneuver in case of collision with one of the corridor's walls

#     :param corridor: corridor
#     :type corridor: CorridorWorld

#     :param pose: initial pose
#     :type pose: list of floats

#     :param tau: turn direction
#     :type tau: float

#     :param R: radius of the arc
#     :type R: float

#     :param wall: wall causing the collision
#     :type wall: int

#     :return: final pose
#     :rtype: Pose object
#     '''
#     # The corner point is used to select one of two possible circumferences
#     x_corner, y_corner = corner_point.x, corner_point.y
#     # Compute the rotation angle to obtain a vertical corridor (with tilt pi/2)
#     # rot_angle = pi/2 - corridor.tilt
#     # Compute the x,y and theta coordinates in the rotated corridor
#     x0, y0, theta0 = absolute_to_relative_pose(corridor, [pose.x, pose.y, pose.theta])

#     # Compute coordinates of the fixed circumference O1
#     circ1 = compute_first_circle(x0, y0, theta0, -tau, R)
#     xc1, yc1 = circ1.xc, circ1.yc

#     # Compute x,y coordinate of the transformed corridor center: reminder that we consider a vertical corridor with center in (0, 0)
#     # cx = cy = 0 

#     # Compute the x coordinate of O2, depending on the wall that causes a collision
#     xc2 = - wall * corridor.width * 0.5 + wall * R

#     # Compute the y coordinate of O2
#     a = 1
#     b = -2*yc1
#     c = xc2**2 + xc1**2 -2*xc2*xc1 + yc1**2 - 4*R**2
#     discr = np.sqrt(b**2 - 4*a*c)
#     yc2_sol1 = (-b + discr)/(2*a)
#     yc2_sol2 = (-b - discr)/(2*a)

#     # Pick the closest circumference to the robot
#     # yc2 = yc2_sol1 if abs(yc2_sol1 - y0) < abs(yc2_sol2 - y0) else yc2_sol2
#     yc2 = yc2_sol1 if abs(yc2_sol1 - y_corner) < abs(yc2_sol2 - y_corner) else yc2_sol2

#     # Compute the relative end pose of the backward maneuver
#     xb_rel, yb_rel = (xc2 + xc1) * 0.5, (yc2 + yc1) * 0.5
#     thetab_rel = wrapPositiveAngle(atan2(yb_rel - yc1, xb_rel - xc1) - tau * 0.5 * pi)

#     # Compute the absolute end pose of the backward maneuver
#     absolute_pose = relative_to_absolute_pose(corridor, [xb_rel, yb_rel, thetab_rel])
#     absolute_circle = relative_to_absolute_pose(corridor, [xc1, yc1, 0])
#     final_bw_pose = Pose(position = Point(x = absolute_pose[0], y = absolute_pose[1]), theta = absolute_pose[2])
#     bw_circle = Circle(center = Point(x = absolute_circle[0], y = absolute_circle[1]), radius = R)

#     return final_bw_pose, bw_circle


from math import atan2, pi


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
    :return: BackwardArc, or None if no construction exists
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

    backward_arc = BackwardArc(
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
    

def compute_traj_to_circle_bicycle_with_fixed_forward_circle(start_maneuvers, bicycle, circle_to_reach, fixed_circle):

    x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(
        fixed_circle.center.x,
        fixed_circle.center.y,
        circle_to_reach.center.x,
        circle_to_reach.center.y,
        fixed_circle.turn_direction,
        circle_to_reach.turn_direction,
        circle_to_reach.radius,
    )

    if isinstance(start_maneuvers[-2], BackwardArc) or isinstance(start_maneuvers[-2], CurvilinearArcUnicycle):
        arc_before_segment = start_maneuvers[-2]
        if len(start_maneuvers) > 2:
            x0, y0, theta0 = start_maneuvers[-3].xf, start_maneuvers[-3].yf, start_maneuvers[-3].thetaf
        else:
            x0, y0, theta0 = arc_before_segment.x0, arc_before_segment.y0, arc_before_segment.theta0

    delta_angle = compute_angular_difference(theta0, theta1)
    if np.sign(delta_angle) == np.sign(fixed_circle.turn_direction):
        if isinstance(arc_before_segment, BackwardArc):
            new_arc = BackwardArc(
                fixed_circle.xc,
                fixed_circle.yc,
                x0, y0, theta0,
                x1, y1, theta1,
                fixed_circle.radius, fixed_circle.turn_direction,
                -bicycle.v_max, -fixed_circle.turn_direction*bicycle.omega_max, bicycle
            )
        else:
            new_arc = CurvilinearArcUnicycle(
                fixed_circle.xc,
                fixed_circle.yc,
                x0, y0, theta0,
                x1, y1, theta1,
                fixed_circle.radius, fixed_circle.turn_direction,
                bicycle.v_max, fixed_circle.turn_direction*bicycle.omega_max, bicycle
            )
    else:
        if isinstance(arc_before_segment, BackwardArc):
            new_arc = CurvilinearArcUnicycle(
                fixed_circle.xc,
                fixed_circle.yc,
                x0, y0, theta0,
                x1, y1, theta1,
                fixed_circle.radius, fixed_circle.turn_direction,
                bicycle.v_max, fixed_circle.turn_direction*bicycle.omega_max, bicycle
            )
        else:
            new_arc = BackwardArc(
                fixed_circle.xc,
                fixed_circle.yc,
                x0, y0, theta0,
                x1, y1, theta1,
                fixed_circle.radius, fixed_circle.turn_direction,
                -bicycle.v_max, -fixed_circle.turn_direction*bicycle.omega_max, bicycle
            )

    #     x0, y0, theta0 = start_maneuvers[1].x0, start_maneuvers[1].y0, start_maneuvers[1].theta0
    #     first_backward_arc = [start_maneuvers[0]]
    #     fixed_circle = Circle(center = Point(x = start_maneuvers[1].xc, y = start_maneuvers[1].yc), radius = start_maneuvers[0].radius)
    #     turn_direction_fixed_circle = start_maneuvers[1].turn_direction
    # else:
    #     x0, y0, theta0 = start_maneuvers[0].x0, start_maneuvers[0].y0, start_maneuvers[0].theta0
    #     first_backward_arc = []
    #     fixed_circle = Circle(center = Point(x = start_maneuvers[0].xc, y = start_maneuvers[0].yc), radius = start_maneuvers[0].radius)
    #     turn_direction_fixed_circle = start_maneuvers[0].turn_direction


    # new_arc = CurvilinearArcUnicycle(
    #     fixed_circle.xc,
    #     fixed_circle.yc,
    #              x0, y0, theta0,
    #              x1, y1, theta1,
    #              circle_to_reach.radius, turn_direction_fixed_circle,
    #              bicycle.v_max,turn_direction_fixed_circle*bicycle.omega_max, bicycle
    #              )
    
    segment = LinearSegmentUnicycle(
        x1, y1, x2, y2, theta1,
        bicycle.v_max, bicycle
    )

    if len(start_maneuvers) > 2:
        new_start_maneuvers = start_maneuvers[:-2] + [new_arc, segment]
    else:   
        new_start_maneuvers = [new_arc, segment]

    # new_start_maneuvers = first_backward_arc + [new_arc, segment] if first_backward_arc else [new_arc, segment]

    # plot_analytical_trajectory(new_start_maneuvers)
    # plt.show(block = True)
    return new_start_maneuvers


def compute_full_traj_bicycle_with_two_fixed_circles(start_maneuvers, end_maneuvers, bicycle, start_fixed_circle, end_fixed_circle):

    if isinstance(start_maneuvers[0], BackwardArc):
        x0, y0, theta0 = start_maneuvers[1].x0, start_maneuvers[1].y0, start_maneuvers[1].theta0
        first_backward_arc = [start_maneuvers[0]]
        first_fixed_circle = Circle(center = Point(x = start_maneuvers[1].xc, y = start_maneuvers[1].yc), radius = start_maneuvers[0].radius)
        turn_direction_first_fixed_circle = start_maneuvers[1].turn_direction
    else:
        x0, y0, theta0 = start_maneuvers[0].x0, start_maneuvers[0].y0, start_maneuvers[0].theta0
        first_backward_arc = []
        first_fixed_circle = Circle(center = Point(x = start_maneuvers[0].xc, y = start_maneuvers[0].yc), radius = start_maneuvers[0].radius)
        turn_direction_first_fixed_circle = start_maneuvers[0].turn_direction

    if isinstance(end_maneuvers[-1], BackwardArc):
        xf, yf, thetaf = end_maneuvers[-2].xf, end_maneuvers[-2].yf, end_maneuvers[-2].thetaf
        last_backward_arc = [end_maneuvers[-1]]
        circle_to_reach = Circle(center = Point(x = end_maneuvers[-2].xc, y = end_maneuvers[-2].yc), radius = end_maneuvers[-1].radius)
        turn_direction_circle_to_reach = end_maneuvers[-2].turn_direction
    else:
        xf, yf, thetaf = end_maneuvers[0].xf, end_maneuvers[0].yf, end_maneuvers[0].thetaf
        last_backward_arc = []
        circle_to_reach = Circle(center = Point(x = end_maneuvers[0].xc, y = end_maneuvers[0].yc), radius = end_maneuvers[0].radius)
        turn_direction_circle_to_reach = end_maneuvers[0].turn_direction

    x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(
        start_fixed_circle.center.x,
        start_fixed_circle.center.y,
        end_fixed_circle.center.x,
        end_fixed_circle.center.y,
        start_fixed_circle.turn_direction,
        end_fixed_circle.turn_direction,
        end_fixed_circle.radius,
    )

    new_first_arc = CurvilinearArcUnicycle(
        first_fixed_circle.xc,
        first_fixed_circle.yc,
                 x0, y0, theta0,
                 x1, y1, theta1,
                 circle_to_reach.radius, turn_direction_first_fixed_circle,
                 bicycle.v_max,turn_direction_first_fixed_circle*bicycle.omega_max, bicycle
                 )
    
    segment = LinearSegmentUnicycle(
        x1, y1, x2, y2, theta1,
        bicycle.v_max, bicycle
    )

    new_last_arc = CurvilinearArcUnicycle(
        circle_to_reach.xc,
        circle_to_reach.yc,
                    x2, y2, theta1,
                    xf, yf, thetaf,
                    circle_to_reach.radius, turn_direction_circle_to_reach,
                    bicycle.v_max,turn_direction_circle_to_reach*bicycle.omega_max, bicycle
                    )
    
    new_start_maneuvers = first_backward_arc + [new_first_arc, segment]  if first_backward_arc or last_backward_arc else [new_first_arc, segment]
    new_end_maneuvers = last_backward_arc + [new_last_arc] if last_backward_arc else [new_last_arc]

    return new_start_maneuvers, new_end_maneuvers


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
    if arc1 is None or segment2 is None:
        raise ValueError("Failed to compute the two maneuvers from start pose to the first intermediate circle due to overlapping circles.")
    start_maneuvers.append(arc1)
    start_maneuvers.append(segment2)
    
    # Adjust the time grid and angles
    for i in range(len(start_maneuvers)-1):
        if start_maneuvers[i+1].time_grid[0] != start_maneuvers[i].time_grid[-1]:
            start_maneuvers[i+1].add_time_offset(abs(start_maneuvers[i+1].time_grid[0] - start_maneuvers[i].time_grid[-1]))
    correct_angles(start_maneuvers)
    return start_maneuvers
