from math import asin, atan2, cos, pi, sin, sqrt

from ..geometry import Circle, Point, Pose
from ..trajectory import CurvilinearArcUnicycle, LinearSegmentUnicycle, TurnOnTheSpot
from ..vehicle import Unicycle
from .geometry_operations import (
    compute_angular_difference_with_turn_direction,
    compute_distance_two_points,
    select_tangency_point_from_point_circle,
)
from .primitives import compute_extreme_poses_arc_line, invert_maneuvers

# compute_extreme_poses_arc_line(xc1, yc1, xc2, yc2, turn1, turn2, R, overlap = False)

def compute_CSC_trajectory(start_pose, end_pose, tau0, tauf, unicycle): 
    R = unicycle.max_radius
    x0 = start_pose.x
    y0 = start_pose.y
    theta0 = start_pose.theta
    xf = end_pose.x
    yf = end_pose.y
    thetaf = end_pose.theta
    v_max = unicycle.v_max
    omega_max = unicycle.omega_max

    xc0, yc0 = x0 + R * cos(theta0 + tau0 * pi * 0.5), y0 + R * sin(theta0 + tau0 * pi * 0.5)
    xcf, ycf = xf + R * cos(thetaf + tauf * pi * 0.5), yf + R * sin(thetaf + tauf * pi * 0.5)

    x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(xc0, yc0, xcf, ycf, tau0, tauf, R)

    delta_theta1 = compute_angular_difference_with_turn_direction(theta0, theta1, tau0)
    delta_theta2 = compute_angular_difference_with_turn_direction(theta2, thetaf, tauf)

    theta_start_C1 = theta0
    theta_end_C1 = theta0 + delta_theta1
    theta_start_S2 = theta_end_C1
    theta_end_S2 = theta_start_S2
    theta_start_C3 = theta_end_S2
    theta_end_C3 = theta_start_C3 + delta_theta2


    # Primitive 2: arc
    C1 = CurvilinearArcUnicycle(
        xc=xc0,
        yc=yc0,
        x0=x0,
        y0=y0,
        theta0=theta_start_C1,
        xf=x1,
        yf=y1,
        thetaf=theta_end_C1,
        radius=R,
        turn_direction=tau0,
        v=v_max,
        omega= tau0 * omega_max,
        unicycle=unicycle,
        t0=0,
        samples_number=30,
    )

    # Primitive 3: segment
    S2 = LinearSegmentUnicycle(
        x0=x1,
        y0=y1,
        xf=x2,
        yf=y2,
        theta=theta_end_S2,
        v=v_max,
        t0=C1.tf,
        unicycle=unicycle,
        samples_number=10,
    )

    C3 = CurvilinearArcUnicycle(
        xc=xcf,
        yc=ycf,
        x0=x2,
        y0=y2,
        theta0=theta_start_C3,
        xf=xf,
        yf=yf,
        thetaf=theta_end_C3,
        radius=R,
        turn_direction=tauf,
        v=v_max,
        omega= tauf * omega_max,
        unicycle=unicycle,
        t0=S2.tf,
        samples_number=30,
    )
    total_time = C1.maneuver_time + S2.maneuver_time + C3.maneuver_time
    return [C1, S2, C3], total_time


def compute_TCSC_trajectory(start_pose, end_pose, tau0, tauf, unicycle): 
    R = unicycle.max_radius
    x0 = start_pose.x
    y0 = start_pose.y
    theta0 = start_pose.theta
    xf = end_pose.x
    yf = end_pose.y
    thetaf = end_pose.theta

    v_max = unicycle.v_max
    omega_max = unicycle.omega_max

    xcf, ycf = xf + R * cos(thetaf + tauf * pi * 0.5), yf + R * sin(thetaf + tauf * pi * 0.5)
    alpha0 = atan2(ycf - y0, xcf - x0)

    if tau0 == tauf: 
        theta0_after_T1 = alpha0 - tau0 * pi * 0.5
        delta_theta_T1 = compute_angular_difference_with_turn_direction(theta0, theta0_after_T1, tau0)
        xc0, yc0 = x0 + R * cos(theta0_after_T1 + tau0 * pi * 0.5), y0 + R * sin(theta0_after_T1 + tau0 * pi * 0.5)
    else: 
        hyp = compute_distance_two_points(Point(xcf, ycf), start_pose.position)
        asin_arg = 2*R/hyp
        if abs(asin_arg) > 1:
            ValueError("The start pose is too close to the final circle for a TCSC trajectory to be optimal.")
        else:
            beta_prime = asin(asin_arg)
        theta0_after_T1 = alpha0 - tauf * (beta_prime - pi * 0.5)
        delta_theta_T1 = compute_angular_difference_with_turn_direction(theta0, theta0_after_T1, tau0)
        xc0, yc0 = x0 + R * cos(theta0_after_T1 + tau0 * pi * 0.5), y0 + R * sin(theta0_after_T1 + tau0 * pi * 0.5)

    x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(xc0, yc0, xcf, ycf, tau0, tauf, R)

    delta_theta_C2 = compute_angular_difference_with_turn_direction(theta0_after_T1, theta1, tau0)
    delta_theta_C3 = compute_angular_difference_with_turn_direction(theta2, thetaf, tauf)

    theta_start_T1 = theta0
    theta_end_T1 = theta_start_T1 + delta_theta_T1
    theta_start_C2 = theta_end_T1
    theta_end_C2 = theta_start_C2 + delta_theta_C2
    theta_start_S3 = theta_end_C2
    theta_end_S3 = theta_start_S3
    theta_start_C4 = theta_end_S3
    theta_end_C4 = theta_start_C4 + delta_theta_C3

    T1 = TurnOnTheSpot(
        x=x0,
        y=y0,
        theta0=theta_start_T1,
        thetaf=theta_end_T1,
        omega=tau0 * omega_max,
        unicycle=unicycle,
        t0=0,
        samples_number=5,
    )

    C2 = CurvilinearArcUnicycle(
        xc=xc0,
        yc=yc0,
        x0=x0,
        y0=y0,
        theta0=theta_start_C2,
        xf=x1,
        yf=y1,
        thetaf=theta_end_C2,
        radius=R,
        turn_direction=tau0,
        v=v_max,
        omega= tau0 * omega_max,
        unicycle=unicycle,
        t0=T1.tf,
        samples_number=30,
    )

    # Primitive 3: segment
    S3 = LinearSegmentUnicycle(
        x0=x1,
        y0=y1,
        xf=x2,
        yf=y2,
        theta=theta_end_S3,
        v=v_max,
        t0=C2.tf,
        unicycle=unicycle,
        samples_number=10,
    )

    C4 = CurvilinearArcUnicycle(
        xc=xcf,
        yc=ycf,
        x0=x2,
        y0=y2,
        theta0=theta_start_C4,
        xf=xf,
        yf=yf,
        thetaf=theta_end_C4,
        radius=R,
        turn_direction=tauf,
        v=v_max,
        omega= tauf * omega_max,
        unicycle=unicycle,
        t0=S3.tf,
        samples_number=30,
    )
    
    total_time = T1.maneuver_time + C2.maneuver_time + S3.maneuver_time + C4.maneuver_time

    return [T1, C2, S3, C4], total_time


def compute_CSCT_trajectory(start_pose, end_pose, tau0, tauf, unicycle): 
    # Later: implement the CSC-T trajectory type
    start_pose_rev = start_pose.reversed()
    end_pose_rev = end_pose.reversed()
    tau0_rev = -tau0
    tauf_rev = -tauf
    reversed_trajectory, total_time = compute_TCSC_trajectory(end_pose_rev, start_pose_rev, tau0_rev, tauf_rev, unicycle)
    [C1, S2, C3, T4] =invert_maneuvers(reversed_trajectory, t0 = 0)
    total_motion_time = C1.maneuver_time + S2.maneuver_time + C3.maneuver_time + T4.maneuver_time
    return [C1, S2, C3, T4], total_motion_time


def compute_TCSCT_trajectory(start_pose, end_pose, tau0, tauf, unicycle): 
    # Later: implement the TCSC-T trajectory type
    R = unicycle.max_radius
    x0 = start_pose.x
    y0 = start_pose.y
    theta0 = start_pose.theta
    xf = end_pose.x
    yf = end_pose.y
    thetaf = end_pose.theta
    v_max = unicycle.v_max
    omega_max = unicycle.omega_max

    if tau0 == tauf: 
        alpha0 = atan2(yf - y0, xf - x0)
        alphaf = atan2(y0 - yf, x0 - xf)

        theta0_after_T1 = alpha0 - tau0 * pi * 0.5
        thetaf_before_T5 = alphaf - tauf * pi * 0.5

        delta_theta_T1 = compute_angular_difference_with_turn_direction(theta0, theta0_after_T1, tau0)
        delta_theta_T5 = compute_angular_difference_with_turn_direction(thetaf_before_T5, thetaf, tauf)

        xc0, yc0 = x0 + R * cos(alpha0), y0 + R * sin(alpha0)
        xcf, ycf = xf + R * cos(alphaf), yf + R * sin(alphaf)

    else: 
        circle_at_final_point = Circle(Point(xf, yf), 2 * R)

        xf_ref, yf_ref = select_tangency_point_from_point_circle(
            Point(x0, y0),
            circle_at_final_point,
            turn_direction=tauf,
        )

        alpha0 = atan2(yf_ref - y0, xf_ref - x0)
        alphaf = atan2(y0 - yf_ref, x0 - xf_ref)

        theta0_after_T1 = alpha0 - tau0 * pi * 0.5
        thetaf_before_T5 = alphaf - tauf * pi * 0.5

        delta_theta_T1 = compute_angular_difference_with_turn_direction(theta0, theta0_after_T1, tau0)
        delta_theta_T5 = compute_angular_difference_with_turn_direction(thetaf_before_T5, thetaf, tauf)

        xc0, yc0 = x0 + R * cos(alpha0), y0 + R * sin(alpha0)
        xcf, ycf = xf + R * cos(alphaf), yf + R * sin(alphaf)


    x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(xc0, yc0, xcf, ycf, tau0, tauf, R)

    delta_theta_C2 = compute_angular_difference_with_turn_direction(theta0_after_T1, theta1, tau0)
    delta_theta_C4 = compute_angular_difference_with_turn_direction(theta2, thetaf_before_T5, tauf)
    
    theta_start_T1 = theta0
    theta_end_T1 = theta_start_T1 + delta_theta_T1
    theta_start_C2 = theta_end_T1
    theta_end_C2 = theta_start_C2 + delta_theta_C2
    theta_start_S3 = theta_end_C2
    theta_end_S3 = theta_start_S3
    theta_start_C4 = theta_end_S3
    theta_end_C4 = theta_start_C4 + delta_theta_C4
    theta_start_T5 = theta_end_C4
    theta_end_T5 = theta_start_T5 + delta_theta_T5


    T1 = TurnOnTheSpot(
        x=x0,
        y=y0,
        theta0=theta_start_T1,
        thetaf=theta_end_T1,
        omega=tau0 * omega_max,
        unicycle=unicycle,
        t0=0,
        samples_number=5,
    )

    C2 = CurvilinearArcUnicycle(
        xc=xc0,
        yc=yc0,
        x0=x0,
        y0=y0,
        theta0=theta_start_C2,
        xf=x1,
        yf=y1,
        thetaf=theta_end_C2,
        radius=R,
        turn_direction=tau0,
        v=v_max,
        omega= tau0 * omega_max,
        unicycle=unicycle,
        t0=T1.tf,
        samples_number=30,
    )

    # Primitive 3: segment
    S3 = LinearSegmentUnicycle(
        x0=x1,
        y0=y1,
        xf=x2,
        yf=y2,
        theta=theta_end_S3,
        v=v_max,
        t0=C2.tf,
        unicycle=unicycle,
        samples_number=10,
    )

    C4 = CurvilinearArcUnicycle(
        xc=xcf,
        yc=ycf,
        x0=x2,
        y0=y2,
        theta0=theta_start_C4,
        xf=xf,
        yf=yf,
        thetaf=theta_end_C4,
        radius=R,
        turn_direction=tauf,
        v=v_max,
        omega= tauf * omega_max,
        unicycle=unicycle,
        t0=S3.tf,
        samples_number=30,
    )

    T5 = TurnOnTheSpot(
        x=xf,
        y=yf,
        theta0=theta_start_T5,
        thetaf=theta_end_T5,
        omega=tauf * omega_max,
        unicycle=unicycle,
        t0=C4.tf,
        samples_number=5,
    )

    total_time = T1.maneuver_time + C2.maneuver_time + S3.maneuver_time + C4.maneuver_time + T5.maneuver_time
    return [T1, C2, S3, C4, T5], total_time


def compute_all_pose_to_pose_trajectories(start_pose, end_pose, unicycle):
    """
    Compute all CSC, TCSC, CSCT and TCSCT trajectory candidates.

    Returns
    -------
    dict
        Dictionary containing trajectory, total time, type, tau0 and tauf.
    """

    R = unicycle.max_radius

    if compute_distance_two_points(
        start_pose.position,
        end_pose.position,
    ) <= 4 * R:
        raise ValueError(
            "Start and end poses are too close: "
            "distance must be greater than 4 * turning radius."
        )

    tau_left = 1
    tau_right = -1

    trajectory_functions = {
        "CSC": compute_CSC_trajectory,
        "TCSC": compute_TCSC_trajectory,
        "CSCT": compute_CSCT_trajectory,
        "TCSCT": compute_TCSCT_trajectory,
    }

    tau_pairs = {
        "left-left": (tau_left, tau_left),
        "left-right": (tau_left, tau_right),
        "right-left": (tau_right, tau_left),
        "right-right": (tau_right, tau_right),
    }

    trajectories = {}

    for trajectory_type, compute_function in trajectory_functions.items():

        for tau_name, (tau0, tauf) in tau_pairs.items():

            name = f"{trajectory_type} {tau_name}"

            try:
                trajectory, total_time = compute_function(
                    start_pose,
                    end_pose,
                    tau0,
                    tauf,
                    unicycle,
                )
            except ValueError as e:
                print(f"  {name}: Skipped: {e}")
                continue

            trajectories[name] = {
                "trajectory": trajectory,
                "time": total_time,
                "type": trajectory_type,
                "tau0": tau0,
                "tauf": tauf,
            }

    return trajectories




    
