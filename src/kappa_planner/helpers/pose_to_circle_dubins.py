from ..trajectory import CurvilinearArcUnicycle, LinearSegmentUnicycle
from .helper_functions import compute_initial_turn_direction, compute_center_coordinates_first_circle, compute_extreme_poses_arc_line_two_radii
from .intersections import compute_intersection_points_between_line_circle
from .collision_avoidance import get_max_radius
from .primitives import compute_extreme_poses_arc_line
from .geometry_operations import compute_distance_two_points

def compute_two_maneuvers(start_pose, unicycle, xc2, yc2, turn2, t0 = 0, turn1 = 0):
    '''
    Compute the two maneuvers required to reach a circumference centered in (xc2, yc2): arc and segment.

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

    :return: primitive1, arc
    :rtype: CurvilinearArcUnicycle
    :return: primitive2, segment
    :rtype: LinearSegmentUnicycle 
    '''
    #Extract variables
    x0, y0, theta0  = start_pose
    r_nominal = unicycle.max_radius
    v_max = unicycle.v_max
    omega_max = unicycle.omega_max
    omega_min = unicycle.omega_min

    # If turn1 is not provided, compute it
    turn1 = compute_initial_turn_direction(xc2, yc2, r_nominal, x0, y0, theta0, turn2) if turn1 == 0 else turn1
    # Compute the coordinates of the first circumference
    xc1, yc1 = compute_center_coordinates_first_circle(x0, y0, theta0, turn1, r_nominal)
    # Compute distance between two circles to detect overlap 
    distance_circles = compute_distance_two_points([xc2, yc2], [xc1, yc1])
    if distance_circles < 2 * r_nominal:
        overlap = True
    else:
        overlap = False
    # Compute the extreme poses for each maneuver
    x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line(xc1, yc1, xc2, yc2, turn1, turn2, r_nominal, overlap = overlap)
     
    #Compute orientations for each primitive
    theta0_p1 = theta0
    thetaf_p1 = theta1
    theta_p2 = theta1

    omega1 = omega_max if turn1 > 0 else omega_min

    # Primitive 1: arc
    primitive1 = CurvilinearArcUnicycle(xc=xc1, yc=yc1, x0 = x0, y0 = y0, theta0 = theta0_p1, xf = x1, yf = y1, thetaf = thetaf_p1, radius = r_nominal, turn_direction = turn1, v = v_max, omega = omega1, unicycle = unicycle, t0 = t0, samples_number = 100)
    # Primitive 2: segment
    primitive2 = LinearSegmentUnicycle(x0=x1, y0=y1, xf=x2, yf=y2, theta=theta_p2, v=v_max, t0 = primitive1.tf, unicycle = unicycle, samples_number=10)

    return primitive1, primitive2


def compute_two_maneuvers_within_corridor(corridor, start_pose, unicycle, xc2, yc2, turn2, t0 = 0, turn1 = 0):
    '''
    Compute the two maneuvers required to reach a circumference centered in (xc2, yc2): arc and segment.

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

    :return: primitive1, arc
    :rtype: CurvilinearArcUnicycle
    :return: primitive2, segment
    :rtype: LinearSegmentUnicycle 
    '''
    #Extract variables
    x0, y0, theta0  = start_pose
    r_nominal = unicycle.max_radius
    v_max = unicycle.v_max
    omega_max = unicycle.omega_max
    omega_min = unicycle.omega_min

    # If turn1 is not provided, compute it
    turn1 = compute_initial_turn_direction(xc2, yc2, r_nominal, x0, y0, theta0, turn2) if turn1 == 0 else turn1
    # Compute the coordinates of the first circumference
    xc1, yc1 = compute_center_coordinates_first_circle(x0, y0, theta0, turn1, r_nominal)
    # Check for collision avoidance 
    lw_intersected, rw_intersected, points_lw, points_rw = compute_intersection_points_between_line_circle(xc = xc1, yc = yc1, radius = r_nominal, corridor = corridor, margin = unicycle.width/2) 
    if lw_intersected:
        R1 = get_max_radius(corridor.shrink(unicycle.width*0.5), start_pose, turn1, wall = 'left')
        xc1, yc1 = compute_center_coordinates_first_circle(x0, y0, theta0, turn1, R1)
    elif rw_intersected:
        R1 = get_max_radius(corridor.shrink(unicycle.width*0.5), start_pose, turn1, wall = 'right')
        xc1, yc1 = compute_center_coordinates_first_circle(x0, y0, theta0, turn1, R1)
    else:
        R1 = r_nominal

    R2 = r_nominal
    # Compute the extreme poses for each maneuver
    x1, y1, theta1, x2, y2, theta2 = compute_extreme_poses_arc_line_two_radii(xc1, yc1, xc2, yc2, turn1, turn2, R1, R2)
     
    #Compute orientations for each primitive
    theta0_p1 = theta0
    thetaf_p1 = theta1
    theta_p2 = theta1

    omega1 = omega_max if turn1 > 0 else omega_min

    # Primitive 1: arc
    primitive1 = CurvilinearArcUnicycle(xc=xc1, yc=yc1, x0 = x0, y0 = y0, theta0 = theta0_p1, xf = x1, yf = y1, thetaf = thetaf_p1, radius = R1, turn_direction = turn1, v = v_max, omega = omega1, unicycle = unicycle, t0 = t0, samples_number = 100)
    # Primitive 2: segment
    primitive2 = LinearSegmentUnicycle(x0=x1, y0=y1, xf=x2, yf=y2, theta=theta_p2, v=v_max, t0 = primitive1.tf, unicycle = unicycle, samples_number=10)

    return primitive1, primitive2