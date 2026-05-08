

def compute_arc_from_one_tangent_and_one_pose(segment1, x3, y3, theta3, turn, xc2, yc2, unicycle):
    '''
    Given two segments, compute the arc in between.

    :param segment1: first segment
    :type segment1: LinearSegmentUnicycle
    :param segment2: second segment
    :type segment2: LinearSegmentUnicycle
    :param turn: turn direction along the arc in between the two segments
    :type turn: float [-1, 1]
    :param xc2: x coordinate of the center of the second circumference, to which the arc belongs
    :type xc2: float
    :param yc2: y coordinate of the center of the second circumference, to which the arc belongs
    :type yc2: float
    :param unicycle: considered unicycle vehicle
    :type unicycle: Unicycle
   
    :return: arc in between the two segments
    :rtype: CurvilinearArcUnicycle 
    '''
    x2, y2 = segment1.end_position
    theta2 = segment1.thetaf
    t0 = segment1.tf
    radius = unicycle.max_radius
    omega = unicycle.omega_max if turn > 0 else unicycle.omega_min
    arc = CurvilinearArcUnicycle(xc=xc2, yc=yc2, x0=x2, y0=y2, theta0=theta2, xf=x3, yf=y3, thetaf=theta3, radius=radius , turn_direction=turn, v=unicycle.v_max, omega=omega, unicycle=unicycle, t0=t0, samples_number=10)
    return arc


def compute_trajectory_intersection_case_without_optimization(corridor1, corridor2, start_pose, end_pose, unicycle):
    x0, y0, theta0 = start_pose
    xf, yf, thetaf = end_pose
    # 1- Compute the final turn direction 
    turn2 = compute_turn_direction([cos(corridor2.tilt), sin(corridor2.tilt)], [cos(thetaf), sin(thetaf)])
    if turn2 == 0:
        turn2 = 1
    # 2- Compute the center of the second circle
    xc2, yc2 = compute_center_coordinates_first_circle(xf, yf, thetaf, turn2, unicycle.max_radius)
    # # 3- Compute the initial turn direction
    # turn1 = compute_initial_turn_direction((xc2, yc2, unicycle.max_radius, x0, y0, theta0, turn2))
    # 4- Compute first three maneuvers from start pose to the final circumference 
    turn_on_the_spot1, arc1, segment1 = compute_three_maneuvers_compact(corridor1, corridor2, start_pose, unicycle, xc2, yc2, turn2, t0 = 0, turn1 = 0)
    arc2 = compute_arc_from_one_tangent_and_one_pose(segment1, xf, yf, thetaf, turn2, xc2, yc2, unicycle)
    # 4- Collect the maneuvers and fix the angles
    maneuvers = [turn_on_the_spot1, arc1, segment1, arc2]
    # Fix the angles of the maneuvers
    correct_angles(maneuvers)

    return maneuvers


def compute_trajectory_unicycle_one_corridor(corridor, start_pose, unicycle, end_pose = None):
    '''
    Compute the time-optimal trajectory within a corridor.

    :param corridor: considered corridor
    :type corridor: CorridorWorld

    :param start_pose: initial pose
    :type start_pose: list of floats

    :param unicycle: considered unicycle vehicle
    :type unicycle: Unicycle

    :param end_pose: final pose
    :type end_pose: list of floats

    :return: sequence of primitives
    :rtype: list of primitives
    '''
    fake_corridor = CorridorWorld(width = corridor.width, height = corridor.height, center = [corridor.center[0] + 5 * corridor.width, corridor.center[1] + 5 * corridor.height], tilt = corridor.tilt)
    # 1- Compute the end pose if it is not provided
    end_pose = compute_end_pose(corridor, unicycle, margin = 0.2 * corridor.height) if end_pose is None else end_pose
    xf, yf, thetaf = end_pose
    
    # 2- Compute the final circumference to reach depending on the position of the vehicle on the corridor (on the left side or on the right side)
    turn = get_corridor_side(corridor, start_pose[:2])
    xc2, yc2 = compute_center_coordinates_first_circle(xf, yf, thetaf, turn, unicycle.max_radius)

    # 3- Compute first three maneuvers from start pose to the final circumference
    turn_on_the_spot1, arc1, segment1 = compute_three_maneuvers_compact(corridor, fake_corridor, start_pose, unicycle, xc2, yc2, turn, t0 = 0, turn1 = 0)
    arc2 = compute_arc_from_one_tangent_and_one_pose(segment1, xf, yf, thetaf, turn, xc2, yc2, unicycle)

    # 4- Collect the maneuvers and fix the angles
    if arc2.iota > pi/2:
        maneuvers = [turn_on_the_spot1, arc1, segment1]
    else:
        maneuvers = [turn_on_the_spot1, arc1, segment1, arc2]

    correct_angles(maneuvers)
    return maneuvers


def get_corridor_side(corridor, position):
    '''
    Get the side of the corridor in which the position is located.

    :param corridor: considered corridor
    :type corridor: CorridorWorld
    :param position: position to check
    :type position: list of floats

    :return: side of the corridor in which the position is located
    :rtype: int
    '''
    xl, yl = corridor.center[0] + corridor.width * 0.5 * cos(corridor.tilt + pi * 0.5), corridor.center[1] + corridor.width * 0.5 * sin(corridor.tilt + pi * 0.5)
    corridor_left = CorridorWorld(width = corridor.width, height = corridor.height, center = [xl, yl], tilt = corridor.tilt)
    if check_point_inside_corridor(corridor_left, position):
        return 1
    else:
        return -1