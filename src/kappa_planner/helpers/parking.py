def project_point_on_line(x1, y1, x2, y2, xp, yp):
    '''
    Project a point onto a line defined by two points.

    The function computes the orthogonal projection of a point P = (xp, yp)
    onto the infinite line passing through A = (x1, y1) and B = (x2, y2).

    :param x1: x-coordinate of first point on the line
    :type x1: float
    :param y1: y-coordinate of first point on the line
    :type y1: float
    :param x2: x-coordinate of second point on the line
    :type x2: float
    :param y2: y-coordinate of second point on the line
    :type y2: float
    :param xp: x-coordinate of the point to project
    :type xp: float
    :param yp: y-coordinate of the point to project
    :type yp: float

    :return: projected point (x, y)
    :rtype: tuple of floats
    '''
    # Direction vector of the line AB
    dx, dy = x2 - x1, y2 - y1

    # Handle degenerate case: A and B are the same point
    if dx == 0 and dy == 0:
        return x1, y1

    # Vector from A to P
    ax, ay = xp - x1, yp - y1

    # Projection scalar
    t = (ax * dx + ay * dy) / (dx * dx + dy * dy)

    # Projected point
    x_proj = x1 + t * dx
    y_proj = y1 + t * dy

    return x_proj, y_proj


def parking_maneuver(corridor1, corridor2, turn, start_pose, end_pose, vehicle, tune_park = 0.2):
    """
    Corridor1 is the lane
    corridor2 is the parking spot
    """
    xf, yf, thetaf = end_pose
    x0, y0, _ = start_pose
    R = vehicle.max_radius
    # 1- Compute the pose along the centerline of the parking spot 
    park_dist = tune_park * R

    
    # 2- Compute the center of the circumference around which to perform the backward arc
    # xA, yA = xp, yp 
    # xB, yB = xf, yf
    xA, yA = corridor2.tail
    xB, yB = corridor2.head
    
    x_proj, y_proj = project_point_on_line(xA, yA, xB, yB, x0, y0)
    xp, yp = x_proj - park_dist * cos(corridor2.tilt), y_proj - park_dist* sin(corridor2.tilt)
    thetap = corridor2.tilt
    
    xP, yP = x0, y0
    D = (xB - xA) * (yP - yA) - (yB - yA) * (xP - xA)
    
    if D >= 0:
        val = -1
    else:
        val = 1    
    
    xc2, yc2 = xp + R * cos(corridor2.tilt + val * pi * 0.5), yp + R * sin(corridor2.tilt + val * pi * 0.5)
    
    # 3- Compute the line parallel to the centerline of the first corridor 
    if wrapPositiveAngle(corridor1.tilt) <= pi*0.5 or wrapPositiveAngle(corridor1.tilt) > 3*pi *0.5:
        val2 = 1
    else:
        val2 = -1

    # Calculate the slope
    if wrapPositiveAngle(corridor1.tilt) == pi*0.5 or wrapPositiveAngle(corridor1.tilt) == 3*pi *0.5 : # Vertical line
        # Vertical line
        A, B, C = -1, 0, corridor1.center[0]
    elif wrapPositiveAngle(corridor1.tilt) == 0 or wrapPositiveAngle(corridor1.tilt) == pi : # Horizontal line
        A, B, C = 0, -1, corridor1.center[1]
    else:
        m1 = tan(corridor1.tilt)
        # Calculate the y-intercept (b) using the point-slope form: y = mx + b
        b = corridor1.center[1] - m1 * corridor1.center[0] if m1 != 0 else 0 
        A, B, C = m1, -1, b 
        

    C_prime = C + val2 * turn * R * sqrt(A**2 + B**2)
    
    # 4- Compute the center of the circumference around which to perform the forward arc
    
    if B == 0: #vertical line
        xc1_1 = -C_prime / A
        xc1_2 = -C_prime / A
        a = 1
        b = -2 * yc2
        c = yc2**2 - 4*R**2 -2*xc2*xc1_1 + xc1_1**2 + xc2**2
        
        discr = sqrt(b**2 - 4*a*c)
        yc1_1 = (-b + discr) / (2 * a)
        yc1_2 = (-b - discr) / (2 * a)
        
        dist1 = (xc1_1 - xp)**2 + (yc1_1 - yp)**2
        dist2 = (xc1_2 - xp)**2 + (yc1_2 - yp)**2
    elif A == 0: #horizontal line
        yc1_1 = -C_prime / B
        yc1_2 = -C_prime / B
        a = 1
        b = -2 * xc2
        c = xc2**2 + yc2**2 + yc1_1**2 - 4*R**2 -2*yc2*yc1_1
        
        discr = sqrt(b**2 - 4*a*c)
        xc1_1 = (-b + discr) / (2 * a)
        xc1_2 = (-b - discr) / (2 * a)
        
        dist1 = (xc1_1 - xp)**2 + (yc1_1 - yp)**2
        dist2 = (xc1_2 - xp)**2 + (yc1_2 - yp)**2
    else:
        a = ((B/A)**2 + 1)
        b = ((2*B*C_prime)/A**2 + (2*xc2*B)/A - 2*yc2)
        c = ((C_prime/A)**2 + xc2**2 + yc2**2 + (2*xc2*C_prime)/A - 4*R**2)
        
        discr = sqrt(b**2 - 4*a*c)
        yc1_1 = (-b + discr) / (2 * a)
        yc1_2 = (-b - discr) / (2 * a)
        
        xc1_1 = (-B * yc1_1 - C_prime) / A
        xc1_2 = (-B * yc1_2 - C_prime) / A
            
        dist1 = (xc1_1 - xp)**2 + (yc1_1 - yp)**2
        dist2 = (xc1_2 - xp)**2 + (yc1_2 - yp)**2
    
    if dist1 < dist2:
        xc1, yc1 = xc1_1, yc1_1
    else:
        xc1, yc1 = xc1_2, yc1_2
        
    # 5- Compute the start pose of the backward arc
    xs, ys = (xc1 + xc2) * 0.5, (yc1 + yc2) * 0.5
    thetas = atan2(yc1 - yc2, xc1 - xc2) - turn * pi * 0.5
    
    # 6- Compute the start maneuver of the forward arc
    x_start, y_start = compute_end_first_segment(A, B, C, xc1, yc1, R)
    
    # 6- Compute the maneuvers
    arc1 = CurvilinearArcUnicycle(xc=xc1, yc=yc1, x0=x_start, y0=y_start,
                                  theta0=corridor1.tilt, xf=xs, yf=ys,
                                  thetaf=thetas, radius=R,
                                  turn_direction=turn, v=vehicle.v_max, omega= turn* vehicle.omega_max,
                                  unicycle=vehicle, t0=0, samples_number=10)
    
    arc2 = BackwardArc(xc=xc2, yc=yc2, x0 = xs, y0 = ys,
                                        theta0 = thetas, xf = xp, yf = yp,
                                        thetaf = thetap, radius = R,
                                        turn_direction = -turn, v = -vehicle.v_max,
                                        omega = turn * vehicle.omega_max, bicycle = vehicle,
                                        t0 = arc1.tf, samples_number = 100)
    segment3 = LinearSegmentUnicycle(x0=xp, y0=yp, xf=xf, yf=yf, theta=thetap, v=vehicle.v_max, t0 = arc2.tf, unicycle = vehicle, samples_number=10)
        
    return [arc1, arc2, segment3]


def exit_parking_slot(corridor1, corridor2, start_pose, unicycle, end_pose = None):
    
    # 1- Compute the end pose if it is not provided
    xf, yf, thetaf = compute_end_pose(corridor1, unicycle, margin = 0.1 * corridor1.height) if end_pose is None else end_pose
    int_point, _ = compute_intersection_two_segments(corridor1.tail, corridor1.head, corridor2.tail, corridor2.head)
    xf, yf = int_point
    thetaf = corridor1.tilt
    # 2- Invert the start pose
    start_pose_inv = [start_pose[0], start_pose[1], wrapPositiveAngle(start_pose[2] + pi)]

    # 3- Compute the final circumference to reach depending on the position of the vehicle on the corridor (on the left side or on the right side)
    turn = get_corridor_side(corridor1, start_pose_inv[:2])
    xc2, yc2 = compute_center_coordinates_first_circle(xf, yf, thetaf, turn, unicycle.max_radius)

    # 3- Compute first three maneuvers from start pose to the final circumference
    T1, C1, S1 = compute_three_maneuvers_compact(corridor1, corridor2, start_pose_inv, unicycle, xc2, yc2, turn, t0 = 0, turn1 = 0)
    C2 = compute_arc_from_one_tangent_and_one_pose(S1, xf, yf, thetaf, turn, xc2, yc2, unicycle)

    # 4- Collect the maneuvers and fix the angles
    maneuvers = [T1, C1, S1, C2]
    # Fix the angles of the maneuvers
    maneuvers[3].change_theta0(maneuvers[2].theta) # Change theta arc2 from theta segment1

    return maneuvers


def compute_three_maneuvers_compact_siemens(corridor, corridor2, start_pose, unicycle, xc2, yc2, turn2, t0 = 0, turn1 = 0):
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
    
    # Make the start theta an angle between in [0, 2*pi]
    start_pose[2] = wrapPositiveAngle(start_pose[2])
    # Extract variables
    x0, y0, theta0  = start_pose
    R, v_max, omega_max, omega_min  = unicycle.max_radius, unicycle.v_max, unicycle.omega_max, unicycle.omega_min

    # 1- If turn1 is not provided as input, compute it in the optimal way
    turn1 = compute_initial_turn_direction(xc2, yc2, R, x0, y0, theta0, turn2) if turn1 == 0 else turn1
    
    dist = sqrt((xc2 - x0)**2 + (yc2 - y0)**2)
    dist = 5
    x_prime = x0 + dist * cos(theta0)
    y_prime = y0 + dist * sin(theta0)
    primitive1 = LinearSegmentUnicycle(x0=x0, y0=y0, xf=x_prime, yf=y_prime, theta=theta0, v=v_max, t0 = 0, unicycle = unicycle, samples_number=10)
    
    xc_prime = x_prime + R * cos(theta0 - turn1 * pi * 0.5)
    yc_prime = y_prime + R * sin(theta0 - turn1 * pi * 0.5)
    primitive3 = compute_segment_between_two_circles(xc_prime, yc_prime, xc2, yc2, turn1, turn1, unicycle)    
    primitive2 = compute_arc_from_two_tangents(primitive1, primitive3, turn1, xc_prime, yc_prime, unicycle)

    return primitive1, primitive2, primitive3


def compute_trajectory_unicycle_two_corridors_siemens(corridor1, corridor2, start_pose, end_pose, unicycle, tune_park = 0.2):
    '''
    Compute the sequence of primitives that build the time-optimal trajectory for a unicycle vehicle within two corridors. 

    :param corridor1: first corridor
    :type corridor1: CorridorWorld
    :param corridor2: second corridor
    :type corridor2: CorridorWorld
    :param start_pose: initial pose within the first corridor
    :type start_pose: list of floats
    :param end_pose: final pose within the second corridor
    :type end_pose: list of floats
    :param unicycle: unicycle vehicle
    :type unicycle: Unicycle

    :return: sequence of primitives
    :rtype: list of primitives
    :return: boolean indicating whether an intersection has been detected
    :rtype: Boolean
    '''
    tic = time.perf_counter()
    x0, y0, theta0 = start_pose
    
    ## Merge two corridors if needed NOT NEEDED check is outside
    # if check_merge_corridors(corridor1, corridor2): # If the two corridors can be merged
    #     new_corridor = get_corridor_from_vector(corridor1.tail, corridor2.head, corridor1.width, add_height = 0)
    #     # Compute the trajectory within the new corridor and return false as the check_intersection Boolean
    #     return compute_trajectory_unicycle_one_corridor(new_corridor, start_pose, unicycle, end_pose), False
    tic1 = time.perf_counter() 

    ## Compute main turn direction tau2
    turn_direction = compute_turn_direction(corridor1.vector, corridor2.vector)

    ## Compute the center of the second circumference
    A1, B1, C1, C1_prime, A2, B2, C2, C2_prime, xc, yc = compute_second_circ_center_line(corridor1, corridor2, unicycle.max_radius, turn_direction)
    
    maneuvers = parking_maneuver(corridor1, corridor2, turn_direction, start_pose, end_pose, unicycle, tune_park = tune_park)
    # tic2 = time.perf_counter()
    # print(f"math formulas time {tic2 - tic1}")
    ## Compute the final point of the first segment
    # x_prime, y_prime = compute_end_first_segment(A1, B1, C1, xc, yc, unicycle.max_radius)
    x_prime, y_prime = compute_end_first_segment(A1, B1, C1, maneuvers[0].xc, maneuvers[0].yc, unicycle.max_radius)
    
    segment1 = LinearSegmentUnicycle(x0=x0, y0=y0, xf=maneuvers[0].x0, yf=maneuvers[0].y0, theta=corridor1.tilt, v=unicycle.v_max, t0 = 0, unicycle = unicycle, samples_number=10)
    maneuvers.insert(0, segment1)
    
    # ## Compute the initial point of the second segment
    # x_primef, y_primef = compute_end_first_segment(A2, B2, C2, xc, yc, unicycle.max_radius)
    # segment2 = LinearSegmentUnicycle(x0=x_primef, y0=y_primef, xf=end_pose[0], yf=end_pose[1], theta=corridor2.tilt, v=unicycle.v_max, t0 = 0, unicycle = unicycle, samples_number=10)

    # # Check whether the intersection case occurs NOT NEEDED
    # # check_intersection = check_intersection_case(segment1, segment2)
    # tic3 = time.perf_counter() 
    # ## Compute the arc that connectis the two segments
    # arc2 = compute_arc_from_two_tangents(segment1, segment2, turn_direction, xc, yc, unicycle)
    # segment2.add_time_offset(arc2.tf)
    # # maneuvers= [segment1, arc2, segment2]
    # # if check_intersection:
    # #     maneuvers = compute_trajectory_intersection_case_without_optimization(corridor1, corridor2, start_pose, end_pose, unicycle)
    # # else:

    
    ## Make sure that the orientation of the vehicle is correct across the primitives
    correct_angles(maneuvers)
    tic4 = time.perf_counter()
    return maneuvers, False

def compute_second_circ_center_line(corridor1, corridor2, R, tau):
    corr1_vertical, corr1_horizontal, corr2_vertical, corr2_horizontal = False, False, False, False
    if wrapPositiveAngle(corridor1.tilt) < pi*0.5 or wrapPositiveAngle(corridor1.tilt) >= 3*pi *0.5:
        val1 = 1
    else:
        val1 = -1
        
    if wrapPositiveAngle(corridor2.tilt) < pi*0.5 or wrapPositiveAngle(corridor2.tilt) >= 3*pi *0.5:
        val2 = 1
    else:
        val2 = -1
    # Line equation for the first centerline
    # Define the points
    x1, y1 = corridor1.center # bottom left corner
    # Calculate the slope
    if wrapPositiveAngle(corridor1.tilt) == pi*0.5 or wrapPositiveAngle(corridor1.tilt) == 3*pi *0.5 : # Vertical line
        A1, B1, C1 = -1, 0, x1
        corr1_vertical = True
    elif wrapPositiveAngle(corridor1.tilt) == 0 or wrapPositiveAngle(corridor1.tilt) == pi : # Horizontal line
        A1, B1, C1 = 0, -1, y1
        corr1_horizontal = True
    else:
        m1 = tan(corridor1.tilt)
        # Calculate the y-intercept (b) using the point-slope form: y = mx + b
        b = y1 - m1 * x1 if m1 != 0 else 0 
        A1, B1, C1 = m1, -1, b 
        
    C1_prime = C1 + val1 * tau * R * sqrt(A1**2 + B1**2)
    
    # Define the points
    x2, y2 = corridor2.center # bottom left corner
    # Calculate the slope
    if wrapPositiveAngle(corridor2.tilt) == pi*0.5 or wrapPositiveAngle(corridor2.tilt) == 3*pi *0.5 : # Vertical line
        # Vertical line
        A2, B2, C2 = -1, 0, x2
        corr2_vertical = True
    elif wrapPositiveAngle(corridor2.tilt) == 0 or wrapPositiveAngle(corridor2.tilt) == pi : # Horizontal line
        A2, B2, C2 = 0, -1, y2
        corr2_horizontal = True
    else:
        m2 = tan(corridor2.tilt)
        # Calculate the y-intercept (b) using the point-slope form: y = mx + b
        b2 = y2 - m2 * x2 if m2 != 0 else 0 
        A2, B2, C2 = m2, -1, b2 
        
    C2_prime = C2 + val2 * tau * R * sqrt(A2**2 + B2**2)
    
    # figure = plot_corridors([corridor1, corridor2])
    # x = np.linspace(-50, 50, 10)
    # x1 = np.linspace(-50, 50, 10)
    # x2 = np.linspace(-50, 50, 10)
    # if A1 == 0:
    #     y1 = -C1_prime/B1 * np.ones(10)
    # elif B1 == 0: 
    #     y1 = np.linspace(-50, 50, 10)
    #     x1 = -C1_prime/A1 * np.ones(10)
    # else:
    #     y1 = (-A1 * x - C1_prime) / B1
    # if A2 == 0:
    #     y2 = -C2_prime/B2 * np.ones(10)
    # elif B2 == 0: 
    #     y2 = np.linspace(-50, 50, 10)
    #     x2 = -C2_prime/A2 * np.ones(10)
    # else:     
    #     y2 = (-A2 * x - C2_prime) / B2
        
    # plt.plot(x1, y1, 'b')
    # plt.plot(x2, y2, 'r')
    # plt.show(block = True)
    
    # # Define the variables
    # xc, yc = sp.symbols('xc yc')
    
    # line1_equation = A1 * xc + B1 * yc + C1_prime
    # line2_equation = A2 * xc + B2 * yc + C2_prime
    
    # # Solve for the intersection point
    # sol = sp.solve([line1_equation, line2_equation], (xc, yc))
    # xc = float(sol[xc])
    # yc = float(sol[yc])
    
    # if A1 == 0: 
    #     xc = C2_prime
    #     yc = B1

    # Case 1 
    if corr1_horizontal and not(corr2_horizontal) and not(corr2_vertical):
        yc = -C1_prime/B1
        xc = (-B2 * yc - C2_prime) / A2
    elif corr1_vertical and not(corr2_horizontal) and not(corr2_vertical):
        xc = -C1_prime/A1
        yc = (-A2 * xc - C2_prime) / B2
    elif corr2_horizontal and not(corr1_horizontal) and not(corr1_vertical):
        yc = -C2_prime/B2
        xc = (-B1 * yc - C1_prime) / A1
    elif corr2_vertical and not(corr1_horizontal) and not(corr1_vertical):
        xc = -C2_prime/A2
        yc = (-A1 * xc - C1_prime) / B1
    elif corr1_horizontal and corr2_vertical:
        yc = -C1_prime/B1
        xc = -C2_prime/A2
    elif corr1_vertical and corr2_horizontal:
        yc = -C2_prime/B2
        xc = -C1_prime/A1
    else:      
        yc = ((A2 * C1_prime)/A1 - C2_prime) / (B2 - (A2 * B1)/A1)
        xc = (-B1 * yc - C1_prime) / A1
        
    return A1, B1, C1, C1_prime, A2, B2, C2, C2_prime, xc, yc


def compute_end_first_segment(A1, B1, C1, xc, yc, R):
    # Define the points
    # x1, y1 = corridor1.center # bottom left corner
    # # Calculate the slope
    # if wrapPositiveAngle(corridor1.tilt) == pi*0.5 or wrapPositiveAngle(corridor1.tilt) == 3*pi *0.5 : # Vertical line
    #     # Vertical line
    #     A1, B1, C1 = 1, 0, x1
    # else:
    #     m1 = tan(corridor1.tilt)
    #     # Calculate the y-intercept (b) using the point-slope form: y = mx + b
    #     b = y1 - m1 * x1 if m1 != 0 else 0 
    #     A1, B1, C1 = m1, -1, b 
    
    if A1 == 0:
        y_line = -C1 / B1
        x_line = xc
    else:   
        a_sol = (B1/A1)**2 + 1
        b_sol = ((2*B1*C1)/A1**2 +(2*xc*B1)/A1 - 2*yc)
        
        y_line = -b_sol / (2 * a_sol)
        x_line = (-B1 * y_line - C1) / A1
            
    # x_line, y_line = sp.symbols('x_line y_line')
    # circ_equation = (x_line - xc)**2 + (y_line - yc)**2 - R**2
    # line_equation = A1 * x_line + B1 * y_line + C1
    # sol = sp.solve([line_equation, circ_equation], (x_line, y_line))
    try:
        # return float(sol[0][0]), float(sol[1][1])
        return x_line, y_line
    except:
        # return float(sp.re(sol[1][0])), float(sp.re(sol[0][1]))
        return x_line, y_line
        # figure = plot_corridors([corridor1])
        # x = np.linspace(-1000, 1000, 100)
        # y1 = (-A1 * x - C1) / B1
        # array_theta = np.linspace(0, 2*np.pi, 100)
        # x_circ = xc + R * np.cos(array_theta)
        # y_circ = yc + R * np.sin(array_theta)

        # plt.plot(xc, yc, 'ro')
        # plt.plot(x_circ, y_circ, 'r')
        # plt.plot(x, y1, 'b')
        # plt.show(block = True)
        # print("Help!!")


### For Siemens
def compute_trajectory_multiple_corridors_siemens(corridor_list, start_pose, end_pose, unicycle, prev_corridor_list = None, prev_trajectory = None, tune_park = 0.2):
    # Initialize vectors with the maneuvers. 
    # For n corridors, the sequence of maneuvers will be:
    # T1, C1, S1, C2, S2, C3, S3, ..., Ci, Si, ..., Cn, Sn, Cn+1, T2
    # Store segments from S2 until Sn-1 (n-2 segments)
    # Store arcs from C3 until Cn-1 (n-3 arcs)
    tic = time.perf_counter() 
    
    ## Given the corridor list, compute the start and end pose
    end_pose = compute_end_pose(corridor_list[-1], unicycle, margin = 0.03 * corridor_list[0].height + 1)
    tic1 = time.perf_counter() 
    
    margin = 0.02 # This margin is used to compute the center of the circumferences both at the intersection between corridors and for obstacle avoidance
    tol = 1e-2

    # Check whether two consecutive corridors are parallel and have the same width. In this case merge them together
    corridor_to_remove = []
    for i in range(len(corridor_list)-1):
        if (abs(corridor_list[i].tilt - corridor_list[i+1].tilt) < tol) and ((wrapPositiveAngle(atan2(corridor_list[i+1].center[1] - corridor_list[i].center[1], corridor_list[i+1].center[0] - corridor_list[i].center[0])) - wrapPositiveAngle(corridor_list[i].tilt))) < tol and (corridor_list[i].width - corridor_list[i+1].width < 1e-2):
            new_corridor = get_corridor_from_vector(corridor_list[i].tail, corridor_list[i+1].head, corridor_list[i].width, add_height = 0)
            corridor_list[i] = new_corridor
            corridor_to_remove = corridor_to_remove + [i+1]
    ind = 0    
    for index in corridor_to_remove:
        corridor_list.pop(index - ind)
        ind += 1
    tic3 = time.perf_counter() 
    
    # Compute the start pose
    x1, y1 = corridor_list[0].center # bottom left corner
    # Calculate the slope
    if wrapPositiveAngle(corridor_list[0].tilt) == pi*0.5 or wrapPositiveAngle(corridor_list[0].tilt) == 3*pi *0.5 : # Vertical line
        # Vertical line
        A, B, C = 1, 0, x1
    else:
        m1 = tan(corridor_list[0].tilt)
        # Calculate the y-intercept (b) using the point-slope form: y = mx + b
        b1 = y1 - m1 * x1 if m1 != 0 else 0 
        A, B, C = m1, -1, b1
    
    # x_start = B1 * (B1 * start_pose[0] - A1 * start_pose[1]) - A1 * C1
    # y_start = A1 * (-B1 * start_pose[0] + A1 * start_pose[1]) - B1 * C1
    
    denom = A**2 + B**2
    x0, y0 = start_pose[0], start_pose[1]
    # Compute the projection coordinates
    x_start = x0 - (A * (A*x0 + B*y0 + C)) / denom
    if A == 0: 
        y_start = corridor_list[0].center[1]
    else:
        y_start = y0 - (B * (A*x0 + B*y0 + C)) / denom
    tic2 = time.perf_counter() 
    
    start_pose = [x_start, y_start, corridor_list[0].tilt]
    # if prev_corridor_list != None:
    #     equal = 0 
    #     if len(prev_corridor_list) == len(corridor_list):
    #         for ind, corridor in enumerate(prev_corridor_list):
    #             if corridor == corridor_list[ind]:
    #                 equal += 1
    #         if equal == len(corridor_list):
    #             new_trajectory = prev_trajectory
    #             x_prime, y_prime = prev_trajectory[0].xf, prev_trajectory[0].yf
    #             new_trajectory[0] = LinearSegmentUnicycle(x0=x0, y0=y0, xf=x_prime, yf=y_prime, theta=corridor_list[0].tilt, v=unicycle.v_max, t0 = 0, unicycle = unicycle, samples_number=10)
    #             return new_trajectory

    if len(corridor_list) == 1:
        return compute_trajectory_unicycle_one_corridor(corridor_list[0], start_pose, unicycle, end_pose)
    elif len(corridor_list) == 2:
        trajectory , _ = compute_trajectory_unicycle_two_corridors_siemens(corridor_list[0], corridor_list[1], start_pose, end_pose, unicycle)
        tic4 = time.perf_counter()
        print(f'''Time to compute the trajectory: {tic4 - tic}''')    
        return [0,0], trajectory
    
    # 1- Compute the turn directions
    turn_direction_vector = compute_turn_direction_vector(corridor_list)
    
    # 2- Compute the corner points NOT NEEDED
    # corner_point_vector = compute_corner_point_vector(corridor_list, turn_direction_vector)
    
    # 3- Compute the centers of the circumferences 
    center_circumference_vector, A1, B1, C1, A2, B2, C2 = compute_center_coordinates_vector_siemens(corridor_list, turn_direction_vector, unicycle)
    
    # figure = plot_corridors(corridor_list)
    # for element in center_circumference_vector:
    #     plt.plot(element[0], element[1], 'ro')
    # plt.show(block = True)
    # # 4- Change position of the second center if the start pose is inside the second circle
    # if obstacle_radius != None and check_point_inside_corridor(corridor_list[0], obstacle_center):
    #     beta = atan2(obstacle_center[1] - start_pose[1], obstacle_center[0] - start_pose[0])
    #     # turn_direction_obstacle = compute_turn_direction([cos(start_pose[2]), sin(start_pose[2])], [cos(beta), sin(beta)])
    #     turn_direction_obstacle = get_corridor_side(corridor_list[0], obstacle_center)
    #     r_prime = unicycle.max_radius - (unicycle.width * 0.5 + obstacle_radius + margin)
    #     x_prime = obstacle_center[0] + r_prime * cos(beta + turn_direction_obstacle * pi * 0.5)
    #     y_prime = obstacle_center[1] + r_prime * sin(beta + turn_direction_obstacle * pi * 0.5)
    #     turn_direction_vector.insert(0, turn_direction_obstacle)
    #     center_circumference_vector.insert(0, [x_prime, y_prime])
    # # xc2, yc2 = center_circumference_vector[0]
    # if ((center_circumference_vector[0][0] - start_pose[0])**2 + (center_circumference_vector[0][1] - start_pose[1])**2) < unicycle.max_radius**2:
    #     xc2, yc2 = start_inside_second_circle_case(corridor_list[0], corridor_list[1], turn_direction_vector[0], corner_point_vector[0], unicycle)
    #     center_circumference_vector[0] = [xc2, yc2]
    # # Eliminate the third circle if there's an initial u-turn
    # # if (abs(compute_angular_difference(corridor_list[0].tilt, corridor_list[1].tilt) - pi/2) < 1e-5 and abs(compute_angular_difference(corridor_list[1].tilt, corridor_list[2].tilt) - pi/2) < 1e-5) or (abs(compute_angular_difference(corridor_list[0].tilt, corridor_list[1].tilt) + pi/2) < 1e-5 and abs(compute_angular_difference(corridor_list[1].tilt, corridor_list[2].tilt) + pi/2) < 1e-5):
    # if is_it_u_turn(corridor_list[0], corridor_list[2], turn_direction_vector[0], turn_direction_vector[1]):
    #     # x1, y1 = center_circumference_vector[0]
    #     # x2, y2 = center_circumference_vector[1]
    #     center_circumference_vector[0] = [(center_circumference_vector[0][0] + center_circumference_vector[1][0]) * 0.5,
    #                                       (center_circumference_vector[0][1] + center_circumference_vector[1][1]) * 0.5]
    #     turn_direction_vector.pop(1)
    #     corner_point_vector.pop(1)
    #     center_circumference_vector.pop(1)
        
    # # 5- Change the position of the penultimate circle if the end pose is inside the penultimate circle
    # xcn_minus_1, ycn_minus_1 = center_circumference_vector[-1]
    # if ((xcn_minus_1 - end_pose[0])**2 + (ycn_minus_1 - end_pose[1])**2) < unicycle.max_radius**2:
    #     xcn_minus_1, ycn_minus_1 = start_inside_second_circle_case(*invert_inputs_start_inside_second_circle_case(corridor_list[-1], corridor_list[-2], turn_direction_vector[-1], corner_point_vector[-1], unicycle))
    #     center_circumference_vector[-1] = [xcn_minus_1, ycn_minus_1]
    # # Eliminate the last third circle if there's a final u-turn
    # if is_it_u_turn(corridor_list[-1], corridor_list[-3], turn_direction_vector[-1], turn_direction_vector[-2]):
    #     center_circumference_vector[-1] = [(center_circumference_vector[-1][0] + center_circumference_vector[-2][0]) * 0.5, 
    #                                        (center_circumference_vector[-1][1] + center_circumference_vector[-2][1]) * 0.5]
    #     turn_direction_vector.pop(-2)
    #     corner_point_vector.pop(-2)
    #     center_circumference_vector.pop(-2)
    # # Eliminate the circles in case of a u-turn
    # center_circumference_to_remove, turn_direction_to_remove, corner_point_to_remove = [], [], []        
    # for i in range(1, len(corridor_list)-2):
    #     if is_it_u_turn(corridor_list[i], corridor_list[i+2], turn_direction_vector[i], turn_direction_vector[i+1]):
    #         center_circumference_vector[i] = [(center_circumference_vector[i][0] + center_circumference_vector[i + 1][0]) * 0.5,
    #                                           (center_circumference_vector[i][1] + center_circumference_vector[i + 1][1]) * 0.5]
    #         center_circumference_to_remove = center_circumference_to_remove + [center_circumference_vector[i + 1]]
    #         turn_direction_to_remove = turn_direction_to_remove + [i + 1]
    #         corner_point_to_remove = corner_point_to_remove + [corner_point_vector[i + 1]]
    # for element in center_circumference_to_remove:
    #     while element in center_circumference_vector:
    #         center_circumference_vector.remove(element)

    # ind = 0    
    # for index in turn_direction_to_remove:
    #     turn_direction_vector.pop(index - ind)
    #     ind += 1
            
    center_circumference_vector, turn_direction_vector = swap_circles(turn_direction_vector, center_circumference_vector, unicycle.max_radius, [start_pose[:2]] + center_circumference_vector + [end_pose[:2]])
    
    # 6- Compute the segments
    segments = [0] * (len(turn_direction_vector)-1)
    for i in range(len(turn_direction_vector)-1):
        segments[i] = compute_segment_between_two_circles(center_circumference_vector[i][0], center_circumference_vector[i][1],
                                                          center_circumference_vector[i+1][0], center_circumference_vector[i+1][1],
                                                          turn_direction_vector[i], turn_direction_vector[i+1], unicycle)
    
    # 7- Check whether there's an intersection between the segments
    segments_to_remove, center_circumference_to_remove, turn_direction_to_remove = [], [], []
    for i in range(len(segments)-1):
        if check_intersection_case(segments[i], segments[i+1]):
            # Substite Si (which goes from Ci to Ci+1) with another Si which goes from Ci to Ci+2
            segments[i] = compute_segment_between_two_circles(center_circumference_vector[i][0], center_circumference_vector[i][1],
                                                              center_circumference_vector[i+2][0], center_circumference_vector[i+2][1],
                                                              turn_direction_vector[i], turn_direction_vector[i+2], unicycle)
            segments_to_remove = segments_to_remove + [segments[i+1]]
            center_circumference_to_remove = center_circumference_to_remove + [center_circumference_vector[i+1]]
            turn_direction_to_remove = turn_direction_to_remove + [i+1]
    
    for element in segments_to_remove:
        while element in segments:
            segments.remove(element)

    for element in center_circumference_to_remove:
        while element in center_circumference_vector:
            center_circumference_vector.remove(element)

    ind = 0    
    for index in turn_direction_to_remove:
        turn_direction_vector.pop(index - ind)
        ind += 1
    
    # 8- Compute the arcs
    arcs = [0] * (len(segments)-1) 
    for i in range(len(segments)-1): 
        arcs[i] = compute_arc_from_two_tangents(segments[i], segments[i+1], turn_direction_vector[i+1], center_circumference_vector[i+1][0], center_circumference_vector[i+1][1], unicycle)
    
    # 9- Compute the first three maneuvers
    x0, y0, theta0 = start_pose
    x_prime, y_prime = compute_end_first_segment(A1, B1, C1, center_circumference_vector[0][0], center_circumference_vector[0][1], unicycle.max_radius)

    S1 = LinearSegmentUnicycle(x0=x0, y0=y0, xf=x_prime, yf=y_prime, theta=corridor_list[0].tilt, v=unicycle.v_max, t0 = 0, unicycle = unicycle, samples_number=10)
    
    # x_primef, y_primef = compute_end_first_segment(A2, B2, C2, center_circumference_vector[-1][0], center_circumference_vector[-1][1], unicycle.max_radius)

    # Sn = LinearSegmentUnicycle(x0=x_primef, y0=y_primef, xf=end_pose[0], yf=end_pose[1], theta=corridor_list[-1].tilt, v=unicycle.v_max, t0 = 0, unicycle = unicycle, samples_number=10)
    # # T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0], corridor_list[1], start_pose, unicycle, center_circumference_vector[0][0], center_circumference_vector[0][1], turn_direction_vector[0], t0 = 0, turn1 = 0)
    # # S1, C1, S2 = compute_three_maneuvers_compact_siemens(corridor_list[0], corridor_list[1], start_pose, unicycle, center_circumference_vector[0][0], center_circumference_vector[0][1], turn_direction_vector[0], t0 = 0, turn1 = 0)
    

    
    # 10- Check whether the intersection case occurs
    # if check_intersection_case(S1, segments[0]):
    #     corridor_list.pop(1)
    #     turn_direction_vector.pop(0)
    #     center_circumference_vector.pop(0)
    #     segments.pop(0)
    #     arcs.pop(0)
    #     T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0], corridor_list[1], start_pose, unicycle, center_circumference_vector[0][0], center_circumference_vector[0][1], turn_direction_vector[0], t0 = 0, turn1 = 0)
   
    C2 = compute_arc_from_two_tangents(S1, segments[0], turn_direction_vector[0], center_circumference_vector[0][0], center_circumference_vector[0][1], unicycle)
    
    # 11- Compute last three maneuvers from end pose to the second circumference and invert them
    # T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1], corridor_list[-2], end_pose, unicycle, center_circumference_vector[-1][0], center_circumference_vector[-1][1], turn_direction_vector[-1], t0 = 0, turn1 = 0))
    # Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
    # # 12- Check whether the intersection case occurs
    # if check_intersection_case(segments[-1], Sn):
    #     corridor_list.pop(-2)
    #     turn_direction_vector.pop(-1)
    #     center_circumference_vector.pop(-1)
    #     segments.pop(-1)
    #     arcs.pop(-1)
    #     T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1], corridor_list[-2], end_pose, unicycle, center_circumference_vector[-1][0], center_circumference_vector[-1][1], turn_direction_vector[-1], t0 = 0, turn1 = 0))
    #     Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
    # Cn = compute_arc_from_two_tangents(segments[-1], Sn, turn_direction_vector[-1], center_circumference_vector[-1][0], center_circumference_vector[-1][1], unicycle)
    
    # 13- Collect the maneuvers
    central_sequence = [0] * (len(segments) + len(arcs))
    j = 0
    for i in range(0, len(central_sequence), 2):
        central_sequence[i]= segments[j]
        j +=1
    j = 0
    for i in range(1, len(central_sequence), 2):
        central_sequence[i]= arcs[j]
        j += 1
    # maneuvers = [primitive1, T1, C1, S1, C2] + central_sequence + [Cn, Sn, Cn_plus_1, T2]
    
    last_maneuvers, _ = compute_trajectory_unicycle_two_corridors_siemens(corridor_list[-2], 
                                                                       corridor_list[-1], 
                                                                       central_sequence[-1].start_pose,
                                                                       end_pose, unicycle, tune_park = tune_park)
    maneuvers = [S1, C2] + central_sequence[:-1] + last_maneuvers#[Cn, Sn]
    
    correct_angles(maneuvers)
    print("executed siemens")
    return center_circumference_vector, maneuvers


def compute_center_coordinates_vector_siemens(corridor_list, turn_direction_vector, unicycle):
    number_corridors = len(corridor_list)
    center_coordinates_vector = [0] * (number_corridors -1)
    for i in range(len(corridor_list)-1):
        A1, B1, C1, C1_prime, A2, B2, C2, C2_prime, xc, yc = compute_second_circ_center_line(corridor_list[i], corridor_list[i+1], unicycle.max_radius, turn_direction_vector[i])
        center_coordinates_vector[i] = [xc, yc]
        if i == 0: 
            A1_sol, B1_sol, C1_sol = A1, B1, C1
        elif i == number_corridors -2:
            A2_sol, B2_sol, C2_sol = A2, B2, C2
            
    return center_coordinates_vector, A1_sol, B1_sol, C1_sol, A2_sol, B2_sol, C2_sol


def send_velocity_commands(start_velocity, end_velocity, maximum_velocity_on_curve, maximum_linear_velocity, linear_acceleration, acceleration_parking, max_parking_velocity, trajectory):
        """
        Send velocity commands to the robot based on the computed trajectory.
        
        :param start_velocity: float, the initial velocity of the robot
        :param end_velocity: float, the final velocity of the robot
        :param maximum_velocity_on_curve: float, the maximum velocity on the curve
        :param maximum_linear_velocity: float, the maximum linear velocity
        :param linear_acceleration: float, the linear acceleration
        :param max_parking_velocity: float, the maximum velocity during parking
        :param trajectory: list, the computed trajectory
        
        :return: list, the velocity commands
        """
        
        # Extract the segments and arcs from the trajectory
        trajectory_until_parking = trajectory[:-4] # Remove the last segment and the two arcs
        segments = trajectory_until_parking[::4]
        arcs = trajectory_until_parking[2::4]
        
        velocity_commands = []
        if len(segments) == 1:
            maximum_velocity_on_curve = max_parking_velocity
            for index, segment in enumerate(segments):
                arc = arcs[index]
                velocity_commands = velocity_commands + compute_velocity_commands_segment_arc(segment, arc, start_velocity, maximum_linear_velocity, maximum_velocity_on_curve, linear_acceleration)
                start_velocity = maximum_velocity_on_curve
                    
        else:
            for index, segment in enumerate(segments):
                arc = arcs[index]
                velocity_commands = velocity_commands + compute_velocity_commands_segment_arc(segment, arc, start_velocity, maximum_linear_velocity, maximum_velocity_on_curve, linear_acceleration)
                start_velocity = maximum_velocity_on_curve
                if index == len(segments) - 2: 
                    maximum_velocity_on_curve = max_parking_velocity
        
        # Remove the last arc
        velocity_commands.pop(-1)
        velocity_commands.pop(-1)
        velocity_commands.pop(-1)
        
        # [-6, -5, -4, -3, -2, -1] -> -2: final segment, -4: final bw arc, -6: final fw arc
        # Compute the velocity profile for the final forward arc
        # from maximum_velocity_on_curve to 0
        final_fw_arc = trajectory[-6] 
        velocity_commands = velocity_commands + compute_velocity_commands_segment(final_fw_arc, start_velocity, max_parking_velocity, 0, acceleration_parking)
        final_bw_arc = trajectory[-4] 
        bw_commands = compute_velocity_commands_segment(final_bw_arc, 0, max_parking_velocity, 0, acceleration_parking)
        # bw_commands[::2] = -bw_commands[::2]
        bw_commands[::3] = [-x for x in bw_commands[::3]]
        bw_commands[1::3] = [-x for x in bw_commands[1::3]]

        velocity_commands = velocity_commands + bw_commands
        final_segment = trajectory[-2]
        velocity_commands = velocity_commands + compute_velocity_commands_segment(final_segment, 0, max_parking_velocity, end_velocity, acceleration_parking)
        
        return velocity_commands
            
def compute_velocity_commands_segment_arc(segment, arc, start_velocity, maximum_linear_velocity, maximum_velocity_on_curve, linear_acceleration):
    v_max = maximum_linear_velocity
    v_min = maximum_velocity_on_curve
    acc = linear_acceleration
    dec = -linear_acceleration
    
    delta_t1 = (v_max - start_velocity) / acc
    delta_t2 = (v_min - v_max) / dec
    # Compute the distance covered during the acceleration and deceleration phases
    # acceleration phase: from start_velocity to v_max
    # deceleration phase: from v_max to v_min
    dist1 = abs(start_velocity * delta_t1 + 0.5 * acc * delta_t1**2)
    dist2 = abs(v_max * delta_t2 + 0.5 * dec * delta_t2**2)
    
    # If the length of the segment is such that the two distances can be covered
    if segment > dist1 + dist2:
        time_segment = (segment - (dist1 + dist2)) / v_max
        time_arc = arc / v_min
        # Send trapeizoidal velocity commands
        vel_commands = [start_velocity, acc, delta_t1,
                        v_max, 0, time_segment,
                        v_max, dec, delta_t2, 
                        v_min, 0, time_arc]

    else: 
        # Compute maximum velocity that can be reached
        numerator = start_velocity**2 + v_min**2 + 2 * acc * segment
        
        v_max = sqrt(numerator / 2)
        
        v_max = sqrt(acc * segment + (start_velocity**2 + v_min**2) / 2)
        # Compute the distance covered during the acceleration and deceleration phases
        # acceleration phase: from start_velocity to v_max
        # deceleration phase: from v_max to v_min
        delta_t1 = (v_max - start_velocity) / acc
        delta_t2 = (v_min - v_max) / dec
        dist1 = abs(start_velocity * delta_t1 + 0.5 * acc * delta_t1**2)
        dist2 = abs(v_max * delta_t2 + 0.5 * dec * delta_t2**2)
        time_arc = arc / v_min
        # Send triangular velocity profile
        vel_commands = [start_velocity, acc, delta_t1,
                        v_max, dec, delta_t2, 
                        v_min, 0, time_arc] 
    
    return vel_commands

def compute_velocity_commands_segment(segment, start_velocity, max_parking_velocity, end_velocity, linear_acceleration):
    v_max = max_parking_velocity
    v_min = end_velocity
    acc = linear_acceleration
    dec = -linear_acceleration
    
    delta_t1 = (v_max - start_velocity) / acc
    delta_t2 = (v_min - v_max) / dec
    dist1 = abs(start_velocity * delta_t1 + 0.5 * acc * delta_t1**2)
    dist2 = abs(v_max * delta_t2 + 0.5 * dec * delta_t2**2)
    
    if segment > dist1 + dist2:
        time_segment = (segment - (dist1 + dist2)) / v_max
        vel_commands = [start_velocity, acc, delta_t1,
                        v_max, 0, time_segment,
                        v_max, dec, delta_t2]
    else: 
        # Compute maximum velocity that can be reached
        numerator = start_velocity**2 + v_min**2 + 2 * acc * segment
        v_max = sqrt(numerator / 2)
        delta_t1 = (v_max - start_velocity) / acc
        delta_t2 = (v_min - v_max) / dec
        dist1 = abs(start_velocity * delta_t1 + 0.5 * acc * delta_t1**2)
        dist2 = abs(v_max * delta_t2 + 0.5 * dec * delta_t2**2)
        vel_commands = [start_velocity, acc, delta_t1,
                        v_max, dec, delta_t2] 
    
    return vel_commands



def send_velocity_commands_space(trajectory, start_velocity, end_velocity, maximum_velocity_on_curve, maximum_linear_velocity, linear_acceleration, delta_max):
    v_max = maximum_linear_velocity
    v_min = maximum_velocity_on_curve
    acc = linear_acceleration
    dec = -linear_acceleration
    
    velocity_commands = []
    acceleration_commands = []
    delta_commands = []
    traveled_distance = 0
    distance_array = []
    for index, maneuver in enumerate(trajectory):
        if isinstance(maneuver, LinearSegmentUnicycle):
            segment_length = maneuver.path_length
            delta_t1 = (v_max - start_velocity) / acc
            delta_t2 = (v_min - v_max) / dec
            # Compute the distance covered during the acceleration and deceleration phases
            # acceleration phase: from start_velocity to v_max
            # deceleration phase: from v_max to v_min
            dist1 = abs(start_velocity * delta_t1 + 0.5 * acc * delta_t1**2)
            dist2 = abs(v_max * delta_t2 + 0.5 * dec * delta_t2**2)
            
                # If the length of the segment is such that the two distances can be covered
            if segment_length > dist1 + dist2:
                dist_mid = segment_length - (dist1 + dist2)
                # Send trapeizoidal velocity commands
                vel_commands = [start_velocity, v_max, v_max]
                acc_commands = [acc, 0, dec]
                del_commands = [0, 0, 0]
                dist_array = [dist1, dist_mid, dist2]
    
            else: 
                # Compute maximum velocity that can be reached
                numerator = start_velocity**2 + v_min**2 + 2 * acc * segment_length
                v_max = sqrt(numerator / 2)
                # v_max = sqrt(acc * segment + (start_velocity**2 + v_min**2) / 2)
                # Compute the distance covered during the acceleration and deceleration phases
                # acceleration phase: from start_velocity to v_max
                # deceleration phase: from v_max to v_min
                delta_t1 = (v_max - start_velocity) / acc
                delta_t2 = (v_min - v_max) / dec
                dist1 = abs(start_velocity * delta_t1 + 0.5 * acc * delta_t1**2)
                dist2 = abs(v_max * delta_t2 + 0.5 * dec * delta_t2**2)
                # Send triangular velocity profile
                vel_commands = [start_velocity, v_max]
                acc_commands = [acc, dec]
                del_commands = [0, 0]
                dist_array = [dist1, dist2]
            
        elif isinstance(maneuver, CurvilinearArcUnicycle):
            vel_commands = [v_min]
            acc_commands = [0]
            del_commands = [maneuver.turn_direction * delta_max]
            dist_array = [maneuver.path_length]
            
        velocity_commands = velocity_commands + vel_commands
        acceleration_commands = acceleration_commands + acc_commands
        delta_commands = delta_commands + del_commands
        distance_array = distance_array + dist_array
        
        start_velocity = v_min
        if index == len(trajectory) - 1:
            v_min = end_velocity     
    
    return velocity_commands, acceleration_commands, delta_commands, distance_array


def compute_distance_traveled_vector(velocity_commands, delta_commands, s_array, v_max = 5, dt = 0.05, s_start = 0):

    # Compute the total time
    time_commands = velocity_commands[2::3] # array that contains the time duration of each maneuver
    total_time = sum(time_commands)
    time_commands.pop(3)
    # Compute the cumulative sum of the time duration of each maneuver
    time_commands_sum = [0]
    for time_quantity in time_commands:
        time_commands_sum.append(time_commands_sum[-1] + time_quantity)
    time_commands_sum.pop(0)

    # Generate the time array containg the points in time 
    time_array = np.arange(0, total_time, dt)
    
    # Extract the velocity commands
    velocities = velocity_commands[0::3]
    
    # Extract the acceleration commands
    accelerations = velocity_commands[1::3]
    
    velocities.pop(3)
    accelerations.pop(3)
    # Initialize the index of maneuver
    ind_vel = 0
    
    # Set initial velocity and acceleration
    v, a = velocities[ind_vel], accelerations[ind_vel]
    
    # Initialize the velocity, acceleration and distance traveled output arrays
    v_in_time = []
    a_in_time = []
    st_array = []
        
    # Main cycle that computes the distance traveled at each time instant
    for ind, time in enumerate(time_array):
        s_traveled = s_start + abs(v * dt + 0.5 * a * dt**2)
        if s_traveled > s_array[-1]: # Avoid problems due to discretization
            print(time)
            time_array = time_array[:ind]
            break
        # Store the found quantities in the output arrays
        st_array.append(s_traveled)
        v_in_time.append(v) 
        a_in_time.append(a) 
        s_start = s_traveled
        # Move to the next maneuver in case the time assigned to the current one is over
        if time > time_commands_sum[ind_vel]:
            ind_vel += 1
            a = accelerations[ind_vel]
            stop = 1
        # Compute the velocity at next time instant as a function of current v and a
        v = v + a * dt
        # if velocities[ind_vel]> 0:
        #     if v > velocities[ind_vel]:
        #         v = velocities[ind_vel]
        # elif velocities[ind_vel] < 0:
        #     if v < velocities[ind_vel]:
        #         v = velocities[ind_vel]
        if v > v_max: 
            v = v_max
            
    ## Compute delta commands in time 
        # Extract delta commands
    deltas = delta_commands[::3]
    
    # array that contains the time duration of each maneuver
    time_deltas = delta_commands[2::3]
    
    
    time_commands_sum = [0]
    for time_quantity in time_deltas:
        time_commands_sum.append(time_commands_sum[-1] + time_quantity)
    time_commands_sum.pop(0)
    
    # Initialize output array
    delta_in_time = []
    
    # Set initial delta
    delta = deltas[0]  
    
    # Initialize index maneuver  
    ind_delta = 0
    
    for ind, time in enumerate(time_array):
        delta_in_time.append(delta)
        if time >= time_commands_sum[ind_delta]:
            ind_delta += 1
            delta = deltas[ind_delta]
            
    return st_array, v_in_time, a_in_time, delta_in_time, time_array


def compute_distance_with_vel_commands(vel_commands, s_start = 0):
    # The distance traveled along an arc of length s or a segment of length s is
    # s = v0 * t + 0.5 * a * t^2
    
    velocities = vel_commands[::3]
    accelerations = vel_commands[1::3]
    time = vel_commands[2::3]
    dist_vector = []
    
    for index, v in enumerate(velocities):
        dist = v * time[index] + 0.5 * accelerations[index] * time[index]**2
        dist_vector.append(abs(dist))
        
    total_dist = sum(dist_vector) + s_start

    return total_dist, dist_vector

def send_delta_commands(vel_commands, delta_max, trajectory, dist_vector):
    time = vel_commands[2::3]
    traj_length = trajectory[::2]
    traj_radius = trajectory[1::2]
    dist_to_check = 0
    time_command = 0
    delta_commands = []
    dist_vector2 = []
    ind = 0
    for index, trajectory_piece in enumerate(traj_length):
        while abs(dist_to_check - trajectory_piece) > 1e-5:
            dist_to_check += dist_vector[ind]
            time_command += time[ind]
            ind += 1
        if abs(traj_radius[index]) >= 100:
            delta_commands = delta_commands + [0, 0, time_command]
            dist_vector2 = dist_vector2 + [dist_to_check]
        else:
            delta_commands = delta_commands + [np.sign(traj_radius[index]) * delta_max, 0, time_command]
            dist_vector2 = dist_vector2 + [dist_to_check]
        dist_to_check = 0
        time_command = 0
        
    return delta_commands, dist_vector2