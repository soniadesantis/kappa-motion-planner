def compute_intersection_points_between_line_circle(x0, y0, xc, yc, turn, radius, corridor, margin = 0):
    '''
    Compute whether a given line and a given circle are intersecting. Specifically, whether a circle with center (xc,yc) intersects the walls of the given corridor.

    :param x0: initial x coordinate of vehicle
    :type x0: float
    :param xc: x coordinate of center of circle
    :type xc: float
    :param yc: y coordinate of center of circle
    :type yc: float
    :param turn: turn direction along the circle
    :type turn: float [-1,1]
    :param radius: v_max/omega_max
    :type radius: float
    :param corridor: current corridor
    :type corridor: CorridorWorld
    :param margin: additional margin to consider for the corridor (the corridor is restricted)
    :type margin: float

    :return: boolean indicating whether the left wall of the corridor is intersecting the given circle
    :rtype: Boolean
    :return: boolean indicating whether the right wall of the corridor is intersecting the given circle
    :rtype: Boolean
    :return: list of intersection points between left wall and circle
    :rtype: list of float
    :return: list of intersection points between right wall and circle
    :rtype: list of float
    '''
    corridor_with_margin = CorridorWorld(width= corridor.width - 2 * margin, height = corridor.height - 2 * margin, center = corridor.center, tilt= corridor.tilt) 
    corners = corridor_with_margin.get_corners()
    left_wall = False
    right_wall = False

    ## Intersection with left wall
    # Define the points
    x1, y1 = corners[2] # bottom left corner
    x2, y2 = corners[3] # top left corner
    # Calculate the slope
    if x1 == x2:
        # Vertical line
        A1, B1, C1 = 1, 0, x1
    else:
        m1 = (y2 - y1)/(x2 - x1)
        # Calculate the y-intercept (b) using the point-slope form: y = mx + b
        b = y1 - m1 * x1 if m1 != 0 else 0 
        A1, B1, C1 = m1, -1, -b 

    # Define the variables
    x, y = sp.symbols('x y')
    # Define the circle
    circle_equation = (x - xc)**2 + (y - yc)**2 - radius**2
    # Define the line
    line_equation = A1 * x + B1 * y - C1
    # Solve for the intersection points
    solutions = sp.solve([circle_equation, line_equation], (x, y))
    # Extract and print the intersection points
    intersection_points_left_wall = []
    for sol in solutions:
        x_val = sol[0]
        y_val = sol[1]
        if not(isinstance(x_val, sp.core.add.Add)) and not(isinstance(y_val, sp.core.add.Add)):
            # Check whether the intersection occurs after an arc of more than pi/2
            iota = compute_central_angle(x0, y0, float(x_val), float(y_val), xc, yc, turn, radius)
            if iota <= pi/2:
                left_wall = True
    #             intersection_points_left_wall.append((x_val, y_val))
    # if intersection_points_left_wall != []:
    #     left_wall = True
    
    ## Intersection with right wall
    # Define the points
    x1, y1 = corners[0] # top right corner
    x2, y2 = corners[1] # bottom right corner
    # Calculate the slope
    if x1 == x2:
        # Vertical line
        A1, B1, C1 = 1, 0, x1
    else:
        m1 = (y2 - y1)/(x2 - x1)
        # Calculate the y-intercept (b) using the point-slope form: y = mx + b
        b = y1 - m1 * x1 if m1 != 0 else 0 
        A1, B1, C1 = m1, -1, -b 

    # Define the variables
    x, y = sp.symbols('x y')
    # Define the circle
    circle_equation = (x - xc)**2 + (y - yc)**2 - radius**2
    # Define the line
    line_equation = A1 * x + B1 * y - C1
    # Solve for the intersection points
    solutions = sp.solve([circle_equation, line_equation], (x, y))
    # Extract and print the intersection points
    intersection_points_right_wall = []
    for sol in solutions:
        x_val = sol[0]
        y_val = sol[1]
        if not(isinstance(x_val, sp.core.add.Add)) and not(isinstance(y_val, sp.core.add.Add)):
            # Check whether the intersection occurs after an arc of more than pi/2
            iota = compute_central_angle(x0, y0, float(x_val), float(y_val), xc, yc, turn, radius)
            if iota <= pi/2:
                right_wall = True
    #             intersection_points_right_wall.append((x_val, y_val))
    # if intersection_points_right_wall != []:
    #     right_wall = True

    return left_wall, right_wall, intersection_points_left_wall, intersection_points_right_wall


def is_it_u_turn_old(corridor1, corridor3, turn1, turn2):
    if turn1 == turn2: 
        if turn1 == 1:
            
            if get_intersection(corridor1.wl, corridor3.wl) == m.inf:
                return True
        else:
            if get_intersection(corridor1.wr, corridor3.wr) == m.inf:
                return True
    return False


def invert_inputs_compute_trajectory_intersection_case_without_optimization(corridor1, corridor2, start_pose, end_pose, unicycle):
    start_pose_inv = [start_pose[0], start_pose[1], start_pose[2] + pi]
    end_pose_inv = [end_pose[0], end_pose[1], end_pose[2] + pi]
    corridor1_inv = CorridorWorld(corridor1.width, corridor1.height, corridor1.center, wrapPositiveAngle(corridor1.tilt + pi))
    corridor2_inv = CorridorWorld(corridor2.width, corridor2.height, corridor2.center, wrapPositiveAngle(corridor2.tilt + pi))
    return corridor1_inv, corridor2_inv, start_pose_inv, end_pose_inv, unicycle


def generate_corridors(corridor1_width, corridor1_height, corridor1_tilt, corridor2_width, corridor2_height, corridor2_tilt, unicycle_radius):
    start_point1 = [0,0]
    end_point1 = [corridor1_height * cos(corridor1_tilt), corridor1_height * sin(corridor1_tilt)]
    start_point2 = [(corridor1_height - 3 * unicycle_radius) * cos(corridor1_tilt), (corridor1_height - 3 * unicycle_radius) * sin(corridor1_tilt)]
    end_point2 = [start_point2[0] + corridor2_height * cos(corridor2_tilt), start_point2[1] + corridor2_height * sin(corridor2_tilt)]

    corridor1 = get_corridor_from_vector(start_point = start_point1, end_point = end_point1, width = corridor1_width)
    corridor2 = get_corridor_from_vector(start_point = start_point2, end_point = end_point2, width = corridor2_width)

    return corridor1, corridor2


# ### For Siemens
# def compute_trajectory_multiple_corridors_siemens(corridor_list, start_pose, end_pose, unicycle, obstacle_center = None, obstacle_radius = None):
#     # Initialize vectors with the maneuvers. 
#     # For n corridors, the sequence of maneuvers will be:
#     # T1, C1, S1, C2, S2, C3, S3, ..., Ci, Si, ..., Cn, Sn, Cn+1, T2
#     # Store segments from S2 until Sn-1 (n-2 segments)
#     # Store arcs from C3 until Cn-1 (n-3 arcs)
#     tic = time.perf_counter()
#     margin = 0.02 # This margin is used to compute the center of the circumferences both at the intersection between corridors and for obstacle avoidance
#     tol = 1e-2
#     # start_pose[2] = wrapPositiveAngle(start_pose[2])
#     print("Well done!!")
#     # Check whether two consecutive corridors are parallel and have the same width. In this case merge them together
#     corridor_to_remove = []
#     for i in range(len(corridor_list)-1):
#         if (abs(corridor_list[i].tilt - corridor_list[i+1].tilt) < tol) and ((wrapPositiveAngle(atan2(corridor_list[i+1].center[1] - corridor_list[i].center[1], corridor_list[i+1].center[0] - corridor_list[i].center[0])) - wrapPositiveAngle(corridor_list[i].tilt))) < tol and (corridor_list[i].width - corridor_list[i+1].width < 1e-2):
#             new_corridor = get_corridor_from_vector(corridor_list[i].tail, corridor_list[i+1].head, corridor_list[i].width, add_height = 0)
#             corridor_list[i] = new_corridor
#             corridor_to_remove = corridor_to_remove + [i+1]
#     tic1 = time.perf_counter()
#     ind = 0    
#     for index in corridor_to_remove:
#         corridor_list.pop(index - ind)
#         ind += 1

#     if len(corridor_list) == 1:
#         return compute_trajectory_unicycle_one_corridor(corridor_list[0], start_pose, unicycle, end_pose)
#     elif len(corridor_list) == 2:
#         trajectory , _ = compute_trajectory_unicycle_two_corridors_siemens(corridor_list[0], corridor_list[1], start_pose, end_pose, unicycle)
#         return [0,0], trajectory
    
#     # 1- Compute the turn directions
#     turn_direction_vector = compute_turn_direction_vector(corridor_list)
    
#     # 2- Compute the corner points
#     corner_point_vector = compute_corner_point_vector(corridor_list, turn_direction_vector)
    
#     # 3- Compute the centers of the circumferences 
#     center_circumference_vector = compute_center_coordinates_vector(corridor_list, turn_direction_vector, corner_point_vector, unicycle, margin = margin)
#     tic2 = time.perf_counter()
#     # 4- Change position of the second center if the start pose is inside the second circle
#     if obstacle_radius != None and check_point_inside_corridor(corridor_list[0], obstacle_center):
#         beta = atan2(obstacle_center[1] - start_pose[1], obstacle_center[0] - start_pose[0])
#         # turn_direction_obstacle = compute_turn_direction([cos(start_pose[2]), sin(start_pose[2])], [cos(beta), sin(beta)])
#         turn_direction_obstacle = get_corridor_side(corridor_list[0], obstacle_center)
#         r_prime = unicycle.max_radius - (unicycle.width * 0.5 + obstacle_radius + margin)
#         x_prime = obstacle_center[0] + r_prime * cos(beta + turn_direction_obstacle * pi * 0.5)
#         y_prime = obstacle_center[1] + r_prime * sin(beta + turn_direction_obstacle * pi * 0.5)
#         turn_direction_vector.insert(0, turn_direction_obstacle)
#         center_circumference_vector.insert(0, [x_prime, y_prime])
#     # xc2, yc2 = center_circumference_vector[0]
#     if ((center_circumference_vector[0][0] - start_pose[0])**2 + (center_circumference_vector[0][1] - start_pose[1])**2) < unicycle.max_radius**2:
#         xc2, yc2 = start_inside_second_circle_case(corridor_list[0], corridor_list[1], turn_direction_vector[0], corner_point_vector[0], unicycle)
#         center_circumference_vector[0] = [xc2, yc2]
#     # Eliminate the third circle if there's an initial u-turn
#     # if (abs(compute_angular_difference(corridor_list[0].tilt, corridor_list[1].tilt) - pi/2) < 1e-5 and abs(compute_angular_difference(corridor_list[1].tilt, corridor_list[2].tilt) - pi/2) < 1e-5) or (abs(compute_angular_difference(corridor_list[0].tilt, corridor_list[1].tilt) + pi/2) < 1e-5 and abs(compute_angular_difference(corridor_list[1].tilt, corridor_list[2].tilt) + pi/2) < 1e-5):
#     if is_it_u_turn(corridor_list[0], corridor_list[2], turn_direction_vector[0], turn_direction_vector[1]):
#         # x1, y1 = center_circumference_vector[0]
#         # x2, y2 = center_circumference_vector[1]
#         center_circumference_vector[0] = [(center_circumference_vector[0][0] + center_circumference_vector[1][0]) * 0.5,
#                                           (center_circumference_vector[0][1] + center_circumference_vector[1][1]) * 0.5]
#         turn_direction_vector.pop(1)
#         corner_point_vector.pop(1)
#         center_circumference_vector.pop(1)
#     tic3 = time.perf_counter()
#     # 5- Change the position of the penultimate circle if the end pose is inside the penultimate circle
#     xcn_minus_1, ycn_minus_1 = center_circumference_vector[-1]
#     if ((xcn_minus_1 - end_pose[0])**2 + (ycn_minus_1 - end_pose[1])**2) < unicycle.max_radius**2:
#         xcn_minus_1, ycn_minus_1 = start_inside_second_circle_case(*invert_inputs_start_inside_second_circle_case(corridor_list[-1], corridor_list[-2], turn_direction_vector[-1], corner_point_vector[-1], unicycle))
#         center_circumference_vector[-1] = [xcn_minus_1, ycn_minus_1]
#     # Eliminate the last third circle if there's a final u-turn
#     if is_it_u_turn(corridor_list[-1], corridor_list[-3], turn_direction_vector[-1], turn_direction_vector[-2]):
#         center_circumference_vector[-1] = [(center_circumference_vector[-1][0] + center_circumference_vector[-2][0]) * 0.5, 
#                                            (center_circumference_vector[-1][1] + center_circumference_vector[-2][1]) * 0.5]
#         turn_direction_vector.pop(-2)
#         corner_point_vector.pop(-2)
#         center_circumference_vector.pop(-2)
#     # Eliminate the circles in case of a u-turn
#     tic4 = time.perf_counter()
#     center_circumference_to_remove, turn_direction_to_remove, corner_point_to_remove = [], [], []        
#     for i in range(1, len(corridor_list)-2):
#         if is_it_u_turn(corridor_list[i], corridor_list[i+2], turn_direction_vector[i], turn_direction_vector[i+1]):
#             center_circumference_vector[i] = [(center_circumference_vector[i][0] + center_circumference_vector[i + 1][0]) * 0.5,
#                                               (center_circumference_vector[i][1] + center_circumference_vector[i + 1][1]) * 0.5]
#             center_circumference_to_remove = center_circumference_to_remove + [center_circumference_vector[i + 1]]
#             turn_direction_to_remove = turn_direction_to_remove + [i + 1]
#             corner_point_to_remove = corner_point_to_remove + [corner_point_vector[i + 1]]
#     for element in center_circumference_to_remove:
#         while element in center_circumference_vector:
#             center_circumference_vector.remove(element)

#     ind = 0    
#     for index in turn_direction_to_remove:
#         turn_direction_vector.pop(index - ind)
#         ind += 1
            
#     center_circumference_vector, turn_direction_vector = swap_circles(turn_direction_vector, center_circumference_vector, unicycle.max_radius, [start_pose[:2]] + center_circumference_vector + [end_pose[:2]])
#     tic5 = time.perf_counter()
#     # 6- Compute the segments
#     segments = [0] * (len(turn_direction_vector)-1)
#     for i in range(len(turn_direction_vector)-1):
#         segments[i] = compute_segment_between_two_circles(center_circumference_vector[i][0], center_circumference_vector[i][1],
#                                                           center_circumference_vector[i+1][0], center_circumference_vector[i+1][1],
#                                                           turn_direction_vector[i], turn_direction_vector[i+1], unicycle)
#     tic6 = time.perf_counter()
#     # 7- Check whether there's an intersection between the segments
#     segments_to_remove, center_circumference_to_remove, turn_direction_to_remove = [], [], []
#     for i in range(len(segments)-1):
#         if check_intersection_case(segments[i], segments[i+1]):
#             # Substite Si (which goes from Ci to Ci+1) with another Si which goes from Ci to Ci+2
#             segments[i] = compute_segment_between_two_circles(center_circumference_vector[i][0], center_circumference_vector[i][1],
#                                                               center_circumference_vector[i+2][0], center_circumference_vector[i+2][1],
#                                                               turn_direction_vector[i], turn_direction_vector[i+2], unicycle)
#             segments_to_remove = segments_to_remove + [segments[i+1]]
#             center_circumference_to_remove = center_circumference_to_remove + [center_circumference_vector[i+1]]
#             turn_direction_to_remove = turn_direction_to_remove + [i+1]
    
#     for element in segments_to_remove:
#         while element in segments:
#             segments.remove(element)

#     for element in center_circumference_to_remove:
#         while element in center_circumference_vector:
#             center_circumference_vector.remove(element)

#     ind = 0    
#     for index in turn_direction_to_remove:
#         turn_direction_vector.pop(index - ind)
#         ind += 1
#     tic7 = time.perf_counter()
#     # 8- Compute the arcs
#     arcs = [0] * (len(segments)-1) 
#     for i in range(len(segments)-1): 
#         arcs[i] = compute_arc_from_two_tangents(segments[i], segments[i+1], turn_direction_vector[i+1], center_circumference_vector[i+1][0], center_circumference_vector[i+1][1], unicycle)
#     tic8 = time.perf_counter()
#     # 9- Compute the first three maneuvers
#     x0, y0, theta0 = start_pose
#     dist = sqrt((xc2 - x0)**2 + (yc2 - y0)**2)
#     dist = 5
#     x_prime = x0 + dist * cos(theta0)
#     y_prime = y0 + dist * sin(theta0)
#     primitive1 = LinearSegmentUnicycle(x0=x0, y0=y0, xf=x_prime, yf=y_prime, theta=theta0, v=unicycle.v_max, t0 = 0, unicycle = unicycle, samples_number=10)
#     start_pose = [x_prime, y_prime, theta0]
#     T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0], corridor_list[1], start_pose, unicycle, center_circumference_vector[0][0], center_circumference_vector[0][1], turn_direction_vector[0], t0 = 0, turn1 = 0)
#     # S1, C1, S2 = compute_three_maneuvers_compact_siemens(corridor_list[0], corridor_list[1], start_pose, unicycle, center_circumference_vector[0][0], center_circumference_vector[0][1], turn_direction_vector[0], t0 = 0, turn1 = 0)
#     tic9 = time.perf_counter()
    
#     # 10- Check whether the intersection case occurs
#     if check_intersection_case(S1, segments[0]):
#         corridor_list.pop(1)
#         turn_direction_vector.pop(0)
#         center_circumference_vector.pop(0)
#         segments.pop(0)
#         arcs.pop(0)
#         T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0], corridor_list[1], start_pose, unicycle, center_circumference_vector[0][0], center_circumference_vector[0][1], turn_direction_vector[0], t0 = 0, turn1 = 0)
#     tic222 = time.perf_counter()
#     C2 = compute_arc_from_two_tangents(S1, segments[0], turn_direction_vector[0], center_circumference_vector[0][0], center_circumference_vector[0][1], unicycle)
#     tic111 = time.perf_counter()
#     # 11- Compute last three maneuvers from end pose to the second circumference and invert them
#     T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1], corridor_list[-2], end_pose, unicycle, center_circumference_vector[-1][0], center_circumference_vector[-1][1], turn_direction_vector[-1], t0 = 0, turn1 = 0))
#     Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
#     # 12- Check whether the intersection case occurs
#     if check_intersection_case(segments[-1], Sn):
#         corridor_list.pop(-2)
#         turn_direction_vector.pop(-1)
#         center_circumference_vector.pop(-1)
#         segments.pop(-1)
#         arcs.pop(-1)
#         T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1], corridor_list[-2], end_pose, unicycle, center_circumference_vector[-1][0], center_circumference_vector[-1][1], turn_direction_vector[-1], t0 = 0, turn1 = 0))
#         Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
#     Cn = compute_arc_from_two_tangents(segments[-1], Sn, turn_direction_vector[-1], center_circumference_vector[-1][0], center_circumference_vector[-1][1], unicycle)
#     tic10 = time.perf_counter()
#     # 13- Collect the maneuvers
#     central_sequence = [0] * (len(segments) + len(arcs))
#     j = 0
#     for i in range(0, len(central_sequence), 2):
#         central_sequence[i]= segments[j]
#         j +=1
#     j = 0
#     for i in range(1, len(central_sequence), 2):
#         central_sequence[i]= arcs[j]
#         j += 1
#     tic11 = time.perf_counter()
#     maneuvers = [primitive1, T1, C1, S1, C2] + central_sequence + [Cn, Sn, Cn_plus_1, T2]
#     correct_angles(maneuvers)
#     print("executed siemens")
#     return center_circumference_vector, maneuvers


def compute_trajectory_multiple_corridors_old(corridor_list, start_pose, end_pose, unicycle, obstacle_center = None, obstacle_radius = None):
    # Initialize vectors with the maneuvers. 
    # For n corridors, the sequence of maneuvers will be:
    # T1, C1, S1, C2, S2, C3, S3, ..., Ci, Si, ..., Cn, Sn, Cn+1, T2
    # Store segments from S2 until Sn-1 (n-2 segments)
    # Store arcs from C3 until Cn-1 (n-3 arcs)
    
    margin = 0.02 # This margin is used to compute the center of the circumferences both at the intersection between corridors and for obstacle avoidance
    tol = 1e-2

    # Check whether two consecutive corridors are parallel and have the same width. In this case merge them together
    corridor_to_remove = []
    for i in range(len(corridor_list)-1):
        if (
            abs(corridor_list[i].tilt - corridor_list[i+1].tilt) < tol
            and (
                wrapPositiveAngle(
                    atan2(
                        corridor_list[i+1].center[1] - corridor_list[i].center[1],
                        corridor_list[i+1].center[0] - corridor_list[i].center[0]
                    )
                )
                - wrapPositiveAngle(corridor_list[i].tilt)
            ) < tol
            and (abs(corridor_list[i].width - corridor_list[i+1].width)< 1e-2)
        ):
        # if (abs(corridor_list[i].tilt - corridor_list[i+1].tilt) < tol) and ((wrapPositiveAngle(atan2(corridor_list[i+1].center[1] - corridor_list[i].center[1], corridor_list[i+1].center[0] - corridor_list[i].center[0])) - wrapPositiveAngle(corridor_list[i].tilt))) < tol and (corridor_list[i].width - corridor_list[i+1].width < 1e-2):
            new_corridor = get_corridor_from_vector(corridor_list[i].tail, corridor_list[i+1].head, corridor_list[i].width, add_height = 0)
            corridor_list[i] = new_corridor
            corridor_to_remove = corridor_to_remove + [i+1]
    
    ind = 0    
    for index in corridor_to_remove:
        corridor_list.pop(index - ind)
        ind += 1

    if len(corridor_list) == 1:
        return [0,0], ocp_function_one_corridor(corridor_list[0], start_pose, end_pose, unicycle) 
    elif len(corridor_list) == 2:
        trajectory , _ = compute_trajectory_unicycle_two_corridors(corridor_list[0], corridor_list[1], start_pose, end_pose, unicycle)
        return [0,0], trajectory
    
    # 1- Compute the turn directions
    turn_direction_vector = compute_turn_direction_vector(corridor_list)

    # If two consecutive corridors have the same tilt but different width, 
    # they don't have to be merged but the turn diretion has to be computed differently
    zero_indices = []
    if np.any(np.array(turn_direction_vector) == 0): # If any turn direction is zero
        zero_indices = np.where(np.array(turn_direction_vector) == 0)[0] 
      
    for ind in zero_indices: # for each couple of corridors with the same tilt
        add_ind = 2
        turn_direction = 0
        while turn_direction == 0:
            if ind + add_ind >= len(corridor_list):
                final_corr = corridor_list[-1]
                angle = atan2((end_pose[1] - final_corr.center[1]), (end_pose[0] - final_corr.center[0]))
                vector2 = [cos(angle), sin(angle)]
                turn_direction = compute_turn_direction(corridor_list[ind].unit_vector, vector2)
            else:
                turn_direction = corridor_list[ind].compute_relative_turn_direction(corridor_list[ind + add_ind])
            add_ind += 1
        turn_direction_vector[ind] = turn_direction

    # 2- Compute the corner points
    corner_point_vector = compute_corner_point_vector(corridor_list, turn_direction_vector)
    
    # 3- Compute the centers of the circumferences 
    center_circumference_vector = compute_center_coordinates_vector(corridor_list, turn_direction_vector, corner_point_vector, unicycle, margin = margin)
    
    # figure = plot_corridors(corridor_list)
    # for element in center_circumference_vector:
    #     plt.plot(element[0], element[1], 'ro')
    # plt.show(block = True)
    
    # 4- Change position of the second center if the start pose is inside the second circle
    if obstacle_radius != None and check_point_inside_corridor(corridor_list[0], obstacle_center):
        beta = atan2(obstacle_center[1] - start_pose[1], obstacle_center[0] - start_pose[0])
        # turn_direction_obstacle = compute_turn_direction([cos(start_pose[2]), sin(start_pose[2])], [cos(beta), sin(beta)])
        turn_direction_obstacle = get_corridor_side(corridor_list[0], obstacle_center)
        r_prime = unicycle.max_radius - (unicycle.width * 0.5 + obstacle_radius + margin)
        x_prime = obstacle_center[0] + r_prime * cos(beta + turn_direction_obstacle * pi * 0.5)
        y_prime = obstacle_center[1] + r_prime * sin(beta + turn_direction_obstacle * pi * 0.5)
        turn_direction_vector.insert(0, turn_direction_obstacle)
        center_circumference_vector.insert(0, [x_prime, y_prime])
    # xc2, yc2 = center_circumference_vector[0]
    
    if ((center_circumference_vector[0][0] - start_pose[0])**2 + (center_circumference_vector[0][1] - start_pose[1])**2) < unicycle.max_radius**2:
        xc2, yc2 = start_inside_second_circle_case(corridor_list[0], corridor_list[1], turn_direction_vector[0], corner_point_vector[0], unicycle)
        center_circumference_vector[0] = [xc2, yc2]
    # Eliminate the third circle if there's an initial u-turn
    # if (abs(compute_angular_difference(corridor_list[0].tilt, corridor_list[1].tilt) - pi/2) < 1e-5 and abs(compute_angular_difference(corridor_list[1].tilt, corridor_list[2].tilt) - pi/2) < 1e-5) or (abs(compute_angular_difference(corridor_list[0].tilt, corridor_list[1].tilt) + pi/2) < 1e-5 and abs(compute_angular_difference(corridor_list[1].tilt, corridor_list[2].tilt) + pi/2) < 1e-5):
    if is_it_u_turn(corridor_list[0], corridor_list[2], turn_direction_vector[0], turn_direction_vector[1]):
        # x1, y1 = center_circumference_vector[0]
        # x2, y2 = center_circumference_vector[1]
        center_circumference_vector[0] = [(center_circumference_vector[0][0] + center_circumference_vector[1][0]) * 0.5,
                                          (center_circumference_vector[0][1] + center_circumference_vector[1][1]) * 0.5]
        turn_direction_vector.pop(1)
        corner_point_vector.pop(1)
        center_circumference_vector.pop(1)

    # 5- Change the position of the penultimate circle if the end pose is inside the penultimate circle
    xcn_minus_1, ycn_minus_1 = center_circumference_vector[-1]
    if ((xcn_minus_1 - end_pose[0])**2 + (ycn_minus_1 - end_pose[1])**2) < unicycle.max_radius**2:
        xcn_minus_1, ycn_minus_1 = start_inside_second_circle_case(*invert_inputs_start_inside_second_circle_case(corridor_list[-1], corridor_list[-2], turn_direction_vector[-1], corner_point_vector[-1], unicycle))
        center_circumference_vector[-1] = [xcn_minus_1, ycn_minus_1]
    # Eliminate the last third circle if there's a final u-turn
    if is_it_u_turn(corridor_list[-1], corridor_list[-3], turn_direction_vector[-1], turn_direction_vector[-2]):
        center_circumference_vector[-1] = [(center_circumference_vector[-1][0] + center_circumference_vector[-2][0]) * 0.5, 
                                           (center_circumference_vector[-1][1] + center_circumference_vector[-2][1]) * 0.5]
        turn_direction_vector.pop(-2)
        corner_point_vector.pop(-2)
        center_circumference_vector.pop(-2)
    # Eliminate the circles in case of a u-turn
    
    center_circumference_to_remove, turn_direction_to_remove, corner_point_to_remove = [], [], []        
    for i in range(1, len(corridor_list)-2):
        if is_it_u_turn(corridor_list[i], corridor_list[i+2], turn_direction_vector[i], turn_direction_vector[i+1]):
            center_circumference_vector[i] = [(center_circumference_vector[i][0] + center_circumference_vector[i + 1][0]) * 0.5,
                                              (center_circumference_vector[i][1] + center_circumference_vector[i + 1][1]) * 0.5]
            center_circumference_to_remove = center_circumference_to_remove + [center_circumference_vector[i + 1]]
            turn_direction_to_remove = turn_direction_to_remove + [i + 1]
            corner_point_to_remove = corner_point_to_remove + [corner_point_vector[i + 1]]
    for element in center_circumference_to_remove:
        while element in center_circumference_vector:
            center_circumference_vector.remove(element)

    ind = 0    
    for index in turn_direction_to_remove:
        turn_direction_vector.pop(index - ind)
        ind += 1
            
    # center_circumference_vector, turn_direction_vector = swap_circles(turn_direction_vector, center_circumference_vector, unicycle.max_radius, [start_pose[:2]] + center_circumference_vector + [end_pose[:2]])
    
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

    # figure = plot_corridors(corridor_list = corridor_list, plot_vectors = True, linestyle = '--', color = 'gray', linewidth = 0.5)

    # 8- Compute the arcs
    arcs = [0] * (len(segments)-1) 
    for i in range(len(segments)-1): 
        arcs[i] = compute_arc_from_two_tangents(segments[i], segments[i+1], turn_direction_vector[i+1], center_circumference_vector[i+1][0], center_circumference_vector[i+1][1], unicycle)

    # 7b: Check whether there's an arc with more than 3/2 pi angle
    arcs_to_remove, segments_to_remove, center_circumference_to_remove, turn_direction_to_remove = [], [], [], []
    for i in range(len(arcs)):
        if abs(arcs[i].iota) > 3 * pi *0.5:
            segments[i] = compute_segment_between_two_circles(center_circumference_vector[i][0], center_circumference_vector[i][1],
                                                    center_circumference_vector[i+2][0], center_circumference_vector[i+2][1],
                                                    turn_direction_vector[i], turn_direction_vector[i+2], unicycle)

            # If the arc is too big, remove it and the two adjacent segments
            arcs_to_remove = arcs_to_remove + [arcs[i]]
            segments_to_remove = segments_to_remove + [segments[i+1]]
            center_circumference_to_remove = center_circumference_to_remove + [center_circumference_vector[i+1]]
            turn_direction_to_remove = turn_direction_to_remove + [i+1]
    
    for element in arcs_to_remove:
        while element in arcs:
            arcs.remove(element)

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
    T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0], corridor_list[1], start_pose, unicycle, center_circumference_vector[0][0], center_circumference_vector[0][1], turn_direction_vector[0], t0 = 0, turn1 = 0)
   
    # 10- Check whether the intersection case occurs
    if check_intersection_case(S1, segments[0]):
        
        corridor_list.pop(1)
        turn_direction_vector.pop(0)
        center_circumference_vector.pop(0)
        segments.pop(0)
        if len(arcs) > 0:
            arcs.pop(0)
        T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0], corridor_list[1], start_pose, unicycle, center_circumference_vector[0][0], center_circumference_vector[0][1], turn_direction_vector[0], t0 = 0, turn1 = 0)
        
    if len(segments) > 0:
        C2 = compute_arc_from_two_tangents(S1, segments[0], turn_direction_vector[0], center_circumference_vector[0][0], center_circumference_vector[0][1], unicycle)

    # 11- Compute last three maneuvers from end pose to the second circumference and invert them
    T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1], corridor_list[-2], end_pose, unicycle, center_circumference_vector[-1][0], center_circumference_vector[-1][1], turn_direction_vector[-1], t0 = 0, turn1 = 0))
    Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
    
    # 12- Check whether the intersection case occurs
    if len(segments) != 0 and check_intersection_case(segments[-1], Sn):
        
        corridor_list.pop(-2)
        turn_direction_vector.pop(-1)
        center_circumference_vector.pop(-1)
        segments.pop(-1)
        if len(arcs) > 0:
            arcs.pop(-1)
        T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1], corridor_list[-2], end_pose, unicycle, center_circumference_vector[-1][0], center_circumference_vector[-1][1], turn_direction_vector[-1], t0 = 0, turn1 = 0))
        Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
    if len(segments) > 0:
        Cn = compute_arc_from_two_tangents(segments[-1], Sn, turn_direction_vector[-1], center_circumference_vector[-1][0], center_circumference_vector[-1][1], unicycle)

    if check_intersection_case(S1, Sn):
        trajectory = ocp_function_free_space(start_pose, end_pose, unicycle)
        return [], trajectory

    if len(segments) == 0: # If there's no segment, it means that the two corridors are connected by an arc
        Cn = compute_arc_from_two_tangents(S1, Sn, turn_direction_vector[-1], center_circumference_vector[-1][0], center_circumference_vector[-1][1], unicycle)
        maneuvers = [T1, C1, S1, Cn, Sn, Cn_plus_1, T2]
        correct_angles(maneuvers)
        return center_circumference_vector, maneuvers
    
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
   
    maneuvers = [T1, C1, S1, C2] + central_sequence + [Cn, Sn, Cn_plus_1, T2]
    correct_angles(maneuvers)
    return center_circumference_vector, maneuvers


### Multiple corridors with waypoints

def compute_trajectory_multiple_corridors_with_waypoints(corridor_list,
                                          start_pose,
                                          end_pose,
                                          unicycle, 
                                          waypoints):
    

# Initialize vectors with the maneuvers. 
    # For n corridors, the solution sequence counts 2n+3 primitives
    # T1, C2, S3, C4, S5, C6, S7, ..., C2n, S2n+1, C2n+2, T2n+3
    
    margin = 0.02 # This margin is used to compute the center of the circumferences both at the intersection between corridors and for obstacle avoidance
    tol = 1e-2

    ### 1- Merge corridors ### (make function)
    
    # Merge two consecutive corridors if  
    # 1) they have the same tilt;
    # 2) they have the same width;
    # 3) their centers belong to a line with same orientation as the corridors' tilt.
     
    corridor_to_remove = []
    for i in range(len(corridor_list)-1):
        if (
            abs(corridor_list[i].tilt - corridor_list[i+1].tilt) < tol
            and (
                wrapPositiveAngle(
                    atan2(
                        corridor_list[i+1].center[1] - corridor_list[i].center[1],
                        corridor_list[i+1].center[0] - corridor_list[i].center[0]
                    )
                )
                - wrapPositiveAngle(corridor_list[i].tilt)
            ) < tol
            and (abs(corridor_list[i].width - corridor_list[i+1].width)< 1e-2)
        ):
            new_corridor = get_corridor_from_vector(corridor_list[i].tail, corridor_list[i+1].head, corridor_list[i].width, add_height = 0)
            corridor_list[i] = new_corridor
            corridor_to_remove = corridor_to_remove + [i+1]
    
    # Remove the corridors after merging, if needed
    ind = 0    
    for index in corridor_to_remove:
        corridor_list.pop(index - ind)
        ind += 1

    # If the resulting corridor list has one or two corridors, use the dedicated functions
    if len(corridor_list) == 1:
        return [0,0], ocp_function_one_corridor(corridor_list[0], start_pose, end_pose, unicycle) 
    elif len(corridor_list) == 2:
        trajectory , _ = compute_trajectory_unicycle_two_corridors(corridor_list[0], corridor_list[1], start_pose, end_pose, unicycle)
        return [0,0], trajectory
    
    # if len(waypoints) != len(corridor_list) - 1:
    #     raise ValueError(f"The number of waypoints must be equal to the number of corridors minus one.\nNumber of corridors: {len(corridor_list)}\nNumber of waypoints: {len(waypoints)}")
    turn_direction_vector = [0]*(len(waypoints))
    corridors_that_matter = []
    corridors_that_matter_ind = []
    center_circumference_vector = [0] * (len(waypoints))

    corridor_ind = 0
    for ind, waypoint in enumerate(waypoints):
        for i in range(len(corridor_list)-2): 
            if check_point_inside_corridor(corridor_list[i], waypoint) and check_point_inside_corridor(corridor_list[i+1], waypoint):
                corridor_ind = i
                corridors_that_matter_ind.append(corridor_ind)
                corridors_that_matter_ind.append(corridor_ind + 1)

                break
        corners1 = corridor_list[corridor_ind].get_corners()
        corners2 = corridor_list[corridor_ind + 1].get_corners()
        if check_point_inside_segment(corners1[2], corners1[3], waypoint) or check_point_inside_segment(corners2[2], corners2[3], waypoint):
            turn_direction_vector[ind] = 1
        else:
            turn_direction_vector[ind] = -1
        xc, yc = compute_center_coordinates_second_circle(waypoint, turn_direction_vector[ind],
                                                          unicycle.max_radius, unicycle.width,
                                                          margin, corridor_list[corridor_ind].tilt,
                                                          corridor_list[corridor_ind+1].tilt,
                                                          corridor_list[corridor_ind],
                                                          corridor_list[corridor_ind+1])
        center_circumference_vector[ind] = [xc, yc]
        # modifica qui
        # figure = plot_corridors(corridor_list)
        # plot_corridors([corridor_list[corridor_ind], corridor_list[corridor_ind + 1]], color = 'r', plot_vectors = True, figure = figure)
        # plt.plot(waypoint[0], waypoint[1], 'ro', label = f"turn direction: {turn_direction_vector[ind]}")
        # plt.legend()
        # plt.show(block = True)

    # corridors_that_matter_ind = list(dict.fromkeys(corridors_that_matter_ind))
    # for i in corridors_that_matter_ind:
    #     corridors_that_matter.append(corridor_list[i])
        # figure = plot_corridors([corridor_list[corridor_ind], corridor_list[corridor_ind + 1]], plot_vectors = True,)
        # plt.plot(waypoint[0], waypoint[1], 'ro', label = f"turn direction: {turn_direction_vector[corridor_ind]}")
        # plt.legend()
        # plt.show(block = True)
        # corridor_ind += 1

    ### 2- Compute the turn directions ###
    # turn_direction_vector = compute_turn_direction_vector(corridor_list)

    # If two consecutive corridors have the same tilt but different width, 
    # they don't have to be merged but the turn diretion has to be computed differently
    zero_indices = []
    if np.any(np.array(turn_direction_vector) == 0): # If any turn direction is zero
        zero_indices = np.where(np.array(turn_direction_vector) == 0)[0].tolist()
    
    # if zero_indices != []:
    #     zero_indices = zero_indices.tolist()

    # If the last turn direction is 0, it means that the last two corridors have same tilt. 
    # First assign a turn direction between the penultimate and the last corridor.
    last_corridor_rotated = None
    if zero_indices != [] and zero_indices[-1] == len(corridor_list)-2:
        add_ind = 2 
        turn_direction = 0
        while turn_direction == 0:
            turn_direction = corridor_list[-add_ind].compute_relative_turn_direction(corridor_list[-1])
            add_ind += 1
        turn_direction_vector[-1] = turn_direction
        zero_indices.pop(-1)
        last_corridor_rotated = corridor_list[-1].rotate_corridor(turn_direction * pi * 0.5)

    for ind in zero_indices: # for each couple of corridors with the same tilt
        add_ind = 2
        turn_direction = 0
        while turn_direction == 0:
            if ind + add_ind >= len(corridor_list) and last_corridor_rotated is not None:
                # final_corr = corridor_list[-1]
                # angle = atan2((end_pose[1] - final_corr.center[1]), (end_pose[0] - final_corr.center[0]))
                # vector2 = [cos(angle), sin(angle)]
                # turn_direction = compute_turn_direction(corridor_list[ind].unit_vector, vector2)
                # turn_direction = corridor_list[ind-1].compute_relative_turn_direction(corridor_list[-1])
                turn_direction = corridor_list[ind].compute_relative_turn_direction(last_corridor_rotated)

            else:
                turn_direction = corridor_list[ind].compute_relative_turn_direction(corridor_list[ind + add_ind])
                # turn_direction = corridor_list[ind-1].compute_relative_turn_direction(corridor_list[ind])
            # turn_direction = corridor_list[ind].compute_relative_turn_direction(corridor_list[ind + add_ind])
            add_ind += 1
        turn_direction_vector[ind] = turn_direction

    ### 3- Compute the corner points ###
    # corner_point_vector = compute_corner_point_vector(corridor_list, turn_direction_vector)
    corner_point_vector = waypoints
    # ### 4- Compute the centers of the intermediate circles ###
    # center_circumference_vector = compute_center_coordinates_vector(corridors_that_matter,
    #                                                                 turn_direction_vector,
    #                                                                 corner_point_vector,
    #                                                                 unicycle, margin = margin)
    # for i, element in enumerate(turn_direction_vector):
    #     print(f"turn direction {i}: {element}")
    # print(f"turn direction vectos: {turn_direction_vector}")
    
    # angle_array = np.linspace(0, 2*pi, 100)
    # figure = plot_corridors(corridor_list, plot_vectors = True,)
    # for waypoint in waypoints:
    #     plt.plot(waypoint[0], waypoint[1], 'go')
    # for element in center_circumference_vector:
    #     plt.plot(element[0], element[1], 'ro')
    #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
    # plt.show(block = True)

    # Merge two circles together if they are too close to each other
    i = 0
    while i < len(center_circumference_vector)-1:
        if compute_distance_two_points(center_circumference_vector[i], center_circumference_vector[i+1]) < unicycle.max_radius:
            center_circumference_vector[i] = [(center_circumference_vector[i][0] + center_circumference_vector[i+1][0]) * 0.5,
                                              (center_circumference_vector[i][1] + center_circumference_vector[i+1][1]) * 0.5]
            turn_direction_vector.pop(i+1)
            corner_point_vector.pop(i+1)
            center_circumference_vector.pop(i+1)
        else:
            i += 1

        
    # figure = plot_corridors(corridor_list, plot_vectors = True,)
    # for element in center_circumference_vector:
    #     plt.plot(element[0], element[1], 'ro')
    #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
    # plt.show(block = True)

    # Change position of the second center if the start pose is inside the second circle
    if compute_distance_two_points(center_circumference_vector[0], start_pose[:2]) < unicycle.max_radius:
        xc_first_circle, yc_first_circle = start_inside_second_circle_case(corridor_list[0], corridor_list[1], turn_direction_vector[0], corner_point_vector[0], unicycle)
        center_circumference_vector[0] = [xc_first_circle, yc_first_circle]
    
    # Change the position of the penultimate circle if the end pose is inside the penultimate circle
    if compute_distance_two_points(center_circumference_vector[-1], end_pose[:2]) < unicycle.max_radius:
        xc_last_circle, yc_last_circle = start_inside_second_circle_case(*invert_inputs_start_inside_second_circle_case(corridor_list[-1], corridor_list[-2], turn_direction_vector[-1], corner_point_vector[-1], unicycle))
        center_circumference_vector[-1] = [xc_last_circle, yc_last_circle]
    
    # attention = 1
    # figure = plot_corridors(corridor_list, plot_vectors = True,)
    # for element in center_circumference_vector:
    #     plt.plot(element[0], element[1], 'ro')
    #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
    # plt.show(block = True)
    # # Eliminate the third circle if there's an initial u-turn
    # if is_it_u_turn(corridor_list[0], corridor_list[2], turn_direction_vector[0], turn_direction_vector[1]):
    #     center_circumference_vector[0] = [(center_circumference_vector[0][0] + center_circumference_vector[1][0]) * 0.5,
    #                                       (center_circumference_vector[0][1] + center_circumference_vector[1][1]) * 0.5]
    #     turn_direction_vector.pop(1)
    #     corner_point_vector.pop(1)
    #     center_circumference_vector.pop(1)

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
            
    # # Eliminate the last third circle if there's a final u-turn
    # if is_it_u_turn(corridor_list[-1], corridor_list[-3], turn_direction_vector[-1], turn_direction_vector[-2]):
    #     center_circumference_vector[-1] = [(center_circumference_vector[-1][0] + center_circumference_vector[-2][0]) * 0.5, 
    #                                        (center_circumference_vector[-1][1] + center_circumference_vector[-2][1]) * 0.5]
    #     turn_direction_vector.pop(-2)
    #     corner_point_vector.pop(-2)
    #     center_circumference_vector.pop(-2)

    center_circumference_vector, turn_direction_vector = swap_circles(turn_direction_vector, center_circumference_vector, unicycle.max_radius, [start_pose[:2]] + center_circumference_vector + [end_pose[:2]])
    
    ### 6- Compute the segments ###
    # Compute first three maneuvers
    T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0],
                                                 corridor_list[1],
                                                 start_pose,
                                                 unicycle,
                                                 center_circumference_vector[0][0],
                                                 center_circumference_vector[0][1],
                                                 turn_direction_vector[0], t0 = 0, turn1 = 0)

    # Compute last three maneuvers 
    T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1],
                                                                       corridor_list[-2],
                                                                       end_pose, unicycle,
                                                                       center_circumference_vector[-1][0],
                                                                       center_circumference_vector[-1][1],
                                                                       turn_direction_vector[-1], t0 = 0, turn1 = 0))
    Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
    # figure = plot_corridors(corridor_list)
    # for element in center_circumference_vector:
    #     plt.plot(element[0], element[1], 'ro')
    #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
    # T1.plot_path(figure)
    # S1.plot_path(figure)
    # Sn.plot_path(figure)
    # T2.plot_path(figure)
    # C1.plot_path(figure)
    # Cn_plus_1.plot_path(figure)
    # plt.show(block = True)

    segments = [0] * (len(turn_direction_vector)+1)
    segments[0] = S1
    segments[-1] = Sn
    for i in range(1, len(turn_direction_vector)): 
        segments[i] = compute_segment_between_two_circles(center_circumference_vector[i-1][0], center_circumference_vector[i-1][1],
                                                          center_circumference_vector[i][0], center_circumference_vector[i][1],
                                                          turn_direction_vector[i-1], turn_direction_vector[i], unicycle)
    

    # # 8- Compute the arcs
    # arcs = [0] * (len(segments)-1) 
    # for i in range(len(segments)-1): 
    #     arcs[i] = compute_arc_from_two_tangents(segments[i], segments[i+1], turn_direction_vector[i], center_circumference_vector[i][0], center_circumference_vector[i][1], unicycle)

    # figure = plot_corridors(corridor_list)
    # plot_corridors([corridor_list[0]], color = 'r', figure = figure, label = "corridor1")
    # plot_corridors([corridor_list[1]], color = 'g', figure = figure, label = "corridor2")
    # plot_corridors([corridor_list[2]], color = 'y', figure = figure, label = "corridor3")
    # plt.plot(corner_point_vector[0][0], corner_point_vector[0][1], 'go', label = "corner point 1")
    # plt.plot(corner_point_vector[1][0], corner_point_vector[1][1], 'mo', label = "corner point 2")
    # plt.plot(corner_point_vector[2][0], corner_point_vector[2][1], 'yo', label = "corner point 3")
    # plt.plot(corner_point_vector[3][0], corner_point_vector[3][1], 'co', label = "corner point 4")


    # for element in center_circumference_vector:
    #     plt.plot(element[0], element[1], 'ro')
    #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
    # # plt.plot(center_circumference_vector[0][0], center_circumference_vector[0][1], 'go', label = "circle 1") # First circle center
    # # plt.plot(center_circumference_vector[1][0], center_circumference_vector[1][1], 'mo', label = "circle 2") # Second circle center
    # # plt.plot(center_circumference_vector[2][0], center_circumference_vector[2][1], 'yo', label = "circle 3") # Third circle center
    # # plt.plot(center_circumference_vector[3][0], center_circumference_vector[3][1], 'co', label = "circle 4") # Last circle center
    # # plt.legend()
    # T1.plot_path(figure)
    # S1.plot_path(figure)
    # Sn.plot_path(figure)
    # T2.plot_path(figure)
    # C1.plot_path(figure)
    # Cn_plus_1.plot_path(figure)
    # for segment in segments:
    #     segment.plot_path(figure)
    # # for arc in arcs:
    # #     arc.plot_path(figure)

    # plt.plot(start_pose[0], start_pose[1], 'ro')
    # plt.plot(end_pose[0], end_pose[1], 'ro')

    # plt.arrow(start_pose[0], start_pose[1], 0.5*cos(start_pose[2]), 0.5*sin(start_pose[2]), head_width = 0.1, head_length = 0.1, fc = 'r', ec = 'r')
    # plt.arrow(end_pose[0], end_pose[1], 0.5*cos(end_pose[2]), 0.5*sin(end_pose[2]), head_width = 0.1, head_length = 0.1, fc = 'r', ec = 'r')

    # plt.show(block = True)

    ### INTERSECTION CHECKS ###

    # Part 1
    while check_intersection_case(segments[0], segments[1]):
            corridor_list.pop(1)
            center_circumference_vector.pop(0)
            turn_direction_vector.pop(0)
            ## Number of circles check
            if len(corridor_list) == 1:
                return [0,0], ocp_function_free_space(start_pose, end_pose, unicycle)
            elif len(corridor_list) == 2:
                T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0],
                                            corridor_list[1],
                                            start_pose,
                                            unicycle,
                                            center_circumference_vector[0][0],
                                            center_circumference_vector[0][1],
                                            turn_direction_vector[0], t0 = 0, turn1 = 0)
                T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1],
                                                                       corridor_list[-2],
                                                                       end_pose, unicycle,
                                                                       center_circumference_vector[-1][0],
                                                                       center_circumference_vector[-1][1],
                                                                       turn_direction_vector[-1], t0 = 0, turn1 = 0))
                Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
                if check_intersection_case(S1, Sn):
                    trajectory = ocp_function_free_space(start_pose, end_pose, unicycle)
                    return [], trajectory
                Cn = compute_arc_from_two_tangents(S1, Sn, turn_direction_vector[-1], center_circumference_vector[-1][0], center_circumference_vector[-1][1], unicycle)
                maneuvers = [T1, C1, S1, Cn, Sn, Cn_plus_1, T2]
                correct_angles(maneuvers)
                return center_circumference_vector, maneuvers
            # Compute T1, C2, S3 from start_pose to circles[0] with turn_direction[0]
            T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0],
                                                        corridor_list[1],
                                                        start_pose,
                                                        unicycle,
                                                        center_circumference_vector[0][0],
                                                        center_circumference_vector[0][1],
                                                        turn_direction_vector[0], t0 = 0, turn1 = 0)
            segments[0] = S1
            segments.pop(1)

    # figure = plot_corridors(corridor_list)
    # for element in center_circumference_vector:
    #     plt.plot(element[0], element[1], 'ro')
    #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
    # plt.plot(center_circumference_vector[0][0], center_circumference_vector[0][1], 'go', label = "circle 1") # First circle center
    # plt.plot(center_circumference_vector[1][0], center_circumference_vector[1][1], 'mo', label = "circle 2") # Second circle center
    # plt.plot(center_circumference_vector[2][0], center_circumference_vector[2][1], 'yo', label = "circle 3") # Third circle center
    # plt.legend()
    # T1.plot_path(figure)
    # S1.plot_path(figure)
    # Sn.plot_path(figure)
    # T2.plot_path(figure)
    # C1.plot_path(figure)
    # Cn_plus_1.plot_path(figure)
    # for segment in segments:
    #     segment.plot_path(figure)
    # # for arc in arcs:
    # #     arc.plot_path(figure)
    # plt.show(block = True)
    
    # Part 2
    i = 1 
    while i < len(segments)-2:
        if check_intersection_case(segments[i], segments[i+1]):
            center_circumference_vector.pop(i)
            turn_direction_vector.pop(i)
            corridor_list.pop(i+1)
            ## Number of circles check
            if len(corridor_list) == 1:
                return [0,0], ocp_function_free_space(start_pose, end_pose, unicycle)
            elif len(corridor_list) == 2:
                T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0],
                                            corridor_list[1],
                                            start_pose,
                                            unicycle,
                                            center_circumference_vector[0][0],
                                            center_circumference_vector[0][1],
                                            turn_direction_vector[0], t0 = 0, turn1 = 0)
                T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1],
                                                                       corridor_list[-2],
                                                                       end_pose, unicycle,
                                                                       center_circumference_vector[-1][0],
                                                                       center_circumference_vector[-1][1],
                                                                       turn_direction_vector[-1], t0 = 0, turn1 = 0))
                Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
                if check_intersection_case(S1, Sn):
                    trajectory = ocp_function_free_space(start_pose, end_pose, unicycle)
                    return [], trajectory
                Cn = compute_arc_from_two_tangents(S1, Sn, turn_direction_vector[-1], center_circumference_vector[-1][0], center_circumference_vector[-1][1], unicycle)
                maneuvers = [T1, C1, S1, Cn, Sn, Cn_plus_1, T2]
                correct_angles(maneuvers)
                return center_circumference_vector, maneuvers
            # Substite Si (which goes from Ci to Ci+1) with another Si which goes from Ci to Ci+2
            segments[i] = compute_segment_between_two_circles(center_circumference_vector[i-1][0], center_circumference_vector[i-1][1],
                                                              center_circumference_vector[i][0], center_circumference_vector[i][1],
                                                              turn_direction_vector[i-1], turn_direction_vector[i], unicycle)
            segments.pop(i+1)
            # segment_removed = True
            # figure = plot_corridors(corridor_list)
            # for element in center_circumference_vector:
            #     plt.plot(element[0], element[1], 'ro')
            #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
            # T1.plot_path(figure)
            # S1.plot_path(figure)
            # Sn.plot_path(figure)
            # T2.plot_path(figure)
            # C1.plot_path(figure)
            # Cn_plus_1.plot_path(figure)
            # for segment in segments:
            #     segment.plot_path(figure)
            # segments[i].plot_path(figure, color = 'm', linewidth = 2)
            # # for arc in arcs:
            # #     arc.plot_path(figure)
            # plt.show(block = True)
        else:
            i += 1

    # Part 3
    while check_intersection_case(segments[-2], segments[-1]):
            # figure = plot_corridors(corridor_list)
            # for element in center_circumference_vector:
            #     plt.plot(element[0], element[1], 'ro')
            #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
            # segments[-2].plot_path(figure)
            # segments[-1].plot_path(figure)
            # plt.show(block = True)
            corridor_list.pop(-2)
            center_circumference_vector.pop(-1)
            turn_direction_vector.pop(-1)
            ## Number of circles check
            if len(corridor_list) == 1:
                return [0,0], ocp_function_free_space(start_pose, end_pose, unicycle)
            elif len(corridor_list) == 2:
                T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0],
                                            corridor_list[1],
                                            start_pose,
                                            unicycle,
                                            center_circumference_vector[0][0],
                                            center_circumference_vector[0][1],
                                            turn_direction_vector[0], t0 = 0, turn1 = 0)
                T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1],
                                                                       corridor_list[-2],
                                                                       end_pose, unicycle,
                                                                       center_circumference_vector[-1][0],
                                                                       center_circumference_vector[-1][1],
                                                                       turn_direction_vector[-1], t0 = 0, turn1 = 0))
                Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
                if check_intersection_case(S1, Sn):
                    trajectory = ocp_function_free_space(start_pose, end_pose, unicycle)
                    return [], trajectory
                Cn = compute_arc_from_two_tangents(S1, Sn, turn_direction_vector[-1], center_circumference_vector[-1][0], center_circumference_vector[-1][1], unicycle)
                maneuvers = [T1, C1, S1, Cn, Sn, Cn_plus_1, T2]
                correct_angles(maneuvers)
                return center_circumference_vector, maneuvers
            T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1],
                                                                       corridor_list[-2],
                                                                       end_pose, unicycle,
                                                                       center_circumference_vector[-1][0],
                                                                       center_circumference_vector[-1][1],
                                                                       turn_direction_vector[-1], t0 = 0, turn1 = 0))
            Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
            segments[-1] = Sn
            segments.pop(-2)    

    # 8- Compute the arcs
    arcs = [0] * (len(segments)-1) 
    for i in range(len(segments)-1): 
        arcs[i] = compute_arc_from_two_tangents(segments[i], segments[i+1], turn_direction_vector[i], center_circumference_vector[i][0], center_circumference_vector[i][1], unicycle)

    if len(segments) == 0: # If there's no segment, it means that the two corridors are connected by an arc
        Cn = compute_arc_from_two_tangents(S1, Sn, turn_direction_vector[-1], center_circumference_vector[-1][0], center_circumference_vector[-1][1], unicycle)
        maneuvers = [T1, C1, S1, Cn, Sn, Cn_plus_1, T2]
        correct_angles(maneuvers)
        return center_circumference_vector, maneuvers
    
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
   
    maneuvers = [T1, C1] + central_sequence + [Cn_plus_1, T2]
    correct_angles(maneuvers)

    # figure = plot_corridors(corridor_list = corridor_list, plot_vectors = True, linestyle = '--', color = 'gray', linewidth = 0.5)

    # for trajectory_piece in maneuvers:
    #     trajectory_piece.plot_path(figure)
    #     # Plot circumference
    #     if isinstance(trajectory_piece, CurvilinearArcUnicycle):
    #         trajectory_piece.plot_circle(figure)

    # for trajectory_piece in maneuvers:
    #     if isinstance(trajectory_piece, TurnOnTheSpot):
    #         print(f'\n\nTurn {trajectory_piece.turn_direction} for {trajectory_piece.maneuver_time}s of {trajectory_piece.delta_angle * 180 / pi} degrees ')
    #     # print(trajectory_piece.theta_trajectory)
    #     print(f'\nPrimitive type: {trajectory_piece.label}')
    #     # print(trajectory_piece.theta_trajectory)

    # plt.plot(start_pose[0], start_pose[1], 'ro')
    # plt.plot(end_pose[0], end_pose[1], 'ro')

    # plt.arrow(start_pose[0], start_pose[1], 0.5*cos(start_pose[2]), 0.5*sin(start_pose[2]), head_width = 0.1, head_length = 0.1, fc = 'r', ec = 'r')
    # plt.arrow(end_pose[0], end_pose[1], 0.5*cos(end_pose[2]), 0.5*sin(end_pose[2]), head_width = 0.1, head_length = 0.1, fc = 'r', ec = 'r')

    # for i in range(len(segments)-2):
    #     if check_intersection_case(segments[i], segments[i+1]):
    #         stop = True
            
    return center_circumference_vector, maneuvers


def get_theta_prova2(pose, xc2, yc2, corridor, R, turn, left_wall, right_wall, margin = 0):
    '''
    This helper function returns the vehicle heading such that the robot does not crash into the wall of the
    corridor. The corridor is first rotated, such that it is vertical (tilt of pi/2).

    :param pose: considered pose
    :type pose: list or np.ndarray
    :param xc2: x coordinate of the center of intermediate circle
    :type xc2: float
    :param yc2: y coordinate of the center of intermediate circle
    :type yc2: float
    :param corridor: considered corridor
    :type corridor: corridor
    :param R: radius of the circumference at which the robot moves at higher speed (R = vmax/omegamax)
    :type R: float
    :param turn: initial turn direction
    :type turn: float [-1, 1]
    :param margin: additional margin to avoid collision with the corridor's walls
    :type margin: float 

    :return: x coordinate of the center of the first circumference
    :rtype: float 
    :return: y coordinate of the center of the first circumference
    :rtype: float S
    :return: initial orientation such that the robot does not collide with the corridor's walls
    :rtype: float 
    '''
    width = corridor.width * 0.5 - margin
    # xc1 = pose[0] + R * cos(pose[2] + turn * pi * 0.5)
    # yc1 = pose[1] + R * sin(pose[2] + turn * pi * 0.5)
    # # Compute whether there is an intersection between the current osculating circle and one of the two walls of the corridor
    # left_wall, right_wall, _, _ = compute_intersection_points_between_line_circle(xc = xc1, yc = yc1, x0 = pose[0], y0 = pose[1], turn = turn, radius = R, corridor = corridor, margin = margin)
    # if not(left_wall) and not(right_wall):
    #     return xc1, yc1, pose[2]
    if left_wall:
        # Compute the rotation angle to obtain a vertical corridor (with tilt pi/2)
        rot_angle = pi*0.5 - corridor.tilt
        # Compute the x,y and theta coordinates in the rotated corridor
        x_tilted, y_tilted, theta_tilted = absolute_to_relative_pose(corridor, pose)
        # Compute x,y coordinate of the transformed corridor center: reminder that we consider a vertical corridor with center in (0, 0)
        cx = 0 # = cy

        # Initialize the two possible theta 
        theta_left_wall1, theta_left_wall2 = theta_tilted, theta_tilted

        ######################################################
        ### Case 1: Left side of the corridor, turn left
        ######################################################
        if x_tilted < cx and turn == 1:
            ### Left wall  
            asin_argument_left_wall = (R - width + (cx-x_tilted))/R
            if abs(asin_argument_left_wall) < 1:
                alpha = asin(asin_argument_left_wall)
                theta_left_wall1 = pi + alpha
                theta_left_wall2 = wrapPositiveAngle(-alpha)

        ######################################################
        ### Case 2: Left side of the corridor, turn right
        ######################################################
        elif x_tilted < cx and turn == -1:
            ### Left wall  
            asin_argument_left_wall = (R - width + (cx-x_tilted))/R
            if abs(asin_argument_left_wall) < 1:
                alpha = asin(asin_argument_left_wall)
                theta_left_wall1 = pi - alpha
                theta_left_wall2 = alpha

        ######################################################
        ### Case 3: Right side of the corridor, turn left
        ######################################################
        elif x_tilted >= cx and turn == 1:
            ### Left wall  
            asin_argument_left_wall = (R - width + (cx-x_tilted))/R
            if abs(asin_argument_left_wall) < 1:
                alpha = asin(asin_argument_left_wall)
                theta_left_wall1 =  wrapPositiveAngle(-alpha)
                theta_left_wall2 = pi + alpha
        
        ######################################################
        ### Case 4: Right side of the corridor, turn right
        ######################################################
        elif x_tilted >= cx and turn == -1:
            ### Left wall  
            asin_argument_left_wall = (R - width + (cx-x_tilted))/R
            if abs(asin_argument_left_wall) < 1:
                alpha = asin(asin_argument_left_wall)
                theta_left_wall1 = alpha 
                theta_left_wall2 = pi - alpha

        ### Select the best theta
        # Get back to actual corridor coordinates
        theta_transformed1 = theta_left_wall1 - rot_angle
        theta_transformed2 = theta_left_wall2 - rot_angle

        # Compute the center of the osculating circle
        xc1_theta_left_wall1 = pose[0] + R * cos(theta_transformed1 + turn * pi * 0.5)
        yc1_theta_left_wall1 = pose[1] + R * sin(theta_transformed1 + turn * pi * 0.5)
        # Compute the center of the osculating circle
        xc1_theta_left_wall2 = pose[0] + R * cos(theta_transformed2 + turn * pi * 0.5)
        yc1_theta_left_wall2 = pose[1] + R * sin(theta_transformed2 + turn * pi * 0.5)

        distance1 = (xc1_theta_left_wall1 - xc2)**2 + (yc1_theta_left_wall1 - yc2)**2
        distance2 = (xc1_theta_left_wall2 - xc2)**2 + (yc1_theta_left_wall2 - yc2)**2

        if distance1 < distance2:
            theta_transformed = theta_transformed1
            xc1 = xc1_theta_left_wall1
            yc1 = yc1_theta_left_wall1
        else:
            theta_transformed = theta_transformed2
            xc1 = xc1_theta_left_wall2
            yc1 = yc1_theta_left_wall2

    else:
        # Compute the rotation angle to obtain a vertical corridor (with tilt pi/2)
        rot_angle = pi*0.5 - corridor.tilt
        # Compute the x,y and theta coordinates in the rotated corridor
        x_tilted, y_tilted, theta_tilted = absolute_to_relative_pose(corridor, pose)
        # Compute x,y coordinate of the transformed corridor center: reminder that we consider a vertical corridor with center in (0, 0)
        cx = cy = 0 

        # Initialize the two possible theta 

        theta_right_wall1, theta_right_wall2 = theta_tilted, theta_tilted

        ######################################################
        ### Case 1: Left side of the corridor, turn left
        ######################################################
        if x_tilted < cx and turn == 1:
            ### Right wall  
            asin_argument_right_wall = (R - width - (cx-x_tilted))/R
            if abs(asin_argument_right_wall) < 1:
                alpha = asin(asin_argument_right_wall)
                theta_right_wall1 = pi - alpha
                theta_right_wall2 = alpha

        ######################################################
        ### Case 2: Left side of the corridor, turn right
        ######################################################
        elif x_tilted < cx and turn == -1:
            ### Right wall  
            asin_argument_right_wall = (R - width - (cx-x_tilted))/R

            if abs(asin_argument_right_wall) < 1:
                alpha = asin(asin_argument_right_wall)
                theta_right_wall1 = pi + alpha
                theta_right_wall2 = wrapPositiveAngle(-alpha)

        ######################################################
        ### Case 3: Right side of the corridor, turn left
        ######################################################
        elif x_tilted >= cx and turn == 1:
            ### Right wall  
            asin_argument_right_wall = (R - width - (cx-x_tilted))/R
            if abs(asin_argument_right_wall) < 1:
                alpha = asin(asin_argument_right_wall)
                theta_right_wall1 = alpha
                theta_right_wall2 = pi - alpha

        ######################################################
        ### Case 4: Right side of the corridor, turn right
        ######################################################
        elif x_tilted >= cx and turn == -1:
            ### Right wall  
            asin_argument_right_wall = (R - width - (cx-x_tilted))/R
            if abs(asin_argument_right_wall) < 1:
                alpha = asin(asin_argument_right_wall)
                theta_right_wall1 = wrapPositiveAngle(-alpha) 
                theta_right_wall2 = pi + alpha

        ### Select the best theta
        # Get back to actual corridor coordinates
        theta_transformed1 = theta_right_wall1 - rot_angle
        theta_transformed2 = theta_right_wall2 - rot_angle

        # Compute the center of the osculating circle
        xc1_theta_right_wall1 = pose[0] + R * cos(theta_transformed1 + turn * pi * 0.5)
        yc1_theta_right_wall1 = pose[1] + R * sin(theta_transformed1 + turn * pi * 0.5)
        # Compute the center of the osculating circle
        xc1_theta_right_wall2 = pose[0] + R * cos(theta_transformed2 + turn * pi * 0.5)
        yc1_theta_right_wall2 = pose[1] + R * sin(theta_transformed2 + turn * pi * 0.5)

        distance1 = (xc1_theta_right_wall1 - xc2)**2 + (yc1_theta_right_wall1 - yc2)**2
        distance2 = (xc1_theta_right_wall2 - xc2)**2 + (yc1_theta_right_wall2 - yc2)**2

        if distance1 < distance2:
            theta_transformed = theta_transformed1
            xc1 = xc1_theta_right_wall1
            yc1 = yc1_theta_right_wall1
        else:
            theta_transformed = theta_transformed2
            xc1 = xc1_theta_right_wall2
            yc1 = yc1_theta_right_wall2
        
    return xc1, yc1, theta_transformed


def compute_initial_turn_on_the_spot_collision_avoidance(start_pose, turn1, corridor1, radius, omega_max, omega_min, margin):
    '''
    Compute the initial angular displacement required to avoid collision with the corridors' walls. If no rotation is needed, the angular displacement will be zero.

    :param start_pose: initial pose
    :type start_pose: list or np.ndarray
    :param turn1: turn direction along the first circumference
    :type turn1: float
    :param corridor1: current corridor
    :type corridor1: CorridorWorld
    :param radius: v_max/omega_max
    :type radius: float
    :param omega_max: maximum angular velocity
    :type omega_max: float
    :param omega_min: minimum angular velocity
    :type omega_min: float
    :param margin: margin to avoid collision with the corridor's walls considering vehicle's footprint
    :type margin: float 

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
    x0, y0, theta0_initial = start_pose
    # Initialize the velocity of the turn-on-the-spot and the delta_angle
    omega_turn_on_the_spot, delta_angle = 0, 0

    ## Turn-on-the-spot if the allowed theta to not crash in the corridor wall is different than the current theta
    xc1, yc1, theta0 = get_theta([x0, y0, theta0_initial], corridor1, radius, turn1, margin)

    # Compute the minimum angle to reach theta0 starting from theta0_initial, if they are different
    if theta0 != theta0_initial:
        delta_angle = atan2(sin(theta0 - theta0_initial), cos(theta0 - theta0_initial))
        omega_turn_on_the_spot = omega_max if delta_angle > 0 else omega_min

    if delta_angle == 0 or efficient_sign(delta_angle) != turn1:
        theta0 = theta0_initial
        delta_angle = 0
        omega_turn_on_the_spot = 0
        xc1, yc1 = compute_center_coordinates_first_circle(x0, y0, theta0, turn1, radius)

    return delta_angle, omega_turn_on_the_spot, xc1, yc1


def get_theta(pose, corridor, R, turn, margin = 0):
    '''
    Compute the vehicle heading such that the robot does not crash into the wall of the
    corridor. The corridor is first rotated, such that it is vertical (tilt of pi/2).

    :param pose: considered pose
    :type pose: list or np.ndarray
    :param corridor: considered corridor
    :type corridor: CorridorWorld
    :param R: radius of the circumference at which the robot moves at higher speed (R = vmax/omegamax)
    :type R: float
    :param turn: initial turn direction
    :type turn: float [-1, 1]
    :param margin: additional margin to avoid collision with the corridor's walls
    :type margin: float 

    :return: x coordinate of the center of the first circumference
    :rtype: float 
    :return: y coordinate of the center of the first circumference
    :rtype: float 
    :return: initial orientation such that the robot does not collide with the corridor's walls
    :rtype: float 
    '''
    # Compute the rotation angle to obtain a vertical corridor (with tilt pi/2)
    rot_angle = pi/2 - corridor.tilt
    # Compute the x,y and theta coordinates in the rotated corridor
    x_tilted, y_tilted, theta_tilted = absolute_to_relative_pose(corridor, pose)

    # Compute x,y coordinate of the transformed corridor center: reminder that we consider a vertical corridor with center in (0, 0)
    cx = cy = 0 

    # Initialize theta
    theta = theta_tilted
    ### Case 1: Left side of the corridor, turn left
    if x_tilted < cx and turn == 1:
        # Left wall  
        asin_argument_left_wall = (R - corridor.width/2 + (cx-x_tilted))/R
        theta_left_wall = theta_tilted
        if abs(asin_argument_left_wall) < 1:
            alpha = asin(asin_argument_left_wall)
            theta_left_wall1 = pi + alpha
            theta_left_wall2 = wrapPositiveAngle(-alpha)
        delta_theta_left_wall = compute_angular_difference(theta_tilted, theta_left_wall)
        if efficient_sign(delta_theta_left_wall) != turn:
            theta_left_wall = theta_tilted
            delta_theta_left_wall = 0
        # Right wall  
        asin_argument_right_wall = (R - corridor.width/2 - (cx-x_tilted))/R
        theta_right_wall = theta_tilted
        if abs(asin_argument_right_wall) < 1:
            alpha = asin(asin_argument_right_wall)
            theta_right_wall = pi - alpha
        delta_theta_right_wall = compute_angular_difference(theta_tilted, theta_right_wall)
        if efficient_sign(delta_theta_right_wall) != turn:
            theta_right_wall = theta_tilted
            delta_theta_right_wall = 0

    ### Case 2: Left side of the corridor, turn right
    elif x_tilted < cx and turn == -1:
        # Left wall  
        asin_argument_left_wall = (R - corridor.width/2 + (cx-x_tilted))/R
        theta_left_wall = theta_tilted
        if abs(asin_argument_left_wall) < 1:
            alpha = asin(asin_argument_left_wall)
            theta_left_wall = pi - alpha
        delta_theta_left_wall = compute_angular_difference(theta_tilted, theta_left_wall)
        if efficient_sign(delta_theta_left_wall) != turn:
            theta_left_wall = theta_tilted
            delta_theta_left_wall = 0
        # Right wall  
        asin_argument_right_wall = (R - corridor.width/2 - (cx-x_tilted))/R
        theta_right_wall = theta_tilted
        if abs(asin_argument_right_wall) < 1:
            alpha = asin(asin_argument_right_wall)
            theta_right_wall = wrapPositiveAngle(-alpha)
        delta_theta_right_wall = compute_angular_difference(theta_tilted, theta_right_wall)
        # if efficient_sign(delta_theta_right_wall) != turn:
        #     theta_right_wall = theta_tilted
        #     delta_theta_right_wall = 0
    
    ### Case 3: Right side of the corridor, turn left
    elif x_tilted >= cx and turn == 1:
        # Left wall  
        asin_argument_left_wall = (R - corridor.width/2 - (cx-x_tilted))/R
        theta_left_wall = theta_tilted
        if abs(asin_argument_left_wall) < 1:
            alpha = asin(asin_argument_left_wall)
            theta_left_wall = alpha
        delta_theta_left_wall = compute_angular_difference(theta_tilted, theta_left_wall)
        if efficient_sign(delta_theta_left_wall) != turn:
            theta_left_wall = theta_tilted
            delta_theta_left_wall = 0
        # Right wall  
        asin_argument_right_wall = (R - corridor.width/2 + (cx-x_tilted))/R
        theta_right_wall = theta_tilted
        if abs(asin_argument_right_wall) < 1:
            alpha = asin(asin_argument_right_wall)
            theta_right_wall = wrapPositiveAngle(-alpha)
        delta_theta_right_wall = compute_angular_difference(theta_tilted, theta_right_wall)
        if efficient_sign(delta_theta_right_wall) != turn:
            theta_right_wall = theta_tilted
            delta_theta_right_wall = 0
    
    ### Case 4: Right side of the corridor, turn right
    elif x_tilted >= cx and turn == -1:
        # Left wall  
        asin_argument_left_wall = (R - corridor.width/2 - (cx-x_tilted))/R
        theta_left_wall = theta_tilted
        if abs(asin_argument_left_wall) < 1:
            alpha = asin(asin_argument_left_wall)
            theta_left_wall = wrapPositiveAngle(-alpha)
        delta_theta_left_wall = compute_angular_difference(theta_tilted, theta_left_wall)
        if efficient_sign(delta_theta_left_wall) != turn:
            theta_left_wall = theta_tilted
            delta_theta_left_wall = 0
        # Right wall  
        asin_argument_right_wall = (R - corridor.width/2 + (cx-x_tilted))/R
        theta_right_wall = theta_tilted
        if abs(asin_argument_right_wall) < 1:
            alpha = asin(asin_argument_right_wall)
            theta_right_wall = alpha 
        delta_theta_right_wall = compute_angular_difference(theta_tilted, theta_right_wall)
        if efficient_sign(delta_theta_right_wall) != turn:
            theta_right_wall = theta_tilted
            delta_theta_right_wall = 0

    # Pick the delta_theta
    theta = theta_left_wall if abs(delta_theta_left_wall) < abs(delta_theta_right_wall) else theta_right_wall
    theta = theta_right_wall
    delta_angle = atan2(sin(theta - theta_tilted), cos(theta - theta_tilted))
    # If the angle you want to reach implies a rotation on a different direction than turn1, don't change theta
    # if efficient_sign(delta_angle) != turn:
    #     theta = pose[2]
    
    # Get back to actual corridor coordinates
    theta_transformed = theta - rot_angle

    # Compute the center of the osculating circle
    xc1 = pose[0] + R * cos(theta_transformed + turn * pi/2)
    yc1 = pose[1] + R * sin(theta_transformed + turn * pi/2)
    
    return xc1, yc1, theta_transformed


def prune_circles_adjust_trajectory_unicycle(
    intermediate_circles,
    segments,
    unicycle,
    start_pose,
    corridor_list,
    end_pose,
    start_maneuvers,
    end_maneuvers,
):
    """
    Prune unnecessary intermediate circles and rebuild/adjust the trajectory accordingly.

    For N corridors:
      - there are (N - 1) intermediate circles
      - there are N connecting segments
    """
    # Remember: at the beginning, 
    # one segment per corridor, 
    # one circle per pair of corridors 
    # N corridors means N segments and N-1 circles
    # segment i in corridor i,  for i = 0, ..., N
    # Segments i and i+1 are adjacent to circle i, for i = 0, ..., N-1
    # Circles i and i+1 are at the extremes of corridor i+1

    processed_finished = False

    while not processed_finished:
        changed = False
        for i in range(len(intermediate_circles)):
            # If an intersection between two segments is found
            if check_intersection_case(segments[i], segments[i+1]): 
                # prune circle i
                removed_index = i
                original_n_circles = len(intermediate_circles)
                # removed_circle = intermediate_circles[i]
                intermediate_circles.remove_at(i)
                if len(intermediate_circles) == 0:
                    return None, None, None, None
                # remove the intersecting segments
                del segments[i : i + 2]
                
                # If the first int circle is pruned, recompute P^init 
                if removed_index == 0: 
                    first_int_circ = intermediate_circles.first
                    corridor1 = corridor_list[0]
                    corridor2 = corridor_list[intermediate_circles.first.index+1]
                    # Compute first three maneuvers
                    T1, C1, S1 = compute_three_maneuvers_compact(corridor1,
                                                                corridor2,
                                                                start_pose,
                                                                unicycle,
                                                                first_int_circ.xc,
                                                                first_int_circ.yc,
                                                                first_int_circ.turn_direction,
                                                                t0 = 0,
                                                                turn1 = 0
                                                                )
                    start_maneuvers = [T1, C1, S1]

                    is_inside, exit_point, exit_t = check_segment_inside_corridors_with_exit_point(
                        S1,
                        corridor_list[0:intermediate_circles.first.index+2]
                    )
                    # segments.insert(0, S1)
                    if is_inside:
                        segments.insert(0, S1)
                    else: 
                        print("Trajectory out of corridor constraints")
                        figure = plot_corridors(corridor_list)
                        plot_analytical_trajectory(start_maneuvers + segments + end_maneuvers + [S1], figure)
                        plt.plot(exit_point[0], exit_point[1], 'ro', label = 'Exit point')
                        plt.show(block = True)
                        raise ValueError("Trajectory out of corridor constraints")
                
                # If a middle circle is pruned, adjust P^mid
                # Since the circle has been removed already, you want to connect the previous circle
                # i-1 to the next circle, which is now i (used to be i+1)
                elif 0 < removed_index < original_n_circles - 1: 
                    circ1 = intermediate_circles[i-1]
                    circ2 = intermediate_circles[i]
                    new_segment = compute_segment_between_two_circles(
                        circ1.xc,
                        circ1.yc,
                        circ2.xc,
                        circ2.yc,
                        circ1.turn_direction,
                        circ2.turn_direction,
                        unicycle,
                        overlap = False
                        )
                    
                    is_inside, exit_point, exit_t = check_segment_inside_corridors_with_exit_point(
                        new_segment,
                        corridor_list[circ1.index:circ2.index+2]
                    )
                    if is_inside:
                        segments.insert(i,new_segment)
                    else: 
                        print("Trajectory out of corridor constraints")
                        figure = plot_corridors(corridor_list)
                        plot_analytical_trajectory(start_maneuvers + segments + end_maneuvers + [new_segment], figure)
                        plt.plot(exit_point[0], exit_point[1], 'ro', label = 'Exit point')
                        plt.show(block = True)
                        raise ValueError("Trajectory out of corridor constraints")

                # If the last int circle is pruned, recompute P^final
                elif removed_index == original_n_circles - 1:
                    last_int_circ = intermediate_circles.last
                    corridor1 = corridor_list[-1]
                    corridor2 = corridor_list[last_int_circ.index]
                    # Compute last three maneuvers 
                    T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor1,
                                                                                    corridor2,
                                                                                    end_pose,
                                                                                    unicycle,
                                                                                    last_int_circ.xc,
                                                                                    last_int_circ.yc,
                                                                                    last_int_circ.turn_direction,
                                                                                    t0 = 0,
                                                                                    turn1 = 0
                                                                                    )
                                                                        )
                    Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
                    end_maneuvers = [Sn, Cn_plus_1, T2]
                    
                    is_inside, exit_point, exit_t = check_segment_inside_corridors_with_exit_point(
                        Sn,
                        corridor_list[last_int_circ.index:]
                    )

                    if is_inside:
                        segments.append(Sn)
                    else: 
                        print("Trajectory out of corridor constraints")
                        figure = plot_corridors(corridor_list)
                        plot_corridors(corridor_list[last_int_circ.index:], figure, color = 'r')
                        plot_analytical_trajectory(start_maneuvers + segments + end_maneuvers + [Sn], figure)
                        plt.plot(exit_point[0], exit_point[1], 'ro', label = 'Exit point')
                        plt.show(block = True)
                        raise ValueError("Trajectory out of corridor constraints")

                changed = True
                break 

        processed_finished = not changed
            
    return intermediate_circles, segments, start_maneuvers, end_maneuvers


def compute_trajectory_multiple_corridors(corridor_list,
                                          start_pose,
                                          end_pose,
                                          unicycle, 
                                          obstacle_center = None, obstacle_radius = None):
    # Initialize vectors with the maneuvers. 
    # For n corridors, the solution sequence counts 2n+3 primitives
    # T1, C2, S3, C4, S5, C6, S7, ..., C2n, S2n+1, C2n+2, T2n+3
    
    margin = 0.0 # This margin is used to compute the center of the circumferences both at the intersection between corridors and for obstacle avoidance
    tol = 1e-2

    # ### 1- Merge corridors ### (make function)
    
    # # Merge two consecutive corridors if  
    # # 1) they have the same tilt;
    # # 2) they have the same width;
    # # 3) their centers belong to a line with same orientation as the corridors' tilt.
     
    # corridor_to_remove = []
    # for i in range(len(corridor_list)-1):
    #     if (
    #         abs(corridor_list[i].tilt - corridor_list[i+1].tilt) < tol
    #         and (
    #             wrapPositiveAngle(
    #                 atan2(
    #                     corridor_list[i+1].center[1] - corridor_list[i].center[1],
    #                     corridor_list[i+1].center[0] - corridor_list[i].center[0]
    #                 )
    #             )
    #             - wrapPositiveAngle(corridor_list[i].tilt)
    #         ) < tol
    #         and (abs(corridor_list[i].width - corridor_list[i+1].width)< 1e-2)
    #     ):
    #         new_corridor = get_corridor_from_vector(corridor_list[i].tail, corridor_list[i+1].head, corridor_list[i].width, add_height = 0)
    #         corridor_list[i] = new_corridor
    #         corridor_to_remove = corridor_to_remove + [i+1]
    
    # # Remove the corridors after merging, if needed
    # ind = 0    
    # for index in corridor_to_remove:
    #     corridor_list.pop(index - ind)
    #     ind += 1

    # # If the resulting corridor list has one or two corridors, use the dedicated functions
    # if len(corridor_list) == 1:
    #     return [0,0], ocp_function_one_corridor(corridor_list[0], start_pose, end_pose, unicycle) 
    # elif len(corridor_list) == 2:
    #     trajectory , _ = compute_trajectory_unicycle_two_corridors(corridor_list[0], corridor_list[1], start_pose, end_pose, unicycle)
    #     return [0,0], trajectory
    
    ### 2- Compute the turn directions ###
    turn_direction_vector = compute_turn_direction_vector(corridor_list)

    # # If two consecutive corridors have the same tilt but different width, 
    # # they don't have to be merged but the turn diretion has to be computed differently
    # zero_indices = []
    # if np.any(np.array(turn_direction_vector) == 0): # If any turn direction is zero
    #     zero_indices = np.where(np.array(turn_direction_vector) == 0)[0].tolist()
    
    # # if zero_indices != []:
    # #     zero_indices = zero_indices.tolist()

    # # If the last turn direction is 0, it means that the last two corridors have same tilt. 
    # # First assign a turn direction between the penultimate and the last corridor.
    # last_corridor_rotated = None
    # if zero_indices != [] and zero_indices[-1] == len(corridor_list)-2:
    #     add_ind = 2 
    #     turn_direction = 0
    #     while turn_direction == 0:
    #         turn_direction = corridor_list[-add_ind].compute_relative_turn_direction(corridor_list[-1])
    #         add_ind += 1
    #     turn_direction_vector[-1] = turn_direction
    #     zero_indices.pop(-1)
    #     last_corridor_rotated = corridor_list[-1].rotate_corridor(turn_direction * pi * 0.5)

    # for ind in zero_indices: # for each couple of corridors with the same tilt
    #     add_ind = 2
    #     turn_direction = 0
    #     while turn_direction == 0:
    #         if ind + add_ind >= len(corridor_list) and last_corridor_rotated is not None:
    #             # final_corr = corridor_list[-1]
    #             # angle = atan2((end_pose[1] - final_corr.center[1]), (end_pose[0] - final_corr.center[0]))
    #             # vector2 = [cos(angle), sin(angle)]
    #             # turn_direction = compute_turn_direction(corridor_list[ind].unit_vector, vector2)
    #             # turn_direction = corridor_list[ind-1].compute_relative_turn_direction(corridor_list[-1])
    #             turn_direction = corridor_list[ind].compute_relative_turn_direction(last_corridor_rotated)

    #         else:
    #             turn_direction = corridor_list[ind].compute_relative_turn_direction(corridor_list[ind + add_ind])
    #             # turn_direction = corridor_list[ind-1].compute_relative_turn_direction(corridor_list[ind])
    #         # turn_direction = corridor_list[ind].compute_relative_turn_direction(corridor_list[ind + add_ind])
    #         add_ind += 1
    #     turn_direction_vector[ind] = turn_direction
    # turn_direction_vector[8] = -1
    # turn_direction_vector[9] = -1
    # turn_direction_vector[5] = -1
    # turn_direction_vector[6] = -1
    ### 3- Compute the corner points ###
    corner_point_vector = compute_corner_point_vector(corridor_list, turn_direction_vector)
    
    ### 4- Compute the centers of the intermediate circles ###
    center_circumference_vector = compute_center_coordinates_vector(corridor_list, turn_direction_vector, corner_point_vector, unicycle, margin = margin)
    # angle_array = np.linspace(0, 2*pi, 100)
    # figure = plot_corridors(corridor_list, plot_vectors = True,)
    # for element in center_circumference_vector:
    #     plt.plot(element[0], element[1], 'ro')
    #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
    # plt.show(block = True)
    # Merge two circles together if they are too close to each other
    # i = 0
    # while i < len(center_circumference_vector)-1:
    #     if compute_distance_two_points(center_circumference_vector[i], center_circumference_vector[i+1]) < unicycle.max_radius:
    #         center_circumference_vector[i] = [(center_circumference_vector[i][0] + center_circumference_vector[i+1][0]) * 0.5,
    #                                           (center_circumference_vector[i][1] + center_circumference_vector[i+1][1]) * 0.5]
    #         turn_direction_vector.pop(i+1)
    #         corner_point_vector.pop(i+1)
    #         center_circumference_vector.pop(i+1)
    #     else:
    #         i += 1

    # figure = plot_corridors(corridor_list, plot_vectors = True,)
    # for element in center_circumference_vector:
    #     plt.plot(element[0], element[1], 'ro')
    #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
    # plt.show(block = True)

    # Change position of the second center if the start pose is inside the second circle
    # if compute_distance_two_points(center_circumference_vector[0], start_pose[:2]) < unicycle.max_radius:
    #     xc_first_circle, yc_first_circle = start_inside_second_circle_case(corridor_list[0], corridor_list[1], turn_direction_vector[0], corner_point_vector[0], unicycle)
    #     center_circumference_vector[0] = [xc_first_circle, yc_first_circle]
    
    # # Change the position of the penultimate circle if the end pose is inside the penultimate circle
    # if compute_distance_two_points(center_circumference_vector[-1], end_pose[:2]) < unicycle.max_radius:
    #     xc_last_circle, yc_last_circle = start_inside_second_circle_case(*invert_inputs_start_inside_second_circle_case(corridor_list[-1], corridor_list[-2], turn_direction_vector[-1], corner_point_vector[-1], unicycle))
    #     center_circumference_vector[-1] = [xc_last_circle, yc_last_circle]
    
    # attension = 1
    # figure = plot_corridors(corridor_list, plot_vectors = True,)
    # for element in center_circumference_vector:
    #     plt.plot(element[0], element[1], 'ro')
    #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
    # plt.show(block = True)
    # # Eliminate the third circle if there's an initial u-turn
    # if is_it_u_turn(corridor_list[0], corridor_list[2], turn_direction_vector[0], turn_direction_vector[1]):
    #     center_circumference_vector[0] = [(center_circumference_vector[0][0] + center_circumference_vector[1][0]) * 0.5,
    #                                       (center_circumference_vector[0][1] + center_circumference_vector[1][1]) * 0.5]
    #     turn_direction_vector.pop(1)
    #     corner_point_vector.pop(1)
    #     center_circumference_vector.pop(1)

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
            
    # # Eliminate the last third circle if there's a final u-turn
    # if is_it_u_turn(corridor_list[-1], corridor_list[-3], turn_direction_vector[-1], turn_direction_vector[-2]):
    #     center_circumference_vector[-1] = [(center_circumference_vector[-1][0] + center_circumference_vector[-2][0]) * 0.5, 
    #                                        (center_circumference_vector[-1][1] + center_circumference_vector[-2][1]) * 0.5]
    #     turn_direction_vector.pop(-2)
    #     corner_point_vector.pop(-2)
    #     center_circumference_vector.pop(-2)

    # center_circumference_vector, turn_direction_vector = swap_circles(turn_direction_vector, center_circumference_vector, unicycle.max_radius, [start_pose[:2]] + center_circumference_vector + [end_pose[:2]])
    
    ### 6- Compute the segments ###
    # Compute first three maneuvers
    T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0],
                                                 corridor_list[1],
                                                 start_pose,
                                                 unicycle,
                                                 center_circumference_vector[0][0],
                                                 center_circumference_vector[0][1],
                                                 turn_direction_vector[0], t0 = 0, turn1 = 0)

    # Compute last three maneuvers 
    T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1],
                                                                       corridor_list[-2],
                                                                       end_pose, unicycle,
                                                                       center_circumference_vector[-1][0],
                                                                       center_circumference_vector[-1][1],
                                                                       turn_direction_vector[-1], t0 = 0, turn1 = 0))
    Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
    # figure = plot_corridors(corridor_list)
    # for element in center_circumference_vector:
    #     plt.plot(element[0], element[1], 'ro')
    #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
    # T1.plot_path(figure)
    # S1.plot_path(figure)
    # Sn.plot_path(figure)
    # T2.plot_path(figure)
    # C1.plot_path(figure)
    # Cn_plus_1.plot_path(figure)
    # plt.show(block = True)

    segments = [0] * (len(turn_direction_vector)+1)
    segments[0] = S1
    segments[-1] = Sn
    for i in range(1, len(turn_direction_vector)): 
        segments[i] = compute_segment_between_two_circles(center_circumference_vector[i-1][0], center_circumference_vector[i-1][1],
                                                          center_circumference_vector[i][0], center_circumference_vector[i][1],
                                                          turn_direction_vector[i-1], turn_direction_vector[i], unicycle)
    

    # # 8- Compute the arcs
    # arcs = [0] * (len(segments)-1) 
    # for i in range(len(segments)-1): 
    #     arcs[i] = compute_arc_from_two_tangents(segments[i], segments[i+1], turn_direction_vector[i], center_circumference_vector[i][0], center_circumference_vector[i][1], unicycle)

    # figure = plot_corridors(corridor_list)
    # plot_corridors([corridor_list[0]], color = 'r', figure = figure, label = "corridor1")
    # plot_corridors([corridor_list[1]], color = 'g', figure = figure, label = "corridor2")
    # plot_corridors([corridor_list[2]], color = 'y', figure = figure, label = "corridor3")
    # plt.plot(corner_point_vector[0][0], corner_point_vector[0][1], 'go', label = "corner point 1")
    # plt.plot(corner_point_vector[1][0], corner_point_vector[1][1], 'mo', label = "corner point 2")
    # plt.plot(corner_point_vector[2][0], corner_point_vector[2][1], 'yo', label = "corner point 3")
    # plt.plot(corner_point_vector[3][0], corner_point_vector[3][1], 'co', label = "corner point 4")


    # for element in center_circumference_vector:
    #     plt.plot(element[0], element[1], 'ro')
    #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
    # # plt.plot(center_circumference_vector[0][0], center_circumference_vector[0][1], 'go', label = "circle 1") # First circle center
    # # plt.plot(center_circumference_vector[1][0], center_circumference_vector[1][1], 'mo', label = "circle 2") # Second circle center
    # # plt.plot(center_circumference_vector[2][0], center_circumference_vector[2][1], 'yo', label = "circle 3") # Third circle center
    # # plt.plot(center_circumference_vector[3][0], center_circumference_vector[3][1], 'co', label = "circle 4") # Last circle center
    # # plt.legend()
    # T1.plot_path(figure)
    # S1.plot_path(figure)
    # Sn.plot_path(figure)
    # T2.plot_path(figure)
    # C1.plot_path(figure)
    # Cn_plus_1.plot_path(figure)
    # for segment in segments:
    #     segment.plot_path(figure)
    # # for arc in arcs:
    # #     arc.plot_path(figure)

    # plt.plot(start_pose[0], start_pose[1], 'ro')
    # plt.plot(end_pose[0], end_pose[1], 'ro')

    # plt.arrow(start_pose[0], start_pose[1], 0.5*cos(start_pose[2]), 0.5*sin(start_pose[2]), head_width = 0.1, head_length = 0.1, fc = 'r', ec = 'r')
    # plt.arrow(end_pose[0], end_pose[1], 0.5*cos(end_pose[2]), 0.5*sin(end_pose[2]), head_width = 0.1, head_length = 0.1, fc = 'r', ec = 'r')

    # plt.show(block = True)

    ### INTERSECTION CHECKS ###

    # Part 1
    while check_intersection_case(segments[0], segments[1]):
            corridor_list.pop(1)
            center_circumference_vector.pop(0)
            turn_direction_vector.pop(0)
            ## Number of circles check
            if len(corridor_list) == 1:
                return "Invalid inputs"
                # return [0,0], ocp_function_free_space(start_pose, end_pose, unicycle)
            elif len(corridor_list) == 2:
                T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0],
                                            corridor_list[1],
                                            start_pose,
                                            unicycle,
                                            center_circumference_vector[0][0],
                                            center_circumference_vector[0][1],
                                            turn_direction_vector[0], t0 = 0, turn1 = 0)
                T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1],
                                                                       corridor_list[-2],
                                                                       end_pose, unicycle,
                                                                       center_circumference_vector[-1][0],
                                                                       center_circumference_vector[-1][1],
                                                                       turn_direction_vector[-1], t0 = 0, turn1 = 0))
                Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
                if check_intersection_case(S1, Sn):
                    # trajectory = ocp_function_free_space(start_pose, end_pose, unicycle)
                    # return [], trajectory
                    return "Invalid inputs"
                Cn = compute_arc_from_two_tangents(S1, Sn, turn_direction_vector[-1], center_circumference_vector[-1][0], center_circumference_vector[-1][1], unicycle)
                maneuvers = [T1, C1, S1, Cn, Sn, Cn_plus_1, T2]
                correct_angles(maneuvers)
                for i in range(len(maneuvers)-1):
                    if maneuvers[i+1].time_grid[0] != maneuvers[i].time_grid[-1]:
                        maneuvers[i+1].add_time_offset(abs(maneuvers[i+1].time_grid[0] - maneuvers[i].time_grid[-1]))
                    if i == len(maneuvers) - 2:
                        break
                return center_circumference_vector, maneuvers
            # Compute T1, C2, S3 from start_pose to circles[0] with turn_direction[0]
            T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0],
                                                        corridor_list[1],
                                                        start_pose,
                                                        unicycle,
                                                        center_circumference_vector[0][0],
                                                        center_circumference_vector[0][1],
                                                        turn_direction_vector[0], t0 = 0, turn1 = 0)
            segments[0] = S1
            segments.pop(1)

    # figure = plot_corridors(corridor_list)
    # for element in center_circumference_vector:
    #     plt.plot(element[0], element[1], 'ro')
    #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
    # plt.plot(center_circumference_vector[0][0], center_circumference_vector[0][1], 'go', label = "circle 1") # First circle center
    # plt.plot(center_circumference_vector[1][0], center_circumference_vector[1][1], 'mo', label = "circle 2") # Second circle center
    # plt.plot(center_circumference_vector[2][0], center_circumference_vector[2][1], 'yo', label = "circle 3") # Third circle center
    # plt.legend()
    # T1.plot_path(figure)
    # S1.plot_path(figure)
    # Sn.plot_path(figure)
    # T2.plot_path(figure)
    # C1.plot_path(figure)
    # Cn_plus_1.plot_path(figure)
    # for segment in segments:
    #     segment.plot_path(figure)
    # # for arc in arcs:
    # #     arc.plot_path(figure)
    # plt.show(block = True)
    
    # Part 2
    i = 1 
    while i < len(segments)-2:
        if check_intersection_case(segments[i], segments[i+1]):
            x_corner, y_corner = get_corner_point(corridor_list[i], corridor_list[i+1], -turn_direction_vector[i])

            xc, yc   = compute_center_coordinates_second_circle([x_corner, y_corner],
                                                                -turn_direction_vector[i],
                                                                unicycle.max_radius, unicycle.width, margin, corridor_list[i].tilt, corridor_list[i+1].tilt, corridor_list[i], corridor_list[i+1])
            center_circumference_vector[i] = [xc, yc]

            segments[i] = compute_segment_between_two_circles(center_circumference_vector[i-1][0], center_circumference_vector[i-1][1],
                                                              center_circumference_vector[i][0], center_circumference_vector[i][1],
                                                              turn_direction_vector[i-1], turn_direction_vector[i], unicycle)
            

            # cambia qui = compute_center_coordinates_





            center_circumference_vector.pop(i)
            turn_direction_vector.pop(i)
            corridor_list.pop(i+1)
            ## Number of circles check
            if len(corridor_list) == 1:
                # return [0,0], ocp_function_free_space(start_pose, end_pose, unicycle)
                return "Invalid inputs"
            elif len(corridor_list) == 2:
                T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0],
                                            corridor_list[1],
                                            start_pose,
                                            unicycle,
                                            center_circumference_vector[0][0],
                                            center_circumference_vector[0][1],
                                            turn_direction_vector[0], t0 = 0, turn1 = 0)
                T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1],
                                                                       corridor_list[-2],
                                                                       end_pose, unicycle,
                                                                       center_circumference_vector[-1][0],
                                                                       center_circumference_vector[-1][1],
                                                                       turn_direction_vector[-1], t0 = 0, turn1 = 0))
                Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
                if check_intersection_case(S1, Sn):
                    # trajectory = ocp_function_free_space(start_pose, end_pose, unicycle)
                    # return [], trajectory
                    return "Invalid inputs"
                Cn = compute_arc_from_two_tangents(S1, Sn, turn_direction_vector[-1], center_circumference_vector[-1][0], center_circumference_vector[-1][1], unicycle)
                maneuvers = [T1, C1, S1, Cn, Sn, Cn_plus_1, T2]
                correct_angles(maneuvers)
                for i in range(len(maneuvers)-1):
                    if maneuvers[i+1].time_grid[0] != maneuvers[i].time_grid[-1]:
                        maneuvers[i+1].add_time_offset(abs(maneuvers[i+1].time_grid[0] - maneuvers[i].time_grid[-1]))
                    if i == len(maneuvers) - 2:
                        break
                return center_circumference_vector, maneuvers
            # Substite Si (which goes from Ci to Ci+1) with another Si which goes from Ci to Ci+2
            segments[i] = compute_segment_between_two_circles(center_circumference_vector[i-1][0], center_circumference_vector[i-1][1],
                                                              center_circumference_vector[i][0], center_circumference_vector[i][1],
                                                              turn_direction_vector[i-1], turn_direction_vector[i], unicycle)
            segments.pop(i+1)
            segment_removed = True
            # figure = plot_corridors(corridor_list)
            # for element in center_circumference_vector:
            #     plt.plot(element[0], element[1], 'ro')
            #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
            # T1.plot_path(figure)
            # S1.plot_path(figure)
            # Sn.plot_path(figure)
            # T2.plot_path(figure)
            # C1.plot_path(figure)
            # Cn_plus_1.plot_path(figure)
            # for segment in segments:
            #     segment.plot_path(figure)
            # segments[i].plot_path(figure, color = 'm', linewidth = 2)
            # # for arc in arcs:
            # #     arc.plot_path(figure)
            # plt.show(block = True)
        else:
            i += 1

    # Part 3
    while check_intersection_case(segments[-2], segments[-1]):
            # figure = plot_corridors(corridor_list)
            # for element in center_circumference_vector:
            #     plt.plot(element[0], element[1], 'ro')
            #     plt.plot(element[0] + unicycle.max_radius * np.cos(angle_array), element[1] + unicycle.max_radius * np.sin(angle_array), 'r--', linewidth = 0.5)
            # segments[-2].plot_path(figure)
            # segments[-1].plot_path(figure)
            # plt.show(block = True)
            corridor_list.pop(-2)
            center_circumference_vector.pop(-1)
            turn_direction_vector.pop(-1)
            ## Number of circles check
            if len(corridor_list) == 1:
                # return [0,0], ocp_function_free_space(start_pose, end_pose, unicycle)
                return "Invalid inputs"
            elif len(corridor_list) == 2:
                T1, C1, S1 = compute_three_maneuvers_compact(corridor_list[0],
                                            corridor_list[1],
                                            start_pose,
                                            unicycle,
                                            center_circumference_vector[0][0],
                                            center_circumference_vector[0][1],
                                            turn_direction_vector[0], t0 = 0, turn1 = 0)
                T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1],
                                                                       corridor_list[-2],
                                                                       end_pose, unicycle,
                                                                       center_circumference_vector[-1][0],
                                                                       center_circumference_vector[-1][1],
                                                                       turn_direction_vector[-1], t0 = 0, turn1 = 0))
                Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
                if check_intersection_case(S1, Sn):
                    # trajectory = ocp_function_free_space(start_pose, end_pose, unicycle)
                    # return [], trajectory
                    return "Invalid inputs"
                Cn = compute_arc_from_two_tangents(S1, Sn, turn_direction_vector[-1], center_circumference_vector[-1][0], center_circumference_vector[-1][1], unicycle)
                maneuvers = [T1, C1, S1, Cn, Sn, Cn_plus_1, T2]
                correct_angles(maneuvers)
                for i in range(len(maneuvers)-1):
                    if maneuvers[i+1].time_grid[0] != maneuvers[i].time_grid[-1]:
                        maneuvers[i+1].add_time_offset(abs(maneuvers[i+1].time_grid[0] - maneuvers[i].time_grid[-1]))
                    if i == len(maneuvers) - 2:
                        break
                return center_circumference_vector, maneuvers
            T2, Cn_plus_1, Sn = compute_three_maneuvers_compact(*invert_inputs(corridor_list[-1],
                                                                       corridor_list[-2],
                                                                       end_pose, unicycle,
                                                                       center_circumference_vector[-1][0],
                                                                       center_circumference_vector[-1][1],
                                                                       turn_direction_vector[-1], t0 = 0, turn1 = 0))
            Sn, Cn_plus_1, T2 = invert_maneuvers([T2, Cn_plus_1, Sn], t0 = 0)
            segments[-1] = Sn
            segments.pop(-2)    

    # 8- Compute the arcs
    arcs = [0] * (len(segments)-1) 
    for i in range(len(segments)-1): 
        arcs[i] = compute_arc_from_two_tangents(segments[i], segments[i+1], turn_direction_vector[i], center_circumference_vector[i][0], center_circumference_vector[i][1], unicycle)

    if len(segments) == 0: # If there's no segment, it means that the two corridors are connected by an arc
        Cn = compute_arc_from_two_tangents(S1, Sn, turn_direction_vector[-1], center_circumference_vector[-1][0], center_circumference_vector[-1][1], unicycle)
        maneuvers = [T1, C1, S1, Cn, Sn, Cn_plus_1, T2]
        correct_angles(maneuvers)
        for i in range(len(maneuvers)-1):
            if maneuvers[i+1].time_grid[0] != maneuvers[i].time_grid[-1]:
                maneuvers[i+1].add_time_offset(abs(maneuvers[i+1].time_grid[0] - maneuvers[i].time_grid[-1]))
            if i == len(maneuvers) - 2:
                break
        return center_circumference_vector, maneuvers
    
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
   
    maneuvers = [T1, C1] + central_sequence + [Cn_plus_1, T2]
    correct_angles(maneuvers)

    # figure = plot_corridors(corridor_list = corridor_list, plot_vectors = True, linestyle = '--', color = 'gray', linewidth = 0.5)

    # for trajectory_piece in maneuvers:
    #     trajectory_piece.plot_path(figure)
    #     # Plot circumference
    #     if isinstance(trajectory_piece, CurvilinearArcUnicycle):
    #         trajectory_piece.plot_circle(figure)

    # for trajectory_piece in maneuvers:
    #     if isinstance(trajectory_piece, TurnOnTheSpot):
    #         print(f'\n\nTurn {trajectory_piece.turn_direction} for {trajectory_piece.maneuver_time}s of {trajectory_piece.delta_angle * 180 / pi} degrees ')
    #     # print(trajectory_piece.theta_trajectory)
    #     print(f'\nPrimitive type: {trajectory_piece.label}')
    #     # print(trajectory_piece.theta_trajectory)

    # plt.plot(start_pose[0], start_pose[1], 'ro')
    # plt.plot(end_pose[0], end_pose[1], 'ro')

    # plt.arrow(start_pose[0], start_pose[1], 0.5*cos(start_pose[2]), 0.5*sin(start_pose[2]), head_width = 0.1, head_length = 0.1, fc = 'r', ec = 'r')
    # plt.arrow(end_pose[0], end_pose[1], 0.5*cos(end_pose[2]), 0.5*sin(end_pose[2]), head_width = 0.1, head_length = 0.1, fc = 'r', ec = 'r')

    # for i in range(len(segments)-2):
    #     if check_intersection_case(segments[i], segments[i+1]):
    #         stop = True

    for i in range(len(maneuvers)-1):
        if maneuvers[i+1].time_grid[0] != maneuvers[i].time_grid[-1]:
            maneuvers[i+1].add_time_offset(abs(maneuvers[i+1].time_grid[0] - maneuvers[i].time_grid[-1]))
        if i == len(maneuvers) - 2:
            break

    return center_circumference_vector, maneuvers


def second_circle(corridor1, corridor2, turn_direction, corner_point, unicycle, start_pose):
    '''
    Compute the center of the second circle. The position of the second circle can vary if
    1. The second corridor is narrow
    2. The start pose is inside the second circle

    :param corridor1: first corridor
    :type corridor1: CorridorWorld

    :param corridor2: second corridor
    :type corridor2: CorridorWorld

    :param turn_direction: turn direction
    :type turn_direction: either [-1, 1]

    :param corner_point: corner point
    :type corner_point: list of floats

    :param unicycle: unicycle model
    :type unicycle: Unicycle

    :param start_pose: start pose
    :type start_pose: list of floats

    :return: x coordinate of the center of the second circle
    :rtype: float

    :return: y coordinate of the center of the second circle
    :rtype: float
    '''

    if corridor2.width > unicycle.width:
        xc2, yc2 = compute_center_coordinates_second_circle(corner_point, turn_direction, unicycle.max_radius, unicycle.width, 0, corridor1.tilt, corridor2.tilt)
    else:
        # If the second corridor is narrow the position of the second circle is adjusted
        # xc2, yc2 = circ_center_narrow_corridors(corridor2.tilt, corner_point, unicycle.max_radius, unicycle.width, turn_direction) ## I REMOVED THE NARROW CORRIDORS FEATURE
        xc2, yc2 = compute_center_coordinates_second_circle(corner_point, turn_direction, unicycle.max_radius, unicycle.width, 0, corridor1.tilt, corridor2.tilt)
    # If the start pose is inside the second circumference, adjust the position of the second circle
    if ((xc2 - start_pose[0])**2 + (yc2 - start_pose[1])**2) < unicycle.max_radius**2:
        xc2, yc2 = start_inside_second_circle_case(corridor1, corridor2, turn_direction, corner_point, unicycle)
    if unicycle.max_radius < unicycle.width * 0.5:
        xc2, yc2 = compute_center_coordinates_second_circle_R_smaller_than_r(corner_point, turn_direction, unicycle.max_radius, unicycle.width, 0, corridor1.tilt, corridor2.tilt)
        
    return xc2, yc2
