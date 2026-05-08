# import numpy as np
# from math import sin, cos, pi, atan2, sqrt, asin, floor
# import sympy as sp
# import math as m

# import time
# import warnings
# import matplotlib.pyplot as plt
# from .plot_helpers import plot_corridors, plot_analytical_trajectory
# from ..Corridor import CorridorWorld
# from ..Trajectory import LinearSegmentUnicycle, CurvilinearArcUnicycle, UnicycleTrajectoryOptimal, TurnOnTheSpot, BackwardArc
# from .helper_functions import compute_distance_two_points, compute_center_coordinates_second_circle_R_smaller_than_r, compute_extreme_poses_arc_line_two_radii, compute_intersection_two_segments, circ_center_narrow_corridors, fit_new_circle, compute_angular_difference_with_turn_direction, wrapPositiveAngle, compute_turn_direction, get_intersection, get_vehicle_vertices, compute_path_coordinates_curvilinear_arc, efficient_sign, compute_angular_difference, compute_direction, include_search, compute_initial_turn_direction, compute_center_coordinates_second_circle, compute_center_coordinates_first_circle, check_point_inside_segment, check_intersection_two_segments, compute_extreme_poses_arc_line
# from ..Geometry import IntermediateCircle, Point, IntermediateCirclesSequence
# from .corridor_prepocessing import plot_intermediate_circle_choices, create_intermediate_circle_choices_sequence, compute_turn_direction_choices, compute_corner_point_choices_and_intersecting_edges, compute_center_coordinate_choices_according_to_edges, compute_center_coordinates_second_circle_according_to_edges, get_corner_point_and_intersecting_edges

# from .poses import (
#     relative_to_absolute_pose,
#     absolute_to_relative_pose,
#     compute_end_pose,
#     compute_start_pose,
#     pose_from_shrunken_corridor_relative_frame,
# )

# from .intersections import (
#     compute_intersection_points_between_line_circle,
#     compute_intersection_points_circle_segment,
# )








    

# def swap_circles(turn_direction_vector, center_circumference_vector, radius, reference_point_vector):
#     for i in range(len(center_circumference_vector)-1):
#         if turn_direction_vector[i] == turn_direction_vector[i+1]:
#             if abs((center_circumference_vector[i][0] - center_circumference_vector[i+1][0])**2 + (center_circumference_vector[i][1] - center_circumference_vector[i+1][1])**2 - 4 * radius**2) < 1e-2:
#                 beta1 = atan2(center_circumference_vector[i][1]- reference_point_vector[i][1], center_circumference_vector[i][0] - reference_point_vector[i][0])
#                 beta2 = atan2(center_circumference_vector[i+1][1]- reference_point_vector[i+1][1], center_circumference_vector[i+1][0] - reference_point_vector[i+1][0])
#                 turn = compute_turn_direction([cos(beta1), sin(beta1)], [cos(beta2), sin(beta2)])
#                 if turn != turn_direction_vector[i]:
#                     center1, turn1 = center_circumference_vector[i+1], turn_direction_vector[i+1]
#                     center2, turn2 = center_circumference_vector[i], turn_direction_vector[i]
#                     center_circumference_vector[i], center_circumference_vector[i+1] = center1, center2
#                     turn_direction_vector[i], turn_direction_vector[i+1] = turn1, turn2
#     return center_circumference_vector, turn_direction_vector


# ## New functions for plotly dash
# def get_path_coordinates(analytical_trajectory):
#     x_coordinates = np.concatenate([maneuver.path_coordinates[:, 0] for maneuver in analytical_trajectory])
#     y_coordinates = np.concatenate([maneuver.path_coordinates[:, 1] for maneuver in analytical_trajectory])
#     return x_coordinates, y_coordinates      

# def get_v_omega_time(analytical_trajectory):
#     time_array = np.concatenate([maneuver.time_grid for maneuver in analytical_trajectory])
#     v_array = np.concatenate([maneuver.forward_velocity for maneuver in analytical_trajectory])
#     omega_array = np.concatenate([maneuver.angular_velocity for maneuver in analytical_trajectory])
#     return v_array, omega_array, time_array






# def compute_maneuver_time_mirrored_circ(xc1, yc1, xc2_mirr, yc2_mirr, tau1, tau2, R, theta0, thetat, v_max, omega_max):
#     # Compute the maneuvers as if the circumference to reach is the mirrored one
#     x1_mirr, y1_mirr, theta1_mirr, x2_mirr, y2_mirr, _ = compute_extreme_poses_arc_line(xc1, yc1, xc2_mirr, yc2_mirr, tau1, -tau2, R)
#     iota1_mirr = compute_angular_difference_with_turn_direction(theta0, theta1_mirr, tau1)
#     iota3_mirr = compute_angular_difference_with_turn_direction(theta1_mirr, thetat, -tau2)
#     d2_mirr = sqrt((x2_mirr - x1_mirr)**2 + (y2_mirr - y1_mirr)**2)

#     total_time = abs(iota1_mirr)/omega_max + d2_mirr/v_max + abs(iota3_mirr)/omega_max
#     return total_time

# def compute_mirr_circ(xc1, yc1, xc2, yc2, R, delta, tau1, tau2, theta0, xt, yt, thetat, v_max, omega_max):
#     # Variables needed for proof: mirrored circumference
#     xc2_mirr = xc2 + (2 * R)*cos(delta - pi * 0.5)
#     yc2_mirr = yc2 + (2 * R)*sin(delta - pi * 0.5)
#     # Compute xt and yt mirrored
#     x1, y1, theta1, x2, y2, _ = compute_extreme_poses_arc_line(xc1, yc1, xc2, yc2, tau1, tau2, R)
#     m = tan(delta)
#     m_perp = -1/m
#     x_prime = (yt - m_perp*xt - y1 + m*x1)/(m - m_perp)
#     y_prime = y1 + m*(x_prime - x1)
#     xt_mirr = 2*x_prime - xt
#     yt_mirr = 2*y_prime - yt
#     thetat_mirr = 2*delta - thetat

#     return xc2_mirr, yc2_mirr, xt_mirr, yt_mirr, thetat_mirr, compute_maneuver_time_mirrored_circ(xc1, yc1, xc2_mirr, yc2_mirr, tau1, tau2, R, theta0, thetat_mirr, v_max, omega_max)

# def compute_trajectory_bicycle_two_corridors(corridor1, corridor2, start_pose, end_pose, bicycle):
#     '''
#     Compute the sequence of primitives that build the time-optimal trajectory for a bicycle vehicle within two corridors. 

#     :param corridor1: first corridor
#     :type corridor1: CorridorWorld
#     :param corridor2: second corridor
#     :type corridor2: CorridorWorld
#     :param start_pose: initial pose within the first corridor
#     :type start_pose: list of floats
#     :param end_pose: final pose within the second corridor
#     :type end_pose: list of floats
#     :param bicycle: bicycle vehicle
#     :type Bicycle: Bicycle

#     :return: sequence of primitives
#     :rtype: list of primitives
#     :return: boolean indicating whether an intersection has been detected
#     :rtype: Boolean
#     '''
#     ## Merge two corridors if needed #TO BE DONE
#     # if check_merge_corridors(corridor1, corridor2): # If the two corridors can be merged
#     #     new_corridor = get_corridor_from_vector(corridor1.tail, corridor2.head, corridor1.width, add_height = 0)
#     #     # Compute the trajectory within the new corridor and return false as the check_intersection Boolean
#     #     return compute_trajectory_unicycle_one_corridor(new_corridor, start_pose, unicycle, end_pose), False

#     ## Compute main turn direction tau2
#     turn_direction = compute_turn_direction(corridor1.vector, corridor2.vector)
#     ## Compute the corner point
#     corner_point = get_corner_point(corridor1, corridor2, turn_direction)
#     ## Compute the center of the second circumference
#     xc2, yc2 = second_circle_bicycle(corridor1, corridor2, turn_direction, corner_point, bicycle, start_pose)
#     ## Compute the first two maneuvers from start pose to the second circumference
#     arc1, segment2 = compute_two_maneuvers_within_corridor(corridor1, start_pose, bicycle, xc2, yc2, turn_direction, t0 = 0, turn1 = 0)
#     # turn_on_the_spot1, arc1, segment1 = compute_three_maneuvers_compact(corridor1, corridor2, start_pose, unicycle, xc2, yc2, turn_direction, t0 = 0, turn1 = 0)
#     ## Compute the last three maneuvers from end pose to the second circumference and invert them
#     arc5, segment4 = compute_two_maneuvers([end_pose[0], end_pose[1], end_pose[2]+pi], bicycle, xc2, yc2, -turn_direction, t0 = 0, turn1 = 0)
#     # raw_maneuvers = compute_three_maneuvers_compact(*invert_inputs(corridor2, corridor1, end_pose, bicycle, xc2, yc2, turn_direction, t0 = 0, turn1 = 0))
#     segment4, arc5 = invert_maneuvers([arc5, segment4], t0 = 0)

#     # Check whether the intersection case occurs
#     # check_intersection = check_intersection_case(segment2, segment4)
#     check_intersection = False

#     if check_intersection:
#         pass
#         # maneuvers = compute_trajectory_intersection_case_without_optimization(corridor1, corridor2, start_pose, end_pose, unicycle)
#     else:
#         # Build the arc along the second circle
#         arc3 = compute_arc_from_two_tangents(segment2, segment4, turn_direction, xc2, yc2, bicycle)
#         segment4.add_time_offset(arc3.tf)
#         arc5.add_time_offset(arc3.tf)
#         trajectory = [arc1, segment2, arc3, segment4, arc5]
    
#     correct_angles(trajectory)
#     return trajectory

# def second_circle_bicycle(corridor1, corridor2, turn_direction, corner_point, bicycle, start_pose):
#     '''
#     Compute the center of the second circle. The position of the second circle can vary if
#     1. The second corridor is narrow
#     2. The start pose is inside the second circle

#     :param corridor1: first corridor
#     :type corridor1: CorridorWorld

#     :param corridor2: second corridor
#     :type corridor2: CorridorWorld

#     :param turn_direction: turn direction
#     :type turn_direction: either [-1, 1]

#     :param corner_point: corner point
#     :type corner_point: list of floats

#     :param bicycle: bicycle model
#     :type bicycle: Bicycle

#     :param start_pose: start pose
#     :type start_pose: list of floats

#     :return: x coordinate of the center of the second circle
#     :rtype: float

#     :return: y coordinate of the center of the second circle
#     :rtype: float
#     '''

#     if corridor2.width > bicycle.width:
#         xc2, yc2 = compute_center_coordinates_second_circle(corner_point, turn_direction, bicycle.max_radius, bicycle.width, 0, corridor1.tilt, corridor2.tilt)
#     else:
#         # If the second corridor is narrow the position of the second circle is adjusted
#         xc2, yc2 = circ_center_narrow_corridors(corridor2.tilt, corner_point, bicycle.max_radius, bicycle.width, turn_direction)
#     # If the start pose is inside the second circumference, adjust the position of the second circle
#     if ((xc2 - start_pose[0])**2 + (yc2 - start_pose[1])**2) < bicycle.max_radius**2:
#         xc2, yc2 = start_inside_second_circle_case(corridor1, corridor2, turn_direction, corner_point, bicycle)
#     return xc2, yc2


# def sample_whole_trajectory(trajectory, ds):
    
#     # Initialize the index of the sampled trajectory
#     ind = 0
    
#     # Compute the length of the whole trajectory as the sum of
#     # the length of each trajectory piece
#     length_whole_traj = 0
#     for trajectory_piece in trajectory:
#         length_whole_traj += trajectory_piece.path_length
    
#     # Compute the total number of samples
#     total_samples_number = floor(length_whole_traj / ds)
    
#     # Initiliaze the array containing the x,y,theta coordinates
#     xs_analytical = []
#     ys_analytical = []
#     thetas_analytical = []
    
#     # The first x,y,theta correspond to the initial 
#     # x,y,theta of the first trajectory piece
#     # xs_analytical[ind] = trajectory[0].x0
#     # ys_analytical[ind] = trajectory[0].y0
#     # thetas_analytical[ind] = trajectory[0].theta0
#     # Initial ds
#     ds_init = 0
    
#     # Start sampling the trajectory
#     for index, trajectory_piece in enumerate(trajectory):
        
#         # Compute the number of samples for the current trajectory piece
#         # Remove the ds initial
#         sample_number_float = (trajectory_piece.path_length - ds_init) / ds
#         sample_number_int = floor(sample_number_float)
#         length_trajectory_piece_sampled = ds * sample_number_int
#         remainder = (trajectory_piece.path_length - ds_init) % ds
#         xs_analytical_traj_piece = [0] * (sample_number_int + 1)
#         ys_analytical_traj_piece = [0] * (sample_number_int + 1)
#         thetas_analytical_traj_piece = [0] * (sample_number_int + 1)
        
#         if isinstance(trajectory_piece, LinearSegmentUnicycle):
#             x0 = trajectory_piece.x0 + ds_init * cos(trajectory_piece.theta0)
#             y0 = trajectory_piece.y0 + ds_init * sin(trajectory_piece.theta0)
#             # thetas_analytical_traj_piece[0] = trajectory_piece.theta0
#         elif isinstance(trajectory_piece, CurvilinearArcUnicycle) or isinstance(trajectory_piece, BackwardArc):
#             dtheta_init = ds_init / trajectory_piece.radius
#             epsilon0 = trajectory_piece.epsilon + trajectory_piece.turn_direction * dtheta_init
#             theta0 = trajectory_piece.theta0 + trajectory_piece.turn_direction * dtheta_init
        
#         if isinstance(trajectory_piece, LinearSegmentUnicycle):
#             xf = x0 + length_trajectory_piece_sampled * cos(trajectory_piece.theta)
#             yf = y0 + length_trajectory_piece_sampled * sin(trajectory_piece.theta)
#             xs_analytical_traj_piece[0:sample_number_int  + 1] = np.linspace(trajectory_piece.x0,
#                                                                  xf,
#                                                                  sample_number_int + 1)    
#             ys_analytical_traj_piece[0:sample_number_int  + 1] = np.linspace(trajectory_piece.y0,
#                                                                  yf,
#                                                                  sample_number_int + 1)
#             thetas_analytical_traj_piece[0:sample_number_int  + 1]  = trajectory_piece.theta0 * np.ones(sample_number_int + 1)
            
#         elif isinstance(trajectory_piece, CurvilinearArcUnicycle) or isinstance(trajectory_piece, BackwardArc):
#             dtheta = ds / trajectory_piece.radius
#             radius = trajectory_piece.radius
#             if isinstance(trajectory_piece, CurvilinearArcUnicycle):
#                 epsilonf = epsilon0 + trajectory_piece.turn_direction * dtheta * sample_number_int
#                 thetaf = theta0 + trajectory_piece.turn_direction * dtheta * sample_number_int
#             else:
#                 epsilonf = epsilon0 - trajectory_piece.turn_direction * dtheta * sample_number_int
#                 thetaf = theta0 - trajectory_piece.turn_direction * dtheta * sample_number_int
#             xc, yc = trajectory_piece.xc, trajectory_piece.yc
#             xs_analytical_traj_piece[0:sample_number_int + 1] = xc + radius * np.cos(np.linspace(epsilon0, epsilonf, sample_number_int + 1))
#             ys_analytical_traj_piece[0:sample_number_int + 1] = yc + radius * np.sin(np.linspace(epsilon0, epsilonf, sample_number_int + 1))
#             thetas_analytical_traj_piece[0:sample_number_int  + 1]  = np.linspace(theta0, thetaf, sample_number_int + 1)
         
#         xs_analytical = np.concatenate((xs_analytical, xs_analytical_traj_piece))
#         ys_analytical = np.concatenate((ys_analytical, ys_analytical_traj_piece))
#         thetas_analytical = np.concatenate((thetas_analytical, thetas_analytical_traj_piece))
#         ds_init = ds - remainder
#         # if index == 4:
#         #     break
        
#     s_array = np.linspace(0, total_samples_number * ds, total_samples_number +1)
        
#     return xs_analytical, ys_analytical, thetas_analytical, s_array











