"""
Geometric utilities for corridor manipulation.

This module contains functions related to the geometric properties of
corridors, such as construction from vectors, point inclusion checks,
and computation of corner points between corridors.
"""

from math import sqrt, atan2, inf, pi

import numpy as np

from ..corridor import CorridorWorld
from .geometry_operations import compute_turn_direction, wrapPositiveAngle
from .intersections import get_intersection
from .poses import absolute_to_relative_pose


def get_corridor_from_vector(start_point, end_point, width, add_height=0):
    '''
    Build a corridor from a centerline vector.

    The corridor is defined by a vector going from `start_point` (tail)
    to `end_point` (head), which represents the centerline of the
    corridor.

    The resulting corridor:
    - is centered at the midpoint of the vector,
    - is oriented along the vector direction,
    - has the specified width,
    - has a length equal to the vector length plus an optional
      additional margin (`add_height`).

    :param start_point: (x, y) coordinates of the tail of the vector
    :type start_point: numpy.ndarray
    :param end_point: (x, y) coordinates of the head of the vector
    :type end_point: numpy.ndarray
    :param width: Width of the corridor [m]
    :type width: float
    :param add_height: Additional length added to the corridor [m].
                       This is summed to the Euclidean distance between
                       `start_point` and `end_point`.
    :type add_height: float

    :return: Constructed corridor
    :rtype: CorridorWorld
    '''
    deltax = end_point[0] - start_point[0]
    deltay = end_point[1] - start_point[1]

    center = [
        start_point[0] + 0.5 * deltax,
        start_point[1] + 0.5 * deltay,
    ]

    tilt = wrapPositiveAngle(atan2(deltay, deltax))
    height = sqrt(deltax**2 + deltay**2) + add_height

    return CorridorWorld(width, height, center, tilt)


def check_point_inside_corridor(corridor, point):
    '''
    Check if a given point lies inside a corridor.

    The point is first expressed in the corridor's local frame, then
    checked against the corridor bounds (width and height).

    :param corridor: corridor to be checked
    :type corridor: CorridorWorld
    :param point: (x, y) coordinates of the point
    :type point: numpy.ndarray

    :return: True if the point is inside the corridor
    :rtype: bool
    '''
    tol = 1e-6

    rel_x, rel_y, _ = absolute_to_relative_pose(corridor, [point[0], point[1], 0.0])

    half_width = 0.5 * corridor.width + tol
    half_height = 0.5 * corridor.height + tol

    return (
        -half_width <= rel_x <= half_width and
        -half_height <= rel_y <= half_height
    )


def shrink_corridor_list(corridor_list, margin):
    """
    Shrink the corridors in the list to avoid collisions.
    
    :param corridor_list: list, the list of corridors to shrink
    :type corridor_list: list of CorridorWorld objects
    :param margin: float, the margin to shrink the corridors
    :type margin: float
    
    :return: list, the shrunk corridors
    """
    # return [corridor.shrink(margin) for corridor in corridor_list]
    shrunk_corridors = []
    for corridor in corridor_list:
        shrunk_corridors.append(corridor.shrink(margin))
        
    return shrunk_corridors


def get_corner_point(corridor1, corridor2, turn):
    '''
    Given two corridors, compute the corner point at the intersection of
    their edges.

    The selected edges depend on the turn direction between the two
    corridors.

    :param corridor1: starting corridor
    :type corridor1: CorridorWorld
    :param corridor2: arrival corridor
    :type corridor2: CorridorWorld
    :param turn: turn direction between the two corridors
    :type turn: float in [-1, 1]

    :return: x and y coordinates of the corner point
    :rtype: numpy.ndarray or None
    '''
    if turn > 0:  # turn left
        select_edges1 = np.array([0, 3])  # F and L faces of corridor1
        select_edges2 = np.array([2, 3])  # B and L faces of corridor2
    elif turn < 0:  # turn right
        select_edges1 = np.array([0, 1])  # F and R faces of corridor1
        select_edges2 = np.array([2, 1])  # B and R faces of corridor2
    else:
        return None

    candidate_points = []

    # Check the intersection points between the selected faces
    for edge2_idx in select_edges2:
        for edge1_idx in select_edges1:
            edge2 = corridor2.W[:, edge2_idx]
            edge1 = corridor1.W[:, edge1_idx]

            intersection_point = get_intersection(edge2, edge1)

            if intersection_point == inf or intersection_point == []:
                continue

            if (
                check_point_inside_corridor(corridor1, intersection_point)
                and check_point_inside_corridor(corridor2, intersection_point)
            ):
                candidate_points.append(np.array(intersection_point))

    if not candidate_points:
        print("No corner point was found between these two corridors")
        return None

    # Select the rightmost or leftmost point depending on the turn direction
    selected_point = candidate_points[0]
    corridor1_tail = np.array([corridor1.tail[0], corridor1.tail[1]])

    for point in candidate_points[1:]:
        turn_direction = compute_turn_direction(
            selected_point - corridor1_tail,
            point - corridor1_tail,
        )

        if turn > 0 and turn_direction > 0:
            selected_point = point
        elif turn < 0 and turn_direction < 0:
            selected_point = point

    return selected_point


def compute_turn_direction_vector(corridor_list):
    '''
    Compute the cross product between vector1 and vector2
    Since they are on a plane, the third element of both vectors is zero
    The resulting vector lies on the plane perpendicular to the x-y plane, therefore
    only the third element is different than zero.
    vector1 = [x1, y1, z1]
    vector2 = [x2, y2, z2]
    vector1 x vector2 = [0 0 x1*y2-x2y1]

    :param vector1: vector 1 in global frame
    :type vector1: numpy.ndarray
    :param vector2: vector 2 in global frame
    :type vector2: numpy.ndarray

    :return: turn_direction, float number larger the zero = turn left, 
            lower than 0 = turn right, equal to 0 = go straight
    :rtype: float64
    '''
    turn_direction_vector = [
    compute_turn_direction(c1.unit_vector, c2.unit_vector)
    for c1, c2 in zip(corridor_list[:-1], corridor_list[1:])
    ]
    return turn_direction_vector


def compute_corner_point_vector(corridor_list, turn_direction_vector):
    '''
    Compute the corner points between the corridors

    :param corridor_list: list of corridors
    :type corridor_list: list of CorridorWorld
    :param turn_direction_vector: vector of turn directions
    :type turn_direction_vector: list of floats

    :return: list of corner points
    :rtype: list of floats
    '''
    corner_point_vector = [0] * (len(corridor_list) - 1)
    for i in range(len(corridor_list)-1):
        corner_point_vector[i] = get_corner_point(
            corridor_list[i],
            corridor_list[i+1],
            turn_direction_vector[i])
    return corner_point_vector


def get_corner_point_and_intersecting_edges(corridor1, corridor2, turn):
    """
    Given two corridors, compute the corner point at the intersection of their edges.
    The selected edges depend on the turn direction between the two corridors.

    Parameters
    ----------
    corridor1 : CorridorWorld
        Starting corridor.
    corridor2 : CorridorWorld
        Arrival corridor.
    turn : float
        Turn direction between the two corridors. Positive = left, negative = right.

    Returns
    -------
    tuple
        (corner_point, intersecting_edges)

        corner_point : np.ndarray shape (2,) or None
            x and y coordinates of the selected corner point.
        intersecting_edges : tuple or list
            (edge_idx_corridor1, edge_idx_corridor2) for the selected corner point,
            or [] if no valid point was found.
    """
    candidates = []  # each item: (intersection_point, edge_idx_corridor1, edge_idx_corridor2)

    if turn == +1:  # turn left
        select_edges1 = (0, 3)  # F and L faces of corridor1
        select_edges2 = (2, 3)  # B and L faces of corridor2
    elif turn == -1:  # turn right
        select_edges1 = (0, 1)  # F and R faces of corridor1
        select_edges2 = (2, 1)  # B and R faces of corridor2
    else:
        return None, []

    # Check intersection points between selected faces
    for i in select_edges2:
        for k in select_edges1:
            w1 = corridor2.W[:, i]
            w2 = corridor1.W[:, k]

            intersection_point = get_intersection(w1, w2)

            # Skip invalid intersections
            if intersection_point == inf or intersection_point == []:
                continue

            if (
                check_point_inside_corridor(corridor1, intersection_point)
                and check_point_inside_corridor(corridor2, intersection_point)
            ):
                candidates.append((np.array(intersection_point), k, i))

    if not candidates:
        print("No corner point was found between two corridors")
        return None, []

    # Initialize with first candidate
    best_point, best_edge1, best_edge2 = candidates[0]
    tail = np.array([corridor1.tail[0], corridor1.tail[1]])

    for point, edge1, edge2 in candidates[1:]:
        turn_direction_new_corner_point = compute_turn_direction(
            best_point - tail,
            point - tail
        )

        if turn > 0 and turn_direction_new_corner_point > 0:
            best_point, best_edge1, best_edge2 = point, edge1, edge2
        elif turn < 0 and turn_direction_new_corner_point < 0:
            best_point, best_edge1, best_edge2 = point, edge1, edge2

    return best_point, (best_edge1, best_edge2)


def compute_corner_point_vector_and_intersecting_edges(corridor_list, turn_direction_vector):
    '''
    Compute the corner points between the corridors

    :param corridor_list: list of corridors
    :type corridor_list: list of CorridorWorld
    :param turn_direction_vector: vector of turn directions
    :type turn_direction_vector: list of floats

    :return: list of corner points
    :rtype: list of floats
    '''
    corner_point_vector = [0] * (len(corridor_list) - 1)
    intersecting_edges = [0] * (len(corridor_list) - 1)
    for i in range(len(corridor_list)-1):
        if turn_direction_vector[i] != 0:
            corner_point_vector[i], intersecting_edges[i] = get_corner_point_and_intersecting_edges(
                corridor_list[i],
                corridor_list[i+1],
                turn_direction_vector[i])
        else: 
            corner_point_vector[i], intersecting_edges[i] = None, None

    return corner_point_vector, intersecting_edges


def remove_zeros_from_turn_direction_vector(corridor_list, final_pose, turn_direction_vector):
    # # check if turn directions are equal to 0
    n = len(corridor_list)
    if turn_direction_vector[-1] == 0:
        # Compute all the intersection points beween the penultimate and the last corridor
        intersection_points = np.empty([0, 2])  # initialize a np.array 
        edges1 = np.array([0, 1, 2, 3]) 
        edges2 = np.array([0, 1, 2, 3])  
        # Check the intersection points between the selected faces of the corridors
        for i in edges2:
            for k in edges1:
                w1 = corridor_list[-2].W[:, i]
                w2 = corridor_list[-1].W[:, k]
                # check whether the two faces don't coincide (can happen especially for 90 degrees turns)
                intersection_point = get_intersection(w1, w2)
                if (intersection_point != inf) and (intersection_point != []):
                    if (check_point_inside_corridor(corridor_list[-2], intersection_point) and check_point_inside_corridor(corridor_list[-1], intersection_point)):
                        intersection_points = np.vstack((intersection_point,intersection_points))
                        #direction = 
        
        if intersection_points.size == 0:
            raise RuntimeError(
                "No valid intersection points found between the last two corridors."
                "Zero turn direction between last two corridors.."
            )
 
    for i, turn in enumerate(turn_direction_vector):
        if turn == 0: 
            if i <= n-3: 
                add_i = 1
                while i + add_i <= n-2: 
                    turn = compute_turn_direction(corridor_list[i].unit_vector, corridor_list[i + add_i].unit_vector)
                    if turn != 0:
                        turn_direction_vector[i] = turn
                        break
                    add_i += 1
            else: # i == n-3
                add_i = 1
                last_corridor_inverted = corridor_list[-1].rotate_corridor(pi)
                while i - add_i >= 0: 

                    turn = compute_turn_direction(last_corridor_inverted.unit_vector.unit_vector, )
                    if turn != 0:
                        turn_direction_vector[i] = turn
                        continue
                    add_i += 1

    return turn_direction_vector


def point_matches_any_corner(point, corners, tol=1e-9):
    for corner in corners:
        dx = point[0] - corner[0]
        dy = point[1] - corner[1]

        if dx * dx + dy * dy <= tol * tol:
            return True

    return False


