from networkx import radius
import numpy as np
from math import sin, cos, pi, sqrt, atan2, asin, acos, tan, floor
import sympy as sp
import math as m
import casadi as cs
import yaml
import time
import warnings
import matplotlib.pyplot as plt
from .plot_helpers import plot_corridors, plot_vehicle, plot_analytical_trajectory
from copy import copy
from ..corridor import CorridorWorld
from ..trajectory import LinearSegmentUnicycle, CurvilinearArcUnicycle, UnicycleTrajectory, UnicycleTrajectoryOptimal, TurnOnTheSpot, BackwardArc
from ..geometry import Circle, IntermediateCircle, Point, Pose, IntermediateCirclesSequence, IntermediateCircleChoice, IntermediateCircleChoicesSequence
from .geometry_operations import compute_angular_difference, wrapPositiveAngle, compute_angular_difference_with_turn_direction, efficient_sign, check_point_inside_segment
from dataclasses import dataclass

@dataclass
class PairConstraint:
    value: float
    dim1: str   # "width" or "length"
    dim2: str

def absolute_to_relative_pose(corridor, absolute_pose):
    '''
    Convert an absolute pose to a pose expressed in the corridor frame.

    The corridor frame is centered at the corridor center, with:
    - x axis along the corridor width,
    - y axis along the corridor length.

    :param corridor: considered corridor
    :type corridor: CorridorWorld
    :param absolute_pose: absolute pose (x, y, theta)
    :type absolute_pose: list or numpy.ndarray

    :return: relative pose (x, y, theta)
    :rtype: list
    '''
    rot_angle = pi * 0.5 - corridor.tilt

    cos_a = cos(rot_angle)
    sin_a = sin(rot_angle)

    abs_x, abs_y, abs_theta = absolute_pose

    dx = abs_x - corridor.center[0]
    dy = abs_y - corridor.center[1]

    rel_x = dx * cos_a - dy * sin_a
    rel_y = dx * sin_a + dy * cos_a
    rel_theta = abs_theta + rot_angle

    return [rel_x, rel_y, rel_theta]

def compute_minimum_widths_and_lengths(corridor_list, vehicle, turn_direction_vector, intersecting_edges, center_circumference_vector):
    """ 
    Compute the minimum widths required for each corridor to guarantee
    collision-free maneuvers for the unicycle/bicycle vehicle.
    :param planner: analytical planner
    :type planner: MotionPlanner object
    :return: list of minimum widths for each corridor
    :rtype: list of floats
    """
    
    R = vehicle.max_radius
    r = vehicle.width * 0.5
    dimensions_list = [0] * (len(corridor_list)-1)

    for i in range(len(corridor_list)-1):
        corridor1 = corridor_list[i]
        corridor2 = corridor_list[i+1]
        edge_corridor1 = intersecting_edges[i][0]
        edge_corridor2 = intersecting_edges[i][1]
        if turn_direction_vector[i] == 1: 
            side_side = (
                edge_corridor1 == 3 and
                edge_corridor2 == 3
            )
            side_back = (
                edge_corridor1 == 3 and
                edge_corridor2 == 2
            )
            front_side = (
                edge_corridor1 == 0 and
                edge_corridor2 == 3
            )
            if side_side: 
                beta = 0.5 * abs(compute_angular_difference(corridor1.tilt, corridor2.tilt))
                q = (R - r) * cos(beta)
                value = r + R - q
                dim1 = "width"
                dim2 = "width"
            elif side_back:
                beta = 0.5 * abs(compute_angular_difference(corridor1.tilt, corridor2.tilt + pi/2))
                q = (R - r) * cos(beta)
                value = r + R - q
                dim1 = "width"
                dim2 = "length"
            elif front_side:
                beta = 0.5 * abs(compute_angular_difference(corridor1.tilt - pi/2, corridor2.tilt))
                q = (R - r) * cos(beta)
                value = r + R - q
                dim1 = "length"
                dim2 = "width"
            else:
                raise ValueError(f"Invalid corridor configuration at index {i}: turn direction {turn_direction_vector[i]}, intersecting edges {intersecting_edges[i]}")
            dimensions_list[i] = PairConstraint(value, dim1, dim2)
        elif turn_direction_vector[i] == -1: 
            side_side = (
                edge_corridor1 == 1 and
                edge_corridor2 == 1
            )
            side_back = (
                edge_corridor1 == 1 and
                edge_corridor2 == 2
            )
            front_side = (
                edge_corridor1 == 0 and
                edge_corridor2 == 1
            )
            if side_side: 
                beta = 0.5 * abs(compute_angular_difference(corridor1.tilt, corridor2.tilt))
                q = (R - r) * cos(beta)
                value = r + R - q
                dim1 = "width"
                dim2 = "width"
            elif side_back:
                beta = 0.5 * abs(compute_angular_difference(corridor1.tilt, corridor2.tilt - pi/2))
                q = (R - r) * cos(beta)
                value = r + R - q
                dim1 = "width"
                dim2 = "length"
            elif front_side:
                beta = 0.5 * abs(compute_angular_difference(corridor1.tilt + pi/2, corridor2.tilt))
                q = (R - r) * cos(beta)
                value = r + R - q
                dim1 = "length"
                dim2 = "width"
            else:
                raise ValueError(f"Invalid corridor configuration at index {i}: turn direction {turn_direction_vector[i]}, intersecting edges {intersecting_edges[i]}")
            dimensions_list[i] = PairConstraint(value, dim1, dim2)

    figure =plot_corridors(corridor_list)
    angle_array = np.linspace(0, 2*pi, 100)
    for c in center_circumference_vector: 
        plt.plot(c[0], c[1], 'ro')
        plt.plot(c[0] + R * np.cos(angle_array), c[1] + R * np.sin(angle_array), 'r--')
    for i in range(len(dimensions_list)-1):
        corridor1 = corridor_list[i]
        corridor2 = corridor_list[i+1]
        if turn_direction_vector[i] == 1:
            if dimensions_list[i].dim1 == "width" and dimensions_list[i].dim2 == "width":
                corner1 = corridor1.get_corners()[2]
                corner2 = corridor1.get_corners()[3]
                corner3 = corridor2.get_corners()[2]
                corner4 = corridor2.get_corners()[3]
                value = dimensions_list[i].value
                plt.plot([corner1[0] + value * cos(corridor1.tilt-pi/2), corner2[0] + value * cos(corridor1.tilt-pi/2)],
                            [corner1[1] + value * sin(corridor1.tilt-pi/2), corner2[1] + value * sin(corridor1.tilt-pi/2)], 'go-')
                plt.plot([corner3[0] + value * cos(corridor2.tilt-pi/2), corner4[0] + value * cos(corridor2.tilt-pi/2)],
                            [corner3[1] + value * sin(corridor2.tilt-pi/2), corner4[1] + value * sin(corridor2.tilt-pi/2)], 'go-')
            elif dimensions_list[i].dim1 == "width" and dimensions_list[i].dim2 == "length":
                corner1 = corridor1.get_corners()[2]
                corner2 = corridor1.get_corners()[3]
                corner3 = corridor2.get_corners()[1]
                corner4 = corridor2.get_corners()[2]
                plt.plot([corner1[0] + value * cos(corridor1.tilt-pi/2), corner2[0] + value * cos(corridor1.tilt-pi/2)],
                [corner1[1] + value * sin(corridor1.tilt-pi/2), corner2[1] + value * sin(corridor1.tilt-pi/2)], 'go-')
                plt.plot([corner3[0] + value * cos(corridor2.tilt), corner4[0] + value * cos(corridor2.tilt)],
                [corner3[1] + value * sin(corridor2.tilt), corner4[1] + value * sin(corridor2.tilt)], 'go-')

    plt.show(block = True)


    s_max_list = [0] * (len(corridor_list)-1)


    return dimensions_list, s_max_list


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




def compute_turn_direction_choices(corridor_list):
    n = len(corridor_list) - 1
    turn_direction_choices = [None] * n

    for i, (c1, c2) in enumerate(zip(corridor_list[:-1], corridor_list[1:])):
        td = compute_turn_direction(c1.unit_vector, c2.unit_vector)
        turn_direction_choices[i] = [td] if td != 0 else [1, -1]

    return turn_direction_choices


def compute_corner_point_choices_and_intersecting_edges(
    corridor_list,
    turn_direction_choices,
):
    """
    Compute possible corner points and intersecting edges between corridors.

    turn_direction_choices[i] is a list, for example:
        [1]
        [-1]
        [1, -1]

    The returned lists have the same nested structure.
    """

    n = len(corridor_list) - 1

    corner_point_choices = [None] * n
    intersecting_edges_choices = [None] * n

    for i in range(n):
        m = len(turn_direction_choices[i])

        corner_point_choices[i] = [None] * m
        intersecting_edges_choices[i] = [None] * m

        for j, turn_direction in enumerate(turn_direction_choices[i]):
            (
                corner_point_choices[i][j],
                intersecting_edges_choices[i][j],
            ) = get_corner_point_and_intersecting_edges(
                corridor_list[i],
                corridor_list[i + 1],
                turn_direction,
            )

    return corner_point_choices, intersecting_edges_choices


def compute_center_coordinate_choices_according_to_edges(
    corridor_list,
    turn_direction_choices,
    corner_point_choices,
    intersecting_edges_choices,
    vehicle,
    margin=0,
):
    n = len(corner_point_choices)

    center_coordinate_choices = [None] * n

    for i in range(n):
        m = len(turn_direction_choices[i])
        center_coordinate_choices[i] = [None] * m

        for j, turn_direction in enumerate(turn_direction_choices[i]):
            xc, yc = compute_center_coordinates_second_circle_according_to_edges(
                corner_point_choices[i][j],
                turn_direction,
                vehicle.max_radius,
                vehicle.width,
                margin,
                intersecting_edges_choices[i][j],
                corridor1=corridor_list[i],
                corridor2=corridor_list[i + 1],
            )

            center_coordinate_choices[i][j] = [xc, yc]

    return center_coordinate_choices








def create_intermediate_circle_choices_sequence(
    intermediate_circle_center_choices,
    turn_direction_choices,
    corner_point_choices,
    radius,
    s_max_circles,
):
    """
    Create the sequence of intermediate circle choices.

    Each outer index i corresponds to one corridor transition.
    Each inner index j corresponds to one candidate circle for that transition.
    """

    n = len(intermediate_circle_center_choices)

    if not (
        len(turn_direction_choices)
        == len(corner_point_choices)
        == len(s_max_circles)
        == n
    ):
        raise ValueError(
            "Impossible to define IntermediateCircleChoicesSequence: "
            "all outer input vectors must have the same length."
        )

    choices = [None] * n

    for i in range(n):
        m = len(intermediate_circle_center_choices[i])

        if not (
            len(turn_direction_choices[i])
            == len(corner_point_choices[i])
            == m
        ):
            raise ValueError(
                f"Impossible to define IntermediateCircleChoice at index {i}: "
                "all inner input vectors must have the same length."
            )

        candidates = [None] * m

        for j in range(m):
            cx, cy = intermediate_circle_center_choices[i][j]
            px, py = corner_point_choices[i][j]
            td = turn_direction_choices[i][j]

            circle = IntermediateCircle(
                center=Point(cx, cy),
                radius=radius,
                corner_point=Point(px, py),
                turn_direction=td,
                index=i,
                s_max=s_max_circles[i],
            )

            candidates[j] = circle

        choices[i] = IntermediateCircleChoice(
            candidates=candidates,
            index=i,
        )

    return IntermediateCircleChoicesSequence(choices)



import matplotlib.pyplot as plt


def plot_intermediate_circle_choices(ax, circle_choices_sequence):
    """
    Plot all intermediate circle candidates on a given matplotlib axis.
    """

    for i, choice in enumerate(circle_choices_sequence):
        is_ambiguous = choice.is_ambiguous

        for j, circle in enumerate(choice):
            xc, yc = circle.center.x, circle.center.y
            r = circle.radius

            # Style depending on ambiguity
            if is_ambiguous:
                color = "orange"
                linestyle = "--"
                alpha = 0.7
            else:
                color = "blue"
                linestyle = "-"
                alpha = 0.8

            # Draw circle
            circ = plt.Circle(
                (xc, yc),
                r,
                fill=False,
                color=color,
                linestyle=linestyle,
                alpha=alpha,
            )
            ax.add_patch(circ)

            # Draw center
            ax.plot(xc, yc, "o", color=color, alpha=alpha)

            # Optional: label (very useful for debugging)
            ax.text(
                xc,
                yc,
                f"{i}:{j}",
                fontsize=8,
                color=color,
            )