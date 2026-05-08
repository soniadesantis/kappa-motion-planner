"""
Input transformation utilities.

This module contains helper functions to invert inputs of planning
functions, typically used to exploit symmetry in motion planning problems.
"""

from math import pi

import numpy as np

from ..corridor import CorridorWorld
from ..geometry import Pose, Point, IntermediateCircle
from .geometry_operations import wrapPositiveAngle


def invert_inputs(
    corridor1,
    corridor2,
    start_pose,
    unicycle,
    xc2,
    yc2,
    turn2,
    t0=0,
    turn1=0,
):
    """
    Invert the inputs to the function compute_three_maneuvers_compact.

    :param corridor1: first corridor
    :type corridor1: Corridor
    :param corridor2: second corridor
    :type corridor2: Corridor
    :param start_pose: initial pose
    :type start_pose: list or np.ndarray
    :param unicycle: vehicle model
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

    :return: inverted inputs for the planner
    :rtype: tuple
    """
    start_pose_inv = [
        start_pose[0],
        start_pose[1],
        wrapPositiveAngle(start_pose[2] + pi),
    ]

    corridor1_inv = CorridorWorld(
        corridor1.width,
        corridor1.height,
        corridor1.center,
        wrapPositiveAngle(corridor1.tilt + pi),
    )

    corridor2_inv = CorridorWorld(
        corridor2.width,
        corridor2.height,
        corridor2.center,
        wrapPositiveAngle(corridor2.tilt + pi),
    )

    return (
        corridor1_inv,
        corridor2_inv,
        start_pose_inv,
        unicycle,
        xc2,
        yc2,
        -turn2,
        t0,
        -turn1,
    )


def invert_inputs_compute_initial_turn_on_the_spot(
    start_pose,
    xc2,
    yc2,
    turn1,
    turn2,
    radius,
    omega_max,
    omega_min,
):
    """
    Invert the inputs to compute_initial_turn_on_the_spot.

    :param start_pose: initial pose
    :type start_pose: list or np.ndarray
    :param xc2: x coordinate of the second circle center
    :type xc2: float
    :param yc2: y coordinate of the second circle center
    :type yc2: float
    :param turn1: first turn direction
    :type turn1: float
    :param turn2: second turn direction
    :type turn2: float
    :param radius: turning radius
    :type radius: float
    :param omega_max: maximum angular velocity
    :type omega_max: float
    :param omega_min: minimum angular velocity
    :type omega_min: float

    :return: inverted inputs
    :rtype: tuple
    """
    start_pose_inv = [
        start_pose[0],
        start_pose[1],
        wrapPositiveAngle(start_pose[2] + pi),
    ]

    return (
        start_pose_inv,
        xc2,
        yc2,
        -turn1,
        -turn2,
        radius,
        omega_max,
        omega_min,
    )


def invert_inputs_compute_initial_turn_on_the_spot_collision_avoidance(
    start_pose,
    turn1,
    corridor,
    radius,
    omega_max,
    omega_min,
    margin,
):
    """
    Invert the inputs to compute_initial_turn_on_the_spot_collision_avoidance.

    :param start_pose: initial pose
    :type start_pose: list or np.ndarray
    :param turn1: turn direction
    :type turn1: float
    :param corridor: current corridor
    :type corridor: CorridorWorld
    :param radius: turning radius
    :type radius: float
    :param omega_max: maximum angular velocity
    :type omega_max: float
    :param omega_min: minimum angular velocity
    :type omega_min: float
    :param margin: safety margin
    :type margin: float

    :return: inverted inputs
    :rtype: tuple
    """
    start_pose_inv = [
        start_pose[0],
        start_pose[1],
        wrapPositiveAngle(start_pose[2] + pi),
    ]

    corridor_inv = CorridorWorld(
        corridor.width,
        corridor.height,
        corridor.center,
        wrapPositiveAngle(corridor.tilt + pi),
    )

    return (
        start_pose_inv,
        -turn1,
        corridor_inv,
        radius,
        omega_max,
        omega_min,
        margin,
    )


def invert_inputs_start_inside_second_circle_case(
    corridor1,
    corridor2,
    turn_direction,
    corner_point,
    unicycle,
):
    """
    Invert inputs for the 'start inside second circle' case.

    :return: inverted inputs
    :rtype: tuple
    """
    corridor1_inv = CorridorWorld(
        corridor1.width,
        corridor1.height,
        corridor1.center,
        wrapPositiveAngle(corridor1.tilt + pi),
    )

    corridor2_inv = CorridorWorld(
        corridor2.width,
        corridor2.height,
        corridor2.center,
        wrapPositiveAngle(corridor2.tilt + pi),
    )

    return (
        corridor1_inv,
        corridor2_inv,
        -turn_direction,
        corner_point,
        unicycle,
    )


def invert_inputs_all(*inputs):
    """
    Invert geometric inputs.
    Supported types are CorridorWorld, Pose, and IntermediateCircle.

    :param inputs: geometric objects to invert
    :type inputs: CorridorWorld | Pose | IntermediateCircle

    :return: inverted objects in the same order as inputs
    :rtype: list
    """
    inverted_inputs = []

    for item in inputs:
        if isinstance(item, CorridorWorld):
            inverted_inputs.append(item.rotate_corridor(pi))

        elif isinstance(item, Pose):
            inverted_inputs.append(
                Pose(
                    position=Point(item.x, item.y),
                    theta=wrapPositiveAngle(item.theta + pi),
                )
            )

        elif isinstance(item, IntermediateCircle):
            inverted_inputs.append(
                IntermediateCircle(
                    center=Point(
                        x=item.center.x,
                        y=item.center.y,
                    ),
                    radius=item.radius,
                    corner_point=Point(
                        x=item.corner_point.x,
                        y=item.corner_point.y,
                    ),
                    turn_direction=-item.turn_direction,
                )
            )

        elif isinstance(item, (list, tuple, np.ndarray)) and np.size(item) == 3:
            x, y, theta = item
            inverted_inputs.append(
                np.array([x, y, wrapPositiveAngle(theta + pi)])
            )

        else:
            raise TypeError(
                "invert_inputs: unsupported input type "
                f"{type(item).__name__}"
            )

    return inverted_inputs