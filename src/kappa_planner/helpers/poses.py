"""
Pose transformation utilities.

This module contains functions to convert poses between different
reference frames (e.g., corridor frame and world frame).
"""

from math import sin, cos, pi


def compute_axis_aligned_boundary_pose(corridor, neighbor, bicycle, initial=True):
    """Place a disk-safe endpoint at the corridor end furthest from its overlap.

    The overlap determines the heading, including corridors whose stored tilt
    points opposite to traversal. Ambiguous symmetric geometry requires an
    explicit pose. Placement alone does not guarantee a feasible connection.
    """
    import numpy as np
    from .baseline_construction import (
        compute_overlap_two_axis_aligned_corridors, compute_safe_overlap,
        compute_boundary_corridor_direction,
    )

    overlap = compute_overlap_two_axis_aligned_corridors(corridor, neighbor)
    safe_overlap = None if overlap is None else compute_safe_overlap(overlap, bicycle.width / 2)
    if safe_overlap is None:
        raise ValueError("Cannot place a default pose without a nonempty safe corridor overlap.")
    direction = compute_boundary_corridor_direction(
        corridor, safe_overlap, corridor.center, initial=initial
    )
    if direction is None:
        raise ValueError("Ambiguous endpoint traversal direction; provide an explicit boundary pose.")
    direction = np.round(direction)
    axis = int(np.argmax(np.abs(direction)))
    corners = np.asarray(corridor.corners)
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    radius = bicycle.width / 2
    # A small additional inset avoids placing the disk exactly on the wall.
    inset = radius + min(radius, 0.05 * (upper[axis] - lower[axis] - 2 * radius))
    outer_sign = -direction[axis] if initial else direction[axis]
    position = np.asarray(corridor.center, dtype=float).copy()
    position[axis] = upper[axis] - inset if outer_sign > 0 else lower[axis] + inset
    return [float(position[0]), float(position[1]), float(np.arctan2(direction[1], direction[0]))]


def relative_to_absolute_pose(corridor, relative_pose):
    '''
    Convert a pose expressed in the corridor frame to an absolute pose.

    The corridor frame is centered at the corridor center, with:
    - x axis along the corridor width,
    - y axis along the corridor length.

    :param corridor: considered corridor
    :type corridor: CorridorWorld
    :param relative_pose: relative pose (x, y, theta), where
                          x in [-width/2, width/2],
                          y in [-height/2, height/2],
                          theta in [0, 2*pi]
    :type relative_pose: list of floats or np.ndarray

    :return: absolute pose (x, y, theta)
    :rtype: list of floats
    '''
    rot_angle = corridor.tilt - pi * 0.5  # inverse of absolute_to_relative rotation

    cos_a = cos(rot_angle)
    sin_a = sin(rot_angle)

    rel_x, rel_y, rel_theta = relative_pose

    x = rel_x * cos_a - rel_y * sin_a + corridor.center[0]
    y = rel_x * sin_a + rel_y * cos_a + corridor.center[1]
    theta = rel_theta + rot_angle

    return [x, y, theta]


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


def compute_end_pose(corridor, vehicle, margin):
    '''
    Compute the goal pose within a given corridor for a unicycle vehicle.

    The x and y coordinates of the goal pose lie on the centerline of the
    corridor. They are computed starting from the midpoint of the forward
    face and shifting backward along the corridor by half the vehicle
    length (to avoid collision) plus an additional margin.

    The goal orientation is equal to the corridor tilt.

    :param corridor: corridor containing the goal position
    :type corridor: CorridorWorld
    :param vehicle: vehicle used to retrieve its length
    :type vehicle: Vehicle
    :param margin: additional backward shift along the centerline
    :type margin: float

    :return: end pose (x, y, theta)
    :rtype: list of three floats
    '''
    offset = 0.5 * corridor.height - 0.5 * vehicle.length - margin

    x = corridor.center[0] + offset * cos(corridor.tilt)
    y = corridor.center[1] + offset * sin(corridor.tilt)
    theta = corridor.tilt

    return [x, y, theta]


def compute_start_pose(corridor, vehicle, margin):
    '''
    Compute the initial pose within a given corridor for a unicycle vehicle.

    The x and y coordinates of the initial pose lie on the centerline of
    the corridor. They are computed starting from the midpoint of the
    backward face and shifting forward along the corridor by half the
    vehicle length (to avoid collision) plus an additional margin.

    The orientation is equal to the corridor tilt.

    :param corridor: corridor containing the initial position
    :type corridor: CorridorWorld
    :param vehicle: vehicle used to retrieve its length
    :type vehicle: Vehicle
    :param margin: additional shift along the centerline
    :type margin: float

    :return: initial pose (x, y, theta)
    :rtype: list of three floats
    '''
    offset = 0.5 * corridor.height - 0.5 * vehicle.length - margin

    x = corridor.center[0] - offset * cos(corridor.tilt)
    y = corridor.center[1] - offset * sin(corridor.tilt)
    theta = corridor.tilt

    return [x, y, theta]


def pose_from_shrunken_corridor_relative_frame(relative_pose_fraction, shrunken_corridor):
    '''
    Convert a normalized pose inside a shrunken corridor to an absolute pose.
    Used when the user provides relative poses to the MotionPlanner.

    The input pose is expressed in a normalized corridor frame:
    - x in [-1, 1] corresponds to the corridor width
    - y in [-1, 1] corresponds to the corridor height
    - theta is kept unchanged

    The function rescales the position to the actual corridor dimensions
    and then converts it to the global (absolute) frame.

    :param relative_pose_fraction: normalized pose (x, y, theta)
    :type relative_pose_fraction: list or numpy.ndarray
    :param shrunken_corridor: corridor with reduced dimensions
    :type shrunken_corridor: CorridorWorld

    :return: absolute pose (x, y, theta)
    :rtype: list of floats
    '''
    rel_x = relative_pose_fraction[0] * shrunken_corridor.width * 0.5
    rel_y = relative_pose_fraction[1] * shrunken_corridor.height * 0.5
    rel_theta = relative_pose_fraction[2]

    relative_pose = [rel_x, rel_y, rel_theta]

    return relative_to_absolute_pose(shrunken_corridor, relative_pose)
