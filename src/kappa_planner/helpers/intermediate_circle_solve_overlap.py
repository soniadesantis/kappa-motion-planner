from ..geometry import Point, IntermediateCircle, IntermediateCirclesSequence, IntermediateCircleChoice, IntermediateCircleChoicesSequence, Circle
from ..corridor import CorridorWorld
from ..vehicle import Unicycle, Bicycle
from .intersections import compute_intersection_two_segments, compute_line_corridor_intersections
from .corridor_geometry import get_corner_point_and_intersecting_edges
from .intermediate_circles_geometry import compute_center_coordinates_second_circle_according_to_edges, compute_center_coordinates_second_circle_given_two_points
from .inputs_check import compute_min_width_s_max_corridor_pair
from .plot_helpers import plot_corridors
from .geometry_operations import (
    select_tangency_point_from_point_circle,
    compute_turn_direction,
    compute_turn_direction_from_three_points,
    efficient_sign,
    compute_distance_two_points
)
from math import sqrt, atan2, cos, sin, asin, tau
from matplotlib import pyplot as plt
import numpy as np


import numpy as np

import numpy as np
import matplotlib.pyplot as plt




def select_preferred_circle(choice: IntermediateCircleChoice) -> IntermediateCircle:
    """
    Select the active circle from one IntermediateCircleChoice.

    If the choice is ambiguous, select the candidate whose turn_direction
    matches choice.preferred_turn_direction.
    Otherwise, return the only candidate.
    """

    if len(choice) == 0:
        raise ValueError("IntermediateCircleChoice has no candidates")

    if not choice.is_ambiguous:
        return choice.first

    if choice.preferred_turn_direction is None:
        raise ValueError(
            f"Ambiguous choice at index {choice.index} has no preferred_turn_direction"
        )

    matching_candidates = [
        circle for circle in choice
        if circle.turn_direction == choice.preferred_turn_direction
    ]

    if len(matching_candidates) != 1:
        raise ValueError(
            f"Expected exactly one candidate with turn_direction "
            f"{choice.preferred_turn_direction}, got {len(matching_candidates)}"
        )

    return matching_candidates[0]


def intermediate_circle_center_at_s(circle, s):
    """
    Return the center Point of an IntermediateCircle at parameter s,
    without mutating the circle.
    """

    if s < 0 or s > circle.s_max:
        raise ValueError(f"s must be in [0, {circle.s_max}]")

    dx = cos(circle.bisector_direction)
    dy = sin(circle.bisector_direction)

    return Point(
        circle.canonical_center.x + s * dx,
        circle.canonical_center.y + s * dy,
    )


def circles_overlap_from_centers(center1, radius1, center2, radius2, tol=1e-9):
    """
    Return True if two circles overlap or touch.
    """

    distance = compute_distance_two_points(center1, center2)

    return distance <= radius1 + radius2 + tol


def intermediate_circles_overlap_at_s(circle1, s1, circle2, s2, tol=1e-9):
    """
    Check overlap between two IntermediateCircle objects
    at arbitrary shift parameters s1 and s2.
    """

    center1 = intermediate_circle_center_at_s(circle1, s1)
    center2 = intermediate_circle_center_at_s(circle2, s2)

    return circles_overlap_from_centers(
        center1,
        circle1.radius,
        center2,
        circle2.radius,
        tol=tol,
    )


def overlap_status_for_intermediate_circles(circle1, circle2, tol=1e-9):
    """
    Check overlap between two intermediate circles in:
    - nominal configuration (s = 0)
    - maximally shifted configuration (s = s_max)
    """

    nominal_overlap = intermediate_circles_overlap_at_s(
        circle1, 0,
        circle2, 0,
        tol=tol,
    )

    max_shift_overlap = intermediate_circles_overlap_at_s(
        circle1, circle1.s_max,
        circle2, circle2.s_max,
        tol=tol,
    )

    return {
        "nominal_overlap": nominal_overlap,
        "max_shift_overlap": max_shift_overlap,
    }



def compute_circle_center_through_two_points_with_radius(
    point1,
    point2,
    radius,
    turn_direction,
    tol=1e-9,
):
    """
    Compute the center of a circle with given radius passing through two points.

    turn_direction:
        +1 selects the center on the left side of point1 -> point2
        -1 selects the center on the right side of point1 -> point2
    """

    if turn_direction not in (-1, 1):
        raise ValueError("turn_direction must be -1 or +1")

    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y

    dx = x2 - x1
    dy = y2 - y1

    d = compute_distance_two_points(point1, point2)

    if d < tol:
        raise ValueError("The two points are coincident; infinitely many circles exist.")

    if d > 2 * radius + tol:
        raise ValueError(
            "No circle with the given radius can pass through both points."
        )

    # Midpoint of the chord
    mx = 0.5 * (x1 + x2)
    my = 0.5 * (y1 + y2)

    # Distance from midpoint to circle center
    half_chord = 0.5 * d
    h_sq = radius**2 - half_chord**2

    # Numerical safety
    if h_sq < 0 and abs(h_sq) < tol:
        h_sq = 0.0

    h = sqrt(h_sq)

    # Unit perpendicular to point1 -> point2
    ux = -dy / d
    uy = dx / d

    if turn_direction == 1:
        cx = mx + h * ux
        cy = my + h * uy
    else:
        cx = mx - h * ux
        cy = my - h * uy

    return Point(cx, cy)


def compute_circle_through_two_points_with_radius(
    point1,
    point2,
    radius,
    turn_direction,
):
    center = compute_circle_center_through_two_points_with_radius(
        point1,
        point2,
        radius,
        turn_direction,
    )

    return Circle(center=center, radius=radius)