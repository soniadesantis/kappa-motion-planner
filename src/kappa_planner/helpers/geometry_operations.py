from math import ceil, floor, pi, asin, atan2, sqrt, sin, cos
from ..geometry import Point


def compute_distance_two_points(point1, point2):
    """Get the Euclidean distance from point1 to point2"""
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def wrapPositiveAngle(angle):
    """
    Wraps any angle (in radians) as a positive angle
    between 0 and 2*pi.

    :param angle: Angle in radians
    :type angle: float

    :return: Wrapped angle in the range [0, 2*pi]
    :rtype: float
    """
    return angle % (2*pi)


def efficient_sign(number):
    """
    Computes the sign of a number efficiently.

    :param number: Number from which to extract the sign
    :type angle: float

    :return: Indication of the sign of a number.
    :rtype: int
    """
    return ceil(number/abs(number)) if number > 0 else floor(number/abs(number)) if number < 0 else 0


def compute_turn_direction(vector1, vector2):
    '''
    Compute the cross product between vector1 and vector2.
    Since vector1 and vector2 lie on the x-y plane, the third element of both vectors is zero.
    Their cross product lies on the plane perpendicular to the x-y plane, therefore
    only the third element is different than zero.
    vector1 = [x1, y1, z1]
    vector2 = [x2, y2, z2]
    vector1 x vector2 = [0 0 x1*y2-x2y1]
    When the two vectors are aligned, their cross product is zero.

    :param vector1: vector 1 in global frame
    :type vector1: numpy.ndarray
    :param vector2: vector 2 in global frame
    :type vector2: numpy.ndarray

    :return: turn_direction, float number larger the zero = turn left, 
            lower than 0 = turn right, equal to 0 = go straight
    :rtype: float64
    '''
    turn = vector1[0]*vector2[1] - vector2[0]*vector1[1]
    if abs(turn) < 1e-5:
        return 0
    else:
        return efficient_sign(turn)
    

def compute_turn_direction_from_three_points(point_prev, point_curr, point_next):
    """
    Compute the turn direction defined by three consecutive points.

    The direction is computed from the vectors:
    - point_prev -> point_curr
    - point_curr -> point_next

    :param point_prev: previous point
    :type point_prev: Point
    :param point_curr: current point
    :type point_curr: Point
    :param point_next: next point
    :type point_next: Point

    :return: turn direction (1 = left, -1 = right, 0 = straight)
    :rtype: int
    """
    vector1 = [
        point_curr.x - point_prev.x,
        point_curr.y - point_prev.y,
    ]

    vector2 = [
        point_next.x - point_curr.x,
        point_next.y - point_curr.y,
    ]

    return compute_turn_direction(vector1, vector2)


def select_tangency_point_from_point_circle(
    start_point,
    circle,
    turn_direction=None,
    tol=1e-9,
):
    if turn_direction is None:
        turn_direction = circle.turn_direction

    hyp = compute_distance_two_points(
        (start_point.x, start_point.y),
        (circle.center.x, circle.center.y),
    )

    if hyp < circle.radius - tol:
        raise ValueError(
            "The start point is inside the circle; no real tangent exists."
        )

    if abs(hyp - circle.radius) <= tol:
        return Point(start_point.x, start_point.y)

    length_tangent = sqrt(hyp * hyp - circle.radius * circle.radius)

    input_to_asin = circle.radius / hyp
    input_to_asin = max(-1.0, min(1.0, input_to_asin))

    beta_angle = asin(input_to_asin)

    angle_to_center = atan2(
        circle.center.y - start_point.y,
        circle.center.x - start_point.x,
    )

    tangent_angle = angle_to_center - turn_direction * beta_angle

    return Point(
        start_point.x + length_tangent * cos(tangent_angle),
        start_point.y + length_tangent * sin(tangent_angle),
    )


def check_point_inside_segment(A, B, P, tol=1e-9):
    x1, y1 = A
    x2, y2 = B
    px, py = P

    # Bounding box check
    if not (
        min(x1, x2) - tol <= px <= max(x1, x2) + tol and
        min(y1, y2) - tol <= py <= max(y1, y2) + tol
    ):
        return False

    # Collinearity (scaled tolerance)
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    length = (x2 - x1)**2 + (y2 - y1)**2

    return abs(cross) <= tol * max(1.0, length)


def compute_angular_difference(theta1, theta2):
    """Get the signed angle from theta1 to theta2"""
    return atan2(sin(theta2 - theta1), cos(theta2 - theta1))


def compute_angular_difference_with_turn_direction(theta1, theta2, turn):
    """Get the signed angle from theta1 to theta2, according to a specified turn direction"""
    delta_angle = atan2(sin(theta2 - theta1), cos(theta2 - theta1))
    if abs(delta_angle) < 1e-5:
        return 0.0
    elif efficient_sign(delta_angle) != turn:
        return turn * (2 * pi - abs(delta_angle))
    else: 
        return delta_angle