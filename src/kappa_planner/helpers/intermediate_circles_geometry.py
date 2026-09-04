from math import atan2, cos, pi, sin, sqrt

from ..geometry import Circle, IntermediateCircle, IntermediateCirclesSequence, Point
from .corridor_geometry import check_point_inside_corridor, get_corner_point
from .geometry_operations import (
    check_point_inside_segment,
    compute_angular_difference,
    compute_distance_two_points,
)


def compute_center_coordinates_vector(corridor_list, turn_direction_vector, corner_point_vector, unicycle, margin = 0):
    center_coordinates_vector = [0] * (len(corner_point_vector))
    for i in range(len(corner_point_vector)):
        xc, yc = compute_center_coordinates_second_circle(corner_point_vector[i], turn_direction_vector[i], unicycle.max_radius, unicycle.width, margin, corridor_list[i].tilt, corridor_list[i+1].tilt, corridor_list[i], corridor_list[i+1])
        center_coordinates_vector[i] = [xc, yc]

    return center_coordinates_vector


def compute_center_coordinates_vector_according_to_edges(corridor_list, turn_direction_vector, corner_point_vector, intersecting_edges, vehicle, margin = 0): 
    center_coordinates_vector = [0] * (len(corner_point_vector))
    for i in range(len(corner_point_vector)):
        if corner_point_vector[i] is None:
            center_coordinates_vector[i] = None
            continue
        xc, yc = compute_center_coordinates_second_circle_according_to_edges(
            corner_point_vector[i],
            turn_direction_vector[i],
            vehicle.max_radius,
            vehicle.width, margin,
            intersecting_edges[i],
            corridor1 = corridor_list[i],
            corridor2 = corridor_list[i+1])
        center_coordinates_vector[i] = [xc, yc]
    return center_coordinates_vector


def start_inside_second_circle_case(corridor1, corridor2, turn, corner_point, unicycle):
    corridor1_shrunken = corridor1.shrink(unicycle.width * 0.5)
    corridor2_shrunken = corridor2.shrink(unicycle.width * 0.5)
    corners2 = corridor2.get_corners()
    corners2_shrunken = corridor2_shrunken.get_corners()
    corner_point_shrunken = get_corner_point(corridor1_shrunken, corridor2_shrunken, turn)
    xi, yi = corner_point_shrunken
    r = unicycle.max_radius
    
    if turn == 1: # turn left
        # Check left face
        if check_point_inside_segment(corners2[2], corners2[3], corner_point):
            x1, y1, x2, y2 = corners2[2][0], corners2[2][1], corners2[3][0], corners2[3][1]
        else:
            x1, y1, x2, y2 = corners2[1][0], corners2[1][1], corners2[2][0], corners2[2][1]
    else:
        # Check right face
        if check_point_inside_segment(corners2[0], corners2[1], corner_point):
            x1, y1, x2, y2 = corners2_shrunken[0][0], corners2_shrunken[0][1], corners2_shrunken[1][0], corners2_shrunken[1][1]
        else:
            x1, y1, x2, y2 = corners2_shrunken[1][0], corners2_shrunken[1][1], corners2_shrunken[2][0], corners2_shrunken[2][1]

    if x1 == x2: # Vertical line
        d = x1
        a = 1
        b = - 2 * yi
        c = xi**2 + d**2 + yi**2 - 2 * xi * d - r**2
        yc1 = (-b + sqrt(b**2 - 4 * a * c))/(2 * a)
        xc1, xc2 = d, d
        yc2 = (-b - sqrt(b**2 - 4 * a * c))/(2 * a)
    elif y1 == y2: # Horizontal line
        d = y1
        a = 1
        b = - 2 * xi
        c = xi**2 + yi**2 + d**2 - 2 * yi * d - r**2

        xc1 = (-b + sqrt(b**2 - 4 * a * c))/(2 * a)
        yc1, yc2 = d, d
        xc2 = (-b - sqrt(b**2 - 4 * a * c))/(2 * a)
    else:
        m = (y2 - y1)/(x2 - x1)
        d = y1 - m * x1
        a = 1 + m**2
        b = 2 * m * d -2 * yi * m - 2 * xi 
        c = xi**2 + yi**2 + d**2 - 2*yi*d - r**2

        xc1 = (-b + sqrt(b**2 - 4 * a * c))/(2 * a)
        yc1 = m * xc1 + d
        xc2 = (-b - sqrt(b**2 - 4 * a * c))/(2 * a)
        yc2 = m * xc2 + d
    
    if check_point_inside_corridor(corridor1_shrunken, [xc1, yc1]):
        return xc2, yc2
    else:
        return xc1, yc1
    

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

    xc2, yc2 = compute_center_coordinates_second_circle(
        corner_point,
        turn_direction,
        unicycle.max_radius,
        unicycle.width,
        0,
        corridor1.tilt,
        corridor2.tilt
        )

    return xc2, yc2


def compute_center_coordinates_second_circle(corner_point, turn, R, vehicle_width, margin, tilt1, tilt2, corridor1 = None, corridor2 = None):
    '''
    Compute the coordinates of the center of a circumference placed at the intersection between two subsequent corridors (intermediate circumference).
    The center is placed at a distance equal to (R - half of vehicle_width - margin) along the bisector between two edges of the two corridors.
    The selected edges depend on the main turn direction between the two corridors. 

    :param corner_point: selected corner point at the intersection between two corridors
    :type corner_point: list of floats or np.ndarray
    :param turn: turn direction along the circumference to be reached
    :type turn: float [-1, 1]
    :param R: radius of the circumference to be reached == radius of arc maneuver of the considered vehicle
    :type R: float
    :param vehicle_width: width of the vehicle
    :type vehicle_width: float
    :param margin: additional margin to avoid collision with the walls of the corridors
    :type margin: float
    :param tilt1: tilt of first corridor
    :type tilt1: float
    :param tilt2: tilt of second corridor
    :type tilt2: float

    :return: x coordinate of the center of the circumference to be reached
    :rtype: float
    :return: y coordinate of the center of the circumference to be reached
    :rtype: float
    '''
    if tilt1 == tilt2 and corridor1 is not None and corridor2 is not None:
        tilt1 = tilt2 - turn * 0.5 * pi
        if turn == 1: 
            if check_point_inside_segment(corridor1.corners[2], corridor1.corners[3], corner_point) and check_point_inside_segment(corridor2.corners[1], corridor2.corners[2], corner_point):
                angle_circle_center_direction = tilt2 + turn * ( pi  * 0.5 +  0.5 * (pi - abs(compute_angular_difference(tilt2, tilt1))))
            else: 
                angle_circle_center_direction = tilt2 + turn * 0.5 * (pi - abs(compute_angular_difference(tilt2, tilt1)))

        elif turn == -1:
            if check_point_inside_segment(corridor1.corners[0], corridor1.corners[1], corner_point) and check_point_inside_segment(corridor2.corners[1], corridor2.corners[2], corner_point):
                angle_circle_center_direction = tilt2 + turn * ( pi  * 0.5 +  0.5 * (pi - abs(compute_angular_difference(tilt2, tilt1))))
            else:
                angle_circle_center_direction = tilt2 + turn * 0.5 * (pi - abs(compute_angular_difference(tilt2, tilt1)))
    
    elif tilt1 == tilt2: 
        # tilt2 = tilt1 + turn * 0.5 * pi
        tilt1 = tilt2 - turn * 0.5 * pi
        angle_circle_center_direction = tilt2 + turn * 0.5 * (pi - abs(compute_angular_difference(tilt2, tilt1)))
    
    else:
        angle_circle_center_direction = tilt2 + turn * 0.5 * (pi - abs(compute_angular_difference(tilt2, tilt1)))


    xc2 = corner_point[0] + (R - 0.5 * vehicle_width - margin) * cos(angle_circle_center_direction)
    yc2 = corner_point[1] + (R - 0.5 * vehicle_width - margin) * sin(angle_circle_center_direction)
    return xc2, yc2


def create_intermediate_circles_sequence(
    intermediate_circle_centers,
    turn_direction_vector,
    corner_point_vector,
    radius,
    s_max_circles,
):
    """
    Create the sequence of intermediate circles used in the trajectory algorithm.
    Each intermediate circle is associated with a corner point and a turn direction.

    :param intermediate_circle_centers: centers of the intermediate circles
    :type intermediate_circle_centers: list of list of floats
    :param turn_direction_vector: turn directions of the intermediate circles
    :type turn_direction_vector: list of int
    :param corner_point_vector: corner points associated with the intermediate circles
    :type corner_point_vector: list of list of floats
    :param radius: radius of the intermediate circles
    :type radius: float

    :return: ordered sequence of intermediate circles
    :rtype: IntermediateCirclesSequence
    """

    n = len(intermediate_circle_centers)

    if not (len(turn_direction_vector) == len(corner_point_vector) == n):
        raise ValueError("Impossible to define IntermediateCirclesSequence:" \
        "all input vectors must have the same length.")

    items = []

    for i in range(n):
        cx, cy = intermediate_circle_centers[i]
        px, py = corner_point_vector[i]
        td = turn_direction_vector[i]

        circle = IntermediateCircle(
            center=Point(cx, cy),
            radius=radius,
            corner_point=Point(px, py),
            turn_direction=td,
            index = i,
            s_max = s_max_circles[i],
        )

        items.append(circle)

    return IntermediateCirclesSequence(items)


def compute_center_coordinates_second_circle_according_to_edges(
    corner_point,
    turn,
    R,
    vehicle_width,
    margin,
    intersecting_edges,
    corridor1,
    corridor2,
):
    '''
    Compute the coordinates of the center of a circumference placed at the
    intersection between two subsequent corridors (intermediate circumference).

    The center is placed at a distance equal to
    (R - half of vehicle_width - margin) along the bisector between two edges
    of the two corridors.

    The selected edges depend on the main turn direction between the two corridors.
    '''
    tilt1 = corridor1.tilt
    tilt2 = corridor2.tilt
    offset = R - 0.5 * vehicle_width - margin

    if turn == 1:

        side_side = (
            intersecting_edges[0] == 3 and
            intersecting_edges[1] == 3
        )
        side_back = (
            intersecting_edges[0] == 3 and
            intersecting_edges[1] == 2
        )
        front_side = (
            intersecting_edges[0] == 0 and
            intersecting_edges[1] == 3
        )

        if side_side:
            # Both corridors' left edges are intersecting
            angle_circle_center_direction = (
                tilt2
                + turn * 0.5
                * (pi - abs(compute_angular_difference(tilt2, tilt1)))
            )

        elif side_back:
            # Corridor1's left edge and corridor2's back edge are intersecting
            corridor_corner = corridor2.get_corners()[2]

            tilt2 = atan2(
                corridor_corner[1] - corner_point[1],
                corridor_corner[0] - corner_point[0],
            )

            angle = 0.5 * abs(
                compute_angular_difference(tilt2, tilt1 + pi)
            )

            angle_circle_center_direction = tilt2 + turn * angle

        elif front_side:
            # Corridor1's front edge and corridor2's left edge are intersecting
            corridor_corner = corridor1.get_corners()[3]

            tilt1 = atan2(
                corridor_corner[1] - corner_point[1],
                corridor_corner[0] - corner_point[0],
            )

            angle = 0.5 * abs(
                compute_angular_difference(tilt2, tilt1)
            )

            angle_circle_center_direction = tilt2 + turn * angle

        else:
            raise ValueError(
                "Invalid edge combination for left turn: "
                f"intersecting_edges={intersecting_edges!r}, "
                "impossible to compute center of intermediate circle"
            )

    elif turn == -1:

        side_side = (
            intersecting_edges[0] == 1 and
            intersecting_edges[1] == 1
        )
        side_back = (
            intersecting_edges[0] == 1 and
            intersecting_edges[1] == 2
        )
        front_side = (
            intersecting_edges[0] == 0 and
            intersecting_edges[1] == 1
        )

        if side_side:
            # Both corridors' right edges are intersecting
            angle_circle_center_direction = (
                tilt2
                + turn * 0.5
                * (pi - abs(compute_angular_difference(tilt2, tilt1)))
            )

        elif side_back:
            # Corridor1's right edge and corridor2's back edge are intersecting
            corridor_corner = corridor2.get_corners()[1]

            tilt2 = atan2(
                corridor_corner[1] - corner_point[1],
                corridor_corner[0] - corner_point[0],
            )

            angle = 0.5 * abs(
                compute_angular_difference(tilt2, tilt1 + pi)
            )

            angle_circle_center_direction = tilt2 + turn * angle

        elif front_side:
            # Corridor1's front edge and corridor2's right edge are intersecting
            corridor_corner = corridor1.get_corners()[0]

            tilt1 = atan2(
                corridor_corner[1] - corner_point[1],
                corridor_corner[0] - corner_point[0],
            )

            angle = 0.5 * abs(
                compute_angular_difference(tilt2, tilt1)
            )

            angle_circle_center_direction = tilt2 + turn * angle

        else:
            raise ValueError(
                "Invalid edge combination for right turn: "
                f"intersecting_edges={intersecting_edges!r}, "
                "impossible to compute center of intermediate circle"
            )

    else:
        raise ValueError(
            f"Invalid turn direction {turn}, expected values are 1 or -1: "
            "impossible to compute center of intermediate circle"
        )

    xc2 = corner_point[0] + offset * cos(angle_circle_center_direction)
    yc2 = corner_point[1] + offset * sin(angle_circle_center_direction)

    return xc2, yc2


def compute_center_coordinates_second_circle_given_two_points(
    corner_point,
    int_point,
    tau,
    R,
    vehicle_width,
    margin=0,
):
    """
    Compute the center of the second circle given a corner point and an
    intersection point.

    The center is placed along the direction from the intersection point
    to the corner point, at a distance defined by the vehicle footprint
    and margin.

    :param corner_point: corner point between corridors [x, y]
    :param int_point: intersection point on corridor edge [x, y]
    :param tau: turn direction (unused here, kept for interface consistency)
    :param R: circle radius
    :param vehicle_width: vehicle width
    :param margin: safety margin
    :return: (xc2, yc2) center of the second circle
    """

    offset = R - 0.5 * vehicle_width - margin

    angle_circle_center_direction = atan2(
        corner_point[1] - int_point[1],
        corner_point[0] - int_point[0],
    )

    xc2 = corner_point[0] + offset * cos(angle_circle_center_direction)
    yc2 = corner_point[1] + offset * sin(angle_circle_center_direction)

    return xc2, yc2


def compute_circle_center_internally_tangent_to_two_circles(
    small_circle1,
    small_circle2,
    radius,
    turn_direction,
    tol=1e-9,
):
    """
    Compute the center of a circle of radius `radius`
    internally tangent to two smaller circles.

    The smaller circles must have equal radius.

    turn_direction:
        +1 selects the center on the left side of
            circle1.center -> circle2.center
        -1 selects the center on the right side
    """

    if turn_direction not in (-1, 1):
        raise ValueError("turn_direction must be -1 or +1")

    r1 = small_circle1.radius
    r2 = small_circle2.radius

    if abs(r1 - r2) > tol:
        raise ValueError(
            "The two small circles must have equal radius."
        )

    r = r1
    R = radius

    if R <= r:
        raise ValueError(
            "The large circle radius must be larger than the small circle radius."
        )

    c1 = small_circle1.center
    c2 = small_circle2.center

    x1, y1 = c1.x, c1.y
    x2, y2 = c2.x, c2.y

    dx = x2 - x1
    dy = y2 - y1

    d = compute_distance_two_points(c1, c2)

    if d < tol:
        raise ValueError(
            "The two small circles are concentric."
        )

    effective_radius = R - r

    if d > 2 * effective_radius + tol:
        raise ValueError(
            "No internally tangent circle with the given radius exists."
        )

    # Midpoint between small-circle centers
    mx = 0.5 * (x1 + x2)
    my = 0.5 * (y1 + y2)

    # Distance from midpoint to big-circle center
    half_chord = 0.5 * d

    h_sq = effective_radius**2 - half_chord**2

    # Numerical safety
    if h_sq < 0 and abs(h_sq) < tol:
        h_sq = 0.0

    h = sqrt(h_sq)

    # Perpendicular unit vector
    ux = -dy / d
    uy = dx / d

    if turn_direction == 1:
        cx = mx + h * ux
        cy = my + h * uy
    else:
        cx = mx - h * ux
        cy = my - h * uy

    return Point(cx, cy)


def compute_circle_internally_tangent_to_two_circles(
    small_circle1,
    small_circle2,
    radius,
    turn_direction,
):
    center = compute_circle_center_internally_tangent_to_two_circles(
        small_circle1,
        small_circle2,
        radius,
        turn_direction,
    )

    return Circle(
        center=center,
        radius=radius,
    )