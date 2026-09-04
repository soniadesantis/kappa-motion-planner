"""
Geometric intersection utilities.

This module provides functions to compute intersections between
geometric primitives such as lines, segments, and circles.
"""

from math import cos, inf, sin, sqrt

from ..corridor import CorridorWorld


def get_intersection(w1, w2):
    '''
    Use Cramer's rule to solve the system of equations
    ``w1.T @ ph = 0 ; w2.T @ ph = 0;`` for p, where ph = [p; 1]

    :param np.array w1: line parameter vector of shape [w0, w1, w2]
    :param np.array w2: line parameter vector of shape [w0, w1, w2]

    :return: intersection of w1 and w2
    :rtype: np.array
    '''
    tol = 1e-3
    denominator = (w1[0]*w2[1] - w1[1]*w2[0])
    if (abs(denominator) < tol):
        if (abs(w2[0]) < tol and abs(w2[2]) > tol) or (abs(w2[0]) > tol and abs(w2[2]) < tol):
            return inf
        elif (abs(w2[0]) > tol and abs(w2[2]) > tol):
            if (abs(w1[0]/w2[0] - w1[2]/w2[2]) < tol):
                return inf
            else: 
                return []
    else:
        return [(w1[1]*w2[2] - w1[2]*w2[1])/denominator,
                (w1[2]*w2[0] - w1[0]*w2[2])/denominator]


def circle_intersection(xc1, yc1, r1, xc2, yc2, r2):

    dist = sqrt((xc2 - xc1)**2 + (yc2 - yc1)**2) 
    tol = 1e-3
    # If the two circles do not intersect because they are too far or one inside the other
    # Too far apart
    if dist > r1 + r2 + tol:
        print("PROBLEMA (no intersection – too far apart)")
        return None

    # One circle inside the other
    elif dist < abs(r1 - r2) - tol:
        print("PROBLEMA (no intersection – one inside another)")
        return None

    # Coincident (infinite intersections)
    elif abs(dist) < tol and abs(r1 - r2) < tol:
        print("PROBLEMA (coincident circles)")
        return None
    
    else:
        a = (r1**2 - r2**2 + dist**2)/ (2*dist)
        if r1**2 - a**2 < tol:
            h = 0
        else:
            h = sqrt(r1**2 - a**2)
        x3 = xc1 + a * (xc2 - xc1) / dist
        y3 = yc1 + a * (yc2 - yc1) / dist
        x4 = x3 + h * (yc2 - yc1) / dist
        y4 = y3 - h * (xc2 - xc1) / dist
        x5 = x3 - h * (yc2 - yc1) / dist
        y5 = y3 + h * (xc2 - xc1) / dist
        return x4, y4, x5, y5
    
    
def compute_intersection_points_between_line_circle(xc, yc, radius, corridor, margin = 0):
    '''
    Compute whether a given line (walls of the given corridor) and a given circle with center (xc,yc) are intersecting. 

    :param xc: x coordinate of center of circle
    :type xc: float
    :param yc: y coordinate of center of circle
    :type yc: float
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
    corridor_with_margin = CorridorWorld(
    width=corridor.width - 2 * margin,
    height=corridor.height - 2 * margin,
    center=corridor.center,
    tilt=corridor.tilt,
    )
    corners = corridor_with_margin.get_corners()
    left_wall = False
    right_wall = False
    intersection_points_left_wall = [[None, None], [None, None]]
    intersection_points_right_wall = [[None, None], [None, None]]
    tol = 1e-5

    # Short explanation: to check whether a line intersects a circle, we need to solve the system of equations of the line and the circle
    # line : y = m * x + c  with m = (y2 - y1)/(x2 - x1) and c = y1 - m * x1
    # circle: (x - h)^2 + (y - k)^2 = r^2  with r = radius, (xc, yc) = (h,k)
    # First express the line as a function of y: x = (y - c)/m
    # Then substitute x in the circle equation: 
    # This is a quadratic equation in y: a * y^2 + b * y + c = 0
    # where a = 1/m^2 + 1, b = -2*c/m^2 - 2*h/m - 2*k, c = c^2/m^2 + 2ch/m + h^2 + k^2 - r^2
    # The discriminant is b^2 - 4*a*c 
    # If the discriminant is negative, there are no intersection points
    # If the discriminant is zero, there is one intersection point
    # If the discriminant is positive, there are two intersection points
    # The equations are modified in case of vertical corridor (x1 = x2) or horizontal corridor (y1 = y2)

    # circle center is (h,k)
    h = xc
    k = yc

    ## Intersection with left wall
    # Define the points
    x1, y1 = corners[2] # bottom left corner
    x2, y2 = corners[3] # top left corner

    ## In case the corridor has a pi/2 or 3pi/2 tilt (vertical corridor)
    if abs(x1 - x2) < tol:
        d = x1
        a = 1
        b = -2*k
        c = d**2 -2*d*h + h**2 + k**2 - radius**2
    
        discriminant = b**2 - 4*a*c

        # discriminant < 0 -> 2 complex solutions -> no intersection points
        # discriminant == 0 -> 1 real solution -> 1 intersection point
        if abs(discriminant) < tol:
            y_sol = -b/(2*a)
            x_sol = d
            intersection_points_left_wall[0] = [x_sol, y_sol]
            left_wall = True
        # discriminant > 0 -> 2 real solutions -> 2 intersection points    
        elif discriminant > tol:
            sqrt_discriminant = sqrt(discriminant)
            y_sol1 = (-b + sqrt_discriminant)/(2*a)
            x_sol1 = d
            intersection_points_left_wall[0] = [x_sol1, y_sol1]
                
            y_sol2 = (-b - sqrt_discriminant)/(2*a)
            x_sol2 = d
            intersection_points_left_wall[1] = [x_sol2, y_sol2]

            left_wall = True
    ## If the corridor has a 0 or pi tilt (horizontal corridor)
    elif abs(y1 - y2) < tol:
        # write code to compute the intersection points between a circle with radius= radius and center (h,k) and a horizontal line y = y1
        a = 1
        b = -2*h
        c = h**2 + k**2 - radius**2 + y1**2 - 2*k*y1
        discriminant = b**2 - 4*a*c

        # discriminant < 0 -> 2 complex solutions -> no intersection points
        # discriminant == 0 -> 1 real solution -> 1 intersection point
        if abs(discriminant) < tol:
            y_sol = y1
            x_sol = -b / (2*a)
            intersection_points_left_wall[0] = [x_sol, y_sol]

            left_wall = True
        # discriminant > 0 -> 2 real solutions -> 2 intersection points    
        elif discriminant > tol:
            sqrt_discriminant = sqrt(discriminant)
            y_sol1 = y1
            x_sol1 = (-b + sqrt_discriminant)/(2*a)
            intersection_points_left_wall[0] = [x_sol1, y_sol1]
            
            y_sol2 = y1
            x_sol2 = (-b - sqrt_discriminant)/(2*a)
            intersection_points_left_wall[1] = [x_sol2, y_sol2]

            left_wall = True

    ## If the corridor is neither horizontal nor vertical
    else:
        # y = m * x + d
        m = (y2 - y1)/(x2 - x1) if x1 != x2 else 0
        d = y1 - m * x1 if m != 0 else 0  
            
        # Solve systems of equations
        a = (1/m**2 + 1)
        b = -(2*d)/m**2 - (2*h)/m - 2 * k 
        c = d**2/m**2 + (2*d*h)/m + h**2 + k**2 - radius**2
        discriminant = b**2 - 4*a*c

        # discriminant < 0 -> 2 complex solutions -> no intersection points
        # discriminant == 0 -> 1 real solution -> 1 intersection point
        if abs(discriminant) < tol:
            y_sol = -b/(2*a)
            x_sol = (y_sol-d)/m
            intersection_points_left_wall[0] = [x_sol, y_sol]

            left_wall = True
        # discriminant > 0 -> 2 real solutions -> 2 intersection points    
        elif discriminant > tol:
            sqrt_discriminant = sqrt(discriminant)
            y_sol1 = (-b + sqrt_discriminant)/(2*a)
            x_sol1 = (y_sol1-d)/m
            intersection_points_left_wall[0] = [x_sol1, y_sol1]
            
            y_sol2 = (-b - sqrt_discriminant)/(2*a)
            x_sol2 = (y_sol2-d)/m
            intersection_points_left_wall[1] = [x_sol2, y_sol2]

            left_wall = True

    ## Intersection with right wall
    # Define the points
    x1, y1 = corners[0] # top right corner
    x2, y2 = corners[1] # bottom right corner

    ## In case the corridor has a pi/2 or 3pi/2 tilt (vertical corridor)
    if abs(x1 - x2) < tol:
        d = x1
        a = 1
        b = -2*k
        c = d**2 -2*d*h + h**2 + k**2 -radius**2
    
        discriminant = b**2 - 4*a*c

        # discriminant < 0 -> 2 complex solutions -> no intersection points
        # discriminant == 0 -> 1 real solution -> 1 intersection point
        if abs(discriminant) < tol:
            y_sol = -b/(2*a)
            x_sol = d
            intersection_points_right_wall[0] = [x_sol, y_sol]

            right_wall = True
        # discriminant > 0 -> 2 real solutions -> 2 intersection points    
        elif discriminant > tol:
            sqrt_discriminant = sqrt(discriminant)
            y_sol1 = (-b + sqrt_discriminant)/(2*a)
            x_sol1 = d
            intersection_points_right_wall[0] = [x_sol1, y_sol1]

            y_sol2 = (-b - sqrt_discriminant)/(2*a)
            x_sol2 = d
            intersection_points_right_wall[1] = [x_sol2, y_sol2]

            right_wall = True

    ## If the corridor has a 0 or pi tilt (horizontal corridor)
    elif abs(y1 - y2) < tol:
        # write code to compute the intersection points between a circle with radius= radius and center (h,k) and a horizontal line y = y1
        a = 1
        b = -2*h
        c = h**2 + k**2 - radius**2 + y1**2 - 2*k*y1
        discriminant = b**2 - 4*a*c

        # discriminant < 0 -> 2 complex solutions -> no intersection points
        # discriminant == 0 -> 1 real solution -> 1 intersection point
        if abs(discriminant) < tol:
            y_sol = y1
            x_sol = -b / (2*a)
            intersection_points_right_wall[0] = [x_sol, y_sol]

            right_wall = True
        # discriminant > 0 -> 2 real solutions -> 2 intersection points    
        elif discriminant > tol:
            sqrt_discriminant = sqrt(discriminant)
            y_sol1 = y1
            x_sol1 = (-b + sqrt_discriminant)/(2*a)
            intersection_points_right_wall[0] = [x_sol1, y_sol1]
                
            y_sol2 = y1
            x_sol2 = (-b - sqrt_discriminant)/(2*a)
            intersection_points_right_wall[1] = [x_sol2, y_sol2]

            right_wall = True
        
    ## If the corridor is neither horizontal nor vertical
    else:
        # y = m * x + d
        m = (y2 - y1)/(x2 - x1) if x1 != x2 else 0
        d = y1 - m * x1 if m != 0 else 0  

            
        # Solve systems of equations
        a = (1/m**2 + 1)
        b = -(2*d)/m**2 - (2*h)/m - 2 * k 
        c = d**2/m**2 + (2*d*h)/m + h**2 + k**2 - radius**2
        discriminant = b**2 - 4*a*c

        # discriminant < 0 -> 2 complex solutions -> no intersection points
        # discriminant == 0 -> 1 real solution -> 1 intersection point
        if abs(discriminant) < tol:
            y_sol = -b/(2*a)
            x_sol = (y_sol-d)/m
            intersection_points_right_wall[0] = [x_sol, y_sol]

            right_wall = True
        # discriminant > 0 -> 2 real solutions -> 2 intersection points    
        elif discriminant > tol:
            sqrt_discriminant = sqrt(discriminant)
            y_sol1 = (-b + sqrt_discriminant)/(2*a)
            x_sol1 = (y_sol1-d)/m
            intersection_points_right_wall[0] = [x_sol1, y_sol1]
       
            y_sol2 = (-b - sqrt_discriminant)/(2*a)
            x_sol2 = (y_sol2-d)/m
            intersection_points_right_wall[1] = [x_sol2, y_sol2]

            right_wall = True
    
    if intersection_points_left_wall[0] == [None, None]:
        intersection_points_left_wall = []
    elif intersection_points_left_wall[1] == [None, None]:
        intersection_points_left_wall.pop(1)
    
    if intersection_points_right_wall[0] == [None, None]:
        intersection_points_right_wall = []
    elif intersection_points_right_wall[1] == [None, None]:
        intersection_points_right_wall.pop(1)

    return left_wall, right_wall, intersection_points_left_wall, intersection_points_right_wall


def compute_intersection_points_circle_segment(x1, y1, x2, y2, xc, yc, radius, tol=1e-5):
    """
    Compute intersection points between the circle centered at (xc, yc) with
    radius 'radius' and the infinite line passing through (x1, y1), (x2, y2).
    Returns a list of 0, 1, or 2 points.
    """
    h, k = xc, yc
    points = []

    # Vertical line
    if abs(x1 - x2) < tol:
        d = x1
        a = 1.0
        b = -2.0 * k
        c = d**2 - 2.0 * d * h + h**2 + k**2 - radius**2
        discriminant = b**2 - 4*a*c

        if abs(discriminant) < tol:
            y_sol = -b / (2*a)
            points.append([d, y_sol])
        elif discriminant > tol:
            sqrt_disc = sqrt(discriminant)
            y_sol1 = (-b + sqrt_disc) / (2*a)
            y_sol2 = (-b - sqrt_disc) / (2*a)
            points.append([d, y_sol1])
            points.append([d, y_sol2])

    # Horizontal line
    elif abs(y1 - y2) < tol:
        a = 1.0
        b = -2.0 * h
        c = h**2 + k**2 - radius**2 + y1**2 - 2.0 * k * y1
        discriminant = b**2 - 4*a*c

        if abs(discriminant) < tol:
            x_sol = -b / (2*a)
            points.append([x_sol, y1])
        elif discriminant > tol:
            sqrt_disc = sqrt(discriminant)
            x_sol1 = (-b + sqrt_disc) / (2*a)
            x_sol2 = (-b - sqrt_disc) / (2*a)
            points.append([x_sol1, y1])
            points.append([x_sol2, y1])

    # General case
    else:
        mm = (y2 - y1) / (x2 - x1)
        d = y1 - mm * x1

        a = 1.0 / mm**2 + 1.0
        b = -(2.0 * d) / mm**2 - (2.0 * h) / mm - 2.0 * k
        c = d**2 / mm**2 + (2.0 * d * h) / mm + h**2 + k**2 - radius**2
        discriminant = b**2 - 4*a*c

        if abs(discriminant) < tol:
            y_sol = -b / (2*a)
            x_sol = (y_sol - d) / mm
            points.append([x_sol, y_sol])
        elif discriminant > tol:
            sqrt_disc = sqrt(discriminant)
            y_sol1 = (-b + sqrt_disc) / (2*a)
            x_sol1 = (y_sol1 - d) / mm
            y_sol2 = (-b - sqrt_disc) / (2*a)
            x_sol2 = (y_sol2 - d) / mm
            points.append([x_sol1, y_sol1])
            points.append([x_sol2, y_sol2])

    return points


def check_intersection_two_segments(A1, A2, B1, B2, tol=1e-12):
    '''
    Check whether two line segments A1-A2 and B1-B2 intersect.

    This function handles:
    - proper intersections (crossing segments)
    - collinear overlaps
    - endpoint touching

    A tolerance is used to handle numerical precision issues.

    :param A1: start point of first segment
    :type A1: tuple or list (x, y)
    :param A2: end point of first segment
    :type A2: tuple or list (x, y)
    :param B1: start point of second segment
    :type B1: tuple or list (x, y)
    :param B2: end point of second segment
    :type B2: tuple or list (x, y)
    :param tol: numerical tolerance
    :type tol: float

    :return: True if segments intersect, False otherwise
    :rtype: bool
    '''
    x1, y1 = A1
    x2, y2 = A2
    x3, y3 = B1
    x4, y4 = B2

    def orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    def on_segment(ax, ay, bx, by, px, py):
        return (
            min(ax, bx) - tol <= px <= max(ax, bx) + tol and
            min(ay, by) - tol <= py <= max(ay, by) + tol
        )

    o1 = orient(x1, y1, x2, y2, x3, y3)
    o2 = orient(x1, y1, x2, y2, x4, y4)
    o3 = orient(x3, y3, x4, y4, x1, y1)
    o4 = orient(x3, y3, x4, y4, x2, y2)

    # Proper intersection
    if ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and \
       ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)):
        return True

    # Collinear / endpoint-touch cases
    if abs(o1) <= tol and on_segment(x1, y1, x2, y2, x3, y3):
        return True
    if abs(o2) <= tol and on_segment(x1, y1, x2, y2, x4, y4):
        return True
    if abs(o3) <= tol and on_segment(x3, y3, x4, y4, x1, y1):
        return True
    if abs(o4) <= tol and on_segment(x3, y3, x4, y4, x2, y2):
        return True

    return False


def check_intersection_case(segment1, segment2, tol=1e-9):
    '''
    Check whether two line segments intersect, excluding the trivial case
    where they only meet at a shared endpoint.

    This function is typically used when consecutive segments share a
    junction point, and we want to ignore that expected contact and only
    detect "true" intersections.

    :param segment1: first segment with attributes start_position and end_position
    :type segment1: object
    :param segment2: second segment with attributes start_position and end_position
    :type segment2: object
    :param tol: numerical tolerance to detect coincident endpoints
    :type tol: float

    :return: True if the segments intersect (excluding shared endpoint), False otherwise
    :rtype: bool
    '''
    A1 = segment1.start_position
    A2 = segment1.end_position
    B1 = segment2.start_position
    B2 = segment2.end_position

    # Ignore trivial case: segments share a common endpoint
    dx = A2[0] - B1[0]
    dy = A2[1] - B1[1]
    if dx * dx + dy * dy <= tol * tol:
        return False

    return check_intersection_two_segments(A1=A1, A2=A2, B1=B1, B2=B2)


def select_closest_intersection(x0, y0, points):
    '''
    Select the closest intersection point to a reference point (x0, y0).

    If the list contains zero or one point, it is returned unchanged.
    If two points are provided, the closest one is selected.

    :param x0: x-coordinate of reference point
    :type x0: float
    :param y0: y-coordinate of reference point
    :type y0: float
    :param points: list of intersection points [[x1, y1], [x2, y2]]
    :type points: list

    :return: list containing the closest point (or original list if <= 1 point)
    :rtype: list
    '''
    if len(points) <= 1:
        return points

    p1, p2 = points[0], points[1]

    d1 = (p1[0] - x0) ** 2 + (p1[1] - y0) ** 2
    d2 = (p2[0] - x0) ** 2 + (p2[1] - y0) ** 2

    return [p1] if d1 < d2 else [p2]


def circles_overlap(xc1, yc1, xc2, yc2, R, tol=1e-3):
    '''
    Check whether two circles of equal radius overlap or touch.

    The function compares the squared distance between the centers
    with the squared sum of the radii (2R).

    A tolerance is used to handle numerical precision issues.

    :param xc1: x-coordinate of first circle center
    :type xc1: float
    :param yc1: y-coordinate of first circle center
    :type yc1: float
    :param xc2: x-coordinate of second circle center
    :type xc2: float
    :param yc2: y-coordinate of second circle center
    :type yc2: float
    :param R: radius of both circles
    :type R: float
    :param tol: numerical tolerance
    :type tol: float

    :return: True if circles overlap or touch, False otherwise
    :rtype: bool
    '''
    dx = xc2 - xc1
    dy = yc2 - yc1

    return dx * dx + dy * dy <= (2 * R) ** 2 + tol


def compute_intersection_two_segments(A1, A2, B1, B2, tol=1e-9):
    """
    Compute the intersection point between two finite 2D segments.

    :param A1: start point of first segment [x, y]
    :param A2: end point of first segment [x, y]
    :param B1: start point of second segment [x, y]
    :param B2: end point of second segment [x, y]
    :param tol: numerical tolerance

    :return: intersection point and boolean indicating whether it lies on both segments
    :rtype: tuple[list[float], bool]
    """
    x1, y1 = A1
    x2, y2 = A2
    x3, y3 = B1
    x4, y4 = B2

    rx = x2 - x1
    ry = y2 - y1
    sx = x4 - x3
    sy = y4 - y3

    denominator = rx * sy - ry * sx

    # Parallel or collinear segments
    if abs(denominator) <= tol:
        return [], False

    qpx = x3 - x1
    qpy = y3 - y1

    t = (qpx * sy - qpy * sx) / denominator
    u = (qpx * ry - qpy * rx) / denominator

    if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
        x_int = x1 + t * rx
        y_int = y1 + t * ry
        return [x_int, y_int], True

    return [], False


def compute_line_corridor_intersections(corner_point, angle, corridor, tol=1e-9):
    """
    Compute intersection points between an infinite line and a corridor.

    The line passes through `corner_point` with orientation `angle`.

    :param corner_point: point on the line [x, y]
    :param angle: line orientation angle [rad]
    :param corridor: corridor to intersect
    :param tol: numerical tolerance

    :return: list of unique intersection points
    :rtype: list[list[float]]
    """
    x0, y0 = corner_point
    dx = cos(angle)
    dy = sin(angle)

    line_length = 1e6
    line_p1 = [x0 - line_length * dx, y0 - line_length * dy]
    line_p2 = [x0 + line_length * dx, y0 + line_length * dy]

    corners = corridor.get_corners()

    edges = [
        (corners[0], corners[3]),
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
    ]

    intersections = []

    for edge_start, edge_end in edges:
        point, intersects = compute_intersection_two_segments(
            line_p1,
            line_p2,
            edge_start,
            edge_end,
            tol=tol,
        )

        if not intersects:
            continue

        is_duplicate = any(
            (point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2 <= tol**2
            for p in intersections
        )

        if not is_duplicate:
            intersections.append(point)

    return intersections
