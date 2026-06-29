import numpy as np
from math import cos, sin, pi

from ..geometry import Point, IntermediateCircle

from .geometry_operations import compute_distance_two_points





def compute_nominal_same_turn_merged_center(
    corner_point1,
    corner_point2,
    R,
    r,
    turn_direction,
    tol=1e-9,
):
    """
    Compute the nominal center of a same-turn merged Intermediate Circle.

    The merged circle has radius R and must contain the two footprint disks
    of radius r centered at corner_point1 and corner_point2.

    The center is chosen as the intersection of two disks of radius D = R - r,
    on the side selected by turn_direction.
    """
    if turn_direction not in (-1, 1):
        raise ValueError("turn_direction must be -1 or +1")

    D = R - r

    p1 = np.array([corner_point1.x, corner_point1.y], dtype=float)
    p2 = np.array([corner_point2.x, corner_point2.y], dtype=float)

    delta = p2 - p1
    distance = np.linalg.norm(delta)

    if distance <= tol:
        raise ValueError("Cannot merge circles with coincident corner points")

    if distance > 2.0 * D + tol:
        return {
            "feasible": False,
            "reason": "corner_points_too_far_for_single_merged_circle",
            "center": None,
            "D": D,
            "distance": distance,
        }

    e_long = delta / distance

    # Left normal of e_long.
    left_normal = np.array([-e_long[1], e_long[0]], dtype=float)

    # Pick side according to turn direction.
    e_side = turn_direction * left_normal

    midpoint = 0.5 * (p1 + p2)

    h_sq = D * D - 0.25 * distance * distance

    if h_sq < 0.0 and h_sq > -tol:
        h_sq = 0.0

    h = np.sqrt(h_sq)

    center_np = midpoint + h * e_side

    return {
        "feasible": True,
        "reason": "ok",
        "center": Point(center_np[0], center_np[1]),
        "D": D,
        "distance": distance,
        "midpoint": Point(midpoint[0], midpoint[1]),
        "e_long": e_long,
        "e_side": e_side,
        "h": h,
    }


def compute_basic_bounds(w1, w2, S):
    """
    Compute the ordinary wall-clearance lower bounds.

    x >= a = max(0, S - w1)
    y >= b = max(0, S - w2)
    """
    a = max(0.0, S - w1)
    b = max(0.0, S - w2)

    return a, b


def compute_safe_half_bounds(w1, w2, R):
    """
    Compute the preferred safe-half lower bounds for the actual
    Intermediate Circle of radius R.

    These enforce that the Intermediate Circle itself, not the swept
    circle of radius S=R+r, stays on the preferred half-side of both
    corridor centerlines.
    """
    h1 = max(0.0, R - 0.5 * w1)
    h2 = max(0.0, R - 0.5 * w2)

    return h1, h2


def point_satisfies_admissible_disk(x, y, D, tol=1e-9):
    """
    Check x^2 + y^2 <= D^2.
    """
    return x * x + y * y <= D * D + tol


def point_avoids_other_intersection(
    center_world,
    other_intersection_point,
    S,
    tol=1e-9,
):
    """
    Check whether the swept circle of radius S centered at center_world
    does not contain the other intersection point.
    """
    if other_intersection_point is None:
        return True

    dx = center_world.x - other_intersection_point.x
    dy = center_world.y - other_intersection_point.y

    return dx * dx + dy * dy >= S * S - tol


def local_to_world_point(corner_point, ex, ey, center_local):
    """
    Convert a local transition-frame point to world coordinates.
    """
    x, y = center_local

    return Point(
        corner_point.x + x * ex[0] + y * ey[0],
        corner_point.y + x * ex[1] + y * ey[1],
    )


def world_to_circle_local(circle, point):
    A = np.column_stack((circle.ex, circle.ey))

    v = np.array([
        point.x - circle.corner_point.x,
        point.y - circle.corner_point.y,
    ], dtype=float)

    return np.linalg.solve(A, v)


def compute_transition_frame(corridor1, corridor2, turn_direction):
    """
    Compute the local transition-frame axes for an Intermediate Circle.

    Convention:
        - local x is perpendicular to corridor1,
        - local y is perpendicular to corridor2,
        - both point toward the side of the turn.

    Therefore:
        C_world = corner_point + x * ex + y * ey

    where:
        ex = normal to corridor1 on the turn side,
        ey = normal to corridor2 on the turn side.

    turn_direction:
        +1 = left turn
        -1 = right turn
    """
    if turn_direction not in (-1, 1):
        raise ValueError("turn_direction must be -1 or +1")

    ex_angle = corridor1.tilt + turn_direction * pi / 2.0
    ey_angle = corridor2.tilt + turn_direction * pi / 2.0

    ex = np.array([cos(ex_angle), sin(ex_angle)], dtype=float)
    ey = np.array([cos(ey_angle), sin(ey_angle)], dtype=float)

    return ex, ey


def generate_local_center_candidates(a, b, h1, h2, D, tol=1e-9):
    """
    Generate local Intermediate Circle center candidates in priority order.

    Priority:
        1. 45-degree point, if it already satisfies the safe-half constraints.
        2. Shifted safe-half point, if safe-half placement is possible.
        3. 45-degree point, if it satisfies ordinary wall-clearance constraints.
        4. Basic shifted point, if ordinary feasibility is possible.
    """
    candidates = []

    q = D / np.sqrt(2.0)

    forty_five_safe_half_feasible = (
        q >= h1 - tol and
        q >= h2 - tol
    )

    safe_half_shifted_feasible = (
        h1 >= a - tol and
        h2 >= b - tol and
        h1 * h1 + h2 * h2 <= D * D + tol
    )

    forty_five_basic_feasible = (
        q >= a - tol and
        q >= b - tol
    )

    basic_shifted_feasible = (
        a * a + b * b <= D * D + tol
    )

    if forty_five_safe_half_feasible:
        candidates.append(
            ("forty_five_safe_half", np.array([q, q], dtype=float))
        )

    if safe_half_shifted_feasible:
        candidates.append(
            ("safe_half_shifted", np.array([h1, h2], dtype=float))
        )

    if forty_five_basic_feasible:
        candidates.append(
            ("forty_five_basic", np.array([q, q], dtype=float))
        )

    if basic_shifted_feasible:
        candidates.append(
            ("basic_shifted", np.array([a, b], dtype=float))
        )

    return candidates


def compute_shift_direction_local(center_local, tol=1e-9):
    """
    Compute the local shift direction.

    The shift direction points from the selected Intermediate Circle center
    back toward the corner point.

    Since the corner is the origin in the local frame, this is:

        -center_local / ||center_local||
    """
    norm = np.linalg.norm(center_local)

    if norm <= tol:
        raise ValueError(
            "Cannot compute shift direction because center_local is too close "
            "to the corner point."
        )

    return -center_local / norm


def local_direction_to_world(ex, ey, direction_local):
    """
    Convert a local direction vector to world coordinates.

    d_world = dx * ex + dy * ey
    """
    dx, dy = direction_local

    direction_world = dx * ex + dy * ey
    norm = np.linalg.norm(direction_world)

    if norm <= 1e-12:
        raise ValueError("Computed zero world shift direction.")

    return direction_world / norm


def compute_s_max_from_corridor_widths(
    center_local,
    shift_direction_local,
    w1,
    w2,
    S,
    tol=1e-9,
):
    """
    Compute the maximum shift allowed by the local swept-circle bounds.

    The shifted center is:

        C(s) = center_local + s * shift_direction_local

    Validity is based on the swept circle of radius S = R + r.

    The local origin is the corner point.

    The center is allowed to move past the corner into the corridor-union
    region, until the swept circle touches the opposite active wall.

    Local validity bounds:

        S - w1 <= x <= w1 - S
        S - w2 <= y <= w2 - S

    :param center_local: Circle center in the local transition frame.
    :param shift_direction_local: Unit shift direction in the local frame.
    :param w1: Width associated with the first corridor side.
    :param w2: Width associated with the second corridor side.
    :param S: Swept-circle radius, equal to R + r.
    :param tol: Numerical tolerance.

    :return: Maximum shift and reason.
    """
    x0, y0 = center_local
    dx, dy = shift_direction_local

    x_min = S - w1
    y_min = S - w2
    x_max = w1 - S
    y_max = w2 - S

    candidates = []

    if dx > tol:
        sx = (x_max - x0) / dx
        candidates.append((sx, "corridor1_upper_width_bound"))
    elif dx < -tol:
        sx = (x_min - x0) / dx
        candidates.append((sx, "corridor1_lower_width_bound"))

    if dy > tol:
        sy = (y_max - y0) / dy
        candidates.append((sy, "corridor2_upper_width_bound"))
    elif dy < -tol:
        sy = (y_min - y0) / dy
        candidates.append((sy, "corridor2_lower_width_bound"))

    if len(candidates) == 0:
        return np.inf, "zero_shift_direction_components"

    positive_candidates = [
        (s, reason)
        for s, reason in candidates
        if s >= -tol
    ]

    if len(positive_candidates) == 0:
        return 0.0, "initial_center_outside_local_shift_bounds"

    s_max, reason = min(positive_candidates, key=lambda item: item[0])

    if s_max < 0:
        s_max = 0.0

    return s_max, reason


def compute_s_max_from_other_intersection(
    center_world,
    shift_direction_world,
    other_intersection_point,
    S,
    tol=1e-9,
):
    """
    Compute the maximum shift allowed by an optional forbidden intersection point.

    The shifted center is:

        C(s) = center_world + s * shift_direction_world

    Validity requires:

        ||C(s) - P|| >= S

    If other_intersection_point is None, this returns infinity.
    """
    if other_intersection_point is None:
        return np.inf, "no_other_intersection_point"

    c0 = np.array([center_world.x, center_world.y], dtype=float)
    d = np.array(shift_direction_world, dtype=float)
    p = np.array([other_intersection_point.x, other_intersection_point.y], dtype=float)

    d_norm = np.linalg.norm(d)
    if d_norm <= tol:
        raise ValueError("shift_direction_world has near-zero norm")

    d = d / d_norm

    m = c0 - p

    # Quadratic:
    # s^2 + 2 * beta * s + gamma = 0
    beta = np.dot(m, d)
    gamma = np.dot(m, m) - S * S

    # If gamma < 0, the initial center is already invalid.
    if gamma < -tol:
        return 0.0, "initial_center_inside_other_intersection_forbidden_disk"

    discriminant = beta * beta - gamma

    # No intersection with forbidden disk along the infinite line.
    if discriminant <= tol:
        return np.inf, "shift_ray_does_not_hit_other_intersection_disk"

    sqrt_disc = np.sqrt(discriminant)

    s_enter = -beta - sqrt_disc
    s_exit = -beta + sqrt_disc

    # Invalid interval is [s_enter, s_exit].
    # We only care about s >= 0.
    if s_exit <= tol:
        # Forbidden disk is behind the shift ray.
        return np.inf, "other_intersection_disk_behind_shift_ray"

    if s_enter <= tol:
        # This means we are at or just outside the forbidden boundary and
        # immediately enter it. Be conservative.
        return 0.0, "shift_immediately_hits_other_intersection_disk"

    return s_enter, "other_intersection_point"


def compute_shift_data(
    center_local,
    center_world,
    ex,
    ey,
    w1,
    w2,
    S,
    other_intersection_point=None,
    tol=1e-9,
):
    """
    Compute shift direction and maximum allowed shift.

    Validity is based on the swept circle of radius S.
    """
    shift_direction_local = compute_shift_direction_local(
        center_local=center_local,
        tol=tol,
    )

    shift_direction_world = local_direction_to_world(
        ex=ex,
        ey=ey,
        direction_local=shift_direction_local,
    )

    s_max_widths, reason_widths = compute_s_max_from_corridor_widths(
        center_local=center_local,
        shift_direction_local=shift_direction_local,
        w1=w1,
        w2=w2,
        S=S,
        tol=tol,
    )

    s_max_other, reason_other = compute_s_max_from_other_intersection(
        center_world=center_world,
        shift_direction_world=shift_direction_world,
        other_intersection_point=other_intersection_point,
        S=S,
        tol=tol,
    )

    if s_max_widths <= s_max_other:
        s_max = s_max_widths
        s_max_reason = reason_widths
    else:
        s_max = s_max_other
        s_max_reason = reason_other

    return {
        "shift_direction_local": shift_direction_local,
        "shift_direction_world": shift_direction_world,
        "s_max": s_max,
        "s_max_reason": s_max_reason,
        "s_max_widths": s_max_widths,
        "s_max_widths_reason": reason_widths,
        "s_max_other_intersection": s_max_other,
        "s_max_other_intersection_reason": reason_other,
    }


def build_intermediate_circle_from_geometry_result(
    geometry_result,
    index=None,
    edge_pair=None,
    door_point=None,
    door_type=None,
    start_angle_arc=None,
    rho=None,
    merged=False,
):
    """
    Build an IntermediateCircle object from the dictionary returned by
    compute_intermediate_circle_geometry.

    :param geometry_result: Result dictionary returned by compute_intermediate_circle_geometry.
    :type geometry_result: dict

    :param index: Optional index of the circle in the corridor sequence.
    :type index: int or None

    :param edge_pair: Optional edge-pair metadata.
    :type edge_pair: object

    :param door_point: Optional door point associated with the circle.
    :type door_point: Point or None

    :param door_type: Optional door type metadata.
    :type door_type: object

    :param start_angle_arc: Optional arc start angle.
    :type start_angle_arc: float or None

    :param rho: Optional radius used to sample arc coordinates.
    :type rho: float or None

    :param merged: Whether the circle is marked as merged.
    :type merged: bool

    :return: IntermediateCircle object.
    :rtype: IntermediateCircle
    """
    if geometry_result is None:
        raise ValueError("geometry_result cannot be None")

    if not geometry_result.get("feasible", False):
        reason = geometry_result.get("reason", "unknown")
        raise ValueError(
            "Cannot build IntermediateCircle from infeasible geometry result. "
            f"Reason: {reason}"
        )

    center = geometry_result.get("center", None)
    corner_point = geometry_result.get("corner_point", None)
    radius = geometry_result.get("R", None)
    turn_direction = geometry_result.get("turn_direction", None)

    if center is None:
        raise ValueError("geometry_result does not contain a valid 'center'")

    if corner_point is None:
        raise ValueError("geometry_result does not contain a valid 'corner_point'")

    if radius is None:
        raise ValueError("geometry_result does not contain a valid radius 'R'")

    if turn_direction not in (-1, 1):
        raise ValueError(
            "geometry_result must contain turn_direction equal to -1 or +1"
        )

    forbidden_point = geometry_result.get("other_intersection_point", None)
    forbidden_points = (
        [] if forbidden_point is None else [forbidden_point]
    )

    w1 = geometry_result.get("w1", None)
    w2 = geometry_result.get("w2", None)
    S = geometry_result.get("S", None)

    circle = IntermediateCircle(
        center=center,
        radius=radius,
        corner_point=corner_point,
        turn_direction=turn_direction,
        index=index,
        s_max=geometry_result.get("s_max", None),
        edge_pair=edge_pair,
        door_point=door_point,
        door_type=door_type,
        start_angle_arc=start_angle_arc,
        rho=rho,
        swept_radius=S,
        admissible_radius=geometry_result.get("D", None),
        lower_bound_x=geometry_result.get("a", None),
        lower_bound_y=geometry_result.get("b", None),
        lower_bound_x_unclamped=S-w1,
        lower_bound_y_unclamped=S-w2,
        ex=geometry_result.get("ex", None),
        ey=geometry_result.get("ey", None),
        forbidden_points=forbidden_points,
        merged=merged,
    )

    # Optional: attach debug/construction metadata.
    # This is useful while testing, but can be removed later if you want
    # IntermediateCircle objects to stay minimal.
    # circle.construction_rule = geometry_result.get("rule", None)
    # circle.center_local = geometry_result.get("center_local", None)
    # circle.shift_direction_local = geometry_result.get("shift_direction_local", None)
    # circle.shift_direction_world = geometry_result.get("shift_direction_world", None)
    # circle.s_max_reason = geometry_result.get("s_max_reason", None)
    # circle.geometry_result = geometry_result

    return circle


def compute_intermediate_circle_geometry(
    corridor1,
    corridor2,
    corner_point,
    turn_direction,
    vehicle,
    other_intersection_point=None,
    tol=1e-9,
):
    """
    Compute the local and world placement of an Intermediate Circle
    between two perpendicular consecutive corridors.

    Inputs
    ------
    corridor1, corridor2:
        Consecutive CorridorWorld objects.

    corner_point:
        Point used as the origin of the local transition frame.

    turn_direction:
        +1 for left turn, -1 for right turn.

    vehicle:
        Vehicle object. The footprint radius is vehicle.width / 2 and
        the maneuver radius is vehicle.max_radius.

    other_intersection_point:
        Optional Point. If provided, the swept circle of radius S=R+r
        must not contain this point.

    Convention
    ----------
    Local x is perpendicular to corridor1.
    Local y is perpendicular to corridor2.

    Therefore:
        x uses corridor1.width
        y uses corridor2.width
    """
    if turn_direction not in (-1, 1):
        raise ValueError("turn_direction must be -1 or +1 for this function")

    # ------------------------------------------------------------
    # Basic geometric quantities
    # ------------------------------------------------------------
    r = vehicle.width / 2.0
    R = vehicle.max_radius
    other_intersection_point_before_check = other_intersection_point

    if R <= r:
        raise ValueError(
            "Turn radius R must be strictly greater than footprint radius r"
        )

    S = R + r
    D = R - r

    if other_intersection_point is not None:
        distance_corner_forbidden = compute_distance_two_points(
            corner_point,
            other_intersection_point,
        )

        if distance_corner_forbidden < 2.0 * r - tol:
            return {
                "feasible": False,
                "reason": "Forbidden point too close to corner point. No arc maneuver can safely fit.",
                "center": None,
                "center_local": None,
                "rule": "infeasible",
                "R": R,
                "r": r,
                "S": S,
                "D": D,
                "turn_direction": turn_direction,
                "corner_point": corner_point,
                "other_intersection_point": other_intersection_point_before_check,

                # No shift data exists because no valid center was selected.
                "shift_direction_local": None,
                "shift_direction_world": None,
                "s_max": None,
                "s_max_reason": None,
                "s_max_widths": None,
                "s_max_widths_reason": None,
                "s_max_other_intersection": None,
                "s_max_other_intersection_reason": None,
            }
        
        elif distance_corner_forbidden >= 2.0 * R - tol:
            other_intersection_point = None

    w1 = corridor1.width
    w2 = corridor2.width

    # ------------------------------------------------------------
    # Bounds in the local transition frame
    # ------------------------------------------------------------
    # Basic wall-clearance bounds use the swept radius S = R + r.
    #
    # a = max(0, S - w1)
    # b = max(0, S - w2)
    a, b = compute_basic_bounds(w1, w2, S)

    # Safe-half preference bounds use the actual Intermediate Circle radius R.
    #
    # h1 = max(0, R - 0.5 * w1)
    # h2 = max(0, R - 0.5 * w2)
    h1, h2 = compute_safe_half_bounds(w1, w2, R)

    # ------------------------------------------------------------
    # Local-to-world transition frame
    # ------------------------------------------------------------
    ex, ey = compute_transition_frame(
        corridor1=corridor1,
        corridor2=corridor2,
        turn_direction=turn_direction,
    )

    # ------------------------------------------------------------
    # Generate candidate centers in priority order
    # ------------------------------------------------------------
    # 1. (q, q) if it satisfies the safe-half constraints
    # 2. (h1, h2) if it satisfies safe-half feasibility
    # 3. (q, q) if it satisfies ordinary wall constraints
    # 4. (a, b) if it satisfies ordinary feasibility
    candidates = generate_local_center_candidates(
        a=a,
        b=b,
        h1=h1,
        h2=h2,
        D=D,
        tol=tol,
    )

    q = D / np.sqrt(2.0)

    # ------------------------------------------------------------
    # Try candidates in priority order.
    #
    # This is intentionally a loop instead of if/elif, because a
    # candidate may satisfy the width constraints but be blocked by
    # other_intersection_point.
    # ------------------------------------------------------------
    blocked_candidates = []

    for rule, center_local in candidates:
        center_world = local_to_world_point(
            corner_point=corner_point,
            ex=ex,
            ey=ey,
            center_local=center_local,
        )

        avoids_other_intersection = point_avoids_other_intersection(
            center_world=center_world,
            other_intersection_point=other_intersection_point,
            S=S,
            tol=tol,
        )

        if avoids_other_intersection:
            # ----------------------------------------------------
            # Compute shift direction and maximum allowed shift.
            #
            # Validity for shifting is based on the swept circle
            # of radius S = R + r.
            # ----------------------------------------------------
            shift_data = compute_shift_data(
                center_local=center_local,
                center_world=center_world,
                ex=ex,
                ey=ey,
                w1=w1,
                w2=w2,
                S=S,
                other_intersection_point=other_intersection_point,
                tol=tol,
            )

            return {
                "feasible": True,
                "reason": "ok",
                "center": center_world,
                "center_local": center_local,
                "rule": rule,
                "R": R,
                "r": r,
                "S": S,
                "D": D,
                "q": q,
                "a": a,
                "b": b,
                "h1": h1,
                "h2": h2,
                "w1": w1,
                "w2": w2,
                "turn_direction": turn_direction,
                "corner_point": corner_point,
                "ex": ex,
                "ey": ey,
                "other_intersection_point": other_intersection_point_before_check,

                # Shift information
                **shift_data,
            }

        blocked_candidates.append(
            {
                "rule": rule,
                "center_local": center_local,
                "center": center_world,
            }
        )

    # ------------------------------------------------------------
    # If no candidate was generated, the width pair itself is infeasible.
    # ------------------------------------------------------------
    if len(candidates) == 0:
        reason = "width_pair_infeasible"

    # ------------------------------------------------------------
    # If candidates existed but all were blocked by the other intersection,
    # this is not a proof that no center exists. It only means the current
    # finite list of preferred candidates failed.
    # ------------------------------------------------------------
    elif other_intersection_point is not None:
        reason = "selected_candidates_blocked_by_other_intersection"

    else:
        reason = "no_valid_candidate_found"

    return {
        "feasible": False,
        "reason": reason,
        "center": None,
        "center_local": None,
        "rule": "infeasible",
        "R": R,
        "r": r,
        "S": S,
        "D": D,
        "q": q,
        "a": a,
        "b": b,
        "h1": h1,
        "h2": h2,
        "w1": w1,
        "w2": w2,
        "turn_direction": turn_direction,
        "corner_point": corner_point,
        "ex": ex,
        "ey": ey,
        "other_intersection_point": other_intersection_point_before_check,
        "candidate_rules": [rule for rule, _ in candidates],
        "blocked_candidates": blocked_candidates,

        # No shift data exists because no valid center was selected.
        "shift_direction_local": None,
        "shift_direction_world": None,
        "s_max": None,
        "s_max_reason": None,
        "s_max_widths": None,
        "s_max_widths_reason": None,
        "s_max_other_intersection": None,
        "s_max_other_intersection_reason": None,
    }



