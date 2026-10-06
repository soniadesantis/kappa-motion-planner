from dataclasses import dataclass

import numpy as np 
from math import sqrt, atan2

from kappa_planner.helpers.intersections import compute_intersection_two_segments
from kappa_planner.helpers.geometry_operations import compute_turn_direction

@dataclass
class AdmissibleWaypointRegion:
    safe_overlap: tuple

    corner_point: np.ndarray = None
    local_x_axis: np.ndarray = None
    local_y_axis: np.ndarray = None

    R: float = None
    r: float = None

@dataclass
class ReachableWaypointRegion:
    admissible_region: AdmissibleWaypointRegion

    x_bounds: tuple
    y_bounds: tuple

@dataclass
class BaselineFillet:
    center: np.ndarray
    start_point: np.ndarray
    end_point: np.ndarray
    start_heading: float
    end_heading: float
    turn_direction: int

@dataclass
class BicycleBaselineResult:
    intersection_points: list
    corridor_overlaps: list
    safe_overlaps: list
    corridor_directions: list
    turn_directions: list
    candidate_corner_points: list
    admissible_regions: list
    reachable_regions: list
    waypoints: list
    fillets: list


@dataclass(frozen=True)
class BicycleBaselineFailure:
    reason: str
    transition_index: int = None


RIGHT = np.array([1, 0])
LEFT  = np.array([-1, 0])
UP    = np.array([0, 1])
DOWN  = np.array([0, -1])

def compute_bicycle_baseline(
    corridor_list,
    bicycle,
    initial_pose=None,
    final_pose=None,
    boundary_connections=False,
    return_failure=False,
):
    """Construct a baseline.

    When ``return_failure`` is false, preserve the original API and return a
    baseline or ``None``.  When true, return ``(baseline, failure)``.  For a
    geometrically valid corridor sequence, failure is either ``A_j_empty`` or
    ``R_j_empty`` and carries the zero-based transition index ``j``.
    """
    def failed(reason, transition_index=None):
        failure = BicycleBaselineFailure(reason, transition_index)
        return (None, failure) if return_failure else None

    n = len(corridor_list)

    if n < 2:
        return failed("invalid_corridor_sequence")

    r = bicycle.width/2
    R = bicycle.max_radius

    # 1. Corridor intersections
    intersection_points = []

    for j in range(n - 1):
        points = compute_intersection_points_two_corridors(
            corridor_list[j],
            corridor_list[j + 1],
        )
        intersection_points.append(points)

    # 2. Corridor overlaps I_j
    corridor_overlaps = []

    for j in range(n - 1):
        overlap = compute_overlap_two_axis_aligned_corridors(
            corridor_list[j],
            corridor_list[j + 1],
        )

        if overlap is None:
            return failed("invalid_corridor_sequence", j)

        corridor_overlaps.append(overlap)

    # 3. Safe overlaps D_j
    safe_overlaps = []

    for j, corridor_overlap in enumerate(corridor_overlaps):
        safe_overlap = compute_safe_overlap(
            corridor_overlap,
            r,
        )

        if safe_overlap is None:
            return failed("invalid_corridor_sequence", j)

        safe_overlaps.append(safe_overlap)

    # 4. Passage directions
    corridor_directions = [None] * n
    for j in range(1, n - 1):
        direction = compute_passage_direction(
            safe_overlaps[j - 1],
            safe_overlaps[j],
        )

        if direction is None:
            return failed("invalid_corridor_sequence", j)

        corridor_directions[j] = direction

    # 5. First and last corridor directions
    corridor_directions[0] = compute_boundary_corridor_direction(
        corridor_list[0],
        safe_overlaps[0],
        initial=True,
    )

    corridor_directions[-1] = compute_boundary_corridor_direction(
        corridor_list[-1],
        safe_overlaps[-1],
        initial=False,
    )

    if corridor_directions[0] is None or corridor_directions[-1] is None:
        return failed("invalid_corridor_sequence")

    # 6. Turn directions tau_j
    turn_directions = []

    for j in range(n - 1):
        tau = compute_corridor_turn_direction(
            corridor_directions[j],
            corridor_directions[j + 1],
        )

        if tau is None:
            return failed("invalid_corridor_sequence", j)

        turn_directions.append(tau)

    # 7. Corner points
    candidate_corner_points = []

    for j in range(n - 1):
        corners = compute_candidate_corner_points(
            intersection_points=intersection_points[j],
            corridor_overlap=corridor_overlaps[j],
            direction_in=corridor_directions[j],
            direction_out=corridor_directions[j + 1],
            turn_direction=turn_directions[j],
        )

        if corners is None:
            return failed("A_j_empty", j)

        candidate_corner_points.append(corners)

    # 8. Local admissible regions A_j
    admissible_regions = []

    for j in range(n - 1):

        tau = turn_directions[j]

        # Aligned transition: A_j = D_j
        if tau == 0:
            region = AdmissibleWaypointRegion(
                safe_overlap=safe_overlaps[j],
            )

        # Turning transition
        else:
            corner_point = candidate_corner_points[j][tau]

            local_x_axis, local_y_axis = compute_admissible_region_local_axes(
                corridor_directions[j],
                corridor_directions[j + 1],
                tau,
            )

            region = AdmissibleWaypointRegion(
                safe_overlap=safe_overlaps[j],
                corner_point=corner_point,
                local_x_axis=local_x_axis,
                local_y_axis=local_y_axis,
                R=R,
                r=r,
            )

        admissible_regions.append(region)

    # 9. Reachable-set propagation R_j
    reachable_regions = []

    # R_1 = A_1
    bounds = compute_region_bounds(
        admissible_regions[0],
    )

    if bounds is None:
        return failed("A_j_empty", 0)

    x_bounds, y_bounds = bounds

    reachable_regions.append(
        ReachableWaypointRegion(
            admissible_region=admissible_regions[0],
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
    )

    # R_2, ..., R_{n-1}
    for j in range(1, n - 1):

        # Distinguish a locally empty A_j from an A_j that becomes empty only
        # after intersecting it with the propagation constraints for R_j.
        if compute_region_bounds(admissible_regions[j]) is None:
            return failed("A_j_empty", j)

        region = compute_reachable_region(
            admissible_region=admissible_regions[j],
            previous_region=reachable_regions[j - 1],
            passage_direction=corridor_directions[j],
            R=R,
        )

        if region is None:
            return failed("R_j_empty", j)

        reachable_regions.append(region)

    # 10. Backward waypoint reconstruction
    waypoints = reconstruct_baseline_waypoints(
        reachable_regions=reachable_regions,
        corridor_directions=corridor_directions,
        R=R,
    )

    if waypoints is None:
        return failed("R_j_empty", n - 2)
    
    # 11. Polyline + radius-R fillets
    fillets = [None] * (n - 1)

    for j in range(n - 1):

        tau = turn_directions[j]

        if tau == 0:
            continue

        fillets[j] = compute_baseline_fillet(
            waypoint=waypoints[j],
            direction_in=corridor_directions[j],
            direction_out=corridor_directions[j + 1],
            turn_direction=tau,
            R=R,
        )

    # 12. Optional boundary connections

    result = BicycleBaselineResult(
        intersection_points=intersection_points,
        corridor_overlaps=corridor_overlaps,
        safe_overlaps=safe_overlaps,
        corridor_directions=corridor_directions,
        turn_directions=turn_directions,
        candidate_corner_points=candidate_corner_points,
        admissible_regions=admissible_regions,
        reachable_regions=reachable_regions,
        waypoints=waypoints,
        fillets=fillets,
    )
    return (result, None) if return_failure else result


def build_baseline_primitive_geometry(
    fillets,
):
    """
    Build the internal baseline as an ordered sequence of
    circular arcs and straight segments.
    """
    turning_fillets = [
        fillet
        for fillet in fillets
        if fillet is not None
    ]

    if not turning_fillets:
        return []

    primitives = []

    for i, fillet in enumerate(turning_fillets):

        primitives.append(
            ("arc", fillet)
        )

        if i == len(turning_fillets) - 1:
            continue

        next_fillet = turning_fillets[i + 1]

        primitives.append(
            (
                "segment",
                fillet.end_point,
                next_fillet.start_point,
            )
        )

    return primitives


def compute_baseline_fillet(
    waypoint,
    direction_in,
    direction_out,
    turn_direction,
    R,
):
    """
    Compute the radius-R circular fillet associated with a turning waypoint.
    """
    if turn_direction not in (-1, 1):
        raise ValueError(
            "A baseline fillet is defined only for a turning waypoint."
        )

    waypoint = np.asarray(waypoint, dtype=float)
    direction_in = np.asarray(direction_in, dtype=float)
    direction_out = np.asarray(direction_out, dtype=float)

    start_point = waypoint - R * direction_in
    end_point = waypoint + R * direction_out

    left_normal = np.array([
        -direction_in[1],
        direction_in[0],
    ])

    center = (
        start_point
        + turn_direction * R * left_normal
    )

    start_heading = atan2(
        direction_in[1],
        direction_in[0],
    )

    end_heading = atan2(
        direction_out[1],
        direction_out[0],
    )

    return BaselineFillet(
        center=center,
        start_point=start_point,
        end_point=end_point,
        start_heading=start_heading,
        end_heading=end_heading,
        turn_direction=turn_direction,
    )


def reconstruct_baseline_waypoints(
    reachable_regions,
    corridor_directions,
    R,
    tol=1e-9,
):
    """
    Reconstruct one compatible waypoint sequence backward through
    the propagated reachable regions.
    """
    n_waypoints = len(reachable_regions)

    waypoints = [None] * n_waypoints

    # ---------------------------------------------------------------
    # Start with any point in the terminal reachable set
    # ---------------------------------------------------------------
    waypoints[-1] = select_point_from_reachable_region(
        reachable_regions[-1],
        tol=tol,
    )

    if waypoints[-1] is None:
        return None

    # ---------------------------------------------------------------
    # Reconstruct backward
    # ---------------------------------------------------------------
    for j in range(n_waypoints - 1, 0, -1):

        current_point = waypoints[j]
        previous_region = reachable_regions[j - 1]
        passage_direction = corridor_directions[j]

        x_current, y_current = current_point

        # -----------------------------------------------------------
        # Right
        # -----------------------------------------------------------
        if np.array_equal(passage_direction, RIGHT):

            feasible_interval = compute_x_slice(
                previous_region,
                y_current,
                tol=tol,
            )

            if feasible_interval is not None:
                feasible_interval = intersect_intervals(
                    feasible_interval,
                    (-np.inf, x_current - 2 * R),
                    tol=tol,
                )

            if feasible_interval is None:
                return None

            x_previous = 0.5 * sum(feasible_interval)

            waypoints[j - 1] = np.array([
                x_previous,
                y_current,
            ])

        # -----------------------------------------------------------
        # Left
        # -----------------------------------------------------------
        elif np.array_equal(passage_direction, LEFT):

            feasible_interval = compute_x_slice(
                previous_region,
                y_current,
                tol=tol,
            )

            if feasible_interval is not None:
                feasible_interval = intersect_intervals(
                    feasible_interval,
                    (x_current + 2 * R, np.inf),
                    tol=tol,
                )

            if feasible_interval is None:
                return None

            x_previous = 0.5 * sum(feasible_interval)

            waypoints[j - 1] = np.array([
                x_previous,
                y_current,
            ])

        # -----------------------------------------------------------
        # Up
        # -----------------------------------------------------------
        elif np.array_equal(passage_direction, UP):

            feasible_interval = compute_y_slice(
                previous_region,
                x_current,
                tol=tol,
            )

            if feasible_interval is not None:
                feasible_interval = intersect_intervals(
                    feasible_interval,
                    (-np.inf, y_current - 2 * R),
                    tol=tol,
                )

            if feasible_interval is None:
                return None

            y_previous = 0.5 * sum(feasible_interval)

            waypoints[j - 1] = np.array([
                x_current,
                y_previous,
            ])

        # -----------------------------------------------------------
        # Down
        # -----------------------------------------------------------
        elif np.array_equal(passage_direction, DOWN):

            feasible_interval = compute_y_slice(
                previous_region,
                x_current,
                tol=tol,
            )

            if feasible_interval is not None:
                feasible_interval = intersect_intervals(
                    feasible_interval,
                    (y_current + 2 * R, np.inf),
                    tol=tol,
                )

            if feasible_interval is None:
                return None

            y_previous = 0.5 * sum(feasible_interval)

            waypoints[j - 1] = np.array([
                x_current,
                y_previous,
            ])

        else:
            raise ValueError("Invalid passage direction.")

    return waypoints


def select_point_from_reachable_region(region, tol=1e-9):
    """
    Select a deterministic point from a nonempty reachable region.
    """
    x_min, x_max = region.x_bounds
    x = 0.5 * (x_min + x_max)

    y_slice = compute_y_slice(
        region,
        x,
        tol=tol,
    )

    if y_slice is None:
        return None

    y = 0.5 * (y_slice[0] + y_slice[1])

    return np.array([x, y])


def compute_x_slice(region, y, tol=1e-9):
    """Return feasible x coordinates of R_j at a fixed y."""
    return compute_reachable_region_slice(
        region,
        fixed_axis=1,
        fixed_value=y,
        tol=tol,
    )


def compute_y_slice(region, x, tol=1e-9):
    """Return feasible y coordinates of R_j at a fixed x."""
    return compute_reachable_region_slice(
        region,
        fixed_axis=0,
        fixed_value=x,
        tol=tol,
    )


def compute_reachable_region_slice(
    region,
    fixed_axis,
    fixed_value,
    tol=1e-9,
):
    """
    Compute an exact axis-aligned slice of a reachable waypoint region.

    :param region:
        Reachable region R_j.
    :type region: ReachableWaypointRegion

    :param fixed_axis:
        0 -> fix x and return the feasible y interval.
        1 -> fix y and return the feasible x interval.
    :type fixed_axis: int

    :param fixed_value:
        Value of the fixed global coordinate.
    :type fixed_value: float

    :return:
        Feasible interval along the other global axis,
        or None if the slice is empty.
    """
    if fixed_axis == 0:
        fixed_bounds = region.x_bounds
        variable_bounds = region.y_bounds
        variable_axis = 1

    elif fixed_axis == 1:
        fixed_bounds = region.y_bounds
        variable_bounds = region.x_bounds
        variable_axis = 0

    else:
        raise ValueError("fixed_axis must be 0 or 1.")

    # Fixed coordinate outside R_j
    if (
        fixed_value < fixed_bounds[0] - tol
        or fixed_value > fixed_bounds[1] + tol
    ):
        return None

    admissible_region = region.admissible_region

    # Aligned transition: A_j = D_j
    if admissible_region.corner_point is None:
        return variable_bounds

    corner = np.asarray(
        admissible_region.corner_point,
        dtype=float,
    )

    local_axes = [
        np.asarray(admissible_region.local_x_axis, dtype=float),
        np.asarray(admissible_region.local_y_axis, dtype=float),
    ]

    # Determine which local coordinate is fixed and which varies.
    if abs(local_axes[0][fixed_axis]) > 1 - tol:
        fixed_local_axis = 0
        variable_local_axis = 1
    else:
        fixed_local_axis = 1
        variable_local_axis = 0

    fixed_coefficient = local_axes[fixed_local_axis][fixed_axis]
    variable_coefficient = local_axes[variable_local_axis][variable_axis]

    fixed_local_value = (
        fixed_coefficient
        * (fixed_value - corner[fixed_axis])
    )

    # Convert the global variable-coordinate bounds to local coordinates.
    local_values = [
        variable_coefficient
        * (variable_bounds[0] - corner[variable_axis]),
        variable_coefficient
        * (variable_bounds[1] - corner[variable_axis]),
    ]

    local_variable_bounds = (
        min(local_values),
        max(local_values),
    )

    R = admissible_region.R
    r = admissible_region.r

    # ---------------------------------------------------------------
    # Fillet-admissibility constraint
    #
    # q_var <= -R
    # OR
    # (q_var + R)^2 + (q_fixed + R)^2 <= (R-r)^2
    # ---------------------------------------------------------------

    if fixed_local_value <= -R + tol:
        feasible_local_bounds = local_variable_bounds

    else:
        radius = R - r
        offset = fixed_local_value + R

        if offset <= radius + tol:
            reach = sqrt(
                max(
                    0.0,
                    radius**2 - offset**2,
                )
            )

            upper = -R + reach

        else:
            upper = -R

        feasible_local_bounds = intersect_intervals(
            local_variable_bounds,
            (-np.inf, upper),
            tol=tol,
        )

        if feasible_local_bounds is None:
            return None

    # Convert back to the global variable coordinate.
    global_values = [
        corner[variable_axis]
        + variable_coefficient * feasible_local_bounds[0],
        corner[variable_axis]
        + variable_coefficient * feasible_local_bounds[1],
    ]

    return (
        min(global_values),
        max(global_values),
    )


def compute_reachable_region(
    admissible_region,
    previous_region,
    passage_direction,
    R,
    tol=1e-9,
):
    """
    Propagate the reachable waypoint region through one corridor.

    :param admissible_region:
        Current local admissible region A_j.
    :type admissible_region: AdmissibleWaypointRegion

    :param previous_region:
        Previous reachable region R_{j-1}.
    :type previous_region: ReachableWaypointRegion

    :param passage_direction:
        Traversal direction through the corridor connecting
        R_{j-1} to A_j.
    :type passage_direction: np.ndarray

    :param R:
        Minimum turning radius.
    :type R: float

    :return:
        Reachable region R_j, or None if the propagation is infeasible.
    :rtype: ReachableWaypointRegion or None
    """
    x_prev_min, x_prev_max = previous_region.x_bounds
    y_prev_min, y_prev_max = previous_region.y_bounds

    x_constraint = None
    y_constraint = None

    # ---------------------------------------------------------------
    # Right
    # ---------------------------------------------------------------
    if np.array_equal(passage_direction, RIGHT):

        x_constraint = (
            x_prev_min + 2 * R,
            np.inf,
        )

        y_constraint = (
            y_prev_min,
            y_prev_max,
        )

    # ---------------------------------------------------------------
    # Left
    # ---------------------------------------------------------------
    elif np.array_equal(passage_direction, LEFT):

        x_constraint = (
            -np.inf,
            x_prev_max - 2 * R,
        )

        y_constraint = (
            y_prev_min,
            y_prev_max,
        )

    # ---------------------------------------------------------------
    # Up
    # ---------------------------------------------------------------
    elif np.array_equal(passage_direction, UP):

        x_constraint = (
            x_prev_min,
            x_prev_max,
        )

        y_constraint = (
            y_prev_min + 2 * R,
            np.inf,
        )

    # ---------------------------------------------------------------
    # Down
    # ---------------------------------------------------------------
    elif np.array_equal(passage_direction, DOWN):

        x_constraint = (
            x_prev_min,
            x_prev_max,
        )

        y_constraint = (
            -np.inf,
            y_prev_max - 2 * R,
        )

    else:
        raise ValueError("Invalid passage direction.")

    # Intersect the propagation constraints with A_j
    bounds = compute_region_bounds(
        admissible_region=admissible_region,
        x_constraint=x_constraint,
        y_constraint=y_constraint,
        tol=tol,
    )

    if bounds is None:
        return None

    x_bounds, y_bounds = bounds

    return ReachableWaypointRegion(
        admissible_region=admissible_region,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
    )


def compute_region_bounds(
    admissible_region,
    x_constraint=None,
    y_constraint=None,
    tol=1e-9,
):
    """
    Compute the global coordinate bounds of an admissible waypoint
    region after imposing additional axis-aligned constraints.

    :param admissible_region: local admissible region A_j
    :type admissible_region: AdmissibleWaypointRegion

    :param x_constraint: optional global x interval
    :type x_constraint: tuple or None

    :param y_constraint: optional global y interval
    :type y_constraint: tuple or None

    :return: (x_bounds, y_bounds), or None if the resulting region is empty
    """
    safe_x_bounds, safe_y_bounds = admissible_region.safe_overlap

    # ---------------------------------------------------------------
    # Intersect D_j with the additional propagation constraints
    # ---------------------------------------------------------------
    if x_constraint is None:
        x_constraint = safe_x_bounds

    if y_constraint is None:
        y_constraint = safe_y_bounds

    x_bounds = intersect_intervals(
        safe_x_bounds,
        x_constraint,
        tol=tol,
    )

    y_bounds = intersect_intervals(
        safe_y_bounds,
        y_constraint,
        tol=tol,
    )

    if x_bounds is None or y_bounds is None:
        return None

    # ---------------------------------------------------------------
    # Aligned transition: A_j = D_j
    # ---------------------------------------------------------------
    if admissible_region.corner_point is None:
        return x_bounds, y_bounds

    # ---------------------------------------------------------------
    # Transform the clipped rectangle to the local corner frame
    # ---------------------------------------------------------------
    corner_point = np.asarray(
        admissible_region.corner_point,
        dtype=float,
    )

    local_x_axis = np.asarray(
        admissible_region.local_x_axis,
        dtype=float,
    )

    local_y_axis = np.asarray(
        admissible_region.local_y_axis,
        dtype=float,
    )

    clipped_corners = [
        np.array([x_bounds[0], y_bounds[0]]),
        np.array([x_bounds[0], y_bounds[1]]),
        np.array([x_bounds[1], y_bounds[0]]),
        np.array([x_bounds[1], y_bounds[1]]),
    ]

    local_coordinates = []

    for point in clipped_corners:
        delta = point - corner_point

        local_coordinates.append([
            np.dot(delta, local_x_axis),
            np.dot(delta, local_y_axis),
        ])

    u_bounds = (
        min(point[0] for point in local_coordinates),
        max(point[0] for point in local_coordinates),
    )

    v_bounds = (
        min(point[1] for point in local_coordinates),
        max(point[1] for point in local_coordinates),
    )

    # ---------------------------------------------------------------
    # A_j is the union of three locally admissible pieces:
    #
    #   u <= -R
    #   v <= -R
    #   disk centered at (-R, -R), radius R-r
    # ---------------------------------------------------------------
    components = []

    R = admissible_region.R
    r = admissible_region.r

    # u <= -R
    component_u_bounds = intersect_intervals(
        u_bounds,
        (-np.inf, -R),
        tol=tol,
    )

    if component_u_bounds is not None:
        components.append(
            (component_u_bounds, v_bounds)
        )

    # v <= -R
    component_v_bounds = intersect_intervals(
        v_bounds,
        (-np.inf, -R),
        tol=tol,
    )

    if component_v_bounds is not None:
        components.append(
            (u_bounds, component_v_bounds)
        )

    # Circular part
    disk_bounds = compute_disk_rectangle_bounds(
        u_bounds=u_bounds,
        v_bounds=v_bounds,
        disk_center=(-R, -R),
        radius=R - r,
        tol=tol,
    )

    if disk_bounds is not None:
        components.append(disk_bounds)

    # Nothing remains after the corner cut
    if not components:
        return None

    # ---------------------------------------------------------------
    # Bounds of the union
    # ---------------------------------------------------------------
    reachable_u_bounds = (
        min(component[0][0] for component in components),
        max(component[0][1] for component in components),
    )

    reachable_v_bounds = (
        min(component[1][0] for component in components),
        max(component[1][1] for component in components),
    )

    return local_bounds_to_global_bounds(
        corner_point=corner_point,
        local_x_axis=local_x_axis,
        local_y_axis=local_y_axis,
        local_x_bounds=reachable_u_bounds,
        local_y_bounds=reachable_v_bounds,
        tol=tol,
    )


def local_bounds_to_global_bounds(
    corner_point,
    local_x_axis,
    local_y_axis,
    local_x_bounds,
    local_y_bounds,
    tol=1e-9,
):
    """
    Convert local coordinate bounds to global x/y bounds.

    Assumes the local axes are axis aligned.
    """
    corner_point = np.asarray(corner_point, dtype=float)
    local_x_axis = np.asarray(local_x_axis, dtype=float)
    local_y_axis = np.asarray(local_y_axis, dtype=float)

    local_axes = [
        (local_x_axis, local_x_bounds),
        (local_y_axis, local_y_bounds),
    ]

    global_bounds = []

    for global_axis in range(2):
        candidates = []

        for axis, bounds in local_axes:
            coefficient = axis[global_axis]

            if abs(coefficient) <= tol:
                continue

            if abs(abs(coefficient) - 1.0) > tol:
                raise ValueError(
                    "Local admissible-region axes must be axis aligned."
                )

            values = [
                corner_point[global_axis] + coefficient * bounds[0],
                corner_point[global_axis] + coefficient * bounds[1],
            ]

            candidates.append(
                (min(values), max(values))
            )

        if len(candidates) != 1:
            raise ValueError(
                "Invalid local admissible-region frame."
            )

        global_bounds.append(candidates[0])

    return global_bounds[0], global_bounds[1]


def intersect_intervals(interval1, interval2, tol=1e-9):
    """
    Compute the intersection of two closed intervals.

    :return: intersected interval, or None if empty
    """
    lower = max(interval1[0], interval2[0])
    upper = min(interval1[1], interval2[1])

    if lower > upper + tol:
        return None

    # Collapse tiny numerical inversions
    if lower > upper:
        midpoint = 0.5 * (lower + upper)
        lower = midpoint
        upper = midpoint

    return lower, upper


def compute_disk_rectangle_bounds(
    u_bounds,
    v_bounds,
    disk_center,
    radius,
    tol=1e-9,
):
    """
    Compute coordinate bounds of the intersection between an
    axis-aligned rectangle and a disk.

    Everything is expressed in the local admissible-region frame.

    :return: (u_bounds, v_bounds), or None if the intersection is empty
    """
    if radius < -tol:
        return None

    radius = max(0.0, radius)

    u_min, u_max = u_bounds
    v_min, v_max = v_bounds
    u_center, v_center = disk_center

    # Closest point of the rectangle to the disk center
    u_closest = min(max(u_center, u_min), u_max)
    v_closest = min(max(v_center, v_min), v_max)

    distance = sqrt(
        (u_closest - u_center) ** 2
        + (v_closest - v_center) ** 2
    )

    if distance > radius + tol:
        return None

    # ---------------------------------------------------------------
    # Extreme u coordinates
    # ---------------------------------------------------------------
    if v_center < v_min:
        distance_v = v_min - v_center
    elif v_center > v_max:
        distance_v = v_center - v_max
    else:
        distance_v = 0.0

    u_reach = sqrt(
        max(0.0, radius**2 - distance_v**2)
    )

    disk_u_bounds = (
        u_center - u_reach,
        u_center + u_reach,
    )

    new_u_bounds = intersect_intervals(
        u_bounds,
        disk_u_bounds,
        tol=tol,
    )

    # ---------------------------------------------------------------
    # Extreme v coordinates
    # ---------------------------------------------------------------
    if u_center < u_min:
        distance_u = u_min - u_center
    elif u_center > u_max:
        distance_u = u_center - u_max
    else:
        distance_u = 0.0

    v_reach = sqrt(
        max(0.0, radius**2 - distance_u**2)
    )

    disk_v_bounds = (
        v_center - v_reach,
        v_center + v_reach,
    )

    new_v_bounds = intersect_intervals(
        v_bounds,
        disk_v_bounds,
        tol=tol,
    )

    if new_u_bounds is None or new_v_bounds is None:
        return None

    return new_u_bounds, new_v_bounds



def contains(self, point, tol=1e-9):
    x, y = point
    x_min, x_max = self.x_bounds
    y_min, y_max = self.y_bounds

    return (
        x_min - tol <= x <= x_max + tol
        and y_min - tol <= y <= y_max + tol
        and self.admissible_region.contains(point, tol=tol)
    )


def compute_admissible_region_local_axes(
    direction_in,
    direction_out,
    turn_direction,
    tol=1e-9,
):
    """
    Compute the local frame used to describe the fillet-admissible
    waypoint region at a turning corridor transition.

    The local axes are directed away from the interiors of the two
    adjacent corridors.

    :param direction_in: traversal direction of the incoming corridor
    :type direction_in: array-like

    :param direction_out: traversal direction of the outgoing corridor
    :type direction_out: array-like

    :param turn_direction: 1 for left turn, -1 for right turn
    :type turn_direction: int

    :param tol: numerical tolerance
    :type tol: float

    :return: local x- and y-axis unit vectors
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    direction_in = np.asarray(direction_in, dtype=float)
    direction_out = np.asarray(direction_out, dtype=float)

    if turn_direction not in (-1, 1):
        raise ValueError(
            "Local admissible-region axes are defined only for turning transitions."
        )

    # The directions must be orthogonal.
    if abs(np.dot(direction_in, direction_out)) > tol:
        raise ValueError(
            "Incoming and outgoing corridor directions must be orthogonal."
        )

    # Check consistency with the prescribed turn direction.
    computed_turn = compute_turn_direction(
        direction_in,
        direction_out,
    )

    if computed_turn != turn_direction:
        raise ValueError(
            "turn_direction is inconsistent with the corridor directions."
        )

    local_x_axis = -direction_in
    local_y_axis = direction_out

    return local_x_axis, local_y_axis


def compute_candidate_corner_points(
    intersection_points,
    corridor_overlap,
    direction_in,
    direction_out,
    turn_direction,
):
    """
    Select the candidate corner points associated with one corridor transition.

    For a turning transition, only the corner consistent with the prescribed
    turn direction is retained.

    For an aligned transition, both the left and right candidate corners
    are retained.

    :return:
        Dictionary mapping possible turn directions to corner points:
            {1: left_corner}
            {-1: right_corner}
            {1: left_corner, -1: right_corner}
        Returns None if the required corner points cannot be identified uniquely.
    :rtype: dict or None
    """
    overlap_midpoint = rectangle_midpoint(corridor_overlap)

    # ---------------------------------------------------------------
    # Turning transition
    # ---------------------------------------------------------------
    if turn_direction != 0:
        candidates = []

        for point in intersection_points:
            point = np.asarray(point, dtype=float)
            vector = point - overlap_midpoint

            side_in = compute_turn_direction(
                direction_in,
                vector,
            )

            side_out = compute_turn_direction(
                direction_out,
                vector,
            )

            # The concave corner must lie on the turning side of both
            # the incoming and outgoing directions.
            if (
                side_in == turn_direction
                and side_out == turn_direction
            ):
                candidates.append(point)

        if len(candidates) != 1:
            return None

        return {
            turn_direction: candidates[0]
        }

   # ---------------------------------------------------------------
    # Aligned transition
    # ---------------------------------------------------------------
    left_candidates = []
    right_candidates = []

    for point in intersection_points:

        point = np.asarray(point, dtype=float)
        vector = point - overlap_midpoint

        side = compute_turn_direction(
            direction_in,
            vector,
        )

        if side == 1:
            left_candidates.append(point)

        elif side == -1:
            right_candidates.append(point)


    # More than one candidate on the same side is ambiguous
    if len(left_candidates) > 1 or len(right_candidates) > 1:
        return None


    corners = {}

    if len(left_candidates) == 1:
        corners[1] = left_candidates[0]

    if len(right_candidates) == 1:
        corners[-1] = right_candidates[0]

    return corners


def compute_corridor_turn_direction(direction_in, direction_out, tol=1e-9):
    """
    Compute the turn direction between two consecutive corridor
    traversal directions.

    :param direction_in: traversal direction of the current corridor
    :type direction_in: array-like

    :param direction_out: traversal direction of the following corridor
    :type direction_out: array-like

    :param tol: numerical tolerance
    :type tol: float

    :return:
        1 for left turn,
        -1 for right turn,
        0 for aligned corridors,
        None for an opposite-direction reversal.
    """
    dot = (
        direction_in[0] * direction_out[0]
        + direction_in[1] * direction_out[1]
    )

    # Opposite directions: 180-degree reversal
    if dot < -tol:
        return None

    return compute_turn_direction(direction_in, direction_out)


def compute_boundary_corridor_direction(
    corridor,
    safe_overlap,
    initial=True,
    tol=1e-9,
):
    """
    Compute the traversal direction of the first or last corridor.

    For the first corridor, the direction points from the corridor center
    toward the first safe overlap.

    For the last corridor, the direction points from the last safe overlap
    toward the corridor center.

    :return: unit direction vector, or None if the direction is ambiguous
    """
    overlap_midpoint = rectangle_midpoint(safe_overlap)
    corridor_center = np.asarray(corridor.center, dtype=float)

    if initial:
        reference_vector = overlap_midpoint - corridor_center
    else:
        reference_vector = corridor_center - overlap_midpoint

    corridor_axis = np.asarray(corridor.unit_vector, dtype=float)

    projection = np.dot(reference_vector, corridor_axis)

    if abs(projection) <= tol:
        return None

    if projection > 0:
        return corridor_axis

    return -corridor_axis


def rectangle_midpoint(rectangle):
    """
    Compute the midpoint of an axis-aligned rectangle represented as
    ((x_min, x_max), (y_min, y_max)).
    """
    (x_min, x_max), (y_min, y_max) = rectangle

    return np.array([
        0.5 * (x_min + x_max),
        0.5 * (y_min + y_max),
    ])


def compute_intersection_points_two_corridors(corridor1, corridor2, tol=1e-9):
    """
    Compute the intersection points between the boundaries of two corridors.

    :param corridor1: first corridor
    :type corridor1: CorridorWorld
    :param corridor2: second corridor
    :type corridor2: CorridorWorld
    :param tol: numerical tolerance
    :type tol: float

    :return: boundary intersection points
    :rtype: list[np.ndarray]
    """
    intersection_points = []

    for edge_index_1 in range(4):
        A1, A2 = corridor1.get_edge_segment(edge_index_1)

        for edge_index_2 in range(4):
            B1, B2 = corridor2.get_edge_segment(edge_index_2)

            point, intersects = compute_intersection_two_segments(
                A1, A2, B1, B2, tol=tol
            )

            if not intersects:
                continue

            point = np.asarray(point, dtype=float)

            # Avoid duplicates, e.g. when the intersection is a corridor corner
            if not any(
                np.linalg.norm(point - existing_point) <= tol
                for existing_point in intersection_points
            ):
                intersection_points.append(point)

    return intersection_points



def compute_overlap_two_axis_aligned_corridors(corridor1, corridor2, tol=1e-9):
    """
    Compute the rectangular overlap between two axis-aligned corridors.

    The overlap is represented as the Cartesian product of an x interval
    and a y interval.

    :param corridor1: first corridor
    :type corridor1: CorridorWorld
    :param corridor2: second corridor
    :type corridor2: CorridorWorld
    :param tol: numerical tolerance
    :type tol: float

    :return: ((x_min, x_max), (y_min, y_max)), or None if the corridors
             do not overlap
    """
    corners1 = corridor1.corners
    corners2 = corridor2.corners

    x1_min = min(point[0] for point in corners1)
    x1_max = max(point[0] for point in corners1)
    y1_min = min(point[1] for point in corners1)
    y1_max = max(point[1] for point in corners1)

    x2_min = min(point[0] for point in corners2)
    x2_max = max(point[0] for point in corners2)
    y2_min = min(point[1] for point in corners2)
    y2_max = max(point[1] for point in corners2)

    x_min = max(x1_min, x2_min)
    x_max = min(x1_max, x2_max)

    y_min = max(y1_min, y2_min)
    y_max = min(y1_max, y2_max)

    # No overlap
    if x_min > x_max + tol or y_min > y_max + tol:
        return None

    # Remove tiny numerical inconsistencies in degenerate overlaps
    if x_min > x_max:
        x_mid = 0.5 * (x_min + x_max)
        x_min = x_mid
        x_max = x_mid

    if y_min > y_max:
        y_mid = 0.5 * (y_min + y_max)
        y_min = y_mid
        y_max = y_mid

    return (x_min, x_max), (y_min, y_max)


def compute_safe_overlap(corridor_overlap, r, tol=1e-9):
    """
    Compute the safe overlap region obtained by eroding an axis-aligned
    rectangular corridor overlap by a disk of radius r.

    :param corridor_overlap:
        Rectangle represented as ((x_min, x_max), (y_min, y_max)).
    :type corridor_overlap: tuple

    :param r:
        Radius of the circular vehicle footprint.
    :type r: float

    :param tol:
        Numerical tolerance.
    :type tol: float

    :return:
        Safe overlap represented as
        ((x_safe_min, x_safe_max), (y_safe_min, y_safe_max)),
        or None if the safe overlap is empty.
    :rtype: tuple or None
    """
    (x_min, x_max), (y_min, y_max) = corridor_overlap

    x_safe_min = x_min + r
    x_safe_max = x_max - r

    y_safe_min = y_min + r
    y_safe_max = y_max - r

    # Empty safe overlap
    if (
        x_safe_min > x_safe_max + tol
        or y_safe_min > y_safe_max + tol
    ):
        return None

    # Collapse very small numerical inversions to a single coordinate
    if x_safe_min > x_safe_max:
        x_mid = 0.5 * (x_safe_min + x_safe_max)
        x_safe_min = x_mid
        x_safe_max = x_mid

    if y_safe_min > y_safe_max:
        y_mid = 0.5 * (y_safe_min + y_safe_max)
        y_safe_min = y_mid
        y_safe_max = y_mid

    return (
        (x_safe_min, x_safe_max),
        (y_safe_min, y_safe_max),
    )


def intervals_overlap(interval1, interval2, tol=1e-9):
    """
    Check whether two closed intervals overlap.

    :param interval1: first interval (min, max)
    :param interval2: second interval (min, max)
    :param tol: numerical tolerance

    :return: True if the intervals overlap
    :rtype: bool
    """
    min1, max1 = interval1
    min2, max2 = interval2

    return max(min1, min2) <= min(max1, max2) + tol


def compute_passage_direction(safe_overlap1, safe_overlap2, tol=1e-9):
    """
    Compute the orthogonal passage direction from one safe overlap
    to the next.

    :param safe_overlap1:
        First safe overlap ((x_min, x_max), (y_min, y_max)).
    :param safe_overlap2:
        Second safe overlap ((x_min, x_max), (y_min, y_max)).
    :param tol:
        Numerical tolerance.

    :return:
        Direction vector [1, 0], [-1, 0], [0, 1], or [0, -1].
        Returns None if no unique orthogonal passage exists.
    :rtype: numpy.ndarray or None
    """
    X1, Y1 = safe_overlap1
    X2, Y2 = safe_overlap2

    x_overlap = intervals_overlap(X1, X2, tol=tol)
    y_overlap = intervals_overlap(Y1, Y2, tol=tol)

    # Both overlap:
    # the safe overlap regions intersect, so the passage is ambiguous
    if x_overlap and y_overlap:
        return None

    # Neither overlaps:
    # no horizontal or vertical segment connects the two regions
    if not x_overlap and not y_overlap:
        return None

    # Horizontal passage
    if y_overlap:
        x1_min, x1_max = X1
        x2_min, x2_max = X2

        if x1_max < x2_min - tol:
            return np.array([1, 0])       # right

        if x2_max < x1_min - tol:
            return np.array([-1, 0])      # left

    # Vertical passage
    if x_overlap:
        y1_min, y1_max = Y1
        y2_min, y2_max = Y2

        if y1_max < y2_min - tol:
            return np.array([0, 1])       # up

        if y2_max < y1_min - tol:
            return np.array([0, -1])      # down

    return None


def validate_baseline_corridor_sequence(
    corridor_list,
    bicycle,
    tol=1e-9,
):
    """
    Check only the geometric assumptions imposed on the corridor
    sequence before running the baseline construction.

    This function does NOT test baseline feasibility.

    In particular, it does not construct or check:
        - candidate corner points,
        - admissible waypoint regions A_j,
        - reachable regions R_j,
        - the terminal condition R_{n-1} != empty,
        - baseline waypoints or fillets.

    It does infer all corridor traversal directions and rejects a 180-degree
    reversal between any adjacent pair.

    Returns
    -------
    bool
        True if the corridor sequence satisfies the assumptions
        required before the baseline construction is attempted.
    """

    n = len(corridor_list)

    # ---------------------------------------------------------------
    # 0. Minimum number of corridors
    # ---------------------------------------------------------------
    if n < 2:
        return False

    r = bicycle.width / 2

    # ---------------------------------------------------------------
    # 1. Corridor overlaps I_j
    # ---------------------------------------------------------------
    corridor_overlaps = []

    for j in range(n - 1):

        overlap = compute_overlap_two_axis_aligned_corridors(
            corridor_list[j],
            corridor_list[j + 1],
            tol=tol,
        )

        if overlap is None:
            return False

        corridor_overlaps.append(overlap)

    # ---------------------------------------------------------------
    # 2. Assumption: nonempty safe overlaps D_j
    # ---------------------------------------------------------------
    safe_overlaps = []

    for overlap in corridor_overlaps:

        safe_overlap = compute_safe_overlap(
            overlap,
            r,
            tol=tol,
        )

        if safe_overlap is None:
            return False

        safe_overlaps.append(safe_overlap)

    # ---------------------------------------------------------------
    # 3. Assumptions on consecutive safe overlaps:
    #
    #    - D_{j-1} and D_j must be disjoint;
    #    - an orthogonal passage must exist between them.
    #
    # compute_passage_direction() already checks both:
    #
    #    x overlap and y overlap -> regions intersect -> invalid
    #    neither overlaps       -> no orthogonal passage -> invalid
    #
    # Exactly one overlap gives a unique horizontal/vertical passage.
    # ---------------------------------------------------------------
    corridor_directions = [None] * n

    for j in range(1, n - 1):

        passage_direction = compute_passage_direction(
            safe_overlaps[j - 1],
            safe_overlaps[j],
            tol=tol,
        )

        if passage_direction is None:
            return False

        corridor_directions[j] = passage_direction

    # ---------------------------------------------------------------
    # 4. Assumption: no 180-degree reversal in the inferred traversal.
    #
    # Include the directions of the first and last corridors so reversals at
    # either boundary transition are rejected as well.
    # ---------------------------------------------------------------
    corridor_directions[0] = compute_boundary_corridor_direction(
        corridor_list[0],
        safe_overlaps[0],
        initial=True,
        tol=tol,
    )
    corridor_directions[-1] = compute_boundary_corridor_direction(
        corridor_list[-1],
        safe_overlaps[-1],
        initial=False,
        tol=tol,
    )

    if corridor_directions[0] is None or corridor_directions[-1] is None:
        return False

    for j in range(n - 1):
        if compute_corridor_turn_direction(
            corridor_directions[j],
            corridor_directions[j + 1],
            tol=tol,
        ) is None:
            return False

    return True
