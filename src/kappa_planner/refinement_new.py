from dataclasses import dataclass
from math import sqrt

import numpy as np

from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class RefinementCircle:
    center: np.ndarray
    radius: float
    turn_direction: int

    corner_point: np.ndarray
    local_x_axis: np.ndarray
    local_y_axis: np.ndarray

    local_coordinates: tuple
    effective_dimensions: tuple | None

    placement_rule: str


@dataclass
class CircleTangent:
    start_circle_index: int
    end_circle_index: int

    start_point: np.ndarray
    end_point: np.ndarray

    start_heading: float
    end_heading: float


@dataclass
class StraightPassageGroup:
    """
    Group associated with one maximal run of aligned transitions.

    The group contains the zero run itself and, when available, the
    genuine-turn transition immediately before and immediately after it.

    The two adjacent genuine-turn circles are kept at their validated
    baseline positions.
    """

    start_index: int
    end_index: int

    zero_start_index: int
    zero_end_index: int

    left_circle_index: int | None
    right_circle_index: int | None


# ===================================================================
# Main refinement function
# ===================================================================

def refine_bicycle_baseline(
    corridor_list,
    bicycle,
    baseline,
):
    """
    Refine a validated bicycle baseline through intermediate circles.

    The refinement applies the following rules:

        1. Find maximal straight-passage groups, i.e. maximal runs of
           transitions with tau = 0.

        2. Keep the genuine-turn circles immediately adjacent to each
           straight-passage group at their validated baseline positions.

        3. Independently position all remaining genuine-turn circles
           through the four local heuristic placement rules, with the
           validated baseline circle as fallback.

        4. Build a collision-free ordered tangent chain. If a required
           tangent does not exist or leaves the eroded corridor union,
           restore both endpoint circles to their baseline positions and
           recheck the preceding connection.

        5. Simplify the tangent chain. Whenever two accepted tangent
           segments intersect, try to skip one or more intermediate
           circles through the longest collision-free direct tangent
           available inside the affected block. If no such skip exists,
           restore the affected active circles to their baseline positions.

    No circle is introduced at an aligned transition.

    :return:
        (circle_groups, straight_passage_groups, active_circle_indices,
        tangents), or None if the refinement cannot be repaired locally.
    """
    n = len(corridor_list)

    R = bicycle.max_radius
    r = bicycle.width / 2

    # ---------------------------------------------------------------
    # 1. Find straight-passage groups
    # ---------------------------------------------------------------
    straight_passage_groups = compute_straight_passage_groups(
        turn_directions=baseline.turn_directions,
    )

    # ---------------------------------------------------------------
    # 2. Genuine-turn circles adjacent to a straight passage
    #    are kept at their validated baseline positions.
    # ---------------------------------------------------------------
    baseline_circle_indices = set()

    for group in straight_passage_groups:

        if group.left_circle_index is not None:
            baseline_circle_indices.add(
                group.left_circle_index
            )

        if group.right_circle_index is not None:
            baseline_circle_indices.add(
                group.right_circle_index
            )

    # ---------------------------------------------------------------
    # 3. Position all genuine-turn circles
    # ---------------------------------------------------------------
    circle_groups = [
        {}
        for _ in range(n - 1)
    ]

    for j in range(n - 1):

        tau = baseline.turn_directions[j]

        # No refinement circle at an aligned transition.
        if tau == 0:
            continue

        # Circle adjacent to a straight-passage group:
        # keep the validated baseline circle.
        if j in baseline_circle_indices:

            circle = compute_baseline_turn_circle(
                baseline=baseline,
                transition_index=j,
                R=R,
                placement_rule="baseline_straight_passage",
            )

        # Otherwise use the independent local heuristic placement.
        else:

            circle = compute_independent_turn_circle(
                baseline=baseline,
                transition_index=j,
                R=R,
                r=r,
            )

        if circle is None:
            return None

        circle_groups[j][tau] = circle

    # Cache the eroded local corridor unions because the same
    # transition pair can be checked repeatedly during repair and
    # simplification.
    safe_union_cache = {}

    # ---------------------------------------------------------------
    # 4. Build the tangent chain and repair incompatible pairs
    # ---------------------------------------------------------------
    active_circle_indices = get_active_circle_indices(
        circle_groups
    )

    tangents = build_and_repair_tangent_chain(
        circle_groups=circle_groups,
        corridor_list=corridor_list,
        baseline=baseline,
        R=R,
        r=r,
        active_circle_indices=active_circle_indices,
        safe_union_cache=safe_union_cache,
    )

    if tangents is None:
        return None

    # ---------------------------------------------------------------
    # 5. Simplify self-intersecting tangent portions by skipping
    #    intermediate circles whenever a safe direct connection exists.
    # ---------------------------------------------------------------
    simplification_result = simplify_tangent_chain(
        circle_groups=circle_groups,
        corridor_list=corridor_list,
        baseline=baseline,
        R=R,
        r=r,
        active_circle_indices=active_circle_indices,
        tangents=tangents,
        safe_union_cache=safe_union_cache,
    )

    if simplification_result is None:
        return None

    (
        active_circle_indices,
        tangents,
    ) = simplification_result

    # ---------------------------------------------------------------
    # 6. Assemble the refined primitive chain
    # ---------------------------------------------------------------
    # TODO: validate/construct the circular arcs between consecutive
    # incoming and outgoing tangency points, then assemble arcs and
    # stored tangent segments.

    return (
        circle_groups,
        straight_passage_groups,
        active_circle_indices,
        tangents,
    )


# ===================================================================
# Straight-passage grouping
# ===================================================================

def compute_straight_passage_groups(
    turn_directions,
):
    """
    Find maximal runs of aligned transitions (tau = 0).

    Each zero run is extended by the genuine turn immediately before it
    and the genuine turn immediately after it, whenever those turns exist.

    Examples
    --------

    [1, 1, 0, 1]

        zero run: [2]
        group:    [1, 2, 3]

    [1, 0, 0, -1]

        zero run: [1, 2]
        group:    [0, 1, 2, 3]

    [0, 0, 1]

        zero run: [0, 1]
        group:    [0, 1, 2]
        left_circle_index = None

    [1, 0, 0]

        zero run: [1, 2]
        group:    [0, 1, 2]
        right_circle_index = None

    Notes
    -----
    Two consecutive straight-passage groups may share one genuine-turn
    transition. This is intentional: that shared turn is adjacent to both
    zero runs and is therefore kept at its baseline position.
    """
    turn_directions = list(turn_directions)

    groups = []

    m = len(turn_directions)

    j = 0

    while j < m:

        # Not the beginning of a zero run.
        if turn_directions[j] != 0:
            j += 1
            continue

        # -----------------------------------------------------------
        # Maximal zero run
        # -----------------------------------------------------------
        zero_start = j

        while (
            j + 1 < m
            and turn_directions[j + 1] == 0
        ):
            j += 1

        zero_end = j

        # -----------------------------------------------------------
        # Genuine turn immediately before the zero run
        # -----------------------------------------------------------
        left_circle_index = None

        if zero_start > 0:

            candidate = zero_start - 1

            if turn_directions[candidate] != 0:
                left_circle_index = candidate

        # -----------------------------------------------------------
        # Genuine turn immediately after the zero run
        # -----------------------------------------------------------
        right_circle_index = None

        if zero_end + 1 < m:

            candidate = zero_end + 1

            if turn_directions[candidate] != 0:
                right_circle_index = candidate

        # -----------------------------------------------------------
        # Full group limits
        # -----------------------------------------------------------
        if left_circle_index is not None:
            start_index = left_circle_index
        else:
            start_index = zero_start

        if right_circle_index is not None:
            end_index = right_circle_index
        else:
            end_index = zero_end

        groups.append(
            StraightPassageGroup(
                start_index=start_index,
                end_index=end_index,
                zero_start_index=zero_start,
                zero_end_index=zero_end,
                left_circle_index=left_circle_index,
                right_circle_index=right_circle_index,
            )
        )

        j += 1

    return groups


# ===================================================================
# Baseline genuine-turn circle
# ===================================================================

def compute_baseline_turn_circle(
    baseline,
    transition_index,
    R,
    placement_rule="baseline",
    tol=1e-9,
):
    """
    Construct a RefinementCircle directly from the validated baseline
    fillet at a genuine turn.
    """
    j = transition_index

    tau = baseline.turn_directions[j]

    if tau not in (-1, 1):
        raise ValueError(
            "A baseline turning circle exists only at a genuine turn."
        )

    baseline_fillet = baseline.fillets[j]

    if baseline_fillet is None:
        return None

    corner_point = np.asarray(
        baseline.candidate_corner_points[j][tau],
        dtype=float,
    )

    admissible_region = baseline.admissible_regions[j]

    local_x_axis = np.asarray(
        admissible_region.local_x_axis,
        dtype=float,
    )

    local_y_axis = np.asarray(
        admissible_region.local_y_axis,
        dtype=float,
    )

    center = np.asarray(
        baseline_fillet.center,
        dtype=float,
    )

    delta = center - corner_point

    x_local = np.dot(
        delta,
        local_x_axis,
    )

    y_local = np.dot(
        delta,
        local_y_axis,
    )

    dimensions = compute_effective_junction_dimensions(
        corridor_overlap=baseline.corridor_overlaps[j],
        corner_point=corner_point,
        local_x_axis=local_x_axis,
        local_y_axis=local_y_axis,
        tol=tol,
    )

    return RefinementCircle(
        center=center,
        radius=R,
        turn_direction=tau,
        corner_point=corner_point,
        local_x_axis=local_x_axis,
        local_y_axis=local_y_axis,
        local_coordinates=(
            x_local,
            y_local,
        ),
        effective_dimensions=dimensions,
        placement_rule=placement_rule,
    )


# ===================================================================
# Independent genuine-turn circle positioning
# ===================================================================

def compute_independent_turn_circle(
    baseline,
    transition_index,
    R,
    r,
    tol=1e-9,
):
    """
    Position the refinement circle associated with a genuine baseline turn.

    The four local heuristic placement rules are attempted first. If none
    can be applied, the validated baseline fillet circle is retained.
    """
    j = transition_index

    tau = baseline.turn_directions[j]

    if tau not in (-1, 1):
        raise ValueError(
            "This helper is defined only for genuine turns."
        )

    # ---------------------------------------------------------------
    # Geometry already determined by the baseline
    # ---------------------------------------------------------------
    corner_point = np.asarray(
        baseline.candidate_corner_points[j][tau],
        dtype=float,
    )

    admissible_region = baseline.admissible_regions[j]

    local_x_axis = np.asarray(
        admissible_region.local_x_axis,
        dtype=float,
    )

    local_y_axis = np.asarray(
        admissible_region.local_y_axis,
        dtype=float,
    )

    # ---------------------------------------------------------------
    # Effective junction dimensions
    # ---------------------------------------------------------------
    dimensions = compute_effective_junction_dimensions(
        corridor_overlap=baseline.corridor_overlaps[j],
        corner_point=corner_point,
        local_x_axis=local_x_axis,
        local_y_axis=local_y_axis,
        tol=tol,
    )

    local_position = None

    if dimensions is not None:

        d_x, d_y = dimensions

        local_position = compute_independent_circle_local_position(
            d_x=d_x,
            d_y=d_y,
            R=R,
            r=r,
            tol=tol,
        )

    # ---------------------------------------------------------------
    # Four local rules succeeded
    # ---------------------------------------------------------------
    if local_position is not None:

        x_local, y_local, placement_rule = local_position

        center = (
            corner_point
            + x_local * local_x_axis
            + y_local * local_y_axis
        )

        return RefinementCircle(
            center=center,
            radius=R,
            turn_direction=tau,
            corner_point=corner_point,
            local_x_axis=local_x_axis,
            local_y_axis=local_y_axis,
            local_coordinates=(
                x_local,
                y_local,
            ),
            effective_dimensions=dimensions,
            placement_rule=placement_rule,
        )

    # ---------------------------------------------------------------
    # Fallback: validated baseline fillet circle
    # ---------------------------------------------------------------
    return compute_baseline_turn_circle(
        baseline=baseline,
        transition_index=j,
        R=R,
        placement_rule="baseline",
        tol=tol,
    )


# ===================================================================
# Collision-free corridor union for tangent validation
# ===================================================================

def corridor_to_axis_aligned_polygon(
    corridor,
):
    """
    Convert an axis-aligned corridor to a Shapely rectangle.

    The Chapter 5 construction assumes axis-aligned corridors, so the
    rectangle is recovered directly from the extrema of its corner points.
    """
    corners = np.asarray(
        corridor.corners,
        dtype=float,
    )

    x_min = np.min(corners[:, 0])
    x_max = np.max(corners[:, 0])
    y_min = np.min(corners[:, 1])
    y_max = np.max(corners[:, 1])

    return box(
        x_min,
        y_min,
        x_max,
        y_max,
    )


def compute_safe_corridor_union(
    corridor_list,
    r,
    quad_segs=32,
):
    """
    Compute the collision-free region for the robot reference point,

        W_free = (union_j C_j) erosion B_r.

    The corridors are unioned *before* erosion. This is important: it
    preserves the rounded collision-free regions that appear at corridor
    junctions and that would be lost by eroding every corridor separately.

    Shapely represents the circular erosion arcs by a polygonal
    approximation. ``quad_segs`` controls the number of segments per
    quarter circle.
    """
    if len(corridor_list) == 0:
        return None

    corridor_polygons = [
        corridor_to_axis_aligned_polygon(corridor)
        for corridor in corridor_list
    ]

    corridor_union = unary_union(
        corridor_polygons
    )

    if corridor_union.is_empty:
        return None

    if r <= 0.0:
        return corridor_union

    # Shapely >= 2 uses quad_segs. The fallback keeps compatibility
    # with older Shapely versions, where the same parameter is called
    # resolution.
    try:
        safe_union = corridor_union.buffer(
            -r,
            quad_segs=quad_segs,
        )
    except TypeError:
        safe_union = corridor_union.buffer(
            -r,
            resolution=quad_segs,
        )

    if safe_union.is_empty:
        return None

    return safe_union


def segment_inside_safe_corridor_union(
    start_point,
    end_point,
    safe_union,
    tol=1e-9,
):
    """
    Check whether the complete finite segment lies inside the eroded
    corridor union.

    This is stronger than checking only the two tangency points. It
    rejects a tangent whose endpoints are collision-free but whose
    interior leaves W_free.
    """
    if safe_union is None:
        return False

    start_point = np.asarray(
        start_point,
        dtype=float,
    )

    end_point = np.asarray(
        end_point,
        dtype=float,
    )

    if np.linalg.norm(
        end_point - start_point
    ) <= tol:
        geometry = Point(
            start_point[0],
            start_point[1],
        )
    else:
        geometry = LineString([
            tuple(start_point),
            tuple(end_point),
        ])

    # ``covers`` includes the boundary. The tiny positive buffer is only
    # a numerical tolerance and should remain much smaller than all
    # geometric modelling tolerances used by the planner.
    if tol > 0.0:
        safe_union_for_test = safe_union.buffer(
            tol
        )
    else:
        safe_union_for_test = safe_union

    return safe_union_for_test.covers(
        geometry
    )


def get_connection_safe_union(
    corridor_list,
    start_circle_index,
    end_circle_index,
    r,
    cache=None,
):
    """
    Return W_free for the local corridor subsequence traversed by a
    circle-to-circle tangent.

    A circle at transition j is associated with the pair (C_j, C_{j+1}).
    Therefore a tangent from transition i to transition k is checked in
    the union C_i union ... union C_{k+1}.
    """
    key = (
        start_circle_index,
        end_circle_index,
    )

    if cache is not None and key in cache:
        return cache[key]

    local_corridors = corridor_list[
        start_circle_index : end_circle_index + 2
    ]

    safe_union = compute_safe_corridor_union(
        corridor_list=local_corridors,
        r=r,
    )

    if cache is not None:
        cache[key] = safe_union

    return safe_union


# ===================================================================
# Tangent construction and local baseline repair
# ===================================================================

def get_active_circle_indices(
    circle_groups,
):
    """
    Return the transition indices containing genuine-turn circles,
    in trajectory order.
    """
    return [
        j
        for j, group in enumerate(circle_groups)
        if len(group) > 0
    ]


def get_single_circle(
    circle_group,
):
    """Return the single genuine-turn circle stored in a group."""
    if len(circle_group) != 1:
        raise ValueError(
            "Each active circle group must contain exactly one circle."
        )

    return next(
        iter(circle_group.values())
    )


def is_baseline_circle(
    circle,
):
    """Return True if the circle already uses a baseline position."""
    return circle.placement_rule in {
        "baseline",
        "baseline_straight_passage",
        "baseline_repair",
    }


def tangent_length(
    tangent,
):
    """Return the length of the stored finite tangent segment."""
    return np.linalg.norm(
        tangent.end_point
        - tangent.start_point
    )


def compute_correct_circle_tangent(
    circle_1,
    circle_2,
    start_circle_index,
    end_circle_index,
    tol=1e-9,
):
    """
    Compute the tangent compatible with an ordered pair of equal-radius
    turning circles.

    Equal turn directions use the compatible external common tangent.
    Opposite turn directions use the compatible internal common tangent.

    The tangent is oriented from circle_1 toward circle_2.

    :return:
        CircleTangent, or None if the required tangent does not exist
        uniquely in the considered geometry.
    """
    center_1 = np.asarray(
        circle_1.center,
        dtype=float,
    )

    center_2 = np.asarray(
        circle_2.center,
        dtype=float,
    )

    R_1 = float(circle_1.radius)
    R_2 = float(circle_2.radius)

    tau_1 = circle_1.turn_direction
    tau_2 = circle_2.turn_direction

    if tau_1 not in (-1, 1):
        raise ValueError(
            "circle_1 must have turn direction +/-1."
        )

    if tau_2 not in (-1, 1):
        raise ValueError(
            "circle_2 must have turn direction +/-1."
        )

    if abs(R_1 - R_2) > tol:
        raise ValueError(
            "This helper assumes equal-radius circles."
        )

    R = 0.5 * (R_1 + R_2)

    delta = (
        center_2
        - center_1
    )

    center_distance = np.linalg.norm(
        delta
    )

    # Coincident equal-radius circles do not define a unique common
    # tangent. In the present refinement they are therefore repaired
    # through the validated baseline.
    if center_distance <= tol:
        return None

    center_direction = (
        delta
        / center_distance
    )

    # ---------------------------------------------------------------
    # Equal turns: compatible external tangent
    # ---------------------------------------------------------------
    if tau_1 == tau_2:

        tangent_direction = (
            center_direction
        )

    # ---------------------------------------------------------------
    # Opposite turns: compatible internal tangent
    # ---------------------------------------------------------------
    else:

        # A nondegenerate internal tangent requires c >= 2R.
        if center_distance < 2.0 * R - tol:
            return None

        ratio = (
            2.0 * R
            / center_distance
        )

        ratio = np.clip(
            ratio,
            -1.0,
            1.0,
        )

        beta = np.arcsin(
            ratio
        )

        center_angle = np.arctan2(
            center_direction[1],
            center_direction[0],
        )

        # This branch is the one compatible with both the ordering
        # circle_1 -> circle_2 and the turn direction of circle_1.
        tangent_angle = (
            center_angle
            + tau_1 * beta
        )

        tangent_direction = np.array([
            np.cos(tangent_angle),
            np.sin(tangent_angle),
        ])

    # ---------------------------------------------------------------
    # Tangency points
    # ---------------------------------------------------------------
    left_normal = np.array([
        -tangent_direction[1],
        tangent_direction[0],
    ])

    radial_direction_1 = (
        -tau_1
        * left_normal
    )

    radial_direction_2 = (
        -tau_2
        * left_normal
    )

    start_point = (
        center_1
        + R * radial_direction_1
    )

    end_point = (
        center_2
        + R * radial_direction_2
    )

    tangent_vector = (
        end_point
        - start_point
    )

    tangent_segment_length = np.linalg.norm(
        tangent_vector
    )

    # At c = 2R for opposite turns, the internal tangent degenerates
    # to one contact point. Keep the limiting tangent heading.
    if tangent_segment_length > tol:

        actual_direction = (
            tangent_vector
            / tangent_segment_length
        )

        if (
            np.dot(
                actual_direction,
                tangent_direction,
            )
            < 1.0 - 1e-7
        ):
            return None

        tangent_heading = np.arctan2(
            actual_direction[1],
            actual_direction[0],
        )

    else:

        tangent_heading = np.arctan2(
            tangent_direction[1],
            tangent_direction[0],
        )

    return CircleTangent(
        start_circle_index=start_circle_index,
        end_circle_index=end_circle_index,
        start_point=start_point,
        end_point=end_point,
        start_heading=tangent_heading,
        end_heading=tangent_heading,
    )


def compute_safe_circle_tangent_between_indices(
    circle_groups,
    corridor_list,
    start_circle_index,
    end_circle_index,
    r,
    safe_union_cache=None,
    tol=1e-9,
):
    """
    Compute the ordered common tangent between two retained circles and
    accept it only if the complete finite tangent segment lies inside the
    eroded union of the corridor subsequence between the two transitions.

    The circles between the two endpoint circles, if any, are deliberately
    irrelevant to this test: when this helper is used for a skip, those
    intermediate circles are being removed from the active chain.
    """
    if start_circle_index >= end_circle_index:
        raise ValueError(
            "start_circle_index must precede end_circle_index."
        )

    circle_1 = get_single_circle(
        circle_groups[start_circle_index]
    )

    circle_2 = get_single_circle(
        circle_groups[end_circle_index]
    )

    tangent = compute_correct_circle_tangent(
        circle_1=circle_1,
        circle_2=circle_2,
        start_circle_index=start_circle_index,
        end_circle_index=end_circle_index,
        tol=tol,
    )

    if tangent is None:
        return None

    safe_union = get_connection_safe_union(
        corridor_list=corridor_list,
        start_circle_index=start_circle_index,
        end_circle_index=end_circle_index,
        r=r,
        cache=safe_union_cache,
    )

    if not segment_inside_safe_corridor_union(
        start_point=tangent.start_point,
        end_point=tangent.end_point,
        safe_union=safe_union,
        tol=tol,
    ):
        return None

    return tangent


def build_and_repair_tangent_chain(
    circle_groups,
    corridor_list,
    baseline,
    R,
    r,
    active_circle_indices=None,
    safe_union_cache=None,
    tol=1e-9,
):
    """
    Build tangents between consecutive active genuine-turn circles.

    A tangent is accepted only when the correct ordered common tangent
    exists and its complete finite segment is contained in the locally
    relevant eroded corridor union.

    If the connection fails, both endpoint circles are restored to their
    validated baseline positions. The preceding connection is then checked
    again because the first endpoint circle may have moved.
    """
    if active_circle_indices is None:
        circle_indices = get_active_circle_indices(
            circle_groups
        )
    else:
        circle_indices = list(
            active_circle_indices
        )

    if len(circle_indices) <= 1:
        return []

    if safe_union_cache is None:
        safe_union_cache = {}

    tangents = [
        None
        for _ in range(len(circle_indices) - 1)
    ]

    k = 0

    while k < len(circle_indices) - 1:

        start_index = circle_indices[k]
        end_index = circle_indices[k + 1]

        tangent = compute_safe_circle_tangent_between_indices(
            circle_groups=circle_groups,
            corridor_list=corridor_list,
            start_circle_index=start_index,
            end_circle_index=end_index,
            r=r,
            safe_union_cache=safe_union_cache,
            tol=tol,
        )

        if tangent is not None:
            tangents[k] = tangent
            k += 1
            continue

        circle_1 = get_single_circle(
            circle_groups[start_index]
        )

        circle_2 = get_single_circle(
            circle_groups[end_index]
        )

        # Both circles are already at baseline: this local repair has
        # no additional action available.
        if (
            is_baseline_circle(circle_1)
            and is_baseline_circle(circle_2)
        ):
            return None

        repaired_circle_1 = compute_baseline_turn_circle(
            baseline=baseline,
            transition_index=start_index,
            R=R,
            placement_rule="baseline_repair",
            tol=tol,
        )

        repaired_circle_2 = compute_baseline_turn_circle(
            baseline=baseline,
            transition_index=end_index,
            R=R,
            placement_rule="baseline_repair",
            tol=tol,
        )

        if (
            repaired_circle_1 is None
            or repaired_circle_2 is None
        ):
            return None

        tau_1 = baseline.turn_directions[start_index]
        tau_2 = baseline.turn_directions[end_index]

        circle_groups[start_index] = {
            tau_1: repaired_circle_1
        }

        circle_groups[end_index] = {
            tau_2: repaired_circle_2
        }

        tangents[k] = None

        if k > 0:
            tangents[k - 1] = None

        k = max(
            0,
            k - 1,
        )

    return tangents


def tangent_segments_have_problematic_intersection(
    tangent_1,
    tangent_2,
    adjacent=False,
    tol=1e-9,
):
    """
    Return True when two finite tangent segments intersect in a way that
    creates a self-intersection of the tangent chain.

    For adjacent tangents, one common endpoint is allowed when the outgoing
    tangency point of the first tangent coincides with the incoming tangency
    point of the second tangent. Any other point intersection or any finite
    overlap is considered problematic.
    """
    def geometry_from_tangent(tangent):
        p0 = np.asarray(
            tangent.start_point,
            dtype=float,
        )
        p1 = np.asarray(
            tangent.end_point,
            dtype=float,
        )

        if np.linalg.norm(p1 - p0) <= tol:
            return Point(
                p0[0],
                p0[1],
            )

        return LineString([
            tuple(p0),
            tuple(p1),
        ])

    geometry_1 = geometry_from_tangent(
        tangent_1
    )
    geometry_2 = geometry_from_tangent(
        tangent_2
    )

    intersection = geometry_1.intersection(
        geometry_2
    )

    if intersection.is_empty:
        return False

    if adjacent:

        expected_1 = np.asarray(
            tangent_1.end_point,
            dtype=float,
        )
        expected_2 = np.asarray(
            tangent_2.start_point,
            dtype=float,
        )

        if np.linalg.norm(
            expected_1 - expected_2
        ) <= tol:

            expected_point = 0.5 * (
                expected_1 + expected_2
            )

            allowed_region = Point(
                expected_point[0],
                expected_point[1],
            ).buffer(
                max(tol, 1e-12)
            )

            if allowed_region.covers(
                intersection
            ):
                return False

    return True


def find_first_problematic_tangent_intersection(
    tangents,
    tol=1e-9,
):
    """
    Find the first pair of tangent segments that produces a self-intersection.

    The pair need not be consecutive. If tangent i intersects tangent j, the
    affected circle block runs from active circle position i through active
    circle position j + 1.

    :return:
        (i, j), or None when the tangent chain has no segment intersections.
    """
    for i in range(len(tangents)):

        for j in range(i + 1, len(tangents)):

            if tangent_segments_have_problematic_intersection(
                tangent_1=tangents[i],
                tangent_2=tangents[j],
                adjacent=(j == i + 1),
                tol=tol,
            ):
                return (
                    i,
                    j,
                )

    return None


def find_longest_safe_skip_in_block(
    active_circle_indices,
    circle_groups,
    corridor_list,
    r,
    block_start_position,
    block_end_position,
    safe_union_cache=None,
    tol=1e-9,
):
    """
    Search a problematic active-circle block for the longest safe direct
    tangent that skips at least one intermediate circle.

    Candidate skips are tested from largest to smallest number of removed
    circles. The first collision-free direct connection is returned.

    :return:
        (start_position, end_position, tangent), or None.
    """
    if (
        block_start_position < 0
        or block_end_position >= len(active_circle_indices)
        or block_start_position >= block_end_position
    ):
        raise ValueError(
            "Invalid active-circle block."
        )

    maximum_span = (
        block_end_position
        - block_start_position
    )

    # span >= 2 means that at least one intermediate circle is skipped.
    for span in range(
        maximum_span,
        1,
        -1,
    ):

        last_start = (
            block_end_position
            - span
        )

        for start_position in range(
            block_start_position,
            last_start + 1,
        ):

            end_position = (
                start_position + span
            )

            start_index = active_circle_indices[
                start_position
            ]
            end_index = active_circle_indices[
                end_position
            ]

            tangent = compute_safe_circle_tangent_between_indices(
                circle_groups=circle_groups,
                corridor_list=corridor_list,
                start_circle_index=start_index,
                end_circle_index=end_index,
                r=r,
                safe_union_cache=safe_union_cache,
                tol=tol,
            )

            if tangent is not None:
                return (
                    start_position,
                    end_position,
                    tangent,
                )

    return None


def restore_active_circle_block_to_baseline(
    active_circle_indices,
    circle_groups,
    baseline,
    R,
    block_start_position,
    block_end_position,
    tol=1e-9,
):
    """
    Restore every non-baseline circle in an active-circle block to its
    validated baseline center.

    :return:
        (success, changed).
    """
    changed = False

    for position in range(
        block_start_position,
        block_end_position + 1,
    ):

        transition_index = active_circle_indices[
            position
        ]

        circle = get_single_circle(
            circle_groups[transition_index]
        )

        if is_baseline_circle(circle):
            continue

        baseline_circle = compute_baseline_turn_circle(
            baseline=baseline,
            transition_index=transition_index,
            R=R,
            placement_rule="baseline_repair",
            tol=tol,
        )

        if baseline_circle is None:
            return (
                False,
                changed,
            )

        tau = baseline.turn_directions[
            transition_index
        ]

        circle_groups[transition_index] = {
            tau: baseline_circle
        }

        changed = True

    return (
        True,
        changed,
    )


def simplify_tangent_chain(
    circle_groups,
    corridor_list,
    baseline,
    R,
    r,
    active_circle_indices,
    tangents,
    safe_union_cache=None,
    tol=1e-9,
):
    """
    Remove self-intersections from the tangent chain through local circle
    skipping.

    For every detected tangent intersection:

        1. identify the active-circle block involved in the crossing;
        2. try the longest safe direct tangent inside that block, allowing
           one or several intermediate circles to be skipped;
        3. if no safe skip exists, restore all active circles in the block
           to their baseline positions and rebuild the local tangent chain;
        4. repeat until no tangent segments intersect.

    The intermediate circles are not used as an additional "push direction"
    criterion. Once they are skipped, the relevant requirement for the new
    straight connection is directly tested: the ordered tangent must exist and
    its complete finite segment must lie inside the eroded corridor union.

    Note that final circular-arc validation between the retained tangency
    points is a separate step and is not performed here.
    """
    active_circle_indices = list(
        active_circle_indices
    )
    tangents = list(
        tangents
    )

    if safe_union_cache is None:
        safe_union_cache = {}

    while True:

        intersection_pair = (
            find_first_problematic_tangent_intersection(
                tangents=tangents,
                tol=tol,
            )
        )

        if intersection_pair is None:
            return (
                active_circle_indices,
                tangents,
            )

        first_tangent, second_tangent = (
            intersection_pair
        )

        # Tangent i connects active circles i -> i+1. Therefore the
        # crossing between tangent i and tangent j affects circle
        # positions i through j+1.
        block_start_position = first_tangent
        block_end_position = second_tangent + 1

        skip_result = find_longest_safe_skip_in_block(
            active_circle_indices=active_circle_indices,
            circle_groups=circle_groups,
            corridor_list=corridor_list,
            r=r,
            block_start_position=block_start_position,
            block_end_position=block_end_position,
            safe_union_cache=safe_union_cache,
            tol=tol,
        )

        # -----------------------------------------------------------
        # A safe direct tangent exists: remove every active circle
        # strictly between its two endpoint circles.
        # -----------------------------------------------------------
        if skip_result is not None:

            (
                skip_start_position,
                skip_end_position,
                _,
            ) = skip_result

            del active_circle_indices[
                skip_start_position + 1 : skip_end_position
            ]

            tangents = build_and_repair_tangent_chain(
                circle_groups=circle_groups,
                corridor_list=corridor_list,
                baseline=baseline,
                R=R,
                r=r,
                active_circle_indices=active_circle_indices,
                safe_union_cache=safe_union_cache,
                tol=tol,
            )

            if tangents is None:
                return None

            continue

        # -----------------------------------------------------------
        # No safe skip exists. Restore the affected active circles to
        # their baseline positions and rebuild the tangent chain.
        # -----------------------------------------------------------
        success, changed = restore_active_circle_block_to_baseline(
            active_circle_indices=active_circle_indices,
            circle_groups=circle_groups,
            baseline=baseline,
            R=R,
            block_start_position=block_start_position,
            block_end_position=block_end_position,
            tol=tol,
        )

        if not success:
            return None

        # The same intersection persists even though every circle in
        # the affected block is already at its baseline position. At
        # this point the refinement is abandoned in favour of the full
        # validated baseline.
        if not changed:
            return None

        tangents = build_and_repair_tangent_chain(
            circle_groups=circle_groups,
            corridor_list=corridor_list,
            baseline=baseline,
            R=R,
            r=r,
            active_circle_indices=active_circle_indices,
            safe_union_cache=safe_union_cache,
            tol=tol,
        )

        if tangents is None:
            return None


# ===================================================================
# Four standard independent-placement rules
# ===================================================================

def compute_independent_circle_local_position(
    d_x,
    d_y,
    R,
    r,
    tol=1e-9,
):
    """
    Apply the four local intermediate-circle placement rules.

    The cases are evaluated in the following order:

        1. preferred 45-degree placement;
        2. preferred shifted placement;
        3. ordinary 45-degree placement;
        4. ordinary shifted placement.

    :return:
        (x_local, y_local, placement_rule), or None if none
        of the four rules is feasible.
    """
    D = R - r
    S = R + r

    if D < -tol:
        return None

    D = max(
        0.0,
        D,
    )

    q = D / sqrt(2.0)

    # ---------------------------------------------------------------
    # 1. Preferred 45-degree placement
    # ---------------------------------------------------------------
    if (
        d_x >= 2.0 * (S - q) - tol
        and d_y >= 2.0 * (S - q) - tol
    ):
        return (
            q,
            q,
            "preferred_45",
        )

    # ---------------------------------------------------------------
    # 2. Preferred shifted placement
    # ---------------------------------------------------------------
    a_hat = max(
        0.0,
        S - d_x / 2.0,
    )

    b_hat = max(
        0.0,
        S - d_y / 2.0,
    )

    if (
        a_hat**2 + b_hat**2
        <= D**2 + tol
    ):
        return (
            a_hat,
            b_hat,
            "preferred_shifted",
        )

    # ---------------------------------------------------------------
    # 3. Ordinary 45-degree placement
    # ---------------------------------------------------------------
    if (
        d_x >= S - q - tol
        and d_y >= S - q - tol
    ):
        return (
            q,
            q,
            "ordinary_45",
        )

    # ---------------------------------------------------------------
    # 4. Ordinary shifted placement
    # ---------------------------------------------------------------
    a = max(
        0.0,
        S - d_x,
    )

    b = max(
        0.0,
        S - d_y,
    )

    if (
        a**2 + b**2
        <= D**2 + tol
    ):
        return (
            a,
            b,
            "ordinary_shifted",
        )

    return None


# ===================================================================
# Effective local overlap dimensions
# ===================================================================

def compute_effective_junction_dimensions(
    corridor_overlap,
    corner_point,
    local_x_axis,
    local_y_axis,
    tol=1e-9,
):
    """
    Compute the effective local overlap dimensions d_x and d_y measured
    from the selected concave corner.

    The overlap must extend from corner_point along the negative local axes.
    """
    (x_min, x_max), (y_min, y_max) = corridor_overlap

    overlap_corners = [
        np.array([x_min, y_min]),
        np.array([x_min, y_max]),
        np.array([x_max, y_min]),
        np.array([x_max, y_max]),
    ]

    corner_point = np.asarray(
        corner_point,
        dtype=float,
    )

    local_x_axis = np.asarray(
        local_x_axis,
        dtype=float,
    )

    local_y_axis = np.asarray(
        local_y_axis,
        dtype=float,
    )

    local_coordinates = []

    for point in overlap_corners:

        delta = point - corner_point

        x_local = np.dot(
            delta,
            local_x_axis,
        )

        y_local = np.dot(
            delta,
            local_y_axis,
        )

        local_coordinates.append(
            (
                x_local,
                y_local,
            )
        )

    # The selected corner must be the positive local corner
    # of the overlap rectangle.
    if any(
        x_local > tol
        or y_local > tol
        for x_local, y_local
        in local_coordinates
    ):
        return None

    d_x = max(
        -x_local
        for x_local, _
        in local_coordinates
    )

    d_y = max(
        -y_local
        for _, y_local
        in local_coordinates
    )

    return (
        d_x,
        d_y,
    )